from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from gcw.config import CacheEncoderMethod, CacheEncoderQuerySource, CacheProcessorConfig, ModelConfig
from gcw.models.bgt import BGT
from gcw.models.cache_processor_ops import (
    expand_cache_head_indices,
    gather_per_cache_head_scores,
    gather_per_cache_head_kv,
    latest_shared_indices,
    rotate_cache_keys,
    select_heads_from_per_cache_head_scores,
    select_per_cache_head_kv_norm_indices_from_scores,
)
from gcw.models.cache_state import CacheState, StaticCacheState
from gcw.models.transformer import RMSNorm, flash_sdpa
from gcw.models.windowed_cache import WindowedCache

CacheLike = CacheState | StaticCacheState | WindowedCache


EncoderMethod = CacheEncoderMethod
EncoderQuerySource = CacheEncoderQuerySource


class ProcessorAttention(nn.Module):
    def __init__(self, config: CacheProcessorConfig):
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("processor hidden_size must be divisible by num_heads")
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError("processor num_heads must be divisible by num_kv_heads")
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        if config.include_self_head_kv:
            self.forward = self.forward_with_self_head_kv
        else:
            self.forward = self.forward_selected_head_kv

    def forward_selected_head_kv(
        self,
        x: torch.Tensor,
        selected_heads: torch.Tensor,
        return_attn_scores: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, cache_heads, _ = x.shape
        if selected_heads.shape[-1] == 0:
            y = torch.zeros_like(x)
            if return_attn_scores:
                return y, x.new_zeros(batch, cache_heads, dtype=torch.float32)
            return y
        q = self.q_proj(x).view(batch, cache_heads, self.num_heads, self.head_dim)
        selected_x = _gather_cache_heads(x, selected_heads)
        k = self.k_proj(selected_x).view(batch, selected_heads.shape[-1], self.num_kv_heads, self.head_dim)
        v = self.v_proj(selected_x).view(batch, selected_heads.shape[-1], self.num_kv_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k, v = (t.permute(0, 2, 1, 3) for t in (k, v))
        q, k = q.to(v.dtype), k.to(v.dtype)
        y = flash_sdpa(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        y = y.permute(0, 2, 1, 3).contiguous().view(batch, cache_heads, -1)
        y = self.o_proj(y)
        if return_attn_scores:
            return y, y.float().pow(2).sum(dim=-1)
        return y

    def forward_with_self_head_kv(
        self,
        x: torch.Tensor,
        selected_heads: torch.Tensor,
        return_attn_scores: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, cache_heads, _ = x.shape
        selected_count = selected_heads.shape[-1]
        groups = self.num_heads // self.num_kv_heads

        q = self.q_proj(x).view(batch, cache_heads, self.num_kv_heads, groups, self.head_dim)
        k_all = self.k_proj(x).view(batch, cache_heads, self.num_kv_heads, self.head_dim)
        v_all = self.v_proj(x).view(batch, cache_heads, self.num_kv_heads, self.head_dim)
        gather_index = selected_heads[:, :, None, None].expand(-1, -1, self.num_kv_heads, self.head_dim)
        k_sel = torch.gather(k_all, dim=1, index=gather_index)
        v_sel = torch.gather(v_all, dim=1, index=gather_index)

        q, k_sel, k_all = q.to(v_all.dtype), k_sel.to(v_all.dtype), k_all.to(v_all.dtype)
        scale = self.head_dim**-0.5
        selected_logits = torch.einsum("bckgd,bskd->bckgs", q, k_sel) * scale
        self_logits = torch.einsum("bckgd,bckd->bckg", q, k_all)[:, :, :, :, None] * scale
        if selected_count:
            query_heads = torch.arange(cache_heads, device=x.device)
            self_is_selected = (selected_heads[:, None, :] == query_heads[None, :, None]).any(dim=-1)
            self_logits = self_logits.masked_fill(self_is_selected[:, :, None, None, None], float("-inf"))

        logits = torch.cat([selected_logits, self_logits], dim=-1)
        attn = torch.softmax(logits, dim=-1)
        if self.training and self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout, training=True)

        selected_attn, self_attn = attn[..., :selected_count], attn[..., selected_count]
        y = torch.einsum("bckgs,bskd->bckgd", selected_attn, v_sel)
        y = y + self_attn[:, :, :, :, None] * v_all[:, :, :, None, :]
        y = y.contiguous().view(batch, cache_heads, self.num_heads * self.head_dim)
        y = self.o_proj(y)
        if return_attn_scores:
            return y, y.float().pow(2).sum(dim=-1)
        return y


class ProcessorLayer(nn.Module):
    def __init__(self, config: CacheProcessorConfig):
        super().__init__()
        self.attn = ProcessorAttention(config)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.input_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        selected_heads: torch.Tensor,
        return_attn_scores: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attn_out = self.attn(self.input_norm(x), selected_heads, return_attn_scores=return_attn_scores)
        if return_attn_scores:
            attn_out, attn_scores = attn_out
        x = x + attn_out
        h = self.post_norm(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        if return_attn_scores:
            return x, attn_scores
        return x


class PerCacheHeadLinear(nn.Module):
    def __init__(self, num_cache_heads: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_cache_heads, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_cache_heads, out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.einsum("bci,cio->bco", x, self.weight)
        if self.bias is not None:
            y = y + self.bias[None, :, :]
        return y


class QueryAttentionTimeEncoder(nn.Module):
    def __init__(self, input_dim: int, num_cache_heads: int, config: CacheProcessorConfig):
        super().__init__()
        self.input_dim = input_dim
        self.encoder_query_source = config.encoder_query_source
        if self.encoder_query_source == "learned":
            self.query = nn.Parameter(torch.empty(num_cache_heads, input_dim))
        elif self.encoder_query_source == "final_token":
            if config.per_cache_head_projections:
                self.query_proj = PerCacheHeadLinear(num_cache_heads, input_dim, input_dim)
            else:
                self.query_proj = nn.Linear(input_dim, input_dim)
        else:
            raise ValueError(f"unknown encoder_query_source: {self.encoder_query_source}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self._query(x)
        scale = self.input_dim**-0.5
        scores = torch.einsum("bcsi,bci->bcs", x, query.to(x.dtype)) * scale
        weights = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        return torch.einsum("bcs,bcsi->bci", weights, x)

    def _query(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_query_source == "learned":
            return self.query[None, :, :].expand(x.shape[0], -1, -1)
        if self.encoder_query_source == "final_token":
            return self.query_proj(x[:, :, -1, :])
        raise ValueError(f"unsupported encoder_query_source: {self.encoder_query_source}")


class CacheProcessor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_cache_heads: int,
        config: CacheProcessorConfig,
        *,
        encoder_method: EncoderMethod = "linear",
        encoder_seq_len: int | None = None,
        output_dim: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.encoder_method = encoder_method
        self.encoder_query_source = config.encoder_query_source
        self.encoder_seq_len = encoder_seq_len or config.encoder_seq_len or config.process_every + max(config.topk, 0)
        self.in_norm = RMSNorm(input_dim, config.norm_eps)
        if encoder_method == "linear":
            in_features = self.encoder_seq_len * input_dim
        elif encoder_method == "query_attention":
            in_features = input_dim
            self.time_encoder = QueryAttentionTimeEncoder(input_dim, num_cache_heads, config)
        else:
            raise ValueError(f"unknown encoder_method: {encoder_method}")
        if config.per_cache_head_projections:
            self.in_proj = PerCacheHeadLinear(num_cache_heads, in_features, config.hidden_size)
        else:
            self.in_proj = nn.Linear(in_features, config.hidden_size)
        self.layers = nn.ModuleList(ProcessorLayer(config) for _ in range(config.num_layers))
        self.out_norm = RMSNorm(config.hidden_size, config.norm_eps)
        if config.per_cache_head_projections:
            self.out_proj = PerCacheHeadLinear(num_cache_heads, config.hidden_size, self.output_dim)
        else:
            self.out_proj = nn.Linear(config.hidden_size, self.output_dim)
        self.gate = nn.Parameter(torch.full((1,), config.gate_init))
        self._selected_head_overlap_totals: dict[int, torch.Tensor] = {}
        self._selected_head_overlap_counts: dict[int, int] = {}
        self.init_weights()

    def init_weights(self, std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, PerCacheHeadLinear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, QueryAttentionTimeEncoder) and hasattr(module, "query"):
                nn.init.normal_(module.query, mean=0.0, std=std)
        nn.init.constant_(self.gate, self.config.gate_init)
        if self.config.out_proj_init == "zero":
            _zero_projection(self.out_proj)

    def compute_update(self, x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        h = self._encode_time(self.in_norm(x))  # TODO: add a skip connection here?
        for iteration in range(self.config.iterations):
            return_attn_scores = iteration + 1 < self.config.iterations
            layer_scores = []
            for layer in self.layers:
                if return_attn_scores:
                    h, attn_scores = self._run_layer(layer, h, selected_heads, return_attn_scores=True)
                    layer_scores.append(attn_scores)
                else:
                    h = self._run_layer(layer, h, selected_heads, return_attn_scores=False)
            if return_attn_scores:
                head_scores = torch.stack(layer_scores, dim=0).mean(dim=0)
                next_selected_heads = _select_heads_from_head_scores(head_scores, self.config.topk_heads)
                self._record_selected_head_overlap(iteration + 1, selected_heads, next_selected_heads)
                selected_heads = next_selected_heads
        return self.out_proj(self.out_norm(h))

    def _run_layer(
        self,
        layer: ProcessorLayer,
        h: torch.Tensor,
        selected_heads: torch.Tensor,
        *,
        return_attn_scores: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(layer, h, selected_heads, return_attn_scores, use_reentrant=False)
        return layer(h, selected_heads, return_attn_scores=return_attn_scores)

    def pop_iteration_metrics(self) -> dict[str, float]:
        metrics = {
            f"cache_processor/selected_head_overlap_after_iter_{iteration}": float(
                (total / self._selected_head_overlap_counts[iteration]).detach().cpu()
            )
            for iteration in sorted(self._selected_head_overlap_totals)
            for total in (self._selected_head_overlap_totals[iteration],)
            if self._selected_head_overlap_counts[iteration] > 0
        }
        self._selected_head_overlap_totals.clear()
        self._selected_head_overlap_counts.clear()
        return metrics

    def _record_selected_head_overlap(
        self,
        iteration: int,
        selected_heads: torch.Tensor,
        next_selected_heads: torch.Tensor,
    ) -> None:
        if selected_heads.shape[-1] == 0 or next_selected_heads.shape[-1] == 0:
            return
        maintained = (selected_heads[:, :, None] == next_selected_heads[:, None, :]).any(dim=-1)
        overlap = maintained.float().mean().detach()
        if iteration in self._selected_head_overlap_totals:
            self._selected_head_overlap_totals[iteration] = self._selected_head_overlap_totals[iteration] + overlap
            self._selected_head_overlap_counts[iteration] += 1
        else:
            self._selected_head_overlap_totals[iteration] = overlap
            self._selected_head_overlap_counts[iteration] = 1

    def scaled_update(self, x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        delta = self.compute_update(x, selected_heads)
        return delta * self._gate_scale(delta.dtype)

    def forward(self, x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        if x.shape[2] == 0:
            raise ValueError("Longshot cache processor requires at least one selected time position")
        residual = x[:, :, -1, -self.output_dim :]
        return residual + self.scaled_update(x, selected_heads)  # TODO: ablate with self-attention

    def _gate_scale(self, dtype: torch.dtype) -> torch.Tensor:
        if self.config.gate_mode == "sigmoid":
            return torch.sigmoid(self.gate).to(dtype)
        if self.config.gate_mode == "direct":
            return self.gate.to(dtype)
        raise ValueError(f"unsupported cache processor gate_mode: {self.config.gate_mode}")

    def _encode_time(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_method == "linear":
            x = _fit_time_length(x, self.encoder_seq_len)
            return self.in_proj(x.flatten(start_dim=2))
        if self.encoder_method == "query_attention":
            return self.in_proj(self.time_encoder(x))
        raise ValueError(f"unknown encoder_method: {self.encoder_method}")


class GlobalCacheProcessor(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        *,
        encoder_method: EncoderMethod | None = None,
        encoder_seq_len: int | None = None,
    ):
        super().__init__()
        if model_config.cache_processor is None:
            raise ValueError("Longshot requires cache_processor config")
        self.config = model_config.cache_processor
        if self.config.processor_mode != "standard":
            raise ValueError("Longshot supports only cache_processor.processor_mode='standard'")
        if self.config.update_stream not in {"kv", "v"}:
            raise ValueError("Longshot supports only cache_processor.update_stream='kv' or 'v'")
        if self.config.score not in {"attn_l2", "kv_norm"}:
            raise ValueError("Longshot supports only cache_processor.score='attn_l2' or 'kv_norm'")
        encoder_method = encoder_method or self.config.encoder_method
        self.num_layers = model_config.num_layers
        self.num_kv_heads = model_config.num_kv_heads
        self.head_dim = int(model_config.head_dim)
        self.rope_theta = model_config.rope_theta
        self.rotary_style = model_config.rotary_style
        self.num_cache_heads = self.num_layers * self.num_kv_heads
        output_dim = self.head_dim if self.config.update_stream == "v" else 2 * self.head_dim
        self.block = CacheProcessor(
            2 * self.head_dim,
            self.num_cache_heads,
            self.config,
            encoder_method=encoder_method,
            encoder_seq_len=encoder_seq_len or self.config.encoder_seq_len,
            output_dim=output_dim,
        )

    def forward(self, cache: CacheLike, *, select_all: bool = False) -> CacheLike:
        if self.config.score == "kv_norm" and not select_all:
            all_indices = expand_cache_head_indices(cache.all_indices(), self.num_cache_heads)
            merged = gather_per_cache_head_kv(
                cache,
                all_indices,
                unrope_keys=self.config.unrope_keys,
                rope_theta=self.rope_theta,
                rotary_style=self.rotary_style,
            )
            indices, selected_heads = self._select_indices_and_heads(cache, merged, select_all)
            if indices.shape[-1] == 0:
                return cache
            index = indices[..., None].expand(-1, -1, -1, merged.shape[-1])
            merged = torch.gather(merged, dim=2, index=index)
        else:
            indices, selected_heads = self._select_indices_and_heads(cache, None, select_all)
            if indices.shape[-1] == 0:
                return cache
            merged = gather_per_cache_head_kv(
                cache,
                indices,
                unrope_keys=self.config.unrope_keys,
                rope_theta=self.rope_theta,
                rotary_style=self.rotary_style,
            )
        if indices.shape[-1] == 0:
            return cache
        if selected_heads is None:
            selected_heads = _select_heads_from_merged(merged, self.config.topk_heads)
        updated = self.block(merged, selected_heads)
        latest = latest_shared_indices(indices)
        if self.config.update_stream == "v":
            values = _split_merged_values(updated[:, :, None, :], self.num_layers, self.num_kv_heads, self.head_dim)
            return _scatter_latest_values(cache, latest, values)
        keys, values = _split_merged(updated[:, :, None, :], self.num_layers, self.num_kv_heads, self.head_dim)
        if self.config.unrope_keys:
            keys = rotate_cache_keys(keys, latest, theta=self.rope_theta, rotary_style=self.rotary_style)
        return _scatter_latest(cache, latest, keys, values)

    def _select_indices_and_heads(
        self,
        cache: CacheLike,
        merged: torch.Tensor | None,
        select_all: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if select_all:
            return expand_cache_head_indices(cache.all_indices(), self.num_cache_heads), cache.all_heads()
        if self.config.score == "kv_norm":
            if merged is None:
                raise ValueError("merged cache is required for kv_norm selection")
            scores = merged.float().pow(2).sum(dim=-1)
            indices = select_per_cache_head_kv_norm_indices_from_scores(scores, self.config.process_every, self.config.topk)
            selected_scores = gather_per_cache_head_scores(scores, indices)
            selected_heads = select_heads_from_per_cache_head_scores(selected_scores, self.config.topk_heads)
            return indices, selected_heads
        indices = expand_cache_head_indices(cache.select_indices(self.config.process_every, self.config.topk), self.num_cache_heads)
        return indices, None


class Longshot(BGT):
    processor_cls = GlobalCacheProcessor
    variant = "longshot"


def _zero_projection(module: nn.Module) -> None:
    nn.init.zeros_(module.weight)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


def _scatter_latest(cache: CacheLike, indices: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> CacheLike:
    if isinstance(cache, WindowedCache):
        index = indices[None, :, None, :, None].expand_as(keys)
        cache.keys = cache.keys.scatter(3, index, keys.to(cache.keys.dtype))
        cache.values = cache.values.scatter(3, index, values.to(cache.values.dtype))
        return cache
    if isinstance(cache, StaticCacheState):
        for layer_idx in range(cache.num_layers):
            cache.scatter_layer(layer_idx, indices, keys[layer_idx], values[layer_idx])
        return cache
    for layer_idx in range(cache.num_layers):
        cache.scatter_layer(layer_idx, indices, keys[layer_idx], values[layer_idx])
    return cache


def _scatter_latest_values(cache: CacheLike, indices: torch.Tensor, values: torch.Tensor) -> CacheLike:
    if isinstance(cache, WindowedCache):
        index = indices[None, :, None, :, None].expand_as(values)
        cache.values = cache.values.scatter(3, index, values.to(cache.values.dtype))
        return cache
    if isinstance(cache, StaticCacheState):
        for layer_idx in range(cache.num_layers):
            target = cache.values[layer_idx][:, :, : cache.length, :]
            index = indices[:, None, :, None].expand_as(values[layer_idx])
            target.scatter_(2, index, values[layer_idx].to(target.dtype))
        return cache
    for layer_idx in range(cache.num_layers):
        index = indices[:, None, :, None].expand_as(values[layer_idx])
        cache.values[layer_idx] = cache.values[layer_idx].scatter(2, index, values[layer_idx].to(cache.values[layer_idx].dtype))
    return cache


def _split_merged(
    x: torch.Tensor,
    num_layers: int,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, cache_heads, seq, _ = x.shape
    split = x.view(batch, num_layers, num_heads, seq, 2, head_dim)
    keys = split[:, :, :, :, 0].permute(1, 0, 2, 3, 4).contiguous()
    values = split[:, :, :, :, 1].permute(1, 0, 2, 3, 4).contiguous()
    return keys, values


def _split_merged_values(
    x: torch.Tensor,
    num_layers: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    batch, cache_heads, seq, _ = x.shape
    return x.view(batch, num_layers, num_heads, seq, head_dim).permute(1, 0, 2, 3, 4).contiguous()


def _fit_time_length(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    current = x.shape[2]
    if current == seq_len:
        return x
    warnings.warn(
        f"Longshot linear encoder expected {seq_len} selected positions but got {current}; "
        "padding/truncating the selected cache history.",
        stacklevel=2,
    )
    if current > seq_len:
        return x[:, :, -seq_len:, :]
    out = x.new_zeros(x.shape[0], x.shape[1], seq_len, x.shape[3])
    out[:, :, :current, :] = x
    return out


def _gather_cache_heads(x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
    if selected_heads.ndim != 2:
        raise ValueError("selected_heads must have shape (batch, selected_heads)")
    if x.ndim != 3:
        raise ValueError("x must have shape (batch, cache_heads, hidden)")
    index = selected_heads[:, :, None].expand(-1, -1, x.shape[2])
    return torch.gather(x, dim=1, index=index)


def _select_heads_from_merged(x: torch.Tensor, topk: int) -> torch.Tensor:
    batch = x.shape[0]
    if topk <= 0:
        return torch.empty(batch, 0, dtype=torch.long, device=x.device)
    scores = x.float().flatten(start_dim=2).norm(dim=-1)
    k = min(topk, scores.shape[-1])
    _, indices = torch.topk(scores, k=k, dim=-1)
    return indices.sort(dim=-1).values


def _select_heads_from_head_scores(scores: torch.Tensor, topk: int) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must have shape (batch, cache_heads)")
    batch = scores.shape[0]
    if topk <= 0:
        return torch.empty(batch, 0, dtype=torch.long, device=scores.device)
    k = min(topk, scores.shape[-1])
    _, indices = torch.topk(scores.float(), k=k, dim=-1)
    return indices.sort(dim=-1).values
