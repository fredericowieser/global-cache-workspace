from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import warnings
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from gcw.config import (
    CacheEncoderMethod,
    CacheEncoderQuerySource,
    CacheProcessorConfig,
    FlowProcessorConfig,
    ModelConfig,
    WorkspaceFlowRefinementConfig,
    linear_attention_following_cache_indices,
)
from gcw.models.cache_processors.base import CacheProcessorBase, cache_processor_recomputation_id
from gcw.models.cache_processors.ops import (
    scatter_latest as _scatter_latest,
    scatter_latest_values as _scatter_latest_values,
    split_merged_cache as _split_merged,
    split_merged_cache_values as _split_merged_values,
    zero_projection,
    expand_cache_head_indices,
    gather_per_cache_head_scores,
    gather_per_cache_head_kv,
    latest_shared_indices,
    rotate_cache_keys,
    select_per_cache_head_kv_norm_indices_from_scores,
)
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.flow import LatentFlow, sinusoidal_time_embedding
from gcw.models.backbones.transformer import RMSNorm, flash_sdpa
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.performance.longshot_scores import (
    chunked_fp32_l2_norm,
    chunked_fp32_squared_l2,
)

CacheLike = CacheState | StaticCacheState | WindowedCache


EncoderMethod = CacheEncoderMethod
EncoderQuerySource = CacheEncoderQuerySource

_GRADIENT_CHECKPOINTING_OVERRIDE: ContextVar[bool | None] = ContextVar(
    "gcw_longshot_gradient_checkpointing_override",
    default=None,
)


@contextmanager
def longshot_gradient_checkpointing_override(enabled: bool) -> Iterator[None]:
    """Override Longshot processor-layer checkpointing for one forward context."""
    token = _GRADIENT_CHECKPOINTING_OVERRIDE.set(enabled)
    try:
        yield
    finally:
        _GRADIENT_CHECKPOINTING_OVERRIDE.reset(token)


def _gradient_checkpointing_enabled(configured: bool) -> bool:
    override = _GRADIENT_CHECKPOINTING_OVERRIDE.get()
    return configured if override is None else override


@dataclass(frozen=True)
class LongshotWorkspaceSource:
    merged: torch.Tensor
    indices: torch.Tensor
    selected_heads: torch.Tensor
    latest_indices: torch.Tensor


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
        self.internal_iteration_score = config.internal_iteration_score
        self.include_self_head_kv = config.include_self_head_kv

    def forward(
        self,
        x: torch.Tensor,
        selected_heads: torch.Tensor,
        return_attn_scores: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.include_self_head_kv:
            return self.forward_with_self_head_kv(x, selected_heads, return_attn_scores)
        return self.forward_selected_head_kv(x, selected_heads, return_attn_scores)

    def forward_with_iteration_scores(
        self,
        x: torch.Tensor,
        selected_heads: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.internal_iteration_score == "cache_head_attention":
            return self.forward_with_cache_head_attention_scores(
                x,
                selected_heads,
                include_self_head_kv=self.include_self_head_kv,
            )
        result = self.forward(x, selected_heads, return_attn_scores=True)
        if not isinstance(result, tuple):
            raise TypeError("iteration scoring requires attention scores")
        return result

    def forward_selected_head_kv(
        self,
        x: torch.Tensor,
        selected_heads: torch.Tensor,
        return_attn_scores: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, cache_heads, _ = x.shape
        score_kv_norm = return_attn_scores and self.internal_iteration_score == "kv_norm"
        assert selected_heads.shape[-1] > 0, "selected_heads must include at least one cache head"
        q = self.q_proj(x).view(batch, cache_heads, self.num_heads, self.head_dim)
        if score_kv_norm:
            k_all = self.k_proj(x).view(batch, cache_heads, self.num_kv_heads, self.head_dim)
            v_all = self.v_proj(x).view(batch, cache_heads, self.num_kv_heads, self.head_dim)
            iteration_scores = k_all.float().pow(2).sum(dim=(2, 3)) + v_all.float().pow(2).sum(dim=(2, 3))
            gather_index = selected_heads[:, :, None, None].expand(-1, -1, self.num_kv_heads, self.head_dim)
            k = torch.gather(k_all, dim=1, index=gather_index)
            v = torch.gather(v_all, dim=1, index=gather_index)
        else:
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
            if score_kv_norm:
                return y, iteration_scores
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
        assert selected_count > 0, "selected_heads must include at least one cache head"
        groups = self.num_heads // self.num_kv_heads

        q = self.q_proj(x).view(batch, cache_heads, self.num_kv_heads, groups, self.head_dim)
        k_all = self.k_proj(x).view(batch, cache_heads, self.num_kv_heads, self.head_dim)
        v_all = self.v_proj(x).view(batch, cache_heads, self.num_kv_heads, self.head_dim)
        if return_attn_scores and self.internal_iteration_score == "kv_norm":
            iteration_scores = k_all.float().pow(2).sum(dim=(2, 3)) + v_all.float().pow(2).sum(dim=(2, 3))
        gather_index = selected_heads[:, :, None, None].expand(-1, -1, self.num_kv_heads, self.head_dim)
        k_sel = torch.gather(k_all, dim=1, index=gather_index)
        v_sel = torch.gather(v_all, dim=1, index=gather_index)

        q, k_sel, k_all = q.to(v_all.dtype), k_sel.to(v_all.dtype), k_all.to(v_all.dtype)
        scale = self.head_dim**-0.5
        selected_logits = torch.einsum("bckgd,bskd->bckgs", q, k_sel) * scale
        self_logits = torch.einsum("bckgd,bckd->bckg", q, k_all)[:, :, :, :, None] * scale
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
            if self.internal_iteration_score == "kv_norm":
                return y, iteration_scores
            return y, y.float().pow(2).sum(dim=-1)
        return y

    def forward_with_cache_head_attention_scores(
        self,
        x: torch.Tensor,
        selected_heads: torch.Tensor,
        *,
        include_self_head_kv: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the attention output plus all-key attention mass per cache head."""
        batch, cache_heads, _ = x.shape
        selected_count = selected_heads.shape[-1]
        assert selected_count > 0, "selected_heads must include at least one cache head"
        groups = self.num_heads // self.num_kv_heads

        q = self.q_proj(x).view(batch, cache_heads, self.num_kv_heads, groups, self.head_dim)
        k_all = self.k_proj(x).view(batch, cache_heads, self.num_kv_heads, self.head_dim)
        v_all = self.v_proj(x).view(batch, cache_heads, self.num_kv_heads, self.head_dim)

        q, k_all = q.to(v_all.dtype), k_all.to(v_all.dtype)
        scale = self.head_dim**-0.5
        logits = torch.einsum("bckgd,bskd->bckgs", q, k_all) * scale
        all_attn = torch.softmax(logits.float(), dim=-1)
        cache_head_attention_scores = all_attn.float().sum(dim=(1, 2, 3))

        logit_index = selected_heads[:, None, None, None, :].expand(
            -1,
            cache_heads,
            self.num_kv_heads,
            groups,
            -1,
        )
        selected_logits = torch.gather(logits, dim=-1, index=logit_index)
        value_index = selected_heads[:, :, None, None].expand(-1, -1, self.num_kv_heads, self.head_dim)
        v_sel = torch.gather(v_all, dim=1, index=value_index)

        if include_self_head_kv:
            query_heads = torch.arange(cache_heads, device=x.device)
            self_index = query_heads[None, :, None, None, None].expand(
                batch,
                -1,
                self.num_kv_heads,
                groups,
                1,
            )
            self_logits = torch.gather(logits, dim=-1, index=self_index)
            self_is_selected = (selected_heads[:, None, :] == query_heads[None, :, None]).any(dim=-1)
            self_logits = self_logits.masked_fill(self_is_selected[:, :, None, None, None], float("-inf"))
            output_logits = torch.cat([selected_logits, self_logits], dim=-1)
        else:
            output_logits = selected_logits

        output_attn = torch.softmax(output_logits, dim=-1)
        if self.training and self.dropout > 0.0:
            output_attn = F.dropout(output_attn, p=self.dropout, training=True)

        selected_attn = output_attn[..., :selected_count]
        y = torch.einsum("bckgs,bskd->bckgd", selected_attn, v_sel)
        if include_self_head_kv:
            self_attn = output_attn[..., selected_count]
            y = y + self_attn[:, :, :, :, None] * v_all[:, :, :, None, :]
        y = y.contiguous().view(batch, cache_heads, self.num_heads * self.head_dim)
        return self.o_proj(y), cache_head_attention_scores


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
        h = self.input_norm(x)
        if return_attn_scores:
            attn_out, attn_scores = self.attn.forward_with_iteration_scores(h, selected_heads)
        else:
            result = self.attn(h, selected_heads)
            if not isinstance(result, torch.Tensor):
                raise TypeError("attention returned scores when none were requested")
            attn_out = result
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
            self.query_proj: nn.Module
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
        flow_config: FlowProcessorConfig | None = None,
        kv_cache_heads: int | None = None,
        model_config: ModelConfig | None = None,
        linear_state_specs: list["LinearStateSpec"] | None = None,
    ):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.encoder_method = encoder_method
        self.encoder_query_source = config.encoder_query_source
        self.encoder_seq_len = encoder_seq_len or config.encoder_seq_len or config.process_every + max(config.topk, 0)
        self.kv_cache_heads = kv_cache_heads or num_cache_heads
        self.in_norm = RMSNorm(input_dim, config.norm_eps)
        if encoder_method == "linear":
            in_features = self.encoder_seq_len * input_dim
        elif encoder_method == "query_attention":
            in_features = input_dim
            self.time_encoder = QueryAttentionTimeEncoder(input_dim, self.kv_cache_heads, config)
        else:
            raise ValueError(f"unknown encoder_method: {encoder_method}")
        self.in_proj: nn.Module
        if config.per_cache_head_projections:
            self.in_proj = PerCacheHeadLinear(self.kv_cache_heads, in_features, config.hidden_size)
        else:
            self.in_proj = nn.Linear(in_features, config.hidden_size)
        self.layers = nn.ModuleList(ProcessorLayer(config) for _ in range(config.num_layers))
        self.out_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.out_proj: nn.Module
        if config.per_cache_head_projections:
            self.out_proj = PerCacheHeadLinear(self.kv_cache_heads, config.hidden_size, self.output_dim)
        else:
            self.out_proj = nn.Linear(config.hidden_size, self.output_dim)
        self.gate = nn.Parameter(torch.full((1,), config.gate_init))
        self.flow = LatentFlow(config.hidden_size, flow_config) if flow_config is not None else None
        self._selected_head_overlap_totals: dict[int, torch.Tensor] = {}
        self._selected_head_overlap_counts: dict[int, int] = {}
        self.linear_state_adapter: LinearStateAdapter | None
        self.init_weights()
        if linear_state_specs:
            if model_config is None:
                raise ValueError("linear_state_specs require model_config")
            self.linear_state_adapter = LinearStateAdapter(model_config, config, linear_state_specs)
        else:
            self.linear_state_adapter = None

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

    def encode_input(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode_time(self.in_norm(x))  # TODO: add a skip connection here?

    def process_hidden(
        self,
        h: torch.Tensor,
        selected_heads: torch.Tensor,
        *,
        topk_heads: int | None = None,
        kv_cache_heads: int | None = None,
        topk_linear_heads: int | None = None,
    ) -> torch.Tensor:
        if self.flow is not None:
            return self.flow(
                h,
                lambda state, _time: self.out_norm(self._run_flow_field(state, selected_heads)),
            )
        topk_heads = self.config.topk_heads if topk_heads is None else topk_heads
        iterations = self.config.iterations if self.config.iteration_mode == "internal" else 1
        for iteration in range(iterations):
            return_attn_scores = iteration + 1 < iterations
            layer_scores = []
            for module in self.layers:
                layer = cast(ProcessorLayer, module)
                result = self._run_layer(layer, h, selected_heads, return_attn_scores=return_attn_scores)
                if return_attn_scores:
                    if not isinstance(result, tuple):
                        raise TypeError("processor layer did not return iteration scores")
                    h, attn_scores = result
                    layer_scores.append(attn_scores)
                else:
                    if not isinstance(result, torch.Tensor):
                        raise TypeError("processor layer returned unexpected iteration scores")
                    h = result
            if return_attn_scores:
                head_scores = torch.stack(layer_scores, dim=0).mean(dim=0)
                if kv_cache_heads is None or topk_linear_heads is None:
                    next_selected_heads = _select_heads_from_head_scores(head_scores, topk_heads)
                else:
                    kv_selected = _select_heads_from_head_scores(
                        head_scores[:, :kv_cache_heads],
                        topk_heads,
                    )
                    if topk_linear_heads > 0:
                        linear_selected = _select_heads_from_head_scores(
                            head_scores[:, kv_cache_heads:],
                            topk_linear_heads,
                        ) + kv_cache_heads
                        next_selected_heads = torch.cat([kv_selected, linear_selected], dim=-1)
                    else:
                        next_selected_heads = kv_selected
                self.record_selected_head_overlap(iteration + 1, selected_heads, next_selected_heads)
                selected_heads = next_selected_heads
        return self.out_norm(h)

    def project_output(self, h: torch.Tensor) -> torch.Tensor:
        return self.out_proj(h)

    def compute_update(self, x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        return self.project_output(self.process_hidden(self.encode_input(x), selected_heads))

    def has_linear_state_adapter(self) -> bool:
        return self.linear_state_adapter is not None

    def linear_state_query(self, layer_idx: int, following_cache_idx: int, context: torch.Tensor) -> torch.Tensor:
        if self.linear_state_adapter is None:
            raise ValueError("linear state adapter is not configured")
        return self.linear_state_adapter.query(layer_idx, following_cache_idx, context)

    def linear_state_query_cache_key(self, layer_idx: int, following_cache_idx: int) -> str:
        if self.linear_state_adapter is None:
            raise ValueError("linear state adapter is not configured")
        return self.linear_state_adapter.query_cache_key(layer_idx, following_cache_idx)

    def linear_state_update_dims(self) -> tuple[int, int]:
        if self.linear_state_adapter is None:
            raise ValueError("linear state adapter is not configured")
        return self.linear_state_adapter.linear_key_dim, self.linear_state_adapter.linear_value_dim

    def forward_with_linear_state_features(
        self,
        x: torch.Tensor,
        kv_hidden: torch.Tensor,
        linear_features: torch.Tensor,
        selected_heads: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.linear_state_adapter is None:
            raise ValueError("linear state adapter is not configured")
        if x.shape[2] == 0:
            raise ValueError("Longshot cache processor requires at least one selected time position")
        hidden = torch.cat([kv_hidden, self.linear_state_adapter.encode(linear_features)], dim=1)
        processed = self.process_hidden(
            hidden,
            selected_heads,
            topk_heads=self.config.topk_heads,
            kv_cache_heads=kv_hidden.shape[1],
            topk_linear_heads=_topk_linear_heads(self.config),
        )
        gate = self._gate_scale(processed.dtype)
        primary_delta = self.project_output(processed[:, : x.shape[1]]) * gate
        auxiliary_delta = self.linear_state_adapter.output_proj(processed[:, x.shape[1] :])
        if self.config.gate_linear_state_delta:
            auxiliary_delta = auxiliary_delta * gate
        residual = x[:, :, -1, -self.output_dim :]
        return residual + primary_delta, auxiliary_delta

    def _run_flow_field(self, state: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        for module in self.layers:
            layer = cast(ProcessorLayer, module)
            result = self._run_layer(layer, state, selected_heads, return_attn_scores=False)
            if not isinstance(result, torch.Tensor):
                raise TypeError("flow field unexpectedly returned iteration scores")
            state = result
        return state

    def _run_layer(
        self,
        layer: ProcessorLayer,
        h: torch.Tensor,
        selected_heads: torch.Tensor,
        *,
        return_attn_scores: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if _gradient_checkpointing_enabled(self.config.gradient_checkpointing) and self.training and torch.is_grad_enabled():
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

    def record_selected_head_overlap(
        self,
        iteration: int,
        selected_heads: torch.Tensor,
        next_selected_heads: torch.Tensor,
    ) -> None:
        if cache_processor_recomputation_id() is not None:
            return
        assert selected_heads.shape[-1] > 0, "selected_heads must include at least one cache head"
        assert next_selected_heads.shape[-1] > 0, "next_selected_heads must include at least one cache head"
        maintained = (selected_heads[:, :, None] == next_selected_heads[:, None, :]).any(dim=-1)
        overlap = maintained.float().mean().detach()
        if iteration in self._selected_head_overlap_totals:
            self._selected_head_overlap_totals[iteration] = self._selected_head_overlap_totals[iteration] + overlap
            self._selected_head_overlap_counts[iteration] += 1
        else:
            self._selected_head_overlap_totals[iteration] = overlap
            self._selected_head_overlap_counts[iteration] = 1

    def scaled_update(self, x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        delta = self.project_output(self.process_hidden(self.encode_input(x), selected_heads))
        return delta * self._gate_scale(delta.dtype)

    def forward_from_hidden(self, x: torch.Tensor, hidden: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        if x.shape[2] == 0:
            raise ValueError("Longshot cache processor requires at least one selected time position")
        residual = x[:, :, -1, -self.output_dim :]
        delta = self.project_output(self.process_hidden(hidden, selected_heads))
        return residual + delta * self._gate_scale(delta.dtype)

    def forward(self, x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        return self.forward_from_hidden(x, self.encode_input(x), selected_heads)  # TODO: ablate with self-attention

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


@dataclass(frozen=True)
class LinearStateSpec:
    layer_idx: int
    following_cache_idx: int
    head_count: int


@dataclass
class LinearStateInputs:
    features: torch.Tensor
    specs: list[LinearStateSpec]


class LinearStateAdapter(nn.Module):
    def __init__(self, model_config: ModelConfig, processor_config: CacheProcessorConfig, specs: list[LinearStateSpec]):
        super().__init__()
        self.query_mode = processor_config.linear_state_query_mode
        self.num_kv_heads = model_config.num_kv_heads
        if model_config.head_dim is None:
            raise ValueError("linear state adapter requires a resolved model head_dim")
        self.full_head_dim = model_config.head_dim
        self.linear_key_dim = int(model_config.linear_key_head_dim or self.full_head_dim)
        self.linear_value_dim = int(model_config.linear_value_head_dim or self.linear_key_dim)
        self.linear_value_heads = int(model_config.linear_num_value_heads or model_config.num_heads)
        self.context_dim = processor_config.hidden_size
        self.feature_dim = self.linear_key_dim + self.linear_value_dim
        self.query_projs = nn.ModuleDict()
        for key in self._query_projection_keys(specs):
            out_dim = self.linear_key_dim if self.query_mode != "per_linear_state" else self.linear_value_heads * self.linear_key_dim
            self.query_projs[key] = nn.Linear(self.context_dim, out_dim)
        self.query_context_norm = RMSNorm(self.context_dim, processor_config.norm_eps)
        self.input_norm = RMSNorm(self.feature_dim, processor_config.norm_eps)
        self.input_proj = nn.Linear(self.feature_dim, processor_config.hidden_size)
        self.output_proj = nn.Linear(processor_config.hidden_size, self.feature_dim)
        self.init_weights(processor_config)

    def init_weights(self, config: CacheProcessorConfig, std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        if config.out_proj_init == "zero":
            _zero_projection(self.output_proj)

    def query(self, layer_idx: int, following_cache_idx: int, context: torch.Tensor) -> torch.Tensor:
        projected = self.query_projs[self.query_cache_key(layer_idx, following_cache_idx)](self.query_context_norm(context))
        if self.query_mode != "per_linear_state":
            query = projected[:, None, :].expand(-1, self.linear_value_heads, -1)
        else:
            query = projected.view(projected.shape[0], self.linear_value_heads, self.linear_key_dim)
        return F.normalize(query.float(), p=2, dim=-1).to(projected.dtype)

    def query_cache_key(self, layer_idx: int, following_cache_idx: int) -> str:
        if self.query_mode == "per_linear_state":
            return f"layer:{layer_idx}"
        if self.query_mode == "shared":
            return f"cache:{following_cache_idx}"
        if self.query_mode == "global":
            return "global"
        raise ValueError(f"unsupported linear_state_query_mode: {self.query_mode}")

    def _query_projection_keys(self, specs: list[LinearStateSpec]) -> list[str]:
        return sorted({self.query_cache_key(spec.layer_idx, spec.following_cache_idx) for spec in specs})

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.input_proj(self.input_norm(features.to(self.input_proj.weight.dtype)))


class GlobalCacheProcessor(CacheProcessorBase):
    def __init__(
        self,
        model_config: ModelConfig,
        *,
        encoder_method: EncoderMethod | None = None,
        encoder_seq_len: int | None = None,
    ):
        super().__init__()
        model_config.ensure_uniform_stored_cache_shape("Longshot")
        if model_config.cache_processor is None:
            raise ValueError("Longshot requires cache_processor config")
        self.config = model_config.cache_processor
        self.workspace_flow_refinement = model_config.workspace_flow_refinement
        if self.config.processor_mode != "standard":
            raise ValueError("Longshot supports only cache_processor.processor_mode='standard'")
        if self.config.update_stream not in {"kv", "v"}:
            raise ValueError("Longshot supports only cache_processor.update_stream='kv' or 'v'")
        if self.config.score not in {"attn_l2", "kv_norm"}:
            raise ValueError("Longshot supports only cache_processor.score='attn_l2' or 'kv_norm'")
        encoder_method = encoder_method or self.config.encoder_method
        self.num_layers = model_config.cache_num_layers
        self.num_kv_heads, self.head_dim = model_config.stored_cache_shape
        self.rope_theta = model_config.rope_theta
        self.rotary_style = model_config.rotary_style
        self.rotary_dim = int(self.head_dim * model_config.partial_rotary_factor)
        self.num_kv_cache_heads = self.num_layers * self.num_kv_heads
        self.linear_state_specs = self._build_linear_state_specs(model_config)
        self.num_linear_state_heads = sum(spec.head_count for spec in self.linear_state_specs)
        self.num_cache_heads = self.num_kv_cache_heads + self.num_linear_state_heads
        output_dim = self.head_dim if self.config.update_stream == "v" else 2 * self.head_dim
        self.block = CacheProcessor(
            2 * self.head_dim,
            self.num_cache_heads,
            self.config,
            encoder_method=encoder_method,
            encoder_seq_len=encoder_seq_len or self.config.encoder_seq_len,
            output_dim=output_dim,
            flow_config=model_config.flow_processor,
            kv_cache_heads=self.num_kv_cache_heads,
            model_config=model_config,
            linear_state_specs=self.linear_state_specs,
        )

    def _build_linear_state_specs(self, model_config: ModelConfig) -> list[LinearStateSpec]:
        if not self.config.linear_state_cache:
            return []
        linear_value_heads = int(model_config.linear_num_value_heads or model_config.num_heads)
        return [
            LinearStateSpec(
                layer_idx=layer_idx,
                following_cache_idx=following_cache_idx,
                head_count=linear_value_heads,
            )
            for layer_idx, following_cache_idx in linear_attention_following_cache_indices(
                model_config,
                require_layer_types=True,
            )
        ]

    def forward(self, cache: CacheLike, *, select_all: bool = False) -> CacheLike:
        if self.config.iteration_mode == "external":
            previous_selected_heads = None
            for iteration in range(1, self.config.iterations + 1):
                cache, selected_heads = self._forward_once(cache, select_all=select_all)
                if previous_selected_heads is not None and selected_heads is not None:
                    self.block.record_selected_head_overlap(iteration - 1, previous_selected_heads, selected_heads)
                previous_selected_heads = selected_heads
            return cache
        cache, _ = self._forward_once(cache, select_all=select_all)
        return cache

    def encode_source(self, cache: CacheLike, *, select_all: bool = False) -> LongshotWorkspaceSource:
        if self.config.score == "kv_norm" and not select_all:
            all_indices = expand_cache_head_indices(cache.all_indices(), self.num_kv_cache_heads)
            merged_all = gather_per_cache_head_kv(
                cache,
                all_indices,
                unrope_keys=self.config.unrope_keys,
                rope_theta=self.rope_theta,
                rotary_style=self.rotary_style,
                rotary_dim=self.rotary_dim,
            )
            scores = chunked_fp32_squared_l2(merged_all)
            indices = select_per_cache_head_kv_norm_indices_from_scores(
                scores,
                self.config.process_every,
                self.config.topk,
            )
            if indices.shape[-1] == 0:
                raise ValueError("Longshot workspace source requires at least one selected position")
            index = indices[..., None].expand(-1, -1, -1, merged_all.shape[-1])
            merged = torch.gather(merged_all, dim=2, index=index)
            head_scores = gather_per_cache_head_scores(scores, indices).float().sum(dim=-1)
        else:
            if select_all:
                indices = expand_cache_head_indices(cache.all_indices(), self.num_kv_cache_heads)
            else:
                indices = expand_cache_head_indices(
                    cache.select_indices(self.config.process_every, self.config.topk),
                    self.num_kv_cache_heads,
                )
            if indices.shape[-1] == 0:
                raise ValueError("Longshot workspace source requires at least one selected position")
            merged = gather_per_cache_head_kv(
                cache,
                indices,
                unrope_keys=self.config.unrope_keys,
                rope_theta=self.rope_theta,
                rotary_style=self.rotary_style,
                rotary_dim=self.rotary_dim,
            )
            head_scores = chunked_fp32_l2_norm(merged)
        if select_all:
            selected_heads = torch.arange(
                self.num_kv_cache_heads,
                dtype=torch.long,
                device=merged.device,
            ).unsqueeze(0).expand(merged.shape[0], -1)
        else:
            selected_heads = _select_heads_from_head_scores(head_scores, self.config.topk_heads)
        return LongshotWorkspaceSource(
            merged=merged,
            indices=indices,
            selected_heads=selected_heads,
            latest_indices=latest_shared_indices(indices),
        )

    def initialise_workspace(self, source: LongshotWorkspaceSource) -> torch.Tensor:
        return source.merged[:, :, -1, -self.block.output_dim :].clone()

    def _validate_workspace_shape(
        self,
        workspace: torch.Tensor,
        source: LongshotWorkspaceSource,
    ) -> None:
        expected_shape = source.merged[:, :, -1, -self.block.output_dim :].shape
        if workspace.shape != expected_shape:
            raise ValueError(
                f"Longshot workspace shape must be {tuple(expected_shape)}, got {tuple(workspace.shape)}"
            )

    def deliberate(self, workspace: torch.Tensor, source: LongshotWorkspaceSource) -> torch.Tensor:
        self._validate_workspace_shape(workspace, source)
        cycle_input = self._workspace_cycle_input(workspace, source)
        return self.block(cycle_input, source.selected_heads)

    def _workspace_cycle_input(
        self,
        workspace: torch.Tensor,
        source: LongshotWorkspaceSource,
    ) -> torch.Tensor:
        frontier = torch.cat(
            [source.merged[:, :, -1, : -self.block.output_dim], workspace],
            dim=-1,
        )
        return torch.cat([source.merged[:, :, :-1], frontier[:, :, None]], dim=2)

    def refine_workspace(
        self,
        workspace: torch.Tensor,
        source: LongshotWorkspaceSource,
    ) -> torch.Tensor:
        """Integrate the opt-in tied workspace field without changing source state."""
        self._validate_workspace_shape(workspace, source)
        config = self.workspace_flow_refinement
        if config is None:
            return workspace
        state = workspace
        step_size = 1.0 / config.solver_steps
        for step in range(config.solver_steps):
            time = step * step_size
            if config.solver == "euler":
                state = state + step_size * self._workspace_velocity(state, time, source, config)
            else:
                first_velocity = self._workspace_velocity(state, time, source, config)
                midpoint = state + 0.5 * step_size * first_velocity
                midpoint_velocity = self._workspace_velocity(
                    midpoint,
                    time + 0.5 * step_size,
                    source,
                    config,
                )
                state = state + step_size * midpoint_velocity
        return state

    def _workspace_velocity(
        self,
        workspace: torch.Tensor,
        time: float,
        source: LongshotWorkspaceSource,
        config: WorkspaceFlowRefinementConfig,
    ) -> torch.Tensor:
        time_condition = sinusoidal_time_embedding(time, workspace.shape[-1], workspace)
        cycle_input = self._workspace_cycle_input(workspace + time_condition, source)
        raw_velocity = self.block.compute_update(cycle_input, source.selected_heads)
        return config.max_velocity * torch.tanh(raw_velocity)

    def write_for_readout(
        self,
        cache: CacheLike,
        source: LongshotWorkspaceSource,
        workspace: torch.Tensor,
    ) -> WindowedCache:
        if self.config.update_stream != "v":
            raise ValueError("Longshot workspace readout initially supports value-only writes")
        if not isinstance(cache, WindowedCache):
            raise TypeError("Longshot workspace readout requires a WindowedCache")
        if cache.linear_states is not None:
            raise ValueError("Longshot workspace readout does not support shared linear cache state")
        self._validate_workspace_shape(workspace, source)
        workspace = self.refine_workspace(workspace, source)
        branch = WindowedCache(
            keys=cache.keys.clone(),
            values=cache.values.clone(),
            scores=cache.scores.clone(),
            linear_states=cache.linear_states,
        )
        values = _split_merged_values(
            workspace[:, :, None],
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
        )
        return cast(WindowedCache, _scatter_latest_values(branch, source.latest_indices, values))

    def _forward_once(self, cache: CacheLike, *, select_all: bool = False) -> tuple[CacheLike, torch.Tensor | None]:
        if self.config.score == "kv_norm" and not select_all:
            all_indices = expand_cache_head_indices(cache.all_indices(), self.num_kv_cache_heads)
            merged = gather_per_cache_head_kv(
                cache,
                all_indices,
                unrope_keys=self.config.unrope_keys,
                rope_theta=self.rope_theta,
                rotary_style=self.rotary_style,
                rotary_dim=self.rotary_dim,
            )
            scores = chunked_fp32_squared_l2(merged)
            indices = select_per_cache_head_kv_norm_indices_from_scores(
                scores, self.config.process_every, self.config.topk
            )
            if indices.shape[-1] == 0:
                return cache, None
            index = indices[..., None].expand(-1, -1, -1, merged.shape[-1])
            merged = torch.gather(merged, dim=2, index=index)
            kv_head_scores = gather_per_cache_head_scores(scores, indices).float().sum(dim=-1)
        else:
            if select_all:
                indices = expand_cache_head_indices(cache.all_indices(), self.num_kv_cache_heads)
            else:
                indices = expand_cache_head_indices(
                    cache.select_indices(self.config.process_every, self.config.topk), self.num_kv_cache_heads
                )
            if indices.shape[-1] == 0:
                return cache, None
            merged = gather_per_cache_head_kv(
                cache,
                indices,
                unrope_keys=self.config.unrope_keys,
                rope_theta=self.rope_theta,
                rotary_style=self.rotary_style,
                rotary_dim=self.rotary_dim,
            )
            kv_head_scores = chunked_fp32_l2_norm(merged)
        latest = latest_shared_indices(indices)
        kv_hidden = self.block.encode_input(merged) if self.block.has_linear_state_adapter() and cache.linear_states else None
        linear_inputs = self._linear_state_inputs(cache, kv_hidden)
        selected_heads = self._select_combined_heads(
            batch=merged.shape[0],
            kv_head_scores=kv_head_scores,
            linear_inputs=linear_inputs,
            select_all=select_all,
            device=merged.device,
        )
        if linear_inputs is None:
            updated = (
                self.block(merged, selected_heads)
                if kv_hidden is None
                else self.block.forward_from_hidden(merged, kv_hidden, selected_heads)
            )
        else:
            if kv_hidden is None:
                raise RuntimeError("linear state inputs require encoded KV cache features")
            updated, linear_delta = self.block.forward_with_linear_state_features(
                merged,
                kv_hidden,
                linear_inputs.features,
                selected_heads,
            )
            cache = self._scatter_linear_state_updates(cache, linear_delta, linear_inputs.specs)
        if self.config.update_stream == "v":
            values = _split_merged_values(updated[:, :, None, :], self.num_layers, self.num_kv_heads, self.head_dim)
            return _scatter_latest_values(cache, latest, values), selected_heads
        keys, values = _split_merged(updated[:, :, None, :], self.num_layers, self.num_kv_heads, self.head_dim)
        if self.config.unrope_keys:
            keys = rotate_cache_keys(
                keys,
                latest,
                theta=self.rope_theta,
                rotary_style=self.rotary_style,
                rotary_dim=self.rotary_dim,
            )
        return _scatter_latest(cache, latest, keys, values), selected_heads

    def _select_combined_heads(
        self,
        *,
        batch: int,
        kv_head_scores: torch.Tensor,
        linear_inputs: LinearStateInputs | None,
        select_all: bool,
        device: torch.device,
    ) -> torch.Tensor:
        if select_all:
            head_count = self.num_kv_cache_heads + (0 if linear_inputs is None else linear_inputs.features.shape[1])
            return torch.arange(head_count, dtype=torch.long, device=device).unsqueeze(0).expand(batch, -1)
        if linear_inputs is None:
            return _select_heads_from_head_scores(kv_head_scores, self.config.topk_heads)
        selected = [_select_heads_from_head_scores(kv_head_scores, self.config.topk_heads)]
        topk_linear = _topk_linear_heads(self.config)
        if topk_linear > 0:
            key_dim = self.block.linear_state_update_dims()[0]
            linear_head_scores = linear_inputs.features[..., key_dim:].float().pow(2).sum(dim=-1)
            selected.append(_select_heads_from_head_scores(linear_head_scores, topk_linear) + self.num_kv_cache_heads)
        return torch.cat(selected, dim=-1)

    def _linear_state_inputs(self, cache: CacheLike, kv_hidden: torch.Tensor | None) -> LinearStateInputs | None:
        if kv_hidden is None or not cache.linear_states:
            return None
        feature_parts: list[torch.Tensor] = []
        active_specs: list[LinearStateSpec] = []
        context_cache: dict[int, torch.Tensor] = {}
        query_cache: dict[tuple[str, int], torch.Tensor] = {}
        for spec in self.linear_state_specs:
            state = cache.linear_states.get(spec.layer_idx)
            if state is None or state[1] is None:
                continue
            recurrent_state = state[1]
            if spec.following_cache_idx not in context_cache:
                context_cache[spec.following_cache_idx] = self._following_kv_hidden_context(kv_hidden, spec.following_cache_idx)
            query_key = self.block.linear_state_query_cache_key(spec.layer_idx, spec.following_cache_idx)
            cache_key = (query_key, spec.following_cache_idx)
            if cache_key not in query_cache:
                query_cache[cache_key] = self.block.linear_state_query(
                    spec.layer_idx,
                    spec.following_cache_idx,
                    context_cache[spec.following_cache_idx],
                )
            query = query_cache[cache_key].to(recurrent_state.dtype)
            readout = torch.einsum("bhk,bhkv->bhv", query, recurrent_state)
            features = torch.cat([query.to(readout.dtype), readout], dim=-1)
            feature_parts.append(features)
            active_specs.append(spec)
        if not feature_parts:
            return None
        return LinearStateInputs(
            features=torch.cat(feature_parts, dim=1),
            specs=active_specs,
        )

    def _following_kv_hidden_context(self, kv_hidden: torch.Tensor, cache_idx: int) -> torch.Tensor:
        start = cache_idx * self.num_kv_heads
        end = start + self.num_kv_heads
        return kv_hidden[:, start:end].mean(dim=1)

    def _scatter_linear_state_updates(
        self,
        cache: CacheLike,
        linear_delta: torch.Tensor,
        specs: list[LinearStateSpec],
    ) -> CacheLike:
        if not self.block.has_linear_state_adapter() or not specs:
            return cache
        states = {} if cache.linear_states is None else dict(cache.linear_states)
        offset = 0
        key_dim, value_dim = self.block.linear_state_update_dims()
        for spec in specs:
            update = linear_delta[:, offset : offset + spec.head_count]
            offset += spec.head_count
            if spec.layer_idx not in states:
                continue
            conv_state, recurrent_state = states[spec.layer_idx]
            if recurrent_state is None:
                continue
            delta_key, delta_value = torch.split(update, [key_dim, value_dim], dim=-1)
            delta_key = F.normalize(delta_key.float(), p=2, dim=-1).to(delta_value.dtype)
            # TODO(qwen3.5-linear-state): Qwen3.5 uses the gated delta-rule state
            # transition with decay/gating terms. This additive normalized-key
            # outer-product is useful as a learned perturbation, but it is not the
            # exact recurrent update used by the native Qwen3.5 linear layer.
            delta_state = torch.einsum("bhk,bhv->bhkv", delta_key.float(), delta_value.float())
            states[spec.layer_idx] = (conv_state, recurrent_state + delta_state.to(recurrent_state.dtype))
        cache.linear_states = states
        return cache


def _zero_projection(module: nn.Module) -> None:
    zero_projection(module, per_head_type=PerCacheHeadLinear)


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


def _topk_linear_heads(config: CacheProcessorConfig) -> int:
    return config.topk_heads if config.topk_linear_heads is None else config.topk_linear_heads


def _select_heads_from_head_scores(scores: torch.Tensor, topk: int) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must have shape (batch, cache_heads)")
    assert topk > 0, "topk must select at least one cache head"
    k = min(topk, scores.shape[-1])
    _, indices = torch.topk(scores.float(), k=k, dim=-1)
    return indices.sort(dim=-1).values
