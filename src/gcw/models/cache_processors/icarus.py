from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from gcw.config import CacheProcessorConfig, FlowProcessorConfig, ModelConfig
from gcw.models.cache_processors.base import CacheProcessorBase
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
    rotary_for_indices,
    rotate_cache_keys,
    select_heads_from_per_cache_head_scores,
    select_per_cache_head_kv_norm_indices_from_scores,
)
from gcw.performance.backends.icarus import (
    IcarusProcessorBackend,
    icarus_processor_backend_metadata,
    resolve_icarus_processor_backend,
    run_icarus_post_selection_region,
)
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.flow import LatentFlow
from gcw.models.backbones.transformer import RMSNorm, apply_rotary, flash_sdpa
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.performance.longshot_scores import (
    chunked_fp32_l2_norm,
    chunked_fp32_squared_l2,
)
from gcw.performance.triton.execution import triton_cache_movement_enabled

CacheLike = CacheState | StaticCacheState | WindowedCache


class ProcessorAttention(nn.Module):
    def __init__(self, config: CacheProcessorConfig):
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                "processor hidden_size must be divisible by num_heads"
            )
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError(
                "processor num_heads must be divisible by num_kv_heads"
            )
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        if config.include_self_head_kv:
            self.forward = self.forward_with_self_head_kv
        else:
            self.forward = self.forward_selected_head_kv

    def forward_selected_head_kv(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        theta: float,
        selected_heads: torch.Tensor,
    ) -> torch.Tensor:
        batch, cache_heads, seq, _ = x.shape
        assert (
            selected_heads.shape[-1] > 0
        ), "selected_heads must include at least one cache head"
        q = self.q_proj(x).view(
            batch, cache_heads, seq, self.num_heads, self.head_dim
        )
        selected_x = _gather_cache_heads(x, selected_heads)
        selected_positions = _gather_cache_head_positions(
            positions, selected_heads
        )
        k = self.k_proj(selected_x).view(
            batch,
            selected_heads.shape[-1],
            seq,
            self.num_kv_heads,
            self.head_dim,
        )
        v = self.v_proj(selected_x).view(
            batch,
            selected_heads.shape[-1],
            seq,
            self.num_kv_heads,
            self.head_dim,
        )
        q_cos, q_sin = rotary_for_indices(
            positions, self.head_dim, theta, q.dtype
        )
        k_cos, k_sin = rotary_for_indices(
            selected_positions, self.head_dim, theta, k.dtype
        )
        q = apply_rotary(q, q_cos, q_sin)
        k = apply_rotary(k, k_cos, k_sin)
        q = q.permute(0, 3, 1, 2, 4).reshape(
            batch, self.num_heads, cache_heads * seq, self.head_dim
        )
        k, v = (
            t.permute(0, 3, 1, 2, 4).reshape(
                batch,
                self.num_kv_heads,
                selected_heads.shape[-1] * seq,
                self.head_dim,
            )
            for t in (k, v)
        )
        q, k = q.to(v.dtype), k.to(v.dtype)
        y = flash_sdpa(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        y = y.view(batch, self.num_heads, cache_heads, seq, self.head_dim)
        y = (
            y.permute(0, 2, 3, 1, 4)
            .contiguous()
            .view(batch, cache_heads, seq, -1)
        )
        return self.o_proj(y)

    def forward_with_self_head_kv(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        theta: float,
        selected_heads: torch.Tensor,
    ) -> torch.Tensor:
        batch, cache_heads, seq, _ = x.shape
        selected_count = selected_heads.shape[-1]
        assert (
            selected_count > 0
        ), "selected_heads must include at least one cache head"
        groups = self.num_heads // self.num_kv_heads

        q = self.q_proj(x).view(
            batch, cache_heads, seq, self.num_kv_heads, groups, self.head_dim
        )
        k_all = self.k_proj(x).view(
            batch, cache_heads, seq, self.num_kv_heads, self.head_dim
        )
        v_all = self.v_proj(x).view(
            batch, cache_heads, seq, self.num_kv_heads, self.head_dim
        )
        q_cos, q_sin = rotary_for_indices(
            positions, self.head_dim, theta, q.dtype
        )
        k_cos, k_sin = rotary_for_indices(
            positions, self.head_dim, theta, k_all.dtype
        )
        q = apply_rotary(q, q_cos.unsqueeze(-2), q_sin.unsqueeze(-2))
        k_all = apply_rotary(k_all, k_cos, k_sin)

        gather_index = selected_heads[:, :, None, None, None].expand(
            -1, -1, seq, self.num_kv_heads, self.head_dim
        )
        k_sel = torch.gather(k_all, dim=1, index=gather_index)
        v_sel = torch.gather(v_all, dim=1, index=gather_index)

        q, k_sel, k_all = (
            q.to(v_all.dtype),
            k_sel.to(v_all.dtype),
            k_all.to(v_all.dtype),
        )
        scale = self.head_dim**-0.5
        selected_logits = (
            torch.einsum("bcnkgd,bsmkd->bcnkgsm", q, k_sel) * scale
        )
        selected_logits = selected_logits.reshape(
            batch,
            cache_heads,
            seq,
            self.num_kv_heads,
            groups,
            selected_count * seq,
        )
        self_logits = torch.einsum("bcnkgd,bcmkd->bcnkgm", q, k_all) * scale
        query_heads = torch.arange(cache_heads, device=x.device)
        self_is_selected = (
            selected_heads[:, None, :] == query_heads[None, :, None]
        ).any(dim=-1)
        self_logits = self_logits.masked_fill(
            self_is_selected[:, :, None, None, None, None], float("-inf")
        )

        logits = torch.cat([selected_logits, self_logits], dim=-1)
        attn = torch.softmax(logits, dim=-1)
        if self.training and self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout, training=True)

        selected_attn = attn[..., : selected_count * seq].reshape(
            batch,
            cache_heads,
            seq,
            self.num_kv_heads,
            groups,
            selected_count,
            seq,
        )
        self_attn = attn[..., selected_count * seq :]
        y = torch.einsum("bcnkgsm,bsmkd->bcnkgd", selected_attn, v_sel)
        y = y + torch.einsum("bcnkgm,bcmkd->bcnkgd", self_attn, v_all)
        y = y.contiguous().view(
            batch, cache_heads, seq, self.num_heads * self.head_dim
        )
        return self.o_proj(y)


class ProcessorLayer(nn.Module):
    def __init__(self, config: CacheProcessorConfig):
        super().__init__()
        self.attn = ProcessorAttention(config)
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.input_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.theta = config.rope_theta

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        selected_heads: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.input_norm(x), positions, self.theta, selected_heads
        )
        h = self.post_norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class PerCacheHeadLinear(nn.Module):
    def __init__(
        self,
        num_cache_heads: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_cache_heads, in_features, out_features)
        )
        self.bias = (
            nn.Parameter(torch.empty(num_cache_heads, out_features))
            if bias
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.einsum("bcsi,cio->bcso", x, self.weight)
        if self.bias is not None:
            y = y + self.bias[None, :, None, :]
        return y


class CacheProcessor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_cache_heads: int,
        config: CacheProcessorConfig,
        *,
        output_dim: int | None = None,
        flow_config: FlowProcessorConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.in_norm = RMSNorm(input_dim, config.norm_eps)
        self.in_proj: nn.Module
        if config.per_cache_head_projections:
            self.in_proj = PerCacheHeadLinear(
                num_cache_heads, input_dim, config.hidden_size
            )
        else:
            self.in_proj = nn.Linear(input_dim, config.hidden_size)
        self.layers = nn.ModuleList(
            ProcessorLayer(config) for _ in range(config.num_layers)
        )
        self.out_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.out_proj: nn.Module
        if config.per_cache_head_projections:
            self.out_proj = PerCacheHeadLinear(
                num_cache_heads, config.hidden_size, self.output_dim
            )
        else:
            self.out_proj = nn.Linear(config.hidden_size, self.output_dim)
        self.gate = nn.Parameter(torch.full((1,), config.gate_init))
        self._gate_value: torch.Tensor | None = None
        self.flow = (
            LatentFlow(config.hidden_size, flow_config)
            if flow_config is not None
            else None
        )
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
        nn.init.constant_(self.gate, self.config.gate_init)
        if self.config.out_proj_init == "zero":
            _zero_projection(self.out_proj)

    def compute_update(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        selected_heads: torch.Tensor,
    ) -> torch.Tensor:
        initial = self.in_proj(self.in_norm(x))
        if self.flow is not None:
            displacement = self.flow(
                initial,
                lambda state, _time: self.out_norm(
                    self._run_layers(state, positions, selected_heads)
                ),
            )
            return self.out_proj(displacement)
        return self.out_proj(
            self.out_norm(self._run_layers(initial, positions, selected_heads))
        )

    def _run_layers(
        self,
        state: torch.Tensor,
        positions: torch.Tensor,
        selected_heads: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            if (
                self.config.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                state = checkpoint(
                    layer, state, positions, selected_heads, use_reentrant=False
                )
            else:
                state = layer(state, positions, selected_heads)
        return state

    def scaled_update(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        selected_heads: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.compute_update(x, positions, selected_heads)
        return delta * self._gate_scale(delta.dtype)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        selected_heads: torch.Tensor,
    ) -> torch.Tensor:
        residual = x[..., -self.output_dim :]
        return residual + self.scaled_update(x, positions, selected_heads)

    def _gate_scale(self, dtype: torch.dtype) -> torch.Tensor:
        if self.config.gate_mode == "sigmoid":
            scale = torch.sigmoid(self.gate).to(dtype)
        elif self.config.gate_mode == "direct":
            scale = self.gate.to(dtype)
        else:
            raise ValueError(
                f"unsupported cache processor gate_mode: {self.config.gate_mode}"
            )
        self._gate_value = scale.detach()
        return scale


class GlobalCacheProcessor(CacheProcessorBase):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        model_config.ensure_uniform_stored_cache_shape("Icarus")
        if model_config.cache_processor is None:
            raise ValueError("Icarus requires cache_processor config")
        self.config = model_config.cache_processor
        self._model_config = model_config
        if self.config.processor_mode != "standard":
            raise ValueError(
                "Icarus supports only cache_processor.processor_mode='standard'"
            )
        if self.config.update_stream not in {"kv", "v"}:
            raise ValueError(
                "Icarus supports only cache_processor.update_stream='kv' or 'v'"
            )
        if self.config.score not in {"attn_l2", "kv_norm"}:
            raise ValueError(
                "Icarus supports only cache_processor.score='attn_l2' or 'kv_norm'"
            )
        self.num_layers = model_config.cache_num_layers
        self.num_kv_heads, self.head_dim = model_config.stored_cache_shape
        self.rope_theta = model_config.rope_theta
        self.rotary_style = model_config.rotary_style
        self.rotary_dim = int(
            self.head_dim * model_config.partial_rotary_factor
        )
        self.num_cache_heads = self.num_layers * self.num_kv_heads
        output_dim = (
            self.head_dim
            if self.config.update_stream == "v"
            else 2 * self.head_dim
        )
        self.block = CacheProcessor(
            2 * self.head_dim,
            self.num_cache_heads,
            self.config,
            output_dim=output_dim,
            flow_config=model_config.flow_processor,
        )
        self.execution_backend: IcarusProcessorBackend = (
            resolve_icarus_processor_backend(model_config)
        )

    def execution_backend_metadata(self) -> dict[str, object]:
        return icarus_processor_backend_metadata(
            self._model_config, self.execution_backend
        )

    def forward(
        self, cache: CacheLike, *, select_all: bool = False
    ) -> CacheLike:
        if self.config.score == "kv_norm" and not select_all:
            all_indices = expand_cache_head_indices(
                cache.all_indices(), self.num_cache_heads
            )
            merged = gather_per_cache_head_kv(
                cache,
                all_indices,
                unrope_keys=self.config.unrope_keys,
                rope_theta=self.rope_theta,
                rotary_style=self.rotary_style,
                rotary_dim=self.rotary_dim,
                backend=self._cache_movement_backend(),
            )
            indices, selected_heads = self._select_indices_and_heads(
                cache, merged, select_all
            )
            if indices.shape[-1] == 0:
                return cache
            index = indices[..., None].expand(-1, -1, -1, merged.shape[-1])
            merged = torch.gather(merged, dim=2, index=index)
        else:
            indices, selected_heads = self._select_indices_and_heads(
                cache, None, select_all
            )
            if indices.shape[-1] == 0:
                return cache
            merged = gather_per_cache_head_kv(
                cache,
                indices,
                unrope_keys=self.config.unrope_keys,
                rope_theta=self.rope_theta,
                rotary_style=self.rotary_style,
                rotary_dim=self.rotary_dim,
                backend=self._cache_movement_backend(),
            )
        if indices.shape[-1] == 0:
            return cache
        if selected_heads is None:
            selected_heads = _select_heads_from_merged(
                merged, self.config.topk_heads
            )
        updated = run_icarus_post_selection_region(
            self.execution_backend,
            self.block,
            merged,
            indices,
            selected_heads,
        )
        latest = latest_shared_indices(indices)
        latest_update = updated[:, :, -1:].contiguous()
        if self.config.update_stream == "v":
            values = _split_merged_values(
                latest_update, self.num_layers, self.num_kv_heads, self.head_dim
            )
            return _scatter_latest_values(cache, latest, values)
        keys, values = _split_merged(
            latest_update, self.num_layers, self.num_kv_heads, self.head_dim
        )
        if self.config.unrope_keys:
            keys = rotate_cache_keys(
                keys,
                latest,
                theta=self.rope_theta,
                rotary_style=self.rotary_style,
                rotary_dim=self.rotary_dim,
            )
        return _scatter_latest(cache, latest, keys, values)

    def _cache_movement_backend(self) -> str:
        return "triton" if triton_cache_movement_enabled(self) else "eager"

    def _select_indices_and_heads(
        self,
        cache: CacheLike,
        merged: torch.Tensor | None,
        select_all: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if select_all:
            return (
                expand_cache_head_indices(
                    cache.all_indices(), self.num_cache_heads
                ),
                cache.all_heads(),
            )
        if self.config.score == "kv_norm":
            if merged is None:
                raise ValueError(
                    "merged cache is required for kv_norm selection"
                )
            scores = chunked_fp32_squared_l2(merged)
            indices = select_per_cache_head_kv_norm_indices_from_scores(
                scores, self.config.process_every, self.config.topk
            )
            selected_scores = gather_per_cache_head_scores(scores, indices)
            selected_heads = select_heads_from_per_cache_head_scores(
                selected_scores, self.config.topk_heads
            )
            return indices, selected_heads
        indices = expand_cache_head_indices(
            cache.select_indices(self.config.process_every, self.config.topk),
            self.num_cache_heads,
        )
        return indices, None


def _zero_projection(module: nn.Module) -> None:
    zero_projection(module, per_head_type=PerCacheHeadLinear)


def _gather_cache_heads(
    x: torch.Tensor, selected_heads: torch.Tensor
) -> torch.Tensor:
    if selected_heads.ndim != 2:
        raise ValueError(
            "selected_heads must have shape (batch, selected_heads)"
        )
    if x.ndim != 4:
        raise ValueError("x must have shape (batch, cache_heads, seq, hidden)")
    index = selected_heads[:, :, None, None].expand(
        -1, -1, x.shape[2], x.shape[3]
    )
    return torch.gather(x, dim=1, index=index)


def _gather_cache_head_positions(
    positions: torch.Tensor, selected_heads: torch.Tensor
) -> torch.Tensor:
    if positions.ndim != 3:
        raise ValueError("positions must have shape (batch, cache_heads, seq)")
    if selected_heads.ndim != 2:
        raise ValueError(
            "selected_heads must have shape (batch, selected_heads)"
        )
    index = selected_heads[:, :, None].expand(-1, -1, positions.shape[2])
    return torch.gather(positions, dim=1, index=index)


def _select_heads_from_merged(x: torch.Tensor, topk: int) -> torch.Tensor:
    assert topk > 0, "topk must select at least one cache head"
    scores = chunked_fp32_l2_norm(x)
    k = min(topk, scores.shape[-1])
    _, indices = torch.topk(scores, k=k, dim=-1)
    return indices.sort(dim=-1).values
