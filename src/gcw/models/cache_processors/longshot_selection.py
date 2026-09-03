from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from gcw.models.backbones.transformer import RMSNorm
from gcw.models.cache_processors.ops import (
    expand_cache_head_indices,
    gather_per_cache_head_kv,
    select_per_cache_head_kv_norm_indices_from_scores,
)
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.performance.kv_score_sidecar import (
    kv_score_sidecar_execution,
    kv_score_sidecar_spec,
    refresh_kv_score_sidecar,
    synchronize_kv_score_sidecar,
)
from gcw.performance.longshot_cache import (
    gather_windowed_per_cache_head_kv,
    longshot_full_history_execution,
    record_longshot_full_history_event,
)
from gcw.performance.longshot_scores import (
    chunked_fp32_concatenated_kv_squared_l2,
    chunked_fp32_squared_l2,
    chunked_fp32_unroped_concatenated_kv_squared_l2,
)
from gcw.performance.triton.execution import triton_cache_movement_enabled

# The cache object any selection path accepts: static, mutable, or windowed.
CacheLike = CacheState | StaticCacheState | WindowedCache


class ContextualHeadSelector(nn.Module):
    """Score encoded cache heads against their shared encoded context."""

    def __init__(
        self,
        hidden_size: int,
        selector_hidden_size: int,
        num_cache_heads: int,
        *,
        norm_eps: float,
        use_identity_bias: bool,
    ) -> None:
        super().__init__()
        self.num_cache_heads = num_cache_heads
        self.scale = selector_hidden_size**-0.5
        self.norm = RMSNorm(hidden_size, norm_eps)
        self.context_proj = nn.Linear(
            hidden_size, selector_hidden_size, bias=False
        )
        self.candidate_proj = nn.Linear(
            hidden_size, selector_hidden_size, bias=False
        )
        if use_identity_bias:
            self.identity_bias = nn.Parameter(torch.zeros(num_cache_heads))
        else:
            self.register_parameter("identity_bias", None)

    def forward(
        self, hidden: torch.Tensor, head_indices: torch.Tensor
    ) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(
                "contextual head selector hidden must have shape "
                "(batch, active_heads, hidden_size)"
            )
        if not hidden.shape[1]:
            raise ValueError(
                "contextual head selector requires at least one active head"
            )
        if head_indices.shape != (hidden.shape[1],):
            raise ValueError(
                "contextual head selector head_indices must be one-dimensional "
                "and match the active head count"
            )
        try:
            torch.iinfo(head_indices.dtype)
        except TypeError:
            raise TypeError(
                "contextual head selector head_indices must have an integer dtype"
            ) from None

        normalized = self.norm(hidden)
        context_hidden = self.context_proj(normalized.mean(dim=1)).float()
        candidate_hidden = self.candidate_proj(normalized).float()
        scores = (
            torch.einsum("bd,bhd->bh", context_hidden, candidate_hidden)
            * self.scale
        )
        if self.identity_bias is not None:  # TODO: Why do we need head_indices? Do they ever get shuffled?
            scores = scores + self.identity_bias.float().index_select(
                0, head_indices.to(self.identity_bias.device, torch.long)
            )
        return scores


def _select_heads_from_head_scores(
    head_scores: torch.Tensor, topk_heads: int
) -> torch.Tensor:
    if topk_heads <= 0:
        raise ValueError(
            f"topk_heads must select at least one cache head, got {topk_heads}"
        )
    if topk_heads >= head_scores.shape[-1]:
        return torch.arange(
            head_scores.shape[-1], device=head_scores.device
        ).expand(head_scores.shape[0], -1)
    return torch.topk(
        head_scores, k=topk_heads, dim=-1, largest=True, sorted=True
    ).indices


def select_kv_norm_history(
    processor: Any,
    cache: CacheLike,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sidecar_policy = kv_score_sidecar_execution(processor)
    if sidecar_policy.enabled:
        policy = longshot_full_history_execution(processor)
        if policy.enabled:
            raise RuntimeError(
                "KV-score sidecar cannot run with Longshot full-history execution"
            )
        if not isinstance(cache, WindowedCache):
            raise TypeError(
                "enabled KV-score sidecar requires contiguous WindowedCache storage"
            )
        synchronize_sidecar(processor, cache)
        state = cache.kv_score_sidecar
        if state is None:
            raise RuntimeError(
                "KV-score sidecar bootstrap did not produce state"
            )
        all_indices = expand_cache_head_indices(
            cache.all_indices(), processor.num_kv_cache_heads
        )
        merged_all = gather_per_cache_head_kv(
            cache,
            all_indices,
            unrope_keys=processor.config.unrope_keys,
            rope_theta=processor.rope_theta,
            rotary_style=processor.rotary_style,
            rotary_dim=processor.rotary_dim,
            backend=cache_movement_backend(processor),
        )
        scores = state.scores
        indices = select_per_cache_head_kv_norm_indices_from_scores(
            scores, processor.config.process_every, processor.config.topk
        )
        index = indices[..., None].expand(-1, -1, -1, merged_all.shape[-1])
        return torch.gather(merged_all, dim=2, index=index), scores, indices

    policy = longshot_full_history_execution(processor)
    if not policy.enabled:
        all_indices = expand_cache_head_indices(
            cache.all_indices(), processor.num_kv_cache_heads
        )
        merged_all = gather_per_cache_head_kv(
            cache,
            all_indices,
            unrope_keys=processor.config.unrope_keys,
            rope_theta=processor.rope_theta,
            rotary_style=processor.rotary_style,
            rotary_dim=processor.rotary_dim,
            backend=cache_movement_backend(processor),
        )
        scores = chunked_fp32_squared_l2(merged_all)
        indices = select_per_cache_head_kv_norm_indices_from_scores(
            scores, processor.config.process_every, processor.config.topk
        )
        index = indices[..., None].expand(-1, -1, -1, merged_all.shape[-1])
        return torch.gather(merged_all, dim=2, index=index), scores, indices

    if not isinstance(cache, WindowedCache):
        raise TypeError(
            "enabled Longshot selected-history execution requires contiguous WindowedCache storage"
        )
    with torch.no_grad():
        if processor.config.unrope_keys:
            scores = chunked_fp32_unroped_concatenated_kv_squared_l2(
                cache.keys,
                cache.values,
                theta=processor.rope_theta,
                rotary_style=processor.rotary_style,
                rotary_dim=processor.rotary_dim,
                max_temp_bytes=policy.max_temp_bytes,
            )
        else:
            scores = chunked_fp32_concatenated_kv_squared_l2(
                cache.keys,
                cache.values,
                max_temp_bytes=policy.max_temp_bytes,
            )
        indices = select_per_cache_head_kv_norm_indices_from_scores(
            scores, processor.config.process_every, processor.config.topk
        )

    keys, values = gather_windowed_per_cache_head_kv(
        cache.keys,
        cache.values,
        indices,
        unrope_keys=processor.config.unrope_keys,
        rope_theta=processor.rope_theta,
        rotary_style=processor.rotary_style,
        rotary_dim=processor.rotary_dim,
    )
    record_longshot_full_history_event(
        processor,
        keys=cache.keys,
        selected_positions=indices.shape[-1],
        max_temp_bytes=policy.max_temp_bytes,
        unrope_keys=processor.config.unrope_keys,
        rotary_dim=processor.rotary_dim,
    )
    return torch.cat([keys, values], dim=-1), scores, indices


def synchronize_sidecar(processor: Any, cache: WindowedCache) -> None:
    spec = kv_score_sidecar_spec(processor)
    cache.kv_score_sidecar = synchronize_kv_score_sidecar(
        cache.kv_score_sidecar, cache.keys, cache.values, spec
    )


def refresh_sidecar(
    processor: Any, cache: CacheLike, positions: torch.Tensor
) -> CacheLike:
    if not kv_score_sidecar_execution(processor).enabled:
        return cache
    if not isinstance(cache, WindowedCache):
        raise TypeError(
            "enabled KV-score sidecar requires contiguous WindowedCache storage"
        )
    spec = kv_score_sidecar_spec(processor)
    cache.kv_score_sidecar = refresh_kv_score_sidecar(
        cache.kv_score_sidecar, cache.keys, cache.values, positions, spec
    )
    return cache


def cache_movement_backend(processor: Any) -> str:
    return "triton" if triton_cache_movement_enabled(processor) else "eager"
