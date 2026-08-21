from __future__ import annotations

from typing import Any
import torch

from gcw.models.cache_processors.ops import (
    expand_cache_head_indices,
    select_per_cache_head_kv_norm_indices_from_scores,
)
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import WindowedCache
import gcw.models.cache_processors.longshot as ls_pkg
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
    chunked_fp32_unroped_concatenated_kv_squared_l2,
)
from gcw.performance.triton.execution import triton_cache_movement_enabled

# The cache object any selection path accepts: static, mutable, or windowed.
CacheLike = CacheState | StaticCacheState | WindowedCache


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
        merged_all = ls_pkg.gather_per_cache_head_kv(
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
        merged_all = ls_pkg.gather_per_cache_head_kv(
            cache,
            all_indices,
            unrope_keys=processor.config.unrope_keys,
            rope_theta=processor.rope_theta,
            rotary_style=processor.rotary_style,
            rotary_dim=processor.rotary_dim,
            backend=cache_movement_backend(processor),
        )
        scores = ls_pkg.chunked_fp32_squared_l2(merged_all)
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
    sync_fn = getattr(
        ls_pkg, "synchronize_kv_score_sidecar", synchronize_kv_score_sidecar
    )
    cache.kv_score_sidecar = sync_fn(
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
    refresh_fn = getattr(
        ls_pkg, "refresh_kv_score_sidecar", refresh_kv_score_sidecar
    )
    cache.kv_score_sidecar = refresh_fn(
        cache.kv_score_sidecar, cache.keys, cache.values, positions, spec
    )
    return cache


def cache_movement_backend(processor: Any) -> str:
    return "triton" if triton_cache_movement_enabled(processor) else "eager"
