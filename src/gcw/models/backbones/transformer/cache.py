from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
import torch

from gcw.models.cache_processors.base import cache_processor_recomputation
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import WindowedCache

if TYPE_CHECKING:
    from gcw.performance.kv_score_sidecar import KVScoreSidecarSpec

CacheLike = CacheState | WindowedCache
ProcessorFn = Callable[..., CacheLike | StaticCacheState]
ProcessorExecutionFactory = Callable[[], AbstractContextManager[ProcessorFn]]
LinearStateLayout = tuple[tuple[int, bool], ...]


def detach_cache(cache: CacheLike) -> CacheLike:
    """Cut autograd history at a recurrent cache boundary.

    Cache processors remain unaware of this training policy: every maintained
    windowed processor consumes either ``CacheState`` or ``WindowedCache``, and
    both cache representations implement the same detached-boundary contract.
    """
    return cache.detach()


@dataclass(frozen=True)
class WindowedCacheLayout:
    sidecar_spec: KVScoreSidecarSpec | None = None
    linear_states: LinearStateLayout = ()


def _flatten_windowed_cache(
    cache: WindowedCache | None,
) -> tuple[tuple[torch.Tensor, ...], WindowedCacheLayout]:
    if cache is None:
        return (), WindowedCacheLayout()
    tensors = [cache.keys, cache.values, cache.scores]
    sidecar_spec = None
    if cache.kv_score_sidecar is not None:
        sidecar_spec = cache.kv_score_sidecar.spec
        tensors.append(cache.kv_score_sidecar.scores)
    linear_layout: list[tuple[int, bool]] = []
    for layer_idx, state in sorted((cache.linear_states or {}).items()):
        recurrent, convolution = state
        has_convolution = convolution is not None
        linear_layout.append((layer_idx, has_convolution))
        tensors.append(recurrent)
        if convolution is not None:
            tensors.append(convolution)
    return tuple(tensors), WindowedCacheLayout(
        sidecar_spec, tuple(linear_layout)
    )


def _unflatten_windowed_cache(
    tensors: tuple[torch.Tensor, ...], layout: WindowedCacheLayout
) -> WindowedCache:
    if len(tensors) < 3:
        raise ValueError(
            "a flattened WindowedCache boundary requires keys, values, and scores"
        )
    position = 3
    sidecar = None
    if layout.sidecar_spec is not None:
        if position >= len(tensors):
            raise ValueError(
                "flattened WindowedCache is missing KV-score sidecar state"
            )
        from gcw.performance.kv_score_sidecar import KVScoreSidecarState

        sidecar = KVScoreSidecarState(
            layout.sidecar_spec, tensors[position].detach()
        )
        position += 1
    linear_states: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}
    for layer_idx, has_convolution in layout.linear_states:
        if position >= len(tensors):
            raise ValueError(
                "flattened WindowedCache is missing a recurrent state tensor"
            )
        recurrent = tensors[position]
        position += 1
        convolution = tensors[position] if has_convolution else None
        position += int(has_convolution)
        linear_states[layer_idx] = (recurrent, convolution)
    if position != len(tensors):
        raise ValueError(
            "flattened WindowedCache contains unexpected state tensors"
        )
    return WindowedCache(
        tensors[0], tensors[1], tensors[2], linear_states or None, sidecar
    )


def _window_checkpoint_contexts():
    return nullcontext(), cache_processor_recomputation()
