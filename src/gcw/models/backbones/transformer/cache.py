from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
import torch

from gcw.models.cache_processors.base import cache_processor_recomputation
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import (
    GroupedWindowedCache,
    WindowedCache,
)

if TYPE_CHECKING:
    from gcw.performance.kv_score_sidecar import KVScoreSidecarSpec

CacheLike = CacheState | WindowedCache | GroupedWindowedCache
ProcessorFn = Callable[..., CacheLike | StaticCacheState]
ProcessorExecutionFactory = Callable[[], AbstractContextManager[ProcessorFn]]
LinearStateLayout = tuple[tuple[int, bool], ...]


def detach_cache(cache: CacheLike) -> CacheLike:
    """Cut autograd history at a recurrent cache boundary.

    List, uniform, and grouped caches share the detached-boundary contract.
    """
    return cache.detach()


@dataclass(frozen=True)
class WindowedCacheLayout:
    sidecar_spec: KVScoreSidecarSpec | None = None
    linear_states: LinearStateLayout = ()
    attention_layers: tuple[int, ...] = ()
    grouped_layers: tuple[tuple[int, ...], ...] = ()


def _flatten_windowed_cache(
    cache: WindowedCache | GroupedWindowedCache | None,
) -> tuple[tuple[torch.Tensor, ...], WindowedCacheLayout]:
    if cache is None:
        return (), WindowedCacheLayout()
    if isinstance(cache, GroupedWindowedCache):
        tensors = [cache.scores]
        for group in cache.groups:
            tensors.extend((group.keys, group.values))
        attention = cache.attention_states or {}
        for layer in sorted(attention):
            tensors.extend(attention[layer])
        return tuple(tensors), WindowedCacheLayout(
            attention_layers=tuple(sorted(attention)),
            grouped_layers=cache.layer_groups,
        )
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
    attention_states = cache.attention_states or {}
    attention_layers = tuple(sorted(attention_states))
    for layer_idx in attention_layers:
        tensors.extend(attention_states[layer_idx])
    return tuple(tensors), WindowedCacheLayout(
        sidecar_spec, tuple(linear_layout), attention_layers
    )


def _unflatten_windowed_cache(
    tensors: tuple[torch.Tensor, ...], layout: WindowedCacheLayout
) -> WindowedCache | GroupedWindowedCache:
    if layout.grouped_layers:
        expected = 1 + 2 * (
            len(layout.grouped_layers) + len(layout.attention_layers)
        )
        if len(tensors) != expected:
            raise ValueError("invalid grouped cache tensor boundary")
        groups = tuple(
            WindowedCache(tensors[1 + 2 * i], tensors[2 + 2 * i], tensors[0])
            for i in range(len(layout.grouped_layers))
        )
        offset = 1 + 2 * len(groups)
        attention = {
            layer: (tensors[offset + 2 * i], tensors[offset + 2 * i + 1])
            for i, layer in enumerate(layout.attention_layers)
        }
        return GroupedWindowedCache(
            groups, layout.grouped_layers, tensors[0], attention or None
        )
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
    attention_states = {}
    for layer_idx in layout.attention_layers:
        if position + 1 >= len(tensors):
            raise ValueError(
                "flattened WindowedCache is missing attention state tensors"
            )
        attention_states[layer_idx] = (tensors[position], tensors[position + 1])
        position += 2
    if position != len(tensors):
        raise ValueError(
            "flattened WindowedCache contains unexpected state tensors"
        )
    return WindowedCache(
        tensors[0],
        tensors[1],
        tensors[2],
        linear_states or None,
        sidecar,
        attention_states or None,
    )


def _window_checkpoint_contexts():
    return nullcontext(), cache_processor_recomputation()
