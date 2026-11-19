from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheShapeGroup:
    num_heads: int
    head_dim: int
    layers: tuple[int, ...]
    head_indices: tuple[int, ...]

    @property
    def name(self) -> str:
        return f"h{self.num_heads}_d{self.head_dim}"


def cache_shape_groups(
    shapes: list[tuple[int, int]],
) -> tuple[CacheShapeGroup, ...]:
    layers: dict[tuple[int, int], list[int]] = {}
    heads: dict[tuple[int, int], list[int]] = {}
    offset = 0
    for index, shape in enumerate(shapes):
        layers.setdefault(shape, []).append(index)
        heads.setdefault(shape, []).extend(range(offset, offset + shape[0]))
        offset += shape[0]
    return tuple(
        CacheShapeGroup(*shape, tuple(layers[shape]), tuple(heads[shape]))
        for shape in sorted(layers)
    )


def cache_group_views(cache, shape_groups):
    from gcw.models.cache_processors.state import CacheState, StaticCacheState
    from gcw.models.cache_processors.windowed import (
        GroupedWindowedCache,
        WindowedCache,
    )

    if isinstance(cache, GroupedWindowedCache):
        if cache.layer_groups != tuple(group.layers for group in shape_groups):
            raise ValueError(
                "cache groups do not match the processor layer layout"
            )
        return cache.groups
    if isinstance(cache, (CacheState, StaticCacheState)):
        return tuple(
            StaticCacheState(
                [cache.keys[i] for i in spec.layers],
                [cache.values[i] for i in spec.layers],
                cache.score_storage,
                cache.length,
            )
            if isinstance(cache, StaticCacheState)
            else CacheState(
                [cache.keys[i] for i in spec.layers],
                [cache.values[i] for i in spec.layers],
                cache.scores,
            )
            for spec in shape_groups
        )
    if isinstance(cache, WindowedCache) and len(shape_groups) == 1:
        return (cache,)
    raise TypeError("mixed-shape processing requires a grouped or list cache")
