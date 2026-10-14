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
