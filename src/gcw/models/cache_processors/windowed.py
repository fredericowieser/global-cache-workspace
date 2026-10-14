from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from gcw.models.cache_processors.state import CacheState, select_l2_indices
from gcw.models.cache_processors.layout import cache_shape_groups

if TYPE_CHECKING:
    from gcw.performance.kv_score_sidecar import KVScoreSidecarState


@dataclass
class WindowedCache:
    keys: torch.Tensor
    values: torch.Tensor
    scores: torch.Tensor
    linear_states: (
        dict[int, tuple[torch.Tensor, torch.Tensor | None]] | None
    ) = None
    kv_score_sidecar: KVScoreSidecarState | None = None
    attention_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None

    @classmethod
    def from_lists(
        cls,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        scores: torch.Tensor,
        linear_states: (
            dict[int, tuple[torch.Tensor, torch.Tensor | None]] | None
        ) = None,
        kv_score_sidecar: KVScoreSidecarState | None = None,
        attention_states: (
            dict[int, tuple[torch.Tensor, torch.Tensor]] | None
        ) = None,
    ) -> "WindowedCache":
        return cls(
            torch.stack(keys, dim=0),
            torch.stack(values, dim=0),
            scores,
            linear_states,
            kv_score_sidecar,
            attention_states,
        )

    @classmethod
    def from_cache_state(cls, cache: CacheState) -> "WindowedCache":
        return cls.from_lists(
            cache.keys,
            cache.values,
            cache.scores,
            cache.linear_states,
            cache.kv_score_sidecar,
            cache.attention_states,
        )

    @property
    def batch_size(self) -> int:
        return self.keys.shape[1]

    @property
    def seq_len(self) -> int:
        return self.keys.shape[3]

    @property
    def num_layers(self) -> int:
        return self.keys.shape[0]

    def layer(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.keys[idx], self.values[idx]

    def detach(self) -> "WindowedCache":
        """Return a cache boundary with no autograd history before it."""
        linear_states = None
        if self.linear_states is not None:
            linear_states = {
                layer_idx: (
                    recurrent.detach(),
                    None if convolution is None else convolution.detach(),
                )
                for layer_idx, (
                    recurrent,
                    convolution,
                ) in self.linear_states.items()
            }
        sidecar = None
        if self.kv_score_sidecar is not None:
            sidecar = replace(
                self.kv_score_sidecar,
                scores=self.kv_score_sidecar.scores.detach(),
            )
        return replace(
            self,
            keys=self.keys.detach(),
            values=self.values.detach(),
            scores=self.scores.detach(),
            linear_states=linear_states,
            kv_score_sidecar=sidecar,
            attention_states=(
                None
                if self.attention_states is None
                else {
                    layer_idx: (key.detach(), value.detach())
                    for layer_idx, (key, value) in self.attention_states.items()
                }
            ),
        )

    def select_indices(
        self, process_every: int = 64, topk: int = 32
    ) -> torch.Tensor:
        return select_l2_indices(
            self.scores, process_every=process_every, topk=topk
        )

    def all_indices(self) -> torch.Tensor:
        seq = self.scores.shape[1]
        return (
            torch.arange(seq, dtype=torch.long, device=self.scores.device)
            .unsqueeze(0)
            .expand(self.scores.shape[0], -1)
        )

    def all_heads(self) -> torch.Tensor:
        heads = self.keys.shape[0] * self.keys.shape[2]
        return (
            torch.arange(heads, dtype=torch.long, device=self.scores.device)
            .unsqueeze(0)
            .expand(self.batch_size, -1)
        )

    def gather_layer(
        self, layer_idx: int, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.layer(layer_idx)
        index = indices[:, None, :, None].expand(-1, k.shape[1], -1, k.shape[3])
        return torch.gather(k, 2, index), torch.gather(v, 2, index)

    def scatter_layer(
        self,
        layer_idx: int,
        indices: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        k, v = self.layer(layer_idx)
        index = indices[:, None, :, None].expand(-1, k.shape[1], -1, k.shape[3])
        new_keys = k.scatter(2, index, keys.to(k.dtype))
        new_values = v.scatter(2, index, values.to(v.dtype))
        self.keys = torch.cat(
            [
                self.keys[:layer_idx],
                new_keys.unsqueeze(0),
                self.keys[layer_idx + 1 :],
            ],
            dim=0,
        )
        self.values = torch.cat(
            [
                self.values[:layer_idx],
                new_values.unsqueeze(0),
                self.values[layer_idx + 1 :],
            ],
            dim=0,
        )
        self.kv_score_sidecar = None

    def to_cache_state(self) -> CacheState:
        return CacheState(
            list(self.keys.unbind(0)),
            list(self.values.unbind(0)),
            self.scores,
            self.linear_states,
            self.kv_score_sidecar,
            self.attention_states,
        )


@dataclass
class GroupedWindowedCache:
    groups: tuple[WindowedCache, ...]
    layer_groups: tuple[tuple[int, ...], ...]
    scores: torch.Tensor
    attention_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None

    @classmethod
    def from_cache_state(cls, cache: CacheState) -> "GroupedWindowedCache":
        if cache.linear_states or cache.kv_score_sidecar is not None:
            raise ValueError(
                "grouped caches do not support linear or sidecar state"
            )
        layout = cache_shape_groups(
            [(key.shape[1], key.shape[3]) for key in cache.keys]
        )
        return cls(
            tuple(
                WindowedCache.from_lists(
                    [cache.keys[i] for i in group.layers],
                    [cache.values[i] for i in group.layers],
                    cache.scores,
                )
                for group in layout
            ),
            tuple(group.layers for group in layout),
            cache.scores,
            cache.attention_states,
        )

    @property
    def batch_size(self) -> int:
        return self.scores.shape[0]

    @property
    def seq_len(self) -> int:
        return self.scores.shape[1]

    @property
    def num_layers(self) -> int:
        return sum(len(layers) for layers in self.layer_groups)

    def layer(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        for group, layers in zip(self.groups, self.layer_groups, strict=True):
            if idx in layers:
                return group.layer(layers.index(idx))
        raise IndexError(idx)

    def to_cache_state(self) -> CacheState:
        pairs = [self.layer(i) for i in range(self.num_layers)]
        return CacheState(
            [key for key, _ in pairs],
            [value for _, value in pairs],
            self.scores,
            attention_states=self.attention_states,
        )

    def detach(self) -> "GroupedWindowedCache":
        return replace(
            self,
            groups=tuple(group.detach() for group in self.groups),
            scores=self.scores.detach(),
            attention_states=(
                None
                if self.attention_states is None
                else {
                    layer: (key.detach(), value.detach())
                    for layer, (key, value) in self.attention_states.items()
                }
            ),
        )

    def select_indices(
        self, process_every: int = 64, topk: int = 32
    ) -> torch.Tensor:
        return select_l2_indices(self.scores, process_every, topk)

    def all_indices(self) -> torch.Tensor:
        return torch.arange(self.seq_len, device=self.scores.device).expand(
            self.batch_size, -1
        )
