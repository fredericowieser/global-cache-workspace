from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from gcw.performance.kv_score_sidecar import KVScoreSidecarState


@dataclass
class CacheState:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    scores: torch.Tensor
    linear_states: (
        dict[int, tuple[torch.Tensor, torch.Tensor | None]] | None
    ) = None
    kv_score_sidecar: KVScoreSidecarState | None = None
    attention_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None

    @property
    def batch_size(self) -> int:
        return self.keys[0].shape[0]

    @property
    def seq_len(self) -> int:
        return self.keys[0].shape[2]

    @property
    def device(self) -> torch.device:
        return self.keys[0].device if self.keys else self.scores.device

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    def layer(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.keys[idx], self.values[idx]

    def detach(self) -> "CacheState":
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
            keys=[key.detach() for key in self.keys],
            values=[value.detach() for value in self.values],
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
        heads = sum(key.shape[1] for key in self.keys)
        return (
            torch.arange(heads, dtype=torch.long, device=self.scores.device)
            .unsqueeze(0)
            .expand(self.batch_size, -1)
        )

    def gather_layer(
        self, layer_idx: int, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.layer(layer_idx)
        index = _expand_cache_indices(indices, k)
        return torch.gather(k, 2, index), torch.gather(v, 2, index)

    def scatter_layer(
        self,
        layer_idx: int,
        indices: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        k, v = self.layer(layer_idx)
        index = _expand_cache_indices(indices, k)
        self.keys[layer_idx] = k.scatter(2, index, keys.to(k.dtype))
        self.values[layer_idx] = v.scatter(2, index, values.to(v.dtype))
        self.kv_score_sidecar = None


def processor_call_count(seq_len: int, process_every: int = 64) -> int:
    if seq_len <= 0:
        return 0
    return max(0, (seq_len + process_every - 1) // process_every - 1)


def select_l2_indices(
    scores: torch.Tensor, process_every: int = 64, topk: int = 32
) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must have shape (batch, seq)")
    batch, seq = scores.shape
    if seq == 0:
        return torch.empty(batch, 0, dtype=torch.long, device=scores.device)
    recent = min(process_every, seq)
    hist_len = seq - recent
    full = (
        torch.arange(seq, device=scores.device).unsqueeze(0).expand(batch, -1)
    )
    recent_idx = full[:, hist_len:]
    if hist_len > 0 and topk > 0:
        k = min(topk, hist_len)
        _, hist_idx = torch.topk(scores[:, :hist_len].float(), k=k, dim=-1)
        selected = torch.cat([hist_idx, recent_idx], dim=1)
    else:
        selected = recent_idx
    return selected.sort(dim=1).values


def _expand_cache_indices(
    indices: torch.Tensor, cache: torch.Tensor
) -> torch.Tensor:
    if indices.ndim != 2:
        raise ValueError("indices must have shape (batch, selected)")
    return indices[:, None, :, None].expand(
        -1, cache.shape[1], -1, cache.shape[3]
    )


class StaticAttentionState:
    """Inference storage for one attention-only layer, optionally a sliding ring."""

    def __init__(
        self, keys: torch.Tensor, values: torch.Tensor, window: int | None
    ):
        self.keys = keys
        self.values = values
        self.window = window
        self.length = 0

    def append(
        self, start: int, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if start != self.length:
            raise ValueError(
                "static attention cache requires contiguous appends"
            )
        end = start + keys.shape[2]
        capacity = self.keys.shape[2]
        if self.window is None:
            if end > capacity:
                raise ValueError("static attention cache capacity exceeded")
            self.keys[:, :, start:end].copy_(keys)
            self.values[:, :, start:end].copy_(values)
            result = (self.keys[:, :, :end], self.values[:, :, :end], 0)
        else:
            first = max(0, start - self.window + 1)
            offset = first % capacity
            count = start - first
            split = min(count, capacity - offset)
            key_parts = [self.keys[:, :, offset : offset + split]]
            value_parts = [self.values[:, :, offset : offset + split]]
            if split < count:
                key_parts.append(self.keys[:, :, : count - split])
                value_parts.append(self.values[:, :, : count - split])
            result = (
                torch.cat((*key_parts, keys), dim=2),
                torch.cat((*value_parts, values), dim=2),
                first,
            )
            retained = min(keys.shape[2], capacity)
            offset = (end - retained) % capacity
            split = min(retained, capacity - offset)
            for target, source in ((self.keys, keys), (self.values, values)):
                source = source[:, :, -retained:]
                target[:, :, offset : offset + split].copy_(
                    source[:, :, :split]
                )
                if split < retained:
                    target[:, :, : retained - split].copy_(source[:, :, split:])
        self.length = end
        return result


class StaticCacheState:
    def __init__(
        self,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        score_storage: torch.Tensor,
        length: int = 0,
        linear_states: (
            dict[int, tuple[torch.Tensor, torch.Tensor | None]] | None
        ) = None,
    ):
        self.keys = keys
        self.values = values
        self.score_storage = score_storage
        self.length = length
        self.linear_states = linear_states
        self.attention_states: dict[int, StaticAttentionState] = {}

    @classmethod
    def allocate(
        cls,
        *,
        batch_size: int,
        num_layers: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        layer_shapes: tuple[tuple[int, int], ...] | None = None,
        attention_shapes: dict[int, tuple[int, int, int | None]] | None = None,
    ) -> "StaticCacheState":
        if max_seq_len <= 0:
            raise ValueError("static cache capacity must be positive")
        shapes = layer_shapes or ((num_kv_heads, head_dim),) * num_layers
        if len(shapes) != num_layers:
            raise ValueError(
                "static cache shapes must match the selected layer count"
            )
        keys = [
            torch.empty(
                (batch_size, heads, max_seq_len, dim),
                device=device,
                dtype=dtype,
            )
            for heads, dim in shapes
        ]
        values = [torch.empty_like(key) for key in keys]
        scores = torch.empty(
            batch_size, max_seq_len, device=device, dtype=torch.float32
        )
        cache = cls(keys, values, scores)
        for layer, (heads, dim, window) in (attention_shapes or {}).items():
            if window is not None and window <= 0:
                raise ValueError("sliding cache window must be positive")
            capacity = (
                max_seq_len if window is None else min(max_seq_len, window)
            )
            key = torch.empty(
                (batch_size, heads, capacity, dim), device=device, dtype=dtype
            )
            cache.attention_states[layer] = StaticAttentionState(
                key, torch.empty_like(key), window
            )
        return cache

    @property
    def capacity(self) -> int:
        return self.score_storage.shape[1]

    def validate_append(self, start: int, steps: int) -> None:
        if start != self.length or steps <= 0 or start + steps > self.capacity:
            raise ValueError(
                "static cache requires a contiguous append within capacity"
            )

    @property
    def batch_size(self) -> int:
        return self.keys[0].shape[0]

    @property
    def seq_len(self) -> int:
        return self.length

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    @property
    def scores(self) -> torch.Tensor:
        return self.score_storage[:, : self.length]

    def layer(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layer_view(idx, self.length)

    def layer_view(
        self, idx: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= end <= self.capacity:
            raise ValueError("static cache view exceeds capacity")
        return self.keys[idx][:, :, :end, :], self.values[idx][:, :, :end, :]

    def write_layer(
        self, idx: int, start: int, keys: torch.Tensor, values: torch.Tensor
    ) -> None:
        self.validate_append(start, keys.shape[2])
        end = start + keys.shape[2]
        self.keys[idx][:, :, start:end, :].copy_(keys.to(self.keys[idx].dtype))
        self.values[idx][:, :, start:end, :].copy_(
            values.to(self.values[idx].dtype)
        )

    def write_scores(self, start: int, scores: torch.Tensor) -> None:
        self.validate_append(start, scores.shape[1])
        end = start + scores.shape[1]
        self.score_storage[:, start:end].copy_(scores.float())
        self.length = max(self.length, end)

    def select_indices(
        self, process_every: int = 64, topk: int = 32
    ) -> torch.Tensor:
        return select_l2_indices(
            self.scores, process_every=process_every, topk=topk
        )

    def all_indices(self) -> torch.Tensor:
        seq = self.length
        return (
            torch.arange(
                seq, dtype=torch.long, device=self.score_storage.device
            )
            .unsqueeze(0)
            .expand(self.batch_size, -1)
        )

    def all_heads(self) -> torch.Tensor:
        heads = sum(key.shape[1] for key in self.keys)
        return (
            torch.arange(
                heads, dtype=torch.long, device=self.score_storage.device
            )
            .unsqueeze(0)
            .expand(self.batch_size, -1)
        )

    def gather_layer(
        self, layer_idx: int, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.layer(layer_idx)
        index = _expand_cache_indices(indices, k)
        return torch.gather(k, 2, index), torch.gather(v, 2, index)

    def scatter_layer(
        self,
        layer_idx: int,
        indices: torch.Tensor,
        keys: torch.Tensor | None,
        values: torch.Tensor,
    ) -> None:
        k = self.keys[layer_idx][:, :, : self.length, :]
        v = self.values[layer_idx][:, :, : self.length, :]
        if indices.ndim != 2:
            raise ValueError("indices must have shape (batch, selected)")
        batch = torch.arange(self.batch_size, device=k.device)[:, None]
        if keys is not None:
            k.transpose(1, 2).index_put_(
                (batch, indices), keys.transpose(1, 2).to(k.dtype)
            )
        v.transpose(1, 2).index_put_(
            (batch, indices), values.transpose(1, 2).to(v.dtype)
        )
