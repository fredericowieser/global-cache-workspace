from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CacheState:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    scores: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.keys[0].shape[0]

    @property
    def seq_len(self) -> int:
        return self.keys[0].shape[2]

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    def layer(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.keys[idx], self.values[idx]

    def append_scores(self, chunk_scores: torch.Tensor) -> "CacheState":
        scores = chunk_scores if self.scores.numel() == 0 else torch.cat([self.scores, chunk_scores], dim=1)
        return CacheState(self.keys, self.values, scores)

    def select_indices(self, process_every: int = 64, topk: int = 32) -> torch.Tensor:
        return select_l2_indices(self.scores, process_every=process_every, topk=topk)

    def all_indices(self) -> torch.Tensor:
        seq = self.scores.shape[1]
        return torch.arange(seq, dtype=torch.long, device=self.scores.device).unsqueeze(0).expand(self.scores.shape[0], -1)

    def gather_layer(self, layer_idx: int, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.layer(layer_idx)
        index = _expand_cache_indices(indices, k)
        return torch.gather(k, 2, index), torch.gather(v, 2, index)

    def scatter_layer(self, layer_idx: int, indices: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> None:
        k, v = self.layer(layer_idx)
        index = _expand_cache_indices(indices, k)
        self.keys[layer_idx] = k.scatter(2, index, keys.to(k.dtype))
        self.values[layer_idx] = v.scatter(2, index, values.to(v.dtype))


def processor_call_count(seq_len: int, process_every: int = 64) -> int:
    if seq_len <= 0:
        return 0
    return max(0, (seq_len + process_every - 1) // process_every - 1)


def select_l2_indices(scores: torch.Tensor, process_every: int = 64, topk: int = 32) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must have shape (batch, seq)")
    batch, seq = scores.shape
    if seq == 0:
        return torch.empty(batch, 0, dtype=torch.long, device=scores.device)
    recent = min(process_every, seq)
    hist_len = seq - recent
    full = torch.arange(seq, device=scores.device).unsqueeze(0).expand(batch, -1)
    recent_idx = full[:, hist_len:]
    if hist_len > 0 and topk > 0:
        k = min(topk, hist_len)
        _, hist_idx = torch.topk(scores[:, :hist_len].float(), k=k, dim=-1)
        selected = torch.cat([hist_idx, recent_idx], dim=1)
    else:
        selected = recent_idx
    return selected.sort(dim=1).values


def _expand_cache_indices(indices: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    if indices.ndim != 2:
        raise ValueError("indices must have shape (batch, selected)")
    return indices[:, None, :, None].expand(-1, cache.shape[1], -1, cache.shape[3])
