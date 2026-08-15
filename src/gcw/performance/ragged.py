"""Closed per-row window descriptors for future exact ragged execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RaggedWindow:
    start: int
    stop: int
    active_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("ragged window bounds must be positive and ordered")
        if not self.active_rows or self.active_rows != tuple(sorted(set(self.active_rows))):
            raise ValueError("ragged window rows must be sorted and unique")


@dataclass(frozen=True)
class RaggedPairPlan:
    """Exact valid lengths and padding-free windows for one physical pair."""

    padded_width: int
    valid_lengths: tuple[int, int]
    process_every: int
    windows: tuple[RaggedWindow, ...]

    def __post_init__(self) -> None:
        if self.padded_width <= 0 or self.process_every <= 0:
            raise ValueError("ragged pair dimensions must be positive")
        if any(length <= 0 or length > self.padded_width for length in self.valid_lengths):
            raise ValueError("ragged valid lengths must lie inside padded width")
        if not self.windows or self.windows[0].start != 0:
            raise ValueError("ragged windows must begin at zero")
        if any(window.stop > self.padded_width for window in self.windows):
            raise ValueError("ragged window exceeds padded width")

    def semantic_boundaries(self, row: int) -> tuple[int, ...]:
        if row not in {0, 1}:
            raise ValueError("ragged pair row must be zero or one")
        return tuple(
            boundary
            for boundary in range(self.process_every, self.valid_lengths[row], self.process_every)
        )


def plan_ragged_pair(attention_mask: torch.Tensor, process_every: int) -> RaggedPairPlan:
    """Validate a prefix mask and derive independent row-window activity."""
    if attention_mask.ndim != 2 or attention_mask.shape[0] != 2:
        raise ValueError("ragged execution requires one physical two-row attention mask")
    if attention_mask.dtype != torch.bool:
        raise TypeError("ragged attention mask must be boolean")
    if process_every <= 0:
        raise ValueError("ragged process_every must be positive")
    lengths_tensor = attention_mask.sum(dim=1, dtype=torch.long)
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    canonical = positions[None, :] < lengths_tensor[:, None]
    if not bool(torch.equal(attention_mask, canonical)):
        raise ValueError("ragged execution requires prefix-shaped attention masks")
    lengths = tuple(int(value) for value in lengths_tensor.cpu().tolist())
    if len(lengths) != 2 or any(length <= 0 for length in lengths):
        raise ValueError("ragged execution requires at least one valid token per row")
    windows = []
    maximum = max(lengths)
    for start in range(0, maximum, process_every):
        stop = min(start + process_every, maximum)
        active = tuple(row for row, length in enumerate(lengths) if start < length)
        windows.append(RaggedWindow(start, stop, active))
    return RaggedPairPlan(
        padded_width=attention_mask.shape[1],
        valid_lengths=(lengths[0], lengths[1]),
        process_every=process_every,
        windows=tuple(windows),
    )
