"""Deterministic adaptive batching for the adjacent-pair SFT objective.

The executor reserves the same 32 two-row source batches as the eager oracle,
never reorders or splits a pair, and only changes how contiguous pairs are
collated into physical model calls. Objective weighting remains outside this
module and is always one pair mean divided by 32.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gcw.data.training import TrainingBatch, TrainingBatchStream
from gcw.performance.config import (
    AdaptiveBatchingConfig,
    PhysicalBatchBandConfig,
)
from gcw.provenance import canonical_json_sha256

PAIRS_PER_STEP = 32
ROWS_PER_PAIR = 2
ROWS_PER_STEP = PAIRS_PER_STEP * ROWS_PER_PAIR
_ADAPTIVE_BATCHING_SCHEMA = "gcw.adaptive_pair_batching.v1"


@dataclass(frozen=True)
class ReservedAdjacentPairs:
    """Exactly 32 source batches, each containing one original adjacent pair."""

    pairs: tuple[TrainingBatch, ...]
    padded_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            len(self.pairs) != PAIRS_PER_STEP
            or len(self.padded_lengths) != PAIRS_PER_STEP
        ):
            raise ValueError(
                f"adaptive batching must reserve exactly {PAIRS_PER_STEP} adjacent pairs"
            )


@dataclass(frozen=True)
class PairGroupPlan:
    """One contiguous range of reserved pairs assigned to a physical batch."""

    pair_start: int
    pair_stop: int
    physical_rows: int
    padded_tokens: int
    band: PhysicalBatchBandConfig

    @property
    def pair_count(self) -> int:
        return self.pair_stop - self.pair_start


@dataclass(frozen=True)
class AdaptivePairStepPlan:
    """A complete, ordered physical plan for one 64-row optimizer step."""

    reservation: ReservedAdjacentPairs
    groups: tuple[PairGroupPlan, ...]

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError(
                "adaptive pair step plan must contain physical groups"
            )
        cursor = 0
        for group in self.groups:
            if (
                group.pair_start != cursor
                or group.pair_stop <= group.pair_start
            ):
                raise ValueError(
                    "adaptive pair groups must be contiguous and ordered"
                )
            if group.physical_rows != group.pair_count * ROWS_PER_PAIR:
                raise ValueError(
                    "adaptive pair group row count does not match its pair range"
                )
            cursor = group.pair_stop
        if cursor != PAIRS_PER_STEP:
            raise ValueError(
                f"adaptive pair groups must cover exactly {PAIRS_PER_STEP} pairs"
            )

    def metrics(self) -> dict[str, float]:
        """Return per-step planning telemetry without model-dependent values."""
        counts = Counter(group.physical_rows for group in self.groups)
        result = {
            "train/adaptive_batching_active": 1.0,
            "train/adaptive_reserved_pairs": float(PAIRS_PER_STEP),
            "train/adaptive_reserved_rows": float(ROWS_PER_STEP),
            "train/adaptive_physical_groups": float(len(self.groups)),
            "train/adaptive_physical_rows_min": float(
                min(group.physical_rows for group in self.groups)
            ),
            "train/adaptive_physical_rows_max": float(
                max(group.physical_rows for group in self.groups)
            ),
            "train/adaptive_padded_tokens_min": float(
                min(group.padded_tokens for group in self.groups)
            ),
            "train/adaptive_padded_tokens_max": float(
                max(group.padded_tokens for group in self.groups)
            ),
            "train/adaptive_source_pair_padded_positions": float(
                sum(
                    length * ROWS_PER_PAIR
                    for length in self.reservation.padded_lengths
                )
            ),
            "train/adaptive_physical_padded_positions": float(
                sum(
                    group.physical_rows * group.padded_tokens
                    for group in self.groups
                )
            ),
        }
        for physical_rows in (2, 4, 8, 16):
            result[f"train/adaptive_groups_rows_{physical_rows}"] = float(
                counts[physical_rows]
            )
        return result


@dataclass(frozen=True)
class AdaptivePhysicalBatch:
    """One collated physical batch and its authenticated contiguous plan."""

    batch: TrainingBatch
    plan: PairGroupPlan


def reserve_adjacent_pairs(
    stream: TrainingBatchStream,
) -> ReservedAdjacentPairs:
    """Consume exactly 32 eager two-row batches without altering their order."""
    pairs: list[TrainingBatch] = []
    padded_lengths: list[int] = []
    for pair_index in range(PAIRS_PER_STEP):
        batch = next(stream)
        _validate_source_pair(batch, pair_index)
        pairs.append(batch)
        padded_lengths.append(batch.x.shape[1])
    return ReservedAdjacentPairs(tuple(pairs), tuple(padded_lengths))


def plan_contiguous_pair_groups(
    reservation: ReservedAdjacentPairs,
    policy: AdaptiveBatchingConfig,
) -> AdaptivePairStepPlan:
    """Greedily choose the largest admitted contiguous group at each pair."""
    if not policy.enabled:
        raise ValueError("adaptive pair planning requires an enabled policy")
    groups: list[PairGroupPlan] = []
    pair_start = 0
    while pair_start < PAIRS_PER_STEP:
        selected: PairGroupPlan | None = None
        for band in policy.bands:
            pair_count = band.physical_rows // ROWS_PER_PAIR
            pair_stop = pair_start + pair_count
            if pair_stop > PAIRS_PER_STEP:
                continue
            padded_tokens = max(
                reservation.padded_lengths[pair_start:pair_stop]
            )
            if padded_tokens <= band.max_padded_tokens:
                selected = PairGroupPlan(
                    pair_start=pair_start,
                    pair_stop=pair_stop,
                    physical_rows=band.physical_rows,
                    padded_tokens=padded_tokens,
                    band=band,
                )
                break
        if selected is None:
            width = reservation.padded_lengths[pair_start]
            raise ValueError(
                f"adaptive batching has no physical band for pair {pair_start} with padded width {width}"
            )
        groups.append(selected)
        pair_start = selected.pair_stop
    return AdaptivePairStepPlan(reservation=reservation, groups=tuple(groups))


def collate_pair_group(
    step: AdaptivePairStepPlan, group: PairGroupPlan
) -> AdaptivePhysicalBatch:
    """Right-pad and concatenate one planned contiguous pair range."""
    if group not in step.groups:
        raise ValueError(
            "pair group does not belong to the supplied adaptive step plan"
        )
    pairs = step.reservation.pairs[group.pair_start : group.pair_stop]
    x = torch.cat(
        [_right_pad(pair.x, group.padded_tokens, 0) for pair in pairs], dim=0
    ).contiguous()
    y = torch.cat(
        [_right_pad(pair.y, group.padded_tokens, -100) for pair in pairs], dim=0
    ).contiguous()
    masks = [pair.attention_mask for pair in pairs]
    if any(mask is None for mask in masks):
        raise ValueError("adaptive pair collation requires attention masks")
    attention_mask = torch.cat(
        [
            _right_pad(mask, group.padded_tokens, False)
            for mask in masks
            if mask is not None
        ],
        dim=0,
    ).contiguous()
    if x.shape != y.shape or x.shape != attention_mask.shape:
        raise RuntimeError(
            "adaptive pair collation produced inconsistent tensor shapes"
        )
    if x.shape != (group.physical_rows, group.padded_tokens):
        raise RuntimeError(
            "adaptive pair collation does not match its plan: "
            f"expected {(group.physical_rows, group.padded_tokens)}, got {tuple(x.shape)}"
        )
    return AdaptivePhysicalBatch(
        batch=TrainingBatch(x=x, y=y, attention_mask=attention_mask),
        plan=group,
    )


def build_adaptive_pair_step(
    stream: TrainingBatchStream,
    policy: AdaptiveBatchingConfig,
) -> AdaptivePairStepPlan:
    """Reserve and plan one complete optimizer step."""
    return plan_contiguous_pair_groups(reserve_adjacent_pairs(stream), policy)


def adaptive_batching_metadata(
    policy: AdaptiveBatchingConfig,
    *,
    global_batch_rows: int,
) -> dict[str, object]:
    """Return stable provenance and resume identity for adaptive execution."""
    execution: dict[str, object] = {
        "schema": _ADAPTIVE_BATCHING_SCHEMA,
        "enabled": policy.enabled,
        "source_pair_rows": ROWS_PER_PAIR,
        "reserved_pairs": PAIRS_PER_STEP if policy.enabled else None,
        "global_batch_rows": global_batch_rows if policy.enabled else None,
        "bands": [
            {
                "max_padded_tokens": band.max_padded_tokens,
                "physical_rows": band.physical_rows,
            }
            for band in policy.bands
        ],
        "planner": "largest_admitted_contiguous_group_v1",
        "ordering": "source_order_no_reordering_no_pair_splitting",
        "group_weighting": "adjacent_pair_loss_sum_divided_by_32",
        "checkpoint_boundary": "optimizer_step_only",
        "distributed_policy": "single_process_only",
        "noisy_diffusion_policy": "rejected_before_execution",
        "dropout_policy": "processor_and_lora_dropout_must_be_zero",
    }
    return {
        **execution,
        "fingerprint_sha256": canonical_json_sha256(execution),
    }


def _validate_source_pair(batch: TrainingBatch, pair_index: int) -> None:
    if batch.x.ndim != 2 or batch.y.ndim != 2 or batch.x.shape != batch.y.shape:
        raise ValueError(
            f"adaptive source pair {pair_index} must have matching two-dimensional x and y"
        )
    if batch.x.shape[0] != ROWS_PER_PAIR:
        raise ValueError(
            f"adaptive source pair {pair_index} must contain exactly {ROWS_PER_PAIR} rows; "
            f"got {batch.x.shape[0]}"
        )
    if batch.x.shape[1] <= 0:
        raise ValueError(
            f"adaptive source pair {pair_index} must contain at least one token position"
        )
    mask = batch.attention_mask
    if mask is None or mask.shape != batch.x.shape:
        raise ValueError(
            f"adaptive source pair {pair_index} requires an attention mask matching x"
        )


def _right_pad(
    tensor: torch.Tensor, width: int, value: int | bool
) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(
            "adaptive pair collation accepts only two-dimensional tensors"
        )
    if tensor.shape[1] > width:
        raise ValueError(
            "adaptive pair collation cannot truncate a source pair"
        )
    return F.pad(tensor, (0, width - tensor.shape[1]), value=value)
