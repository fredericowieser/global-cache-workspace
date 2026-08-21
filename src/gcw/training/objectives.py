"""Scientific training objectives independent of physical execution policy."""

from __future__ import annotations

from typing import Any

import torch

from gcw.config import LossBackend
from gcw.performance.checkpointing import (
    longshot_gradient_checkpointing_context,
)


def model_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_reduction: str = "mean",
    loss_backend: LossBackend = "reference",
    attention_mask: torch.Tensor | None = None,
    activation_checkpoint_windows: int = 0,
    activation_checkpoint_min_padded_sequence_length: int = 0,
    longshot_gradient_checkpointing: bool | None = None,
) -> torch.Tensor:
    """Invoke one model loss while preserving all explicit execution controls."""
    kwargs: dict[str, Any] = {
        "loss_reduction": loss_reduction,
        "loss_backend": loss_backend,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if activation_checkpoint_windows:
        kwargs["activation_checkpoint_windows"] = activation_checkpoint_windows
        kwargs["activation_checkpoint_min_padded_sequence_length"] = (
            activation_checkpoint_min_padded_sequence_length
        )
    with longshot_gradient_checkpointing_context(
        longshot_gradient_checkpointing
    ):
        return model(input_ids, targets, **kwargs)


def row_loss_sums_and_counts(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_backend: LossBackend = "reference",
    attention_mask: torch.Tensor | None = None,
    activation_checkpoint_windows: int = 0,
    activation_checkpoint_min_padded_sequence_length: int = 0,
    longshot_gradient_checkpointing: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one answer-token CE sum and supervised-token count per row."""
    row_sums = model_loss(
        model,
        input_ids,
        targets,
        loss_reduction="row_sum",
        loss_backend=loss_backend,
        attention_mask=attention_mask,
        activation_checkpoint_windows=activation_checkpoint_windows,
        activation_checkpoint_min_padded_sequence_length=activation_checkpoint_min_padded_sequence_length,
        longshot_gradient_checkpointing=longshot_gradient_checkpointing,
    )
    if row_sums.shape != (targets.shape[0],):
        raise RuntimeError(
            f"row_sum loss returned shape {tuple(row_sums.shape)} for batch {targets.shape[0]}"
        )
    return row_sums, (targets >= 0).sum(dim=1).to(dtype=row_sums.dtype)


def adjacent_pair_loss_sum_from_rows(
    row_sums: torch.Tensor,
    row_counts: torch.Tensor,
) -> torch.Tensor:
    """Sum adjacent pair token means without averaging physical groups."""
    if (
        row_sums.ndim != 1
        or row_counts.ndim != 1
        or row_sums.shape != row_counts.shape
    ):
        raise ValueError(
            "adjacent pair row sums and counts must be matching one-dimensional tensors"
        )
    if row_sums.shape[0] < 2 or row_sums.shape[0] % 2:
        raise ValueError(
            "adjacent pair objective requires an even physical batch with at least two rows"
        )
    pair_sums = row_sums.reshape(-1, 2).sum(dim=1)
    pair_counts = row_counts.reshape(-1, 2).sum(dim=1)
    return (pair_sums / pair_counts.clamp_min(1.0)).sum()


def adjacent_pair_loss_sum(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_backend: LossBackend = "reference",
    attention_mask: torch.Tensor | None = None,
    activation_checkpoint_windows: int = 0,
    activation_checkpoint_min_padded_sequence_length: int = 0,
    longshot_gradient_checkpointing: bool | None = None,
) -> torch.Tensor:
    """Return the sum of original adjacent-pair token-mean losses."""
    row_sums, row_counts = row_loss_sums_and_counts(
        model,
        input_ids,
        targets,
        loss_backend=loss_backend,
        attention_mask=attention_mask,
        activation_checkpoint_windows=activation_checkpoint_windows,
        activation_checkpoint_min_padded_sequence_length=activation_checkpoint_min_padded_sequence_length,
        longshot_gradient_checkpointing=longshot_gradient_checkpointing,
    )
    return adjacent_pair_loss_sum_from_rows(row_sums, row_counts)


def logical_pair_grouped_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_backend: LossBackend = "reference",
    attention_mask: torch.Tensor | None = None,
    activation_checkpoint_windows: int = 0,
    activation_checkpoint_min_padded_sequence_length: int = 0,
    longshot_gradient_checkpointing: bool | None = None,
) -> torch.Tensor:
    """Average adjacent two-row token-mean losses within one physical group."""
    row_sums, row_counts = row_loss_sums_and_counts(
        model,
        input_ids,
        targets,
        loss_backend=loss_backend,
        attention_mask=attention_mask,
        activation_checkpoint_windows=activation_checkpoint_windows,
        activation_checkpoint_min_padded_sequence_length=activation_checkpoint_min_padded_sequence_length,
        longshot_gradient_checkpointing=longshot_gradient_checkpointing,
    )
    pair_sums = row_sums.reshape(-1, 2).sum(dim=1)
    pair_counts = row_counts.reshape(-1, 2).sum(dim=1)
    return (pair_sums / pair_counts.clamp_min(1.0)).mean()
