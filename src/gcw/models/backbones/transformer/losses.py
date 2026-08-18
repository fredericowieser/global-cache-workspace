from __future__ import annotations

from typing import Callable, cast
import torch
import torch.nn.functional as F

from gcw.models.backbones.transformer.layers import Linear


def answer_only_loss_from_hidden(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    targets: torch.Tensor,
    loss_reduction: str,
    *,
    tie_embeddings: bool,
    embed_weight: torch.Tensor,
    lm_head: Linear | None,
) -> torch.Tensor:
    """Compute loss without projecting prompt or padding positions through the vocabulary head."""
    chunk_size = 256 if x.is_cuda else 64
    loss_sum = x.new_zeros((), dtype=torch.float32)
    row_loss_sums = x.new_zeros((x.shape[0],), dtype=torch.float32)
    token_count = x.new_zeros((), dtype=torch.float32)
    per_token: list[torch.Tensor] = []
    any_valid = False
    for start in range(0, x.shape[1], chunk_size):
        end = start + chunk_size
        hidden_chunk = x[:, start:end]
        target_chunk = targets[:, start:end]
        flat_targets = target_chunk.reshape(-1)
        valid = flat_targets >= 0
        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if valid_indices.numel() > 0:
            any_valid = True
            flat_hidden = hidden_chunk.reshape(-1, hidden_chunk.shape[-1])
            selected_hidden = flat_hidden.index_select(0, valid_indices)
            logits = logits_fn(selected_hidden)
            selected_targets = flat_targets.index_select(0, valid_indices)
            selected_loss = F.cross_entropy(logits, selected_targets, reduction="none")
            chunk_loss = logits.new_zeros(flat_targets.shape).index_copy(0, valid_indices, selected_loss)
        else:
            chunk_loss = x.new_zeros(flat_targets.shape, dtype=torch.float32)
        loss_sum = loss_sum + chunk_loss.sum()
        token_count = token_count + valid.sum().to(torch.float32)
        if loss_reduction == "row_sum":
            row_loss_sums = row_loss_sums + chunk_loss.reshape(target_chunk.shape).sum(dim=1)
        if loss_reduction == "none":
            per_token.append(chunk_loss.reshape(target_chunk.shape))
    if not any_valid:
        head = embed_weight if tie_embeddings else cast(Linear, lm_head).weight
        connected_zero = (x.reshape(-1)[:1].float().sum() + head.reshape(-1)[:1].float().sum()) * 0.0
        if loss_reduction == "none":
            return torch.cat(per_token, dim=1) + connected_zero
        if loss_reduction == "row_sum":
            return row_loss_sums + connected_zero
        return loss_sum + connected_zero
    if loss_reduction == "none":
        return torch.cat(per_token, dim=1)
    if loss_reduction == "row_sum":
        return row_loss_sums
    if loss_reduction == "sum":
        return loss_sum
    return loss_sum / token_count.clamp_min(1.0)


def loss_from_hidden_reference(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    targets: torch.Tensor,
    loss_reduction: str,
) -> torch.Tensor:
    """Eager loss oracle which projects every position, including ignored positions."""
    chunk_size = 256 if x.is_cuda else 64
    loss_sum = x.new_zeros((), dtype=torch.float32)
    row_loss_sums = x.new_zeros((x.shape[0],), dtype=torch.float32)
    token_count = x.new_zeros((), dtype=torch.float32)
    per_token: list[torch.Tensor] = []
    for start in range(0, x.shape[1], chunk_size):
        end = start + chunk_size
        logits = logits_fn(x[:, start:end])
        target_chunk = targets[:, start:end]
        chunk_targets = target_chunk.reshape(-1).clone()
        valid = chunk_targets >= 0
        chunk_targets[~valid] = -100
        chunk_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            chunk_targets,
            ignore_index=-100,
            reduction="none",
        )
        loss_sum = loss_sum + chunk_loss.sum()
        token_count = token_count + valid.sum().to(torch.float32)
        if loss_reduction == "row_sum":
            row_loss_sums = row_loss_sums + chunk_loss.reshape(target_chunk.shape).sum(dim=1)
        if loss_reduction == "none":
            per_token.append(chunk_loss.reshape(target_chunk.shape))
    if loss_reduction == "none":
        return torch.cat(per_token, dim=1)
    if loss_reduction == "row_sum":
        return row_loss_sums
    if loss_reduction == "sum":
        return loss_sum
    return loss_sum / token_count.clamp_min(1.0)
