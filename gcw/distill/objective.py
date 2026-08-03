from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ForwardKLMetrics:
    loss: torch.Tensor
    unclipped_kl: torch.Tensor
    clipping_fraction: torch.Tensor
    student_entropy: torch.Tensor
    teacher_entropy: torch.Tensor


def clipped_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    answer_mask: torch.Tensor,
    *,
    temperature: float = 1.1,
    token_clip: float | None = 0.05,
    sequence_chunk_size: int = 64,
) -> torch.Tensor:
    return clipped_forward_kl_with_metrics(
        student_logits,
        teacher_logits,
        answer_mask,
        temperature=temperature,
        token_clip=token_clip,
        sequence_chunk_size=sequence_chunk_size,
    ).loss


def clipped_forward_kl_with_metrics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    answer_mask: torch.Tensor,
    *,
    temperature: float = 1.1,
    token_clip: float | None = 0.05,
    sequence_chunk_size: int = 64,
) -> ForwardKLMetrics:
    """Released OPSD forward KL with chunked exact diagnostics."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if student_logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if answer_mask.shape != student_logits.shape[:2]:
        raise ValueError("answer_mask must have shape [batch, sequence]")
    if answer_mask.dtype != torch.bool:
        raise TypeError("answer_mask must be boolean")
    valid_tokens = answer_mask.sum()
    if int(valid_tokens.item()) == 0:
        raise ValueError("answer_mask must include at least one answer token")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if token_clip is not None and token_clip <= 0:
        raise ValueError("token_clip must be positive or null")
    if sequence_chunk_size <= 0:
        raise ValueError("sequence_chunk_size must be positive")

    device = student_logits.device
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    unclipped_sum = torch.zeros_like(loss_sum)
    student_entropy_sum = torch.zeros_like(loss_sum)
    teacher_entropy_sum = torch.zeros_like(loss_sum)
    clipped_count = torch.zeros((), dtype=torch.long, device=device)
    sequence_length = student_logits.shape[1]
    vocabulary_size = student_logits.shape[2]
    for start in range(0, sequence_length, sequence_chunk_size):
        stop = min(start + sequence_chunk_size, sequence_length)
        mask = answer_mask[:, start:stop]
        mask_float = mask.to(torch.float32)
        student_log_probs = F.log_softmax(student_logits[:, start:stop].float() / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(
            teacher_logits[:, start:stop].detach().float() / temperature,
            dim=-1,
        )
        per_vocab = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        unclipped_sum = unclipped_sum + (per_vocab.detach().sum(dim=-1) * mask_float).sum()
        with torch.no_grad():
            student_entropy = -(student_log_probs.exp() * student_log_probs).sum(dim=-1)
            teacher_entropy = -(teacher_log_probs.exp() * teacher_log_probs).sum(dim=-1)
            student_entropy_sum += (student_entropy * mask_float).sum()
            teacher_entropy_sum += (teacher_entropy * mask_float).sum()
            if token_clip is not None:
                clipped_count += ((per_vocab > token_clip) & mask.unsqueeze(-1)).sum()
        clipped = per_vocab.clamp(max=token_clip) if token_clip is not None else per_vocab
        loss_sum = loss_sum + (clipped.sum(dim=-1) * mask_float).sum()

    denominator = valid_tokens.to(torch.float32)
    clipping_denominator = denominator * vocabulary_size
    return ForwardKLMetrics(
        loss=loss_sum / denominator,
        unclipped_kl=unclipped_sum / denominator,
        clipping_fraction=clipped_count.to(torch.float32) / clipping_denominator,
        student_entropy=student_entropy_sum / denominator,
        teacher_entropy=teacher_entropy_sum / denominator,
    )
