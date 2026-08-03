from __future__ import annotations

from dataclasses import dataclass

import torch

from gcw.distill.objective import exact_forward_kl_with_metrics
from gcw.distill.prompt import ChatPrompt
from gcw.distill.session import BGTDecodeSession
from gcw.distill.teacher import teacher_answer_logits
from gcw.models.bgt import BGT


@dataclass(frozen=True)
class DistillationExample:
    student_prompt: ChatPrompt
    teacher_prompt: ChatPrompt
    teacher_reasoning_ids: tuple[int, ...]
    response_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.teacher_reasoning_ids:
            raise ValueError("distillation example requires teacher reasoning")
        if not self.response_ids:
            raise ValueError("distillation example requires a student response")


@dataclass(frozen=True)
class DistillationLossResult:
    loss: torch.Tensor
    unclipped_kl: torch.Tensor
    clipping_fraction: torch.Tensor
    student_entropy: torch.Tensor
    teacher_entropy: torch.Tensor
    answer_tokens: int
    event_cache_lengths: tuple[int, ...]
    event_relative_l2_displacements: tuple[float, ...]


def distillation_example_loss(
    model: BGT,
    example: DistillationExample,
    session: BGTDecodeSession,
    *,
    temperature: float,
    token_clip: float | None,
    sequence_chunk_size: int = 64,
) -> DistillationLossResult:
    """Compute CoT-conditioned forward KL on one on-policy student response."""
    student = session.replay(example.student_prompt, example.response_ids)
    teacher_context = example.teacher_prompt.full_ids + example.teacher_reasoning_ids
    teacher = teacher_answer_logits(model.backbone, teacher_context, example.response_ids)
    if teacher.device != student.logits.device:
        teacher = teacher.to(student.logits.device)
    answer_mask = torch.ones(student.logits.shape[:2], dtype=torch.bool, device=student.logits.device)
    objective = exact_forward_kl_with_metrics(
        student.logits,
        teacher,
        answer_mask,
        temperature=temperature,
        token_clip=token_clip,
        sequence_chunk_size=sequence_chunk_size,
    )
    return DistillationLossResult(
        loss=objective.loss,
        unclipped_kl=objective.unclipped_kl,
        clipping_fraction=objective.clipping_fraction,
        student_entropy=objective.student_entropy,
        teacher_entropy=objective.teacher_entropy,
        answer_tokens=len(example.response_ids),
        event_cache_lengths=student.event_cache_lengths,
        event_relative_l2_displacements=student.event_relative_l2_displacements,
    )
