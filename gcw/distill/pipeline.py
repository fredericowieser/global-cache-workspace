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
    teacher_response_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.teacher_reasoning_ids:
            raise ValueError("distillation example requires teacher reasoning")
        if not self.response_ids:
            raise ValueError("distillation example requires a student response")
        if self.teacher_response_ids is not None and not self.teacher_response_ids:
            raise ValueError("teacher visible response must not be empty")


@dataclass(frozen=True)
class DistillationLossResult:
    loss: torch.Tensor
    unclipped_kl: torch.Tensor
    clipping_fraction: torch.Tensor
    student_entropy: torch.Tensor
    teacher_entropy: torch.Tensor
    answer_tokens: int
    student_trajectory_tokens: int
    teacher_trajectory_tokens: int
    event_cache_lengths: tuple[int, ...]
    event_relative_l2_displacements: tuple[float, ...]
    teacher_trajectory_event_cache_lengths: tuple[int, ...]
    teacher_trajectory_event_relative_l2_displacements: tuple[float, ...]


def distillation_example_loss(
    model: BGT,
    example: DistillationExample,
    session: BGTDecodeSession,
    *,
    temperature: float,
    token_clip: float | None,
    sequence_chunk_size: int = 64,
    visible_trajectory_mode: str = "student",
) -> DistillationLossResult:
    """Compute CoT-conditioned forward KL on shared visible response prefixes."""
    if visible_trajectory_mode not in {"student", "teacher", "both"}:
        raise ValueError("visible_trajectory_mode must be student, teacher, or both")
    if visible_trajectory_mode == "both" and torch.is_grad_enabled():
        raise RuntimeError(
            "both visible trajectories require serial backward; use the distillation trainer"
        )
    if visible_trajectory_mode in {"teacher", "both"} and example.teacher_response_ids is None:
        raise ValueError("teacher visible trajectory mode requires teacher_response_ids")

    trajectory_ids: list[tuple[str, tuple[int, ...]]] = []
    if visible_trajectory_mode in {"student", "both"}:
        trajectory_ids.append(("student", example.response_ids))
    if visible_trajectory_mode in {"teacher", "both"}:
        assert example.teacher_response_ids is not None
        trajectory_ids.append(("teacher", example.teacher_response_ids))

    teacher_context = example.teacher_prompt.full_ids + example.teacher_reasoning_ids
    objectives = []
    event_cache_lengths: dict[str, tuple[int, ...]] = {}
    event_displacements: dict[str, tuple[float, ...]] = {}
    for name, response_ids in trajectory_ids:
        student = session.replay(example.student_prompt, response_ids)
        teacher = teacher_answer_logits(model.backbone, teacher_context, response_ids)
        if teacher.device != student.logits.device:
            teacher = teacher.to(student.logits.device)
        answer_mask = torch.ones(
            student.logits.shape[:2], dtype=torch.bool, device=student.logits.device
        )
        objectives.append(
            exact_forward_kl_with_metrics(
                student.logits,
                teacher,
                answer_mask,
                temperature=temperature,
                token_clip=token_clip,
                sequence_chunk_size=sequence_chunk_size,
            )
        )
        event_cache_lengths[name] = student.event_cache_lengths
        event_displacements[name] = student.event_relative_l2_displacements
        del answer_mask, teacher, student

    def mean_metric(name: str) -> torch.Tensor:
        return torch.stack([getattr(objective, name) for objective in objectives]).mean()

    primary_name = "student" if "student" in event_cache_lengths else "teacher"
    student_tokens = len(example.response_ids) if "student" in event_cache_lengths else 0
    teacher_tokens = (
        len(example.teacher_response_ids or ()) if "teacher" in event_cache_lengths else 0
    )
    return DistillationLossResult(
        loss=mean_metric("loss"),
        unclipped_kl=mean_metric("unclipped_kl"),
        clipping_fraction=mean_metric("clipping_fraction"),
        student_entropy=mean_metric("student_entropy"),
        teacher_entropy=mean_metric("teacher_entropy"),
        answer_tokens=student_tokens + teacher_tokens,
        student_trajectory_tokens=student_tokens,
        teacher_trajectory_tokens=teacher_tokens,
        event_cache_lengths=event_cache_lengths[primary_name],
        event_relative_l2_displacements=event_displacements[primary_name],
        teacher_trajectory_event_cache_lengths=event_cache_lengths.get("teacher", ()),
        teacher_trajectory_event_relative_l2_displacements=event_displacements.get(
            "teacher", ()
        ),
    )
