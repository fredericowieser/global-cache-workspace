from __future__ import annotations

from typing import Protocol

import torch


class TeacherBackbone(Protocol):
    def parameters(self, recurse: bool = True): ...

    def forward_full(self, input_ids: torch.Tensor) -> torch.Tensor: ...


def split_teacher_reasoning(
    generated_ids: tuple[int, ...],
    *,
    close_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """Keep the teacher trajectory through its first thinking-close separator."""
    if not close_ids:
        raise ValueError("thinking close separator must not be empty")
    stop = _find_subsequence(generated_ids, close_ids)
    if stop is None:
        raise ValueError("teacher generation is missing the thinking close separator")
    return generated_ids[: stop + len(close_ids)]


@torch.no_grad()
def teacher_answer_logits(
    backbone: TeacherBackbone,
    teacher_context_ids: tuple[int, ...],
    response_ids: tuple[int, ...],
) -> torch.Tensor:
    """Score student answer tokens after privileged teacher reasoning context."""
    if not teacher_context_ids:
        raise ValueError("teacher context must not be empty")
    if not response_ids:
        raise ValueError("teacher scoring requires at least one response token")
    device = _module_device(backbone)
    input_ids = torch.tensor([teacher_context_ids + response_ids], dtype=torch.long, device=device)
    logits = backbone.forward_full(input_ids)
    start = len(teacher_context_ids) - 1
    stop = start + len(response_ids)
    aligned = logits[:, start:stop, :]
    if aligned.shape[1] != len(response_ids):
        raise RuntimeError("teacher logits did not align with every response token")
    return aligned.detach()


def _find_subsequence(sequence: tuple[int, ...], pattern: tuple[int, ...]) -> int | None:
    last_start = len(sequence) - len(pattern)
    for start in range(last_start + 1):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return None


def _module_device(module: TeacherBackbone) -> torch.device:
    try:
        parameter = next(iter(module.parameters()))
    except StopIteration:
        return torch.device("cpu")
    return parameter.device
