from __future__ import annotations

import torch
import torch.nn.functional as F


def clipped_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    answer_mask: torch.Tensor,
    *,
    temperature: float = 1.1,
    token_clip: float | None = 0.05,
) -> torch.Tensor:
    """Released-OPSD forward KL over aligned student-answer positions.

    Clipping applies to each unreduced vocabulary contribution before masking and
    token normalisation. Teacher logits are detached by construction.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if student_logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if answer_mask.shape != student_logits.shape[:2]:
        raise ValueError("answer_mask must match the batch and sequence dimensions")
    if answer_mask.dtype != torch.bool:
        raise TypeError("answer_mask must be boolean")
    valid_tokens = answer_mask.sum()
    if not bool(valid_tokens):
        raise ValueError("KL requires at least one answer token")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if token_clip is not None and token_clip <= 0:
        raise ValueError("token_clip must be positive or null")

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.detach() / temperature, dim=-1)
    per_vocab = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    if token_clip is not None:
        per_vocab = per_vocab.clamp(max=token_clip)
    return per_vocab[answer_mask].sum() / valid_tokens
