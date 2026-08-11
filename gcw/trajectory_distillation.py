from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

from gcw.distill.prompt import format_chat_prompt
from gcw.models.longshot import Longshot
from gcw.models.windowed_cache import WindowedCache
from gcw.process_branch import (
    InvalidProcessBranchExample,
    ProcessBranchTokenExample,
    branch_next_token_loss,
    build_process_branch_state,
)
from gcw.process_data import ProcessBranchRecord, ProcessDataError, read_process_branch_records_bytes


@dataclass(frozen=True)
class TrajectoryReplayLosses:
    """Token-normalised losses for the two generation policies used at evaluation."""

    total_loss: torch.Tensor
    rationale_loss: torch.Tensor
    direct_answer_loss: torch.Tensor
    thinking_answer_loss: torch.Tensor
    answer_loss: torch.Tensor
    cycles: int


def authenticate_trajectory_artifact(
    source_bytes: bytes,
    *,
    expected_artifact_sha256: str,
    expected_source_ids_sha256: str,
) -> list[ProcessBranchRecord]:
    """Authenticate one immutable trajectory JSONL snapshot before parsing it."""
    if not isinstance(source_bytes, bytes):
        raise TypeError("trajectory authentication requires an immutable bytes snapshot")
    if hashlib.sha256(source_bytes).hexdigest() != expected_artifact_sha256:
        raise ProcessDataError("trajectory artifact sha256 mismatch")
    records = read_process_branch_records_bytes(source_bytes)
    encoded_source_ids = json.dumps(
        [record.source_id for record in records],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if hashlib.sha256(encoded_source_ids).hexdigest() != expected_source_ids_sha256:
        raise ProcessDataError("trajectory ordered source-ID sha256 mismatch")
    return records


def tokenize_trajectory_record(
    record: ProcessBranchRecord,
    tokenizer: Any,
    *,
    prompt_suffix: str,
    rationale_joiner: str,
    reasoning_close_text: str,
    max_prompt_tokens: int,
    max_trajectory_tokens: int,
    max_answer_tokens: int,
    device: torch.device | str = "cpu",
) -> ProcessBranchTokenExample:
    """Build aligned thinking-on and thinking-off teacher-forcing trajectories.

    The user-message cache is shared by both modes. ``step_response_prefix_ids``
    stores the tokenizer's exact thinking-on assistant trigger and
    ``answer_response_prefix_ids`` stores its thinking-off trigger. The single
    reasoning tensor is the full canonical rationale followed by the thinking
    close marker. Keeping the close marker in the rationale region makes the
    answer loss cover only answer tokens in both modes.
    """
    prompt = record.prompt
    if prompt_suffix.strip():
        prompt = f"{prompt}\n\n{prompt_suffix.strip()}"
    try:
        thinking = format_chat_prompt(tokenizer, prompt, enable_thinking=True)
        direct = format_chat_prompt(tokenizer, prompt, enable_thinking=False)
    except (TypeError, ValueError) as exc:
        raise InvalidProcessBranchExample("invalid_chat_template") from exc
    if thinking.base_ids != direct.base_ids:
        raise InvalidProcessBranchExample("thinking_modes_change_user_prompt")
    if len(thinking.base_ids) > max_prompt_tokens:
        raise InvalidProcessBranchExample("prompt_too_long")

    rationale_text = rationale_joiner.join(record.reasoning_steps).strip()
    if not rationale_text:
        raise InvalidProcessBranchExample("empty_rationale")
    rationale_ids = [int(token_id) for token_id in tokenizer.encode(rationale_text)]
    close_ids = [int(token_id) for token_id in tokenizer.encode(reasoning_close_text)]
    if not rationale_ids:
        raise InvalidProcessBranchExample("empty_rationale")
    if not close_ids:
        raise InvalidProcessBranchExample("empty_reasoning_close")
    trajectory_ids = rationale_ids + close_ids
    if len(trajectory_ids) > max_trajectory_tokens:
        raise InvalidProcessBranchExample("trajectory_too_long")

    answer_ids = [int(token_id) for token_id in tokenizer.encode(record.final_answer)]
    answer_ids.append(int(tokenizer.eot_id))
    if len(answer_ids) > max_answer_tokens:
        raise InvalidProcessBranchExample("answer_too_long")

    return ProcessBranchTokenExample(
        prompt_ids=torch.tensor(thinking.base_ids, dtype=torch.long, device=device),
        step_response_prefix_ids=torch.tensor(
            thinking.trigger_ids, dtype=torch.long, device=device
        ),
        answer_response_prefix_ids=torch.tensor(
            direct.trigger_ids, dtype=torch.long, device=device
        ),
        reasoning_step_ids=(
            torch.tensor(trajectory_ids, dtype=torch.long, device=device),
        ),
        final_answer_ids=torch.tensor(answer_ids, dtype=torch.long, device=device),
    )


def _masked_trajectory_targets(
    *,
    prefix_length: int,
    rationale_length: int,
    answer_length: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rationale/answer masks aligned to prefix + trajectory[:-1] inputs."""
    if min(prefix_length, rationale_length, answer_length) <= 0:
        raise ValueError("trajectory prefix, rationale, and answer must all be non-empty")
    target_length = prefix_length - 1 + rationale_length + answer_length
    rationale_targets = torch.full((1, target_length), -1, dtype=torch.long, device=device)
    answer_targets = torch.full((1, target_length), -1, dtype=torch.long, device=device)
    rationale_start = prefix_length - 1
    answer_start = rationale_start + rationale_length
    # Sentinel token IDs are replaced by actual targets in trajectory_replay_example_loss.
    rationale_targets[:, rationale_start:answer_start] = 0
    answer_targets[:, answer_start:] = 0
    return rationale_targets, answer_targets


def _thinking_trajectory_losses(
    model: Longshot,
    branch_cache: WindowedCache,
    example: ProcessBranchTokenExample,
) -> tuple[torch.Tensor, torch.Tensor]:
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise ValueError("trajectory replay requires a frozen backbone")
    if len(example.reasoning_step_ids) != 1:
        raise ValueError("trajectory replay requires exactly one full-rationale tensor")
    prefix = example.step_response_prefix_ids.unsqueeze(0)
    rationale = example.reasoning_step_ids[0].unsqueeze(0)
    answer = example.final_answer_ids.unsqueeze(0)
    trajectory = torch.cat((rationale, answer), dim=1)
    inputs = torch.cat((prefix, trajectory[:, :-1]), dim=1)
    rationale_targets, answer_targets = _masked_trajectory_targets(
        prefix_length=prefix.shape[1],
        rationale_length=rationale.shape[1],
        answer_length=answer.shape[1],
        device=inputs.device,
    )
    target_offset = prefix.shape[1] - 1
    rationale_end = target_offset + rationale.shape[1]
    rationale_targets[:, target_offset:rationale_end] = rationale
    answer_targets[:, rationale_end:] = answer
    hidden, _ = model.backbone.forward_chunk(
        inputs,
        branch_cache.seq_len,
        branch_cache,
        logits=False,
        cache_format="windowed",
    )
    return (
        model.backbone.loss_from_hidden(hidden, rationale_targets, "mean"),
        model.backbone.loss_from_hidden(hidden, answer_targets, "mean"),
    )


def trajectory_replay_example_loss(
    model: Longshot,
    example: ProcessBranchTokenExample,
    *,
    cycles: int,
    rationale_weight: float,
    answer_weight: float,
) -> TrajectoryReplayLosses:
    """Distil a canonical rationale plus answer into the processed prompt cache.

    Two answer losses deliberately match deployment: thinking-on predicts the
    answer after replaying the canonical rationale, while thinking-off predicts
    the answer directly. The arithmetic mean prevents the longer rationale from
    drowning out either answer policy. No shuffled-workspace hinge is optimised.
    """
    if rationale_weight < 0.0 or answer_weight <= 0.0:
        raise ValueError("trajectory replay weights must be non-negative/positive")
    state = build_process_branch_state(model, example, cycles=cycles)
    processor = model.cache_processor
    write_for_readout = getattr(processor, "write_for_readout", None)
    if not callable(write_for_readout):
        raise TypeError("trajectory replay requires a Longshot cache processor")
    branch_cache = write_for_readout(
        state.prompt_cache,
        state.source,
        state.workspace,
    )
    if not isinstance(branch_cache, WindowedCache):
        raise TypeError("trajectory replay readout requires a WindowedCache")
    rationale_loss, thinking_answer_loss = _thinking_trajectory_losses(
        model, branch_cache, example
    )
    direct_answer_loss = branch_next_token_loss(
        model.backbone,
        branch_cache,
        start_pos=example.prompt_ids.numel(),
        response_prefix_ids=example.answer_response_prefix_ids.unsqueeze(0),
        target_ids=example.final_answer_ids.unsqueeze(0),
    )
    combined_answer_loss = 0.5 * (direct_answer_loss + thinking_answer_loss)
    total_loss = rationale_weight * rationale_loss + answer_weight * combined_answer_loss
    return TrajectoryReplayLosses(
        total_loss=total_loss,
        rationale_loss=rationale_loss,
        direct_answer_loss=direct_answer_loss,
        thinking_answer_loss=thinking_answer_loss,
        answer_loss=combined_answer_loss,
        cycles=state.cycles,
    )
