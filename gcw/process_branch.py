from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from gcw.models.longshot import GlobalCacheProcessor, Longshot, LongshotWorkspaceSource
from gcw.models.transformer import TF
from gcw.models.windowed_cache import WindowedCache
from gcw.process_data import ProcessBranchRecord


class InvalidProcessBranchExample(ValueError):
    pass


def _longshot_processor(model: Longshot) -> GlobalCacheProcessor:
    return cast(GlobalCacheProcessor, model.cache_processor)


@dataclass(frozen=True)
class ProcessBranchTokenExample:
    prompt_ids: torch.Tensor
    step_response_prefix_ids: torch.Tensor
    answer_response_prefix_ids: torch.Tensor
    reasoning_step_ids: tuple[torch.Tensor, ...]
    final_answer_ids: torch.Tensor


@dataclass(frozen=True)
class ProcessBranchLosses:
    total_loss: torch.Tensor
    process_loss: torch.Tensor
    answer_loss: torch.Tensor
    cycles: int


@dataclass(frozen=True)
class ProcessBranchState:
    prompt_cache: WindowedCache
    source: LongshotWorkspaceSource
    workspace: torch.Tensor
    cycles: int


def build_process_branch_state(
    model: Longshot,
    example: ProcessBranchTokenExample,
    *,
    cycles: int,
) -> ProcessBranchState:
    if cycles < 0:
        raise ValueError("process-branch cycles must be non-negative")
    prompt_ids = example.prompt_ids.unsqueeze(0)
    with torch.no_grad():
        _, prompt_cache = model.backbone.forward_chunk(
            prompt_ids,
            0,
            None,
            logits=False,
            cache_format="windowed",
        )
    if not isinstance(prompt_cache, WindowedCache):
        raise TypeError("process-branch prefill requires a WindowedCache")
    processor = _longshot_processor(model)
    source = processor.encode_source(prompt_cache, select_all=True)
    workspace = processor.initialise_workspace(source)
    for _ in range(cycles):
        workspace = processor.deliberate(workspace, source)
    return ProcessBranchState(
        prompt_cache=prompt_cache,
        source=source,
        workspace=workspace,
        cycles=cycles,
    )


def branch_answer_loss(
    model: Longshot,
    example: ProcessBranchTokenExample,
    state: ProcessBranchState,
    *,
    workspace_override: torch.Tensor | None = None,
) -> torch.Tensor:
    workspace = state.workspace if workspace_override is None else workspace_override
    if workspace.shape != state.workspace.shape:
        raise ValueError("workspace override shape must match the recipient workspace")
    answer_cache = _longshot_processor(model).write_for_readout(
        state.prompt_cache,
        state.source,
        workspace,
    )
    return branch_next_token_loss(
        model.backbone,
        answer_cache,
        start_pos=example.prompt_ids.numel(),
        response_prefix_ids=example.answer_response_prefix_ids.unsqueeze(0),
        target_ids=example.final_answer_ids.unsqueeze(0),
    )


def tokenize_process_record(
    record: ProcessBranchRecord,
    tokenizer: Any,
    *,
    step_response_prefix: str,
    answer_response_prefix: str,
    prompt_suffix: str,
    cycle_count: int,
    max_prompt_tokens: int,
    max_step_tokens: int,
    max_answer_tokens: int,
    device: torch.device | str = "cpu",
) -> ProcessBranchTokenExample:
    if cycle_count <= 0:
        raise ValueError("cycle_count must be positive")
    if len(record.reasoning_steps) < cycle_count:
        raise InvalidProcessBranchExample("insufficient_reasoning_steps")
    prompt = record.prompt
    if prompt_suffix.strip():
        prompt = f"{prompt}\n\n{prompt_suffix.strip()}"
    prompt_ids = tokenizer.encode(prompt)
    step_prefix_ids = tokenizer.encode(step_response_prefix)
    answer_prefix_ids = tokenizer.encode(answer_response_prefix)
    if not step_prefix_ids:
        raise InvalidProcessBranchExample("empty_step_response_prefix")
    if not answer_prefix_ids:
        raise InvalidProcessBranchExample("empty_answer_response_prefix")
    if not prompt_ids:
        raise InvalidProcessBranchExample("empty_prompt")
    if len(prompt_ids) > max_prompt_tokens:
        raise InvalidProcessBranchExample("prompt_too_long")

    step_ids = []
    for step in record.reasoning_steps[:cycle_count]:
        encoded = tokenizer.encode(step) + [int(tokenizer.eot_id)]
        if len(encoded) > max_step_tokens:
            raise InvalidProcessBranchExample("reasoning_step_too_long")
        step_ids.append(torch.tensor(encoded, dtype=torch.long, device=device))
    answer_ids = tokenizer.encode(record.final_answer) + [int(tokenizer.eot_id)]
    if len(answer_ids) > max_answer_tokens:
        raise InvalidProcessBranchExample("answer_too_long")

    return ProcessBranchTokenExample(
        prompt_ids=torch.tensor(prompt_ids, dtype=torch.long, device=device),
        step_response_prefix_ids=torch.tensor(step_prefix_ids, dtype=torch.long, device=device),
        answer_response_prefix_ids=torch.tensor(answer_prefix_ids, dtype=torch.long, device=device),
        reasoning_step_ids=tuple(step_ids),
        final_answer_ids=torch.tensor(answer_ids, dtype=torch.long, device=device),
    )


def branch_next_token_loss(
    backbone: TF,
    branch_cache: WindowedCache,
    *,
    start_pos: int,
    response_prefix_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("process-branch decoding requires a frozen backbone")
    if response_prefix_ids.ndim != 2 or target_ids.ndim != 2:
        raise ValueError("response prefix and target IDs must be rank-2 tensors")
    if response_prefix_ids.shape[0] != target_ids.shape[0]:
        raise ValueError("response prefix and target batch sizes must match")
    if response_prefix_ids.shape[1] == 0:
        raise ValueError("response prefix must contain at least one token")
    if target_ids.shape[1] == 0:
        raise ValueError("branch target must contain at least one token")
    if start_pos != branch_cache.seq_len:
        raise ValueError(
            f"branch start position {start_pos} must equal cache length {branch_cache.seq_len}"
        )

    input_ids = torch.cat([response_prefix_ids, target_ids[:, :-1]], dim=1)
    masked_prefix = target_ids.new_full(
        (target_ids.shape[0], response_prefix_ids.shape[1] - 1),
        -1,
    )
    targets = torch.cat([masked_prefix, target_ids], dim=1)
    hidden, _ = backbone.forward_chunk(
        input_ids,
        start_pos,
        branch_cache,
        logits=False,
        cache_format="windowed",
    )
    return backbone.loss_from_hidden(hidden, targets, "mean")


def process_branch_example_loss(
    model: Longshot,
    example: ProcessBranchTokenExample,
    *,
    process_weight: float,
    answer_weight: float,
) -> ProcessBranchLosses:
    prompt_ids = example.prompt_ids.unsqueeze(0)
    step_response_prefix_ids = example.step_response_prefix_ids.unsqueeze(0)
    with torch.no_grad():
        _, prompt_cache = model.backbone.forward_chunk(
            prompt_ids,
            0,
            None,
            logits=False,
            cache_format="windowed",
        )
    if not isinstance(prompt_cache, WindowedCache):
        raise TypeError("process-branch prefill requires a WindowedCache")

    processor = _longshot_processor(model)
    source = processor.encode_source(prompt_cache, select_all=True)
    workspace = processor.initialise_workspace(source)
    process_losses = []
    for step_ids in example.reasoning_step_ids:
        workspace = processor.deliberate(workspace, source)
        branch_cache = processor.write_for_readout(prompt_cache, source, workspace)
        process_losses.append(
            branch_next_token_loss(
                model.backbone,
                branch_cache,
                start_pos=prompt_ids.shape[1],
                response_prefix_ids=step_response_prefix_ids,
                target_ids=step_ids.unsqueeze(0),
            )
        )

    state = ProcessBranchState(
        prompt_cache=prompt_cache,
        source=source,
        workspace=workspace,
        cycles=len(process_losses),
    )
    answer_loss = branch_answer_loss(model, example, state)
    if process_losses:
        process_loss = torch.stack(process_losses).mean()
    else:
        process_loss = answer_loss.new_zeros(())
    total_loss = process_weight * process_loss + answer_weight * answer_loss
    return ProcessBranchLosses(
        total_loss=total_loss,
        process_loss=process_loss,
        answer_loss=answer_loss,
        cycles=len(process_losses),
    )
