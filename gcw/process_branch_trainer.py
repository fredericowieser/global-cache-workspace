from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time
from typing import Any

import torch

from gcw.checkpoint import load_backbone_checkpoint, load_checkpoint, save_checkpoint
from gcw.config import TrainConfig, to_dict
from gcw.data.tokenizers import get_tokenizer
from gcw.models import build_model
from gcw.models.longshot import Longshot
from gcw.process_branch import (
    InvalidProcessBranchExample,
    ProcessBranchTokenExample,
    branch_answer_loss,
    build_process_branch_state,
    process_branch_example_loss,
    tokenize_process_record,
)
from gcw.process_data import ProcessBranchRecord, read_process_branch_records
from gcw.state_hash import tensor_state_sha256


def train_process_branch(config: TrainConfig) -> None:
    process = config.process_branch
    if config.mode != "process_branch" or process is None:
        raise ValueError("train_process_branch requires process_branch mode and configuration")
    if config.data.data_dir is None:
        raise ValueError("process_branch requires data.data_dir")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "metrics.jsonl"
    final_checkpoint = output_dir / "final"
    (output_dir / "run_config.json").write_text(
        json.dumps(to_dict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tokenizer_name = config.data.tokenizer
    if tokenizer_name == "hf":
        if not config.data.tokenizer_source:
            raise ValueError("process_branch tokenizer='hf' requires data.tokenizer_source")
        tokenizer_name = f"hf:{config.data.tokenizer_source}"
    tokenizer = get_tokenizer(tokenizer_name)
    if tokenizer.get_vocab_size() > config.model.vocab_size:
        raise ValueError("process_branch tokenizer vocabulary exceeds model vocabulary")
    records = read_process_branch_records(
        config.data.data_dir,
        skip_rows=config.data.sft_skip_rows,
        take_rows=config.data.sft_take_rows,
    )
    if not records:
        raise ValueError("process_branch training slice is empty")
    heldout_records = (
        read_process_branch_records(
            config.data.data_dir,
            skip_rows=process.heldout_skip_rows,
            take_rows=process.heldout_take_rows,
        )
        if process.heldout_take_rows
        else []
    )
    overlap = {record.source_id for record in records} & {
        record.source_id for record in heldout_records
    }
    if overlap:
        raise ValueError(f"process_branch train/held-out overlap: {sorted(overlap)[:4]}")

    torch.manual_seed(config.seed)
    random_generator = random.Random(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model)
    if not isinstance(model, Longshot):
        raise TypeError("process_branch training requires a Longshot model")
    model.init_weights()
    load_backbone_checkpoint(model, config.backbone_checkpoint or "")
    model.backbone.requires_grad_(False)
    model.backbone.eval()
    model.cache_processor.train()
    model.to(device)
    if config.bf16 and device.type == "cuda":
        model.backbone.to(dtype=torch.bfloat16)

    valid_records, rejection_reasons = _valid_records(records, tokenizer, config)
    if not valid_records:
        raise ValueError("process_branch has no valid token-bounded records")
    valid_heldout, heldout_rejection_reasons = _valid_records(heldout_records, tokenizer, config)
    if heldout_records and len(valid_heldout) < 2:
        raise ValueError("process_branch requires at least two valid held-out records")

    parameters = [parameter for parameter in model.cache_processor.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    initial_processor_sha256 = tensor_state_sha256(model.cache_processor.state_dict())
    started = time.monotonic()
    optimizer_steps = 0
    accumulated = 0
    seen_examples = 0
    metric_sums = {"total": 0.0, "process": 0.0, "answer": 0.0, "cycles": 0.0}
    order = list(range(len(valid_records)))
    optimizer.zero_grad(set_to_none=True)

    with metrics_path.open("x", encoding="utf-8") as metrics_handle:
        while optimizer_steps < process.max_optimizer_steps:
            random_generator.shuffle(order)
            for record_index in order:
                cycle_count = process.cycle_choices[seen_examples % len(process.cycle_choices)]
                example = tokenize_process_record(
                    valid_records[record_index],
                    tokenizer,
                    step_response_prefix=process.step_response_prefix,
                    answer_response_prefix=config.data.response_prefix,
                    prompt_suffix=config.data.prompt_suffix,
                    cycle_count=cycle_count,
                    max_prompt_tokens=process.max_prompt_tokens,
                    max_step_tokens=process.max_step_tokens,
                    max_answer_tokens=process.max_answer_tokens,
                    device=device,
                )
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if config.bf16 and device.type == "cuda"
                    else nullcontext()
                )
                with context:
                    losses = process_branch_example_loss(
                        model,
                        example,
                        process_weight=process.process_weight,
                        answer_weight=process.answer_weight,
                    )
                if not bool(torch.isfinite(losses.total_loss)):
                    raise FloatingPointError("process_branch loss is non-finite")
                (losses.total_loss / process.gradient_accumulation_steps).backward()
                accumulated += 1
                seen_examples += 1
                metric_sums["total"] += float(losses.total_loss.detach().float())
                metric_sums["process"] += float(losses.process_loss.detach().float())
                metric_sums["answer"] += float(losses.answer_loss.detach().float())
                metric_sums["cycles"] += losses.cycles
                if accumulated < process.gradient_accumulation_steps:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                payload = {
                    "train/step": optimizer_steps,
                    "train/loss": metric_sums["total"] / accumulated,
                    "train/process_loss": metric_sums["process"] / accumulated,
                    "train/answer_loss": metric_sums["answer"] / accumulated,
                    "train/cycles": metric_sums["cycles"] / accumulated,
                    "train/grad_norm": float(grad_norm.detach().float()),
                    "train/examples": seen_examples,
                    "train/elapsed_seconds": time.monotonic() - started,
                }
                metrics_handle.write(json.dumps(payload, sort_keys=True) + "\n")
                metrics_handle.flush()
                accumulated = 0
                metric_sums = {"total": 0.0, "process": 0.0, "answer": 0.0, "cycles": 0.0}
                if optimizer_steps >= process.max_optimizer_steps:
                    break

    final_processor_sha256 = tensor_state_sha256(model.cache_processor.state_dict())
    heldout_controls = (
        _evaluate_heldout_controls(model, valid_heldout, tokenizer, config, device)
        if valid_heldout
        else None
    )
    train_state: dict[str, Any] = {
        "kind": "gcw.process_branch.train.v1",
        "optimizer_steps": optimizer_steps,
        "seen_examples": seen_examples,
        "source_records": len(records),
        "valid_records": len(valid_records),
        "rejected_records": len(records) - len(valid_records),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "heldout_source_records": len(heldout_records),
        "heldout_valid_records": len(valid_heldout),
        "heldout_rejection_reasons": dict(sorted(heldout_rejection_reasons.items())),
        "initial_processor_sha256": initial_processor_sha256,
        "final_processor_sha256": final_processor_sha256,
        "dtype": (
            "mixed_bfloat16_backbone_float32_processor"
            if config.bf16 and device.type == "cuda"
            else "float32"
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    if heldout_controls is not None:
        train_state["heldout_controls"] = heldout_controls
    save_checkpoint(final_checkpoint, model, config.model, train_state)
    reloaded, reloaded_config, _ = load_checkpoint(final_checkpoint, device="cpu")
    if not isinstance(reloaded, Longshot) or reloaded_config != config.model:
        raise RuntimeError("process_branch checkpoint reload changed the model contract")
    reloaded_hash = tensor_state_sha256(reloaded.cache_processor.state_dict())
    if reloaded_hash != final_processor_sha256:
        raise RuntimeError("process_branch checkpoint reload changed processor tensors")
    train_state["checkpoint_reload_verified"] = True
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(train_state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(train_state, sort_keys=True))


@torch.no_grad()
def _evaluate_heldout_controls(
    model: Longshot,
    records: list[ProcessBranchRecord],
    tokenizer,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float | int]:
    process = config.process_branch
    if process is None:
        raise ValueError("process_branch configuration is missing")
    cycles = max(process.cycle_choices)
    examples: list[ProcessBranchTokenExample] = [
        tokenize_process_record(
            record,
            tokenizer,
            step_response_prefix=process.step_response_prefix,
            answer_response_prefix=config.data.response_prefix,
            prompt_suffix=config.data.prompt_suffix,
            cycle_count=cycles,
            max_prompt_tokens=process.max_prompt_tokens,
            max_step_tokens=process.max_step_tokens,
            max_answer_tokens=process.max_answer_tokens,
            device=device,
        )
        for record in records
    ]
    model.cache_processor.eval()
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if config.bf16 and device.type == "cuda"
        else nullcontext()
    )
    with context:
        matched_states = [
            build_process_branch_state(model, example, cycles=cycles) for example in examples
        ]
        off_states = [
            build_process_branch_state(model, example, cycles=0) for example in examples
        ]
        matched = [
            float(branch_answer_loss(model, example, state).detach().float())
            for example, state in zip(examples, matched_states, strict=True)
        ]
        processor_off = [
            float(branch_answer_loss(model, example, state).detach().float())
            for example, state in zip(examples, off_states, strict=True)
        ]
        shuffled = [
            float(
                branch_answer_loss(
                    model,
                    example,
                    state,
                    workspace_override=matched_states[(index + 1) % len(matched_states)].workspace,
                )
                .detach()
                .float()
            )
            for index, (example, state) in enumerate(
                zip(examples, matched_states, strict=True)
            )
        ]
    model.cache_processor.train()
    matched_mean = sum(matched) / len(matched)
    off_mean = sum(processor_off) / len(processor_off)
    shuffled_mean = sum(shuffled) / len(shuffled)
    return {
        "examples": len(examples),
        "cycles": cycles,
        "matched_answer_loss": matched_mean,
        "processor_off_answer_loss": off_mean,
        "shuffled_answer_loss": shuffled_mean,
        "matched_minus_processor_off": matched_mean - off_mean,
        "matched_minus_shuffled": matched_mean - shuffled_mean,
    }


def _valid_records(
    records: list[ProcessBranchRecord],
    tokenizer,
    config: TrainConfig,
) -> tuple[list[ProcessBranchRecord], Counter[str]]:
    process = config.process_branch
    if process is None:
        raise ValueError("process_branch configuration is missing")
    maximum_cycles = max(process.cycle_choices)
    valid = []
    reasons: Counter[str] = Counter()
    for record in records:
        try:
            tokenize_process_record(
                record,
                tokenizer,
                step_response_prefix=process.step_response_prefix,
                answer_response_prefix=config.data.response_prefix,
                prompt_suffix=config.data.prompt_suffix,
                cycle_count=maximum_cycles,
                max_prompt_tokens=process.max_prompt_tokens,
                max_step_tokens=process.max_step_tokens,
                max_answer_tokens=process.max_answer_tokens,
            )
        except InvalidProcessBranchExample as exc:
            reasons[str(exc)] += 1
            continue
        valid.append(record)
    return valid, reasons
