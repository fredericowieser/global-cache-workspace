#!/usr/bin/env python3
"""Train and fully evaluate the frozen-Qwen unified flow-position processor."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcw.checkpoint import load_checkpoint  # noqa: E402
from gcw.config import CacheProcessorConfig, WorkspaceFlowRefinementConfig  # noqa: E402
from gcw.continuous_thought import frozen_qwen_backbone  # noqa: E402
from gcw.data.tokenizers import get_tokenizer  # noqa: E402
from gcw.process_branch import (  # noqa: E402
    AnswerOnlyTokenExample,
    InvalidProcessBranchExample,
    tokenize_answer_only_record,
)
from gcw.process_data import (  # noqa: E402
    AnswerOnlyRecord,
    read_answer_only_records,
    read_process_branch_records,
)
from gcw.process_eval import extract_numeric_answer  # noqa: E402
from gcw.unified_flow_position import (  # noqa: E402
    UnifiedFlowPositionConfig,
    UnifiedFlowPositionProcessor,
    build_unified_flow_position_processor,
    generate_unified_answer,
    load_unified_weights,
    save_unified_checkpoint,
    unified_answer_loss,
)

SCHEMA = "gcw.unified_flow_position.experiment.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen-Qwen unified flow-workspace + continuous-position train/eval.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "eval", "both"), default="both")
    parser.add_argument("--checkpoint", type=Path, help="processor checkpoint for eval-only mode")
    parser.add_argument("--allow-non-cuda", action="store_true", help="CPU smoke/debug only")
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"config must use schema {SCHEMA}")
    for section in ("paths", "processor", "flow_refinement", "hybrid", "training", "evaluation", "runtime"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"config section {section!r} must be an object")
    return payload


def _device(runtime: dict[str, Any], allow_non_cuda: bool) -> torch.device:
    requested = str(runtime.get("device", "cuda"))
    if requested == "cuda" and not torch.cuda.is_available():
        if not allow_non_cuda:
            raise RuntimeError("canonical run requires CUDA; use --allow-non-cuda only for smoke tests")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type != "cuda" and not allow_non_cuda:
        raise ValueError("canonical run requires a CUDA device")
    return device


def _build(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, Any, UnifiedFlowPositionProcessor, Any]:
    paths = config["paths"]
    loaded, model_config, backbone_state = load_checkpoint(paths["backbone_checkpoint"], device=device)
    backbone = frozen_qwen_backbone(loaded)
    dtype = config["runtime"].get("dtype", "bfloat16")
    if dtype == "bfloat16" and device.type == "cuda":
        backbone.to(dtype=torch.bfloat16)
    elif dtype == "float32" or device.type == "cpu":
        backbone.float()
    else:
        raise ValueError("runtime.dtype must be bfloat16 or float32")
    processor_config = CacheProcessorConfig(**config["processor"])
    hybrid_config = UnifiedFlowPositionConfig(model_width=model_config.hidden_size, **config["hybrid"])
    flow_config = WorkspaceFlowRefinementConfig(**config["flow_refinement"])
    processor = build_unified_flow_position_processor(
        model_config,
        cache_processor=processor_config,
        hybrid=hybrid_config,
        flow_refinement=flow_config,
    ).to(device=device, dtype=torch.float32)
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise RuntimeError("backbone freeze failed")
    tokenizer = get_tokenizer(f"hf:{paths['tokenizer']}")
    if tokenizer.get_vocab_size() > backbone.config.vocab_size:
        raise ValueError("tokenizer vocabulary exceeds backbone vocabulary")
    return loaded, backbone, processor, (model_config, backbone_state, tokenizer)


def _autocast(device: torch.device, backbone: torch.nn.Module):
    enabled = device.type == "cuda" and next(backbone.parameters()).dtype == torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if enabled else nullcontext()


def _tokenize(
    record: AnswerOnlyRecord,
    tokenizer,
    *,
    prompt_suffix: str,
    response_prefix: str,
    max_prompt_tokens: int,
    max_answer_tokens: int,
    device: torch.device,
):
    return tokenize_answer_only_record(
        record,
        tokenizer,
        prompt_suffix=prompt_suffix,
        answer_response_prefix=response_prefix,
        max_prompt_tokens=max_prompt_tokens,
        max_answer_tokens=max_answer_tokens,
        device=device,
    )


def _tokenize_eval_mode(
    record: AnswerOnlyRecord,
    tokenizer,
    mode: dict[str, Any],
    *,
    max_prompt_tokens: int,
    max_answer_tokens: int,
    device: torch.device,
) -> AnswerOnlyTokenExample:
    """Use Qwen's real enable_thinking chat-template switch when configured."""
    if not mode.get("use_chat_template", False):
        return _tokenize(
            record,
            tokenizer,
            prompt_suffix=str(mode["prompt_suffix"]),
            response_prefix=str(mode["response_prefix"]),
            max_prompt_tokens=max_prompt_tokens,
            max_answer_tokens=max_answer_tokens,
            device=device,
        )
    enable_thinking = mode.get("enable_thinking")
    if not isinstance(enable_thinking, bool):
        raise ValueError("chat-template thinking mode requires boolean enable_thinking")
    content = record.prompt
    suffix = str(mode["prompt_suffix"]).strip()
    if suffix:
        content = f"{content}\n\n{suffix}"
    template_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if not template_ids:
        raise InvalidProcessBranchExample("empty_chat_template")
    # The complete assistant generation marker is part of the prompt. Active
    # processing therefore happens after (not inside) the real Qwen template;
    # generation can read logits directly from the final continuous position.
    prompt_ids = template_ids
    if len(prompt_ids) > max_prompt_tokens:
        raise InvalidProcessBranchExample("prompt_too_long")
    answer_ids = tokenizer.encode(record.final_answer) + [int(tokenizer.eot_id)]
    if len(answer_ids) > max_answer_tokens:
        raise InvalidProcessBranchExample("answer_too_long")
    return AnswerOnlyTokenExample(
        prompt_ids=torch.tensor(prompt_ids, dtype=torch.long, device=device),
        answer_response_prefix_ids=torch.empty(0, dtype=torch.long, device=device),
        final_answer_ids=torch.tensor(answer_ids, dtype=torch.long, device=device),
    )


def train(
    config: dict[str, Any],
    loaded: torch.nn.Module,
    backbone,
    processor: UnifiedFlowPositionProcessor,
    tokenizer,
    device: torch.device,
    output: Path,
) -> Path:
    settings = config["training"]
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    records = read_answer_only_records(config["paths"]["train_data"])
    requested = int(settings["train_rows"])
    if len(records) < requested:
        raise ValueError(f"training data has {len(records)} rows, expected at least {requested}")
    records = records[:requested]
    examples = []
    accepted_records = []
    for record in records:
        try:
            examples.append(
                _tokenize(
                    record,
                    tokenizer,
                    prompt_suffix=settings["prompt_suffix"],
                    response_prefix=settings["response_prefix"],
                    max_prompt_tokens=int(settings["max_prompt_tokens"]),
                    max_answer_tokens=int(settings["max_answer_tokens"]),
                    device=device,
                )
            )
            accepted_records.append(record)
        except InvalidProcessBranchExample:
            continue
    if len(examples) != requested:
        raise ValueError(f"canonical training requires all {requested} rows; only {len(examples)} are eligible")

    processor.train()
    optimizer = torch.optim.AdamW(
        processor.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    accumulation = int(settings["gradient_accumulation_steps"])
    epochs = int(settings["epochs"])
    if accumulation <= 0 or epochs <= 0:
        raise ValueError("epochs and gradient_accumulation_steps must be positive")
    contrast_weight = float(settings["shuffle_contrastive_weight"])
    margin = float(settings["shuffle_contrastive_margin"])
    metrics_path = output / "train_metrics.jsonl"
    output.mkdir(parents=True, exist_ok=False)
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    examples_seen = 0
    started = time.perf_counter()
    max_backbone_grad_tensors = 0
    with metrics_path.open("x", encoding="utf-8") as metrics:
        for epoch in range(epochs):
            order = list(range(len(examples)))
            random.Random(seed + epoch).shuffle(order)
            for offset, index in enumerate(order):
                example = examples[index]
                donor_example = examples[order[(offset + 1) % len(order)]]
                with _autocast(device, backbone):
                    donor = processor.prepare_source(backbone, donor_example.prompt_ids.unsqueeze(0))
                    matched = unified_answer_loss(
                        backbone,
                        processor,
                        prompt_ids=example.prompt_ids.unsqueeze(0),
                        response_prefix_ids=example.answer_response_prefix_ids.unsqueeze(0),
                        target_ids=example.final_answer_ids.unsqueeze(0),
                        active=True,
                    )
                    shuffled = unified_answer_loss(
                        backbone,
                        processor,
                        prompt_ids=example.prompt_ids.unsqueeze(0),
                        response_prefix_ids=example.answer_response_prefix_ids.unsqueeze(0),
                        target_ids=example.final_answer_ids.unsqueeze(0),
                        active=True,
                        source_override=donor,
                    )
                    contrast = torch.relu(margin + matched.loss - shuffled.loss)
                    objective = matched.loss + contrast_weight * contrast
                (objective / accumulation).backward()
                examples_seen += 1
                should_step = examples_seen % accumulation == 0 or (
                    epoch + 1 == epochs and offset + 1 == len(order)
                )
                grad_norm = None
                if should_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        processor.parameters(), float(settings["grad_clip"])
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                backbone_grads = sum(parameter.grad is not None for parameter in loaded.parameters())
                max_backbone_grad_tensors = max(max_backbone_grad_tensors, backbone_grads)
                row = {
                    "contrastive_penalty": float(contrast.detach().float()),
                    "elapsed_seconds": time.perf_counter() - started,
                    "epoch": epoch + 1,
                    "example": offset + 1,
                    "grad_norm": None if grad_norm is None else float(grad_norm.detach().float()),
                    "matched_loss": float(matched.loss.detach().float()),
                    "objective": float(objective.detach().float()),
                    "optimizer_step": optimizer_steps,
                    "shuffled_loss": float(shuffled.loss.detach().float()),
                }
                metrics.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                if should_step:
                    metrics.flush()
                    print(
                        f"train step={optimizer_steps} examples={examples_seen}/{epochs * len(order)} "
                        f"loss={row['matched_loss']:.4f}",
                        file=sys.stderr,
                        flush=True,
                    )
    if max_backbone_grad_tensors:
        raise RuntimeError("frozen backbone received gradients")
    checkpoint = output / "checkpoint"
    summary = {
        "backbone_grad_tensors": max_backbone_grad_tensors,
        "elapsed_seconds": time.perf_counter() - started,
        "epochs": epochs,
        "examples": examples_seen,
        "optimizer_steps": optimizer_steps,
        "processor_parameters": processor.trainable_parameter_count,
        "schema": "gcw.unified_flow_position.train.v1",
    }
    construction = {
        "cache_processor": config["processor"],
        "flow_refinement": config["flow_refinement"],
    }
    save_unified_checkpoint(checkpoint, processor, construction=construction, train_state=summary)
    (output / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return checkpoint


def _canonical_answer(text: str) -> str | None:
    answer = extract_numeric_answer(text)
    if answer is None:
        return None
    if not isinstance(answer, Fraction):
        raise TypeError("numeric answer parser returned non-Fraction")
    return str(answer.numerator) if answer.denominator == 1 else f"{answer.numerator}/{answer.denominator}"


def _correct(prediction: str, target: str) -> bool:
    predicted = _canonical_answer(prediction)
    expected = _canonical_answer(target)
    return predicted is not None and expected is not None and predicted == expected


def evaluate(
    config: dict[str, Any],
    backbone,
    processor: UnifiedFlowPositionProcessor,
    tokenizer,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    settings = config["evaluation"]
    records = read_process_branch_records(config["paths"]["eval_data"])
    expected_rows = int(settings["expected_rows"])
    if len(records) != expected_rows:
        raise ValueError(f"full GSM8K evaluation requires {expected_rows} rows, found {len(records)}")
    conditions = tuple(settings["conditions"])
    if conditions != ("off", "matched", "shuffled"):
        raise ValueError("canonical conditions must be [off, matched, shuffled]")
    modes = settings["thinking_modes"]
    if set(modes) != {"thinking_off", "thinking_on"}:
        raise ValueError("evaluation must define thinking_off and thinking_on")
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "eval_rows.jsonl"
    if rows_path.exists():
        raise FileExistsError(rows_path)
    processor.eval()
    aggregates: dict[str, dict[str, dict[str, int | float]]] = {}
    started = time.perf_counter()
    with rows_path.open("x", encoding="utf-8") as rows_file, torch.inference_mode():
        for mode_name in ("thinking_off", "thinking_on"):
            mode = modes[mode_name]
            examples = [
                _tokenize_eval_mode(
                    AnswerOnlyRecord(row.source_id, row.prompt, row.final_answer),
                    tokenizer,
                    mode,
                    max_prompt_tokens=int(settings["max_prompt_tokens"]),
                    max_answer_tokens=int(settings["max_answer_tokens"]),
                    device=device,
                )
                for row in records
            ]
            counts = {condition: 0 for condition in conditions}
            generated_counts = {condition: 0 for condition in conditions}
            truncated = {condition: 0 for condition in conditions}
            for index, (record, example) in enumerate(zip(records, examples, strict=True)):
                donor_index = (index + 1) % len(examples)
                with _autocast(device, backbone):
                    donor = processor.prepare_source(
                        backbone, examples[donor_index].prompt_ids.unsqueeze(0)
                    )
                    results = {}
                    for condition in conditions:
                        generated = generate_unified_answer(
                            backbone,
                            processor,
                            prompt_ids=example.prompt_ids,
                            response_prefix_ids=example.answer_response_prefix_ids,
                            active=condition != "off",
                            source_override=donor if condition == "shuffled" else None,
                            max_new_tokens=int(settings["max_new_tokens"]),
                            eos_token_id=int(tokenizer.eot_id),
                            valid_vocab_size=tokenizer.get_vocab_size(),
                        )
                        ids = [int(value) for value in generated.token_ids.tolist()]
                        prediction = tokenizer.decode(ids)
                        is_correct = _correct(prediction, record.final_answer)
                        is_truncated = len(ids) == int(settings["max_new_tokens"]) and (
                            not ids or ids[-1] != int(tokenizer.eot_id)
                        )
                        counts[condition] += int(is_correct)
                        generated_counts[condition] += len(ids)
                        truncated[condition] += int(is_truncated)
                        results[condition] = {
                            "added_positions": generated.trace.added_positions,
                            "answer": _canonical_answer(prediction),
                            "answer_start_position": generated.answer_start_position,
                            "correct": is_correct,
                            "generated_tokens": len(ids),
                            "prediction": prediction,
                            "source_mode": generated.trace.source_mode,
                            "truncated": is_truncated,
                        }
                rows_file.write(
                    json.dumps(
                        {
                            "conditions": results,
                            "mode": mode_name,
                            "row_index": index,
                            "source_id": record.source_id,
                            "target": record.final_answer,
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                rows_file.flush()
                if (index + 1) % 16 == 0:
                    print(
                        f"eval {mode_name} {index + 1}/{len(records)}",
                        file=sys.stderr,
                        flush=True,
                    )
            aggregates[mode_name] = {
                condition: {
                    "accuracy": counts[condition] / len(records),
                    "correct": counts[condition],
                    "mean_generated_tokens": generated_counts[condition] / len(records),
                    "rows": len(records),
                    "truncated": truncated[condition],
                }
                for condition in conditions
            }
    condition_scores = {
        condition: sum(float(aggregates[mode][condition]["accuracy"]) for mode in modes) / len(modes)
        for condition in ("off", "matched")
    }
    selected_condition = max(("off", "matched"), key=lambda condition: (condition_scores[condition], condition == "matched"))
    summary = {
        "aggregate": aggregates,
        "elapsed_seconds": time.perf_counter() - started,
        "expected_rows_per_mode": expected_rows,
        "processor_parameters": processor.trainable_parameter_count,
        "schema": "gcw.unified_flow_position.gsm8k_full_eval.v1",
        "selection": {
            "criterion": "mean exact generated-answer accuracy across thinking_off and thinking_on",
            "condition_scores": condition_scores,
            "selected_condition": selected_condition,
            "selected_method": (
                "unified_flow_workspace_continuous_position_v1"
                if selected_condition == "matched"
                else "frozen_qwen_no_processor"
            ),
            "shuffled_is_control_only": True,
        },
        "thinking_modes": list(modes),
        "total_generations": expected_rows * len(modes) * len(conditions),
    }
    (output / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    device = _device(config["runtime"], args.allow_non_cuda)
    loaded, backbone, processor, extras = _build(config, device)
    model_config, backbone_state, tokenizer = extras
    paths = config["paths"]
    for key in ("backbone_checkpoint", "tokenizer"):
        if not Path(paths[key]).exists():
            raise FileNotFoundError(paths[key])
    output = Path(paths["output"])
    checkpoint = args.checkpoint
    if args.mode in {"train", "both"}:
        if output.exists():
            raise FileExistsError(output)
        checkpoint = train(config, loaded, backbone, processor, tokenizer, device, output)
    if args.mode in {"eval", "both"}:
        if checkpoint is None:
            checkpoint = output / "checkpoint"
        train_state = load_unified_weights(checkpoint, processor, device=device)
        eval_output = output / "full-gsm8k-eval"
        summary = evaluate(config, backbone, processor, tokenizer, device, eval_output)
    else:
        train_state = {}
        summary = {}
    receipt = {
        "backbone_config": asdict(model_config),
        "backbone_train_state": backbone_state,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "device": str(device),
        "evaluation": summary,
        "processor_parameters": processor.trainable_parameter_count,
        "train_state": train_state,
    }
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    receipt = run(build_parser().parse_args(argv))
    print(json.dumps(receipt, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
