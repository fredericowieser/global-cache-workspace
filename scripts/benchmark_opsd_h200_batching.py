#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import torch

from gcw.config import load_train_config
from gcw.data.tokenizers import TransformersTokenizer, get_tokenizer
from gcw.distill.data import iter_openmath_prompts
from gcw.distill.prompt import format_chat_prompt, format_teacher_prompt
from gcw.distill.session import BGTDecodeSession
from gcw.distill.teacher import sample_backbone_completions
from gcw.models import build_model
from gcw.models.bgt import BGT
from gcw.train import load_backbone_checkpoint


def _atomic_write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _timed(device: torch.device, fn):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    value = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return value, elapsed, peak


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens <= 0:
        raise ValueError("tokens must be positive")
    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(","))
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("batch sizes must be positive")

    config = load_train_config(args.config)
    distill = config.distillation
    if distill is None:
        raise ValueError("benchmark requires distillation configuration")
    tokenizer = get_tokenizer(config.data.tokenizer)
    if not isinstance(tokenizer, TransformersTokenizer):
        raise TypeError("benchmark requires a TransformersTokenizer")
    record = next(
        iter(
            iter_openmath_prompts(
                config.data.data_dir or "",
                skip_rows=config.data.sft_skip_rows,
                take_rows=1,
            )
        )
    )
    student_prompt = format_chat_prompt(tokenizer, record.prompt, enable_thinking=False)
    teacher_prompt = format_teacher_prompt(
        tokenizer, record.prompt, reasoning_open_text=distill.reasoning_open_text
    )

    torch.manual_seed(config.seed)
    model = build_model(config.model)
    if not isinstance(model, BGT):
        raise TypeError("benchmark requires BGT")
    load_backbone_checkpoint(model, config.backbone_checkpoint or "")
    model.backbone.requires_grad_(False)
    model.backbone.eval()
    model.cache_processor.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if config.bf16 and device.type == "cuda":
        model.backbone.to(dtype=torch.bfloat16)
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    rows = []
    for batch_size in batch_sizes:
        row: dict[str, Any] = {"batch_size": batch_size}
        try:
            with context:
                teacher_generators = tuple(
                    _generator(device, config.seed + 2 * index)
                    for index in range(batch_size)
                )
                teacher, teacher_seconds, teacher_peak = _timed(
                    device,
                    lambda: sample_backbone_completions(
                        model.backbone,
                        (teacher_prompt.full_ids,) * batch_size,
                        max_new_tokens=args.tokens,
                        eos_token_id=tokenizer.eot_id,
                        stop_ids=None,
                        temperature=distill.temperature,
                        top_p=distill.top_p,
                        top_k=distill.top_k,
                        valid_vocab_size=tokenizer.get_vocab_size(),
                        generators=teacher_generators,
                    ),
                )
                serial_teacher = tuple(
                    sample_backbone_completions(
                        model.backbone,
                        (teacher_prompt.full_ids,),
                        max_new_tokens=args.tokens,
                        eos_token_id=tokenizer.eot_id,
                        stop_ids=None,
                        temperature=distill.temperature,
                        top_p=distill.top_p,
                        top_k=distill.top_k,
                        valid_vocab_size=tokenizer.get_vocab_size(),
                        generators=(_generator(device, config.seed + 2 * index),),
                    )[0]
                    for index in range(batch_size)
                )
            teacher_tokens = sum(len(tokens) for tokens in teacher)
            row.update(
                {
                    "teacher_status": "passed",
                    "teacher_seconds": teacher_seconds,
                    "teacher_completion_tokens": teacher_tokens,
                    "teacher_tokens_per_second": teacher_tokens / teacher_seconds,
                    "teacher_peak_cuda_memory_bytes": teacher_peak,
                    "teacher_batch_matches_serial": teacher == serial_teacher,
                }
            )
        except torch.OutOfMemoryError as exc:
            row.update({"teacher_status": "oom", "teacher_error": str(exc)})
            if device.type == "cuda":
                torch.cuda.empty_cache()

        try:
            with context:
                session = BGTDecodeSession(
                    model,
                    process_every=distill.response_process_every,
                    select_all=distill.boundary_selection == "all",
                    valid_vocab_size=tokenizer.get_vocab_size(),
                    processor_enabled=True,
                )
                student_generators = tuple(
                    _generator(device, config.seed + 2 * index + 1)
                    for index in range(batch_size)
                )
                student, student_seconds, student_peak = _timed(
                    device,
                    lambda: session.rollout_many(
                        (student_prompt,) * batch_size,
                        max_response_tokens=args.tokens,
                        eos_token_id=tokenizer.eot_id,
                        temperature=distill.temperature,
                        top_p=distill.top_p,
                        top_k=distill.top_k,
                        generators=student_generators,
                        greedy=False,
                    ),
                )
                serial_student = tuple(
                    session.rollout(
                        student_prompt,
                        max_response_tokens=args.tokens,
                        eos_token_id=tokenizer.eot_id,
                        temperature=distill.temperature,
                        top_p=distill.top_p,
                        top_k=distill.top_k,
                        generator=_generator(device, config.seed + 2 * index + 1),
                    )
                    for index in range(batch_size)
                )
            student_tokens = sum(len(result.response_ids) for result in student)
            row.update(
                {
                    "student_status": "passed",
                    "student_seconds": student_seconds,
                    "student_response_tokens": student_tokens,
                    "student_tokens_per_second": student_tokens / student_seconds,
                    "student_peak_cuda_memory_bytes": student_peak,
                    "student_batch_matches_serial": tuple(
                        result.response_ids for result in student
                    )
                    == tuple(result.response_ids for result in serial_student),
                    "student_event_schedules_match_serial": tuple(
                        result.event_cache_lengths for result in student
                    )
                    == tuple(
                        result.event_cache_lengths for result in serial_student
                    ),
                }
            )
        except torch.OutOfMemoryError as exc:
            row.update({"student_status": "oom", "student_error": str(exc)})
            if device.type == "cuda":
                torch.cuda.empty_cache()

        row["status"] = (
            "passed"
            if row.get("teacher_status") == row.get("student_status") == "passed"
            else "partial"
        )
        rows.append(row)

    payload = {
        "schema_version": 1,
        "kind": "opsd_h200_exact_shape_batch_benchmark",
        "source_revision": args.source_revision,
        "config": str(args.config.resolve()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "tokens_per_sequence": args.tokens,
        "rows": rows,
    }
    _atomic_write_new(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
