from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from gcw.config import DistillationConfig, load_train_config, to_dict
from gcw.data.tokenizers import TransformersTokenizer, get_tokenizer
from gcw.distill.data import PromptRecord, iter_openmath_prompts
from gcw.distill.prompt import format_teacher_prompt
from gcw.distill.teacher import (
    parse_native_teacher_trajectory,
    sample_backbone_completions,
)
from gcw.distill.traces import TeacherTraceStore
from gcw.models import build_model
from gcw.models.bgt import BGT
from gcw.train import load_backbone_checkpoint


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _validate_native_caps(
    reasoning_tokens: int,
    completion_tokens: int,
    distill: DistillationConfig,
) -> None:
    if reasoning_tokens > distill.max_teacher_reasoning_tokens:
        raise ValueError("native teacher reasoning exceeds max_teacher_reasoning_tokens")
    if completion_tokens > distill.max_teacher_completion_tokens:
        raise ValueError("native teacher completion exceeds max_teacher_completion_tokens")


@torch.no_grad()
def precompute_native_teacher_records(
    model: BGT,
    tokenizer: TransformersTokenizer,
    records: Sequence[PromptRecord],
    distill: DistillationConfig,
    trace_store: TeacherTraceStore,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, int]:
    """Populate native teacher traces in exact-length generation buckets."""
    if distill.teacher_trajectory_mode != "native_full":
        raise ValueError("batched teacher precompute requires native_full trajectories")
    if batch_size <= 0:
        raise ValueError("teacher precompute batch_size must be positive")
    record_list = tuple(records)
    report = {
        "records": len(record_list),
        "cached": 0,
        "generated": 0,
        "valid": 0,
        "invalid": 0,
        "completion_tokens": 0,
        "generation_batches": 0,
        "singleton_batches": 0,
        "max_observed_batch_size": 0,
    }
    open_ids = tuple(tokenizer.encode(distill.reasoning_open_text))
    close_ids = tuple(tokenizer.encode(distill.reasoning_close_text))
    prepared: list[tuple[PromptRecord, Any]] = []
    for record in record_list:
        failure = trace_store.get_failure(
            source_index=record.source_index, prompt=record.prompt
        )
        if failure is not None:
            report["cached"] += 1
            report["invalid"] += 1
            continue
        cached = trace_store.get_native_trajectory(
            source_index=record.source_index, prompt=record.prompt
        )
        if cached is not None:
            report["cached"] += 1
            report["valid"] += 1
            continue
        try:
            teacher_prompt = format_teacher_prompt(
                tokenizer,
                record.prompt,
                reasoning_open_text=distill.reasoning_open_text,
            )
        except ValueError as exc:
            trace_store.put_failure(
                source_index=record.source_index,
                prompt=record.prompt,
                reason=str(exc),
            )
            report["invalid"] += 1
            continue
        prepared.append((record, teacher_prompt))

    grouped: dict[int, list[tuple[PromptRecord, Any]]] = {}
    for item in prepared:
        grouped.setdefault(len(item[1].full_ids), []).append(item)
    for group in grouped.values():
        for offset in range(0, len(group), batch_size):
            chunk = group[offset : offset + batch_size]
            report["generation_batches"] += 1
            report["singleton_batches"] += int(len(chunk) == 1)
            report["max_observed_batch_size"] = max(
                report["max_observed_batch_size"], len(chunk)
            )
            generated = sample_backbone_completions(
                model.backbone,
                tuple(item[1].full_ids for item in chunk),
                max_new_tokens=distill.max_teacher_completion_tokens,
                eos_token_id=tokenizer.eot_id,
                stop_ids=None,
                temperature=distill.temperature,
                top_p=distill.top_p,
                top_k=distill.top_k,
                valid_vocab_size=tokenizer.get_vocab_size(),
                generators=tuple(
                    _generator(device, seed + 2 * item[0].source_index)
                    for item in chunk
                ),
            )
            report["generated"] += len(generated)
            report["completion_tokens"] += sum(len(tokens) for tokens in generated)
            for (record, teacher_prompt), completion in zip(
                chunk, generated, strict=True
            ):
                try:
                    trajectory = parse_native_teacher_trajectory(
                        completion,
                        teacher_trigger_ids=teacher_prompt.trigger_ids,
                        open_ids=open_ids,
                        close_ids=close_ids,
                        eos_token_id=tokenizer.eot_id,
                    )
                    _validate_native_caps(
                        len(trajectory.reasoning_ids),
                        len(trajectory.full_completion_ids),
                        distill,
                    )
                except ValueError as exc:
                    trace_store.put_failure(
                        source_index=record.source_index,
                        prompt=record.prompt,
                        reason=str(exc),
                    )
                    report["invalid"] += 1
                    continue
                trace_store.put_native_trajectory(
                    source_index=record.source_index,
                    prompt=record.prompt,
                    reasoning_ids=trajectory.reasoning_ids,
                    visible_response_ids=trajectory.visible_response_ids,
                    full_completion_ids=trajectory.full_completion_ids,
                )
                report["valid"] += 1
    trace_store.sync()
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def precompute_teacher_traces_from_config(
    config_path: str | Path,
    *,
    batch_size: int,
    source_revision: str,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve(strict=True)
    config = load_train_config(config_path)
    distill = config.distillation
    if config.mode != "cot_distillation" or distill is None:
        raise ValueError("teacher precompute requires cot_distillation configuration")
    if not distill.require_precomputed_teacher_traces:
        raise ValueError("configuration must require precomputed teacher traces")
    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise ValueError("source_revision must be a full lowercase Git SHA")
    output_dir = Path(config.output_dir)
    if (output_dir / "metrics.jsonl").exists() or (output_dir / "final").exists():
        raise FileExistsError("refusing to precompute into an already-started training run")
    checkpoint = Path(config.backbone_checkpoint or "") / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing backbone checkpoint weights: {checkpoint}")
    tokenizer = get_tokenizer(config.data.tokenizer)
    if not isinstance(tokenizer, TransformersTokenizer):
        raise TypeError("teacher precompute requires a TransformersTokenizer")
    open_ids = tuple(tokenizer.encode(distill.reasoning_open_text))
    close_ids = tuple(tokenizer.encode(distill.reasoning_close_text))
    if not open_ids or not close_ids:
        raise ValueError("teacher reasoning delimiters must tokenize non-empty")

    torch.manual_seed(config.seed)
    model = build_model(config.model)
    if not isinstance(model, BGT):
        raise TypeError("teacher precompute requires a BGT model")
    load_backbone_checkpoint(model, config.backbone_checkpoint or "")
    model.backbone.requires_grad_(False)
    model.backbone.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if config.bf16 and device.type == "cuda":
        model.backbone.to(dtype=torch.bfloat16)

    from gcw.distill.trainer import _teacher_fingerprint

    fingerprint = _teacher_fingerprint(config, open_ids, close_ids, tokenizer)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "teacher-traces.jsonl"
    trace_store = TeacherTraceStore(trace_path, fingerprint=fingerprint)
    records = tuple(
        iter_openmath_prompts(
            config.data.data_dir or "",
            skip_rows=config.data.sft_skip_rows,
            take_rows=config.data.sft_take_rows,
        )
    )
    started = time.monotonic()
    report = precompute_native_teacher_records(
        model,
        tokenizer,
        records,
        distill,
        trace_store,
        batch_size=batch_size,
        seed=config.seed,
        device=device,
    )
    elapsed = time.monotonic() - started
    config_bytes = json.dumps(
        to_dict(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opsd_native_teacher_trace_precompute",
        "source_revision": source_revision,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "teacher_fingerprint": fingerprint,
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": _sha256_file(trace_path),
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "completion_tokens_per_second": (
            report["completion_tokens"] / elapsed if elapsed > 0 else 0.0
        ),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        ),
        **report,
    }
    _atomic_write_json(output_dir / "teacher-precompute.json", manifest)
    return manifest
