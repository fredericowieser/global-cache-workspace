from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any, TypeVar

import torch

from gcw.config import DistillationConfig, load_train_config, to_dict
from gcw.data.tokenizers import TransformersTokenizer, get_tokenizer
from gcw.distill.data import PromptRecord, iter_openmath_prompts
from gcw.distill.prompt import format_teacher_prompt
from gcw.distill.session import sample_next_token
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


_T = TypeVar("_T")


def fixed_batch_chunks(items: Sequence[_T], batch_size: int) -> tuple[tuple[_T, ...], ...]:
    """Return a stable contiguous plan; item lengths never affect chunk membership."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("teacher precompute batch_size must be a positive integer")
    item_list = tuple(items)
    return tuple(
        item_list[offset : offset + batch_size]
        for offset in range(0, len(item_list), batch_size)
    )


def left_pad_hf_prompts(
    prompts: Sequence[tuple[int, ...]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build HF inputs with real tokens right-aligned and positions starting at zero."""
    prompt_list = tuple(prompts)
    if not prompt_list or any(not prompt for prompt in prompt_list):
        raise ValueError("HF teacher generation requires non-empty prompts")
    lengths = torch.tensor(
        [len(prompt) for prompt in prompt_list], dtype=torch.long, device=device
    )
    max_prompt_length = max(len(prompt) for prompt in prompt_list)
    input_ids = torch.full(
        (len(prompt_list), max_prompt_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, prompt in enumerate(prompt_list):
        start = max_prompt_length - len(prompt)
        input_ids[row, start:] = torch.tensor(prompt, dtype=torch.long, device=device)
        attention_mask[row, start:] = 1
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.clamp_min_(0)
    return input_ids, attention_mask, position_ids, lengths


@torch.inference_mode()
def sample_hf_dynamic_completions(
    model: Any,
    prompt_ids: Sequence[tuple[int, ...]],
    *,
    max_new_tokens: int,
    eos_token_id: int,
    temperature: float,
    top_p: float,
    top_k: int,
    valid_vocab_size: int,
    generators: Sequence[torch.Generator],
    device: torch.device,
) -> tuple[tuple[int, ...], ...]:
    """Sample a variable-length, left-padded HF batch with independent RNG streams."""
    prompts = tuple(prompt_ids)
    generator_list = tuple(generators)
    if len(generator_list) != len(prompts):
        raise ValueError("HF teacher generation requires one generator per prompt")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    input_ids, attention_mask, position_ids, lengths = left_pad_hf_prompts(
        prompts, pad_token_id=eos_token_id, device=device
    )
    maximum_positions = getattr(model.config, "max_position_embeddings", None)
    if maximum_positions is not None and int(lengths.max().item()) + max_new_tokens > int(
        maximum_positions
    ):
        raise ValueError("HF teacher prompt and completion exceed model context length")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    next_logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values
    generated: list[list[int]] = [[] for _ in prompts]
    finished = [False] * len(prompts)
    max_prompt_length = input_ids.shape[1]

    for generated_index in range(max_new_tokens):
        sampled_rows: list[torch.Tensor] = []
        for row, generator in enumerate(generator_list):
            if finished[row]:
                sampled_rows.append(
                    torch.tensor(eos_token_id, dtype=torch.long, device=device)
                )
            else:
                sampled_rows.append(
                    sample_next_token(
                        next_logits[row : row + 1],
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        valid_vocab_size=valid_vocab_size,
                        generator=generator,
                    )[0]
                )
        sampled = torch.stack(sampled_rows)
        for row in range(len(prompts)):
            if finished[row]:
                continue
            token_id = int(sampled[row].item())
            generated[row].append(token_id)
            finished[row] = token_id == eos_token_id
        if all(finished) or generated_index + 1 == max_new_tokens:
            break
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (len(prompts), 1), dtype=attention_mask.dtype, device=device
                ),
            ],
            dim=1,
        )
        outputs = model(
            input_ids=sampled[:, None],
            attention_mask=attention_mask,
            position_ids=(lengths + generated_index)[:, None],
            cache_position=torch.tensor(
                [max_prompt_length + generated_index], dtype=torch.long, device=device
            ),
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
    return tuple(tuple(row) for row in generated)


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


@torch.inference_mode()
def precompute_hf_dynamic_native_teacher_records(
    model: Any,
    tokenizer: TransformersTokenizer,
    records: Sequence[PromptRecord],
    distill: DistillationConfig,
    trace_store: TeacherTraceStore,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, int]:
    """Populate native traces from a fixed, variable-prompt-length HF batch plan."""
    if distill.teacher_trace_backend != "hf_dynamic":
        raise ValueError("HF dynamic precompute requires teacher_trace_backend='hf_dynamic'")
    if distill.teacher_trajectory_mode != "native_full":
        raise ValueError("HF dynamic precompute requires native_full trajectories")
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
    prepared: list[tuple[PromptRecord, Any, bool]] = []
    for record in record_list:
        failure = trace_store.get_failure(
            source_index=record.source_index, prompt=record.prompt
        )
        cached = (
            None
            if failure is not None
            else trace_store.get_native_trajectory(
                source_index=record.source_index, prompt=record.prompt
            )
        )
        try:
            teacher_prompt = format_teacher_prompt(
                tokenizer,
                record.prompt,
                reasoning_open_text=distill.reasoning_open_text,
            )
        except ValueError as exc:
            if failure is None and cached is None:
                trace_store.put_failure(
                    source_index=record.source_index,
                    prompt=record.prompt,
                    reason=str(exc),
                )
            else:
                report["cached"] += 1
            report["invalid"] += 1
            continue
        is_cached = failure is not None or cached is not None
        if is_cached:
            report["cached"] += 1
            if failure is not None:
                report["invalid"] += 1
            else:
                report["valid"] += 1
        prepared.append((record, teacher_prompt, is_cached))

    for chunk in fixed_batch_chunks(prepared, batch_size):
        # Replaying the complete original chunk keeps resumed generation on the same
        # numerical batch plan instead of shifting uncached rows between batches.
        if all(item[2] for item in chunk):
            continue
        report["generation_batches"] += 1
        report["singleton_batches"] += int(len(chunk) == 1)
        report["max_observed_batch_size"] = max(
            report["max_observed_batch_size"], len(chunk)
        )
        generated = sample_hf_dynamic_completions(
            model,
            tuple(item[1].full_ids for item in chunk),
            max_new_tokens=distill.max_teacher_completion_tokens,
            eos_token_id=tokenizer.eot_id,
            temperature=distill.temperature,
            top_p=distill.top_p,
            top_k=distill.top_k,
            valid_vocab_size=tokenizer.get_vocab_size(),
            generators=tuple(
                _generator(device, seed + 2 * item[0].source_index) for item in chunk
            ),
            device=device,
        )
        report["completion_tokens"] += sum(len(tokens) for tokens in generated)
        for (record, teacher_prompt, is_cached), completion in zip(
            chunk, generated, strict=True
        ):
            if is_cached:
                continue
            report["generated"] += 1
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
    batch_size: int | None = None,
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
    if distill.teacher_trace_backend == "hf_dynamic":
        if batch_size is not None and batch_size != distill.teacher_trace_batch_size:
            raise ValueError(
                "--batch-size must match teacher_trace_batch_size for hf_dynamic"
            )
        resolved_batch_size = distill.teacher_trace_batch_size
    else:
        resolved_batch_size = (
            distill.teacher_trace_batch_size if batch_size is None else batch_size
        )
    if resolved_batch_size <= 0:
        raise ValueError("teacher precompute batch_size must be positive")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if distill.teacher_trace_backend == "gcw_static":
        teacher_model: Any = build_model(config.model)
        if not isinstance(teacher_model, BGT):
            raise TypeError("teacher precompute requires a BGT model")
        load_backbone_checkpoint(teacher_model, config.backbone_checkpoint or "")
        teacher_model.backbone.requires_grad_(False)
        teacher_model.backbone.eval()
        teacher_model.to(device)
        if config.bf16 and device.type == "cuda":
            teacher_model.backbone.to(dtype=torch.bfloat16)
        manifest_checkpoint = str(checkpoint.resolve())
    else:
        try:
            from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "hf_dynamic teacher precompute requires the transformers extra"
            ) from exc
        hf_checkpoint = Path(distill.teacher_trace_checkpoint or "").resolve(strict=True)
        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "attn_implementation": "sdpa",
        }
        if config.bf16 and device.type == "cuda":
            load_kwargs["dtype"] = torch.bfloat16
        teacher_model = AutoModelForCausalLM.from_pretrained(
            hf_checkpoint, **load_kwargs
        )
        teacher_model.requires_grad_(False)
        teacher_model.to(device)
        teacher_model.eval()
        manifest_checkpoint = str(hf_checkpoint)

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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.monotonic()
    if distill.teacher_trace_backend == "gcw_static":
        report = precompute_native_teacher_records(
            teacher_model,
            tokenizer,
            records,
            distill,
            trace_store,
            batch_size=resolved_batch_size,
            seed=config.seed,
            device=device,
        )
    else:
        report = precompute_hf_dynamic_native_teacher_records(
            teacher_model,
            tokenizer,
            records,
            distill,
            trace_store,
            batch_size=resolved_batch_size,
            seed=config.seed,
            device=device,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
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
        "teacher_trace_backend": distill.teacher_trace_backend,
        "teacher_trace_checkpoint": manifest_checkpoint,
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": _sha256_file(trace_path),
        "batch_size": resolved_batch_size,
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
