from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from gcw.checkpoint import load_backbone_checkpoint, save_checkpoint
from gcw.config import DistillationConfig, ModelConfig, TrainConfig, to_dict
from gcw.data.tokenizers import TransformersTokenizer, get_tokenizer
from gcw.distill.data import PromptRecord, iter_openmath_prompts
from gcw.distill.pipeline import DistillationExample, distillation_example_loss
from gcw.distill.prompt import format_chat_prompt
from gcw.distill.session import BGTDecodeSession
from gcw.distill.teacher import sample_backbone_completion, split_teacher_reasoning
from gcw.distill.traces import TeacherTraceStore
from gcw.models import build_model
from gcw.models.bgt import BGT


class InvalidDistillationExample(ValueError):
    """A data-dependent prompt/trace failure that may be skipped by policy."""


def train_distillation(config: TrainConfig) -> None:
    """Run one finite prompt epoch of on-policy self-distillation."""
    distill = config.distillation
    if config.mode != "cot_distillation" or distill is None:
        raise ValueError("train_distillation requires cot_distillation configuration")
    if config.data.data_dir is None:
        raise ValueError("cot_distillation requires data.data_dir")

    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    final_checkpoint = output_dir / "final"
    if metrics_path.exists() or final_checkpoint.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing distillation run in {output_dir}; "
            "teacher traces may be prepared there, but metrics/final must be absent"
        )
    checkpoint_path = Path(config.backbone_checkpoint or "") / "model.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing backbone checkpoint weights: {checkpoint_path}")

    tokenizer = get_tokenizer(config.data.tokenizer)
    if not isinstance(tokenizer, TransformersTokenizer):
        raise TypeError("cot_distillation requires a TransformersTokenizer")
    if tokenizer.get_vocab_size() > config.model.vocab_size:
        raise ValueError(
            f"tokenizer vocabulary {tokenizer.get_vocab_size()} exceeds model vocabulary "
            f"{config.model.vocab_size}"
        )
    close_ids = tuple(tokenizer.encode(distill.reasoning_close_text))
    if not close_ids:
        raise ValueError("reasoning_close_text tokenised to an empty sequence")
    trace_fingerprint = _teacher_fingerprint(config, close_ids, tokenizer)

    model = build_model(config.model)
    if not isinstance(model, BGT):
        raise TypeError("cot_distillation requires a BGT model")
    load_backbone_checkpoint(model, config.backbone_checkpoint or "")
    model.backbone.requires_grad_(False)
    model.backbone.eval()
    model.cache_processor.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if config.bf16 and device.type == "cuda":
        model.backbone.to(dtype=torch.bfloat16)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(to_dict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    trace_store = TeacherTraceStore(output_dir / "teacher-traces.jsonl", fingerprint=trace_fingerprint)
    optimizer = _optimizer(model, config)
    select_all = distill.boundary_selection == "all"
    decode_session = BGTDecodeSession(
        model,
        process_every=distill.response_process_every,
        select_all=select_all,
        valid_vocab_size=tokenizer.get_vocab_size(),
    )

    seen = 0
    valid = 0
    skipped = 0
    answer_tokens = 0
    optimizer_steps = 0
    accumulated = 0
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)

    records = iter_openmath_prompts(
        config.data.data_dir,
        skip_rows=config.data.sft_skip_rows,
        take_rows=config.data.sft_take_rows,
    )
    with metrics_path.open("x", encoding="utf-8") as metrics:
        for record in records:
            seen += 1
            try:
                with _autocast_context(config, device):
                    example, trace_cached = _build_example(
                        model,
                        tokenizer,
                        record,
                        distill,
                        trace_store,
                        close_ids,
                        decode_session,
                        seed=config.seed,
                        device=device,
                    )
                with _autocast_context(config, device):
                    result = distillation_example_loss(
                        model,
                        example,
                        decode_session,
                        temperature=distill.temperature,
                        token_clip=distill.token_clip,
                        sequence_chunk_size=distill.kl_sequence_chunk_size,
                    )
                if not bool(torch.isfinite(result.loss)):
                    raise FloatingPointError("distillation loss is non-finite")
                result.loss.backward()
                if valid == 0 and any(parameter.grad is not None for parameter in model.backbone.parameters()):
                    raise RuntimeError("frozen teacher/student backbone accumulated gradients")
                accumulated += 1
                valid += 1
                answer_tokens += result.answer_tokens
                grad_norm: float | None = None
                if accumulated == distill.gradient_accumulation_steps:
                    grad_norm = _optimizer_step(model, optimizer, accumulated, config.grad_clip)
                    optimizer_steps += 1
                    accumulated = 0

                _write_metric(
                    metrics,
                    {
                        "source_index": record.source_index,
                        "status": "trained",
                        "loss": float(result.loss.detach().float().item()),
                        "unclipped_kl": float(result.unclipped_kl.item()),
                        "clipping_fraction": float(result.clipping_fraction.item()),
                        "student_entropy": float(result.student_entropy.item()),
                        "teacher_entropy": float(result.teacher_entropy.item()),
                        "teacher_reasoning_tokens": len(example.teacher_reasoning_ids),
                        "answer_tokens": result.answer_tokens,
                        "cache_events": list(result.event_cache_lengths),
                        "teacher_trace_cached": trace_cached,
                        "grad_norm": grad_norm,
                        "optimizer_steps": optimizer_steps,
                    },
                )
            except InvalidDistillationExample as exc:
                if not distill.skip_invalid_traces:
                    raise
                skipped += 1
                _write_metric(
                    metrics,
                    {
                        "source_index": record.source_index,
                        "status": "skipped",
                        "reason": str(exc),
                        "optimizer_steps": optimizer_steps,
                    },
                )

        if accumulated:
            _optimizer_step(model, optimizer, accumulated, config.grad_clip)
            optimizer_steps += 1

    trace_store.sync()
    if valid == 0:
        raise RuntimeError("distillation epoch produced no valid training examples")
    elapsed = time.monotonic() - started
    train_state: dict[str, Any] = {
        "mode": "cot_distillation",
        "epoch": 1,
        "seen_prompts": seen,
        "valid_prompts": valid,
        "skipped_prompts": skipped,
        "answer_tokens": answer_tokens,
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": elapsed,
        "answer_tokens_per_second": answer_tokens / elapsed,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "teacher_trace_fingerprint": trace_fingerprint,
        "dtype": "bfloat16" if config.bf16 and device.type == "cuda" else "float32",
    }
    if config.bf16 and device.type == "cuda":
        model.cache_processor.to(dtype=torch.bfloat16)
    _publish_checkpoint(final_checkpoint, model, config.model, train_state)
    _atomic_write_json(output_dir / "summary.json", train_state)
    print(json.dumps(train_state, sort_keys=True))


def _build_example(
    model: BGT,
    tokenizer: TransformersTokenizer,
    record: PromptRecord,
    distill: DistillationConfig,
    trace_store: TeacherTraceStore,
    close_ids: tuple[int, ...],
    decode_session: BGTDecodeSession,
    *,
    seed: int,
    device: torch.device,
) -> tuple[DistillationExample, bool]:
    try:
        student_prompt = format_chat_prompt(tokenizer, record.prompt, enable_thinking=False)
        teacher_prompt = format_chat_prompt(tokenizer, record.prompt, enable_thinking=True)
    except ValueError as exc:
        raise InvalidDistillationExample(str(exc)) from exc
    reasoning_ids = trace_store.get(source_index=record.source_index, prompt=record.prompt)
    trace_cached = reasoning_ids is not None
    if reasoning_ids is not None and any(token_id >= tokenizer.get_vocab_size() for token_id in reasoning_ids):
        raise ValueError(f"cached teacher trace contains out-of-vocabulary IDs for source_index={record.source_index}")
    if reasoning_ids is None:
        teacher_generator = _generator(device, seed + 2 * record.source_index)
        try:
            generated = sample_backbone_completion(
                model.backbone,
                teacher_prompt.full_ids,
                max_new_tokens=distill.max_teacher_reasoning_tokens,
                eos_token_id=tokenizer.eot_id,
                stop_ids=close_ids,
                temperature=distill.temperature,
                top_p=distill.top_p,
                top_k=distill.top_k,
                valid_vocab_size=tokenizer.get_vocab_size(),
                generator=teacher_generator,
            )
            reasoning_ids = split_teacher_reasoning(generated, close_ids=close_ids)
        except ValueError as exc:
            raise InvalidDistillationExample(str(exc)) from exc
        trace_store.put(
            source_index=record.source_index,
            prompt=record.prompt,
            reasoning_ids=reasoning_ids,
        )

    student_generator = _generator(device, seed + 2 * record.source_index + 1)
    try:
        rollout = decode_session.rollout(
            student_prompt,
            max_response_tokens=distill.max_response_tokens,
            eos_token_id=tokenizer.eot_id,
            temperature=distill.temperature,
            top_p=distill.top_p,
            top_k=distill.top_k,
            generator=student_generator,
        )
    except ValueError as exc:
        raise InvalidDistillationExample(str(exc)) from exc
    teacher_scoring_tokens = len(teacher_prompt.full_ids) + len(reasoning_ids) + len(rollout.response_ids)
    if teacher_scoring_tokens > model.config.sequence_len:
        raise InvalidDistillationExample(
            f"teacher context and student response require {teacher_scoring_tokens} tokens but model sequence_len is "
            f"{model.config.sequence_len}"
        )
    return (
        DistillationExample(
            student_prompt=student_prompt,
            teacher_prompt=teacher_prompt,
            teacher_reasoning_ids=reasoning_ids,
            response_ids=rollout.response_ids,
        ),
        trace_cached,
    )


def _optimizer(model: BGT, config: TrainConfig) -> torch.optim.AdamW:
    params = [parameter for parameter in model.cache_processor.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("BGT cache processor has no trainable parameters")
    decay = [parameter for parameter in params if parameter.ndim >= 2]
    no_decay = [parameter for parameter in params if parameter.ndim < 2]
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": config.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )


def _optimizer_step(
    model: BGT,
    optimizer: torch.optim.AdamW,
    accumulated: int,
    grad_clip: float,
) -> float:
    if accumulated <= 0:
        raise ValueError("accumulated example count must be positive")
    scale = 1.0 / accumulated
    params: list[torch.nn.Parameter] = []
    for parameter in model.cache_processor.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if accumulated != 1:
            gradient.mul_(scale)
        params.append(parameter)
    if not params:
        raise RuntimeError("BGT cache processor produced no gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm.detach().float().item())


def _teacher_fingerprint(
    config: TrainConfig,
    close_ids: tuple[int, ...],
    tokenizer: TransformersTokenizer,
) -> str:
    distill = config.distillation
    if distill is None:
        raise ValueError("teacher fingerprint requires distillation configuration")
    checkpoint = Path(config.backbone_checkpoint or "") / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing backbone checkpoint weights: {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    payload = {
        "schema": "gcw-teacher-trace-v2",
        "checkpoint_sha256": digest.hexdigest(),
        "model": to_dict(config.model),
        "tokenizer": config.data.tokenizer,
        "tokenizer_identity_sha256": tokenizer.identity_sha256(),
        "teacher_generation": {
            "seed": config.seed,
            "teacher_thinking": distill.teacher_thinking,
            "temperature": distill.temperature,
            "top_p": distill.top_p,
            "top_k": distill.top_k,
            "max_teacher_reasoning_tokens": distill.max_teacher_reasoning_tokens,
            "eos_token_id": tokenizer.eot_id,
            "reasoning_close_text": distill.reasoning_close_text,
            "close_ids": list(close_ids),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _autocast_context(config: TrainConfig, device: torch.device):
    if config.bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _write_metric(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
    handle.flush()


def _publish_checkpoint(
    destination: Path,
    model: BGT,
    model_config: ModelConfig,
    train_state: dict[str, Any],
) -> None:
    temporary = Path(tempfile.mkdtemp(prefix=".final-", dir=destination.parent))
    save_checkpoint(temporary, model, model_config, train_state)
    for child in temporary.iterdir():
        if child.is_file():
            with child.open("rb") as handle:
                os.fsync(handle.fileno())
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}-",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
