from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch

from gcw.checkpoint import load_backbone_checkpoint, load_checkpoint, save_checkpoint
from gcw.config import DistillationConfig, ModelConfig, TrainConfig, to_dict
from gcw.data.tokenizers import TransformersTokenizer, get_tokenizer
from gcw.distill.data import PromptRecord, iter_openmath_prompts
from gcw.distill.pipeline import (
    DistillationExample,
    DistillationLossResult,
    distillation_example_loss,
)
from gcw.distill.prompt import format_chat_prompt, format_teacher_prompt
from gcw.distill.session import BGTDecodeSession
from gcw.distill.teacher import (
    NativeTeacherTrajectory,
    parse_native_teacher_trajectory,
    sample_backbone_completion,
    split_teacher_reasoning,
)
from gcw.distill.traces import TeacherTraceStore
from gcw.models import build_model
from gcw.models.bgt import BGT
from gcw.prompt_identity import normalized_prompt_text_sha256, prompt_text_sha256
from gcw.state_hash import tensor_state_sha256


class InvalidDistillationExample(ValueError):
    """A data-dependent prompt/trace failure that may be skipped by policy."""


def train_distillation(config: TrainConfig) -> None:
    """Run one finite prompt epoch of on-policy self-distillation."""
    distill = config.distillation
    if config.mode != "cot_distillation" or distill is None:
        raise ValueError("train_distillation requires cot_distillation configuration")
    if config.data.data_dir is None:
        raise ValueError("cot_distillation requires data.data_dir")

    output_dir = Path(config.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    final_checkpoint = output_dir / "final"
    staging_checkpoint = output_dir / ".final-overfit-staging"
    locked_examples_path = output_dir / "locked-examples.json"
    if (
        metrics_path.exists()
        or final_checkpoint.exists()
        or staging_checkpoint.exists()
        or (distill.overfit_diagnostic and locked_examples_path.exists())
    ):
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
    open_ids = tuple(tokenizer.encode(distill.reasoning_open_text))
    if not open_ids:
        raise ValueError("reasoning_open_text tokenised to an empty sequence")
    close_ids = tuple(tokenizer.encode(distill.reasoning_close_text))
    if not close_ids:
        raise ValueError("reasoning_close_text tokenised to an empty sequence")
    trace_fingerprint = _teacher_fingerprint(config, open_ids, close_ids, tokenizer)

    torch.manual_seed(config.seed)
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
    initial_processor_sha256 = _processor_state_sha256(model)
    initial_processor_gate = _processor_gate(model)
    source_revision, source_clean = _repository_state()

    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    run_config_path.write_text(
        json.dumps(to_dict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    run_config_sha256 = _sha256_file(run_config_path)

    trace_store = TeacherTraceStore(output_dir / "teacher-traces.jsonl", fingerprint=trace_fingerprint)
    optimizer = _optimizer(model, config)
    select_all = distill.boundary_selection == "all"
    decode_session = BGTDecodeSession(
        model,
        process_every=distill.response_process_every,
        select_all=select_all,
        valid_vocab_size=tokenizer.get_vocab_size(),
        record_displacements=(
            distill.overfit_diagnostic or distill.record_cache_displacements
        ),
    )

    diagnostic_examples: list[tuple[PromptRecord, DistillationExample]] | None = None
    overfit_diagnostic: dict[str, Any] | None = None
    if distill.overfit_diagnostic:
        diagnostic_examples, lock_report = _prepare_locked_examples(
            model,
            tokenizer,
            config,
            distill,
            trace_store,
            close_ids,
            decode_session,
            device,
        )
        initial_metrics = _evaluate_locked_examples(
            model,
            [example for _, example in diagnostic_examples],
            decode_session,
            config,
            device,
        )
        initial_probe, probe_tokens, probe_events = _student_logit_probe_sha256(
            model,
            [example for _, example in diagnostic_examples],
            decode_session,
            config,
            device,
        )
        overfit_diagnostic = {
            **lock_report,
            "locked_examples": len(diagnostic_examples),
            "initial": initial_metrics,
            "initial_processor_sha256": _processor_state_sha256(model),
            "initial_logit_probe_sha256": initial_probe,
            "logit_probe_response_tokens": probe_tokens,
            "logit_probe_event_cache_lengths": probe_events,
        }
        _atomic_write_json(
            locked_examples_path,
            _locked_examples_payload(diagnostic_examples),
        )

    seen = 0
    valid = 0
    skipped = 0
    answer_tokens = 0
    optimizer_steps = 0
    accumulated = 0
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)

    with metrics_path.open("x", encoding="utf-8") as metrics:
        for epoch in range(1, distill.epochs + 1):
            for record, locked_example in _epoch_training_items(config, diagnostic_examples):
                seen += 1
                processor_gate_before_example = _processor_gate(model)
                try:
                    if locked_example is None:
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
                    else:
                        example = locked_example
                        trace_cached = True
                    result = _backward_distillation_example(
                        model,
                        example,
                        decode_session,
                        config,
                        device,
                    )
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
                            "epoch": epoch,
                            "source_index": record.source_index,
                            "prompt_text_sha256": prompt_text_sha256(record.prompt),
                            "normalized_prompt_text_sha256": normalized_prompt_text_sha256(
                                record.prompt
                            ),
                            "status": "trained",
                            "loss": float(result.loss.detach().float().item()),
                            "unclipped_kl": float(result.unclipped_kl.item()),
                            "clipping_fraction": float(result.clipping_fraction.item()),
                            "student_entropy": float(result.student_entropy.item()),
                            "teacher_entropy": float(result.teacher_entropy.item()),
                            "teacher_reasoning_tokens": len(example.teacher_reasoning_ids),
                            "teacher_visible_response_tokens": len(
                                example.teacher_response_ids or ()
                            ),
                            "visible_trajectory_mode": distill.visible_trajectory_mode,
                            "answer_tokens": result.answer_tokens,
                            "student_trajectory_tokens": result.student_trajectory_tokens,
                            "teacher_trajectory_tokens": result.teacher_trajectory_tokens,
                            "cache_events": list(result.event_cache_lengths),
                            "teacher_trajectory_cache_events": list(
                                result.teacher_trajectory_event_cache_lengths
                            ),
                            "cache_relative_l2_displacements": list(
                                result.event_relative_l2_displacements
                            ),
                            "teacher_trajectory_cache_relative_l2_displacements": list(
                                result.teacher_trajectory_event_relative_l2_displacements
                            ),
                            "teacher_trace_cached": trace_cached,
                            "grad_norm": grad_norm,
                            "processor_gate_before_example": processor_gate_before_example,
                            "processor_gate_after_example": _processor_gate(model),
                            "optimizer_steps": optimizer_steps,
                        },
                    )
                    del result
                except InvalidDistillationExample as exc:
                    if not distill.skip_invalid_traces:
                        raise
                    skipped += 1
                    _write_metric(
                        metrics,
                        {
                            "epoch": epoch,
                            "source_index": record.source_index,
                            "prompt_text_sha256": prompt_text_sha256(record.prompt),
                            "normalized_prompt_text_sha256": normalized_prompt_text_sha256(
                                record.prompt
                            ),
                            "status": "skipped",
                            "reason": str(exc),
                            "optimizer_steps": optimizer_steps,
                        },
                    )

            if accumulated:
                _optimizer_step(model, optimizer, accumulated, config.grad_clip)
                optimizer_steps += 1
                accumulated = 0

    trace_store.sync()
    if valid == 0:
        raise RuntimeError("distillation epoch produced no valid training examples")
    if overfit_diagnostic is not None and diagnostic_examples is not None:
        locked = [example for _, example in diagnostic_examples]
        overfit_diagnostic["pre_save"] = _evaluate_locked_examples(
            model,
            locked,
            decode_session,
            config,
            device,
        )
        pre_save_processor_hash = _processor_state_sha256(model)
        overfit_diagnostic["pre_save_processor_sha256"] = pre_save_processor_hash
        overfit_diagnostic["processor_changed_from_initialization"] = (
            pre_save_processor_hash != overfit_diagnostic["initial_processor_sha256"]
        )
        if not overfit_diagnostic["processor_changed_from_initialization"]:
            raise RuntimeError("overfit diagnostic did not change the cache processor state")
        pre_save_probe, pre_probe_tokens, pre_probe_events = _student_logit_probe_sha256(
            model,
            locked,
            decode_session,
            config,
            device,
        )
        if (
            pre_probe_tokens != overfit_diagnostic["logit_probe_response_tokens"]
            or pre_probe_events != overfit_diagnostic["logit_probe_event_cache_lengths"]
        ):
            raise RuntimeError("logit probe coverage changed before checkpoint publication")
        overfit_diagnostic["pre_save_logit_probe_sha256"] = pre_save_probe
    elapsed = time.monotonic() - started
    train_state: dict[str, Any] = {
        "mode": "cot_distillation",
        "epoch": distill.epochs,
        "seen_prompts": seen,
        "valid_prompts": valid,
        "skipped_prompts": skipped,
        "answer_tokens": answer_tokens,
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": elapsed,
        "answer_tokens_per_second": answer_tokens / elapsed,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "teacher_trace_fingerprint": trace_fingerprint,
        "tokenizer_source": config.data.tokenizer.removeprefix("hf:"),
        "tokenizer_identity_sha256": tokenizer.identity_sha256(),
        "source_revision": source_revision,
        "source_clean": source_clean,
        "run_config_sha256": run_config_sha256,
        "training_seed": config.seed,
        "processor_initialization_recipe": (
            "torch.manual_seed(training_seed); build_model(model_config); v1"
        ),
        "torch_version": torch.__version__,
        "initial_processor_sha256": initial_processor_sha256,
        "initial_processor_gate": initial_processor_gate,
        "record_cache_displacements": distill.record_cache_displacements,
        "dtype": (
            "mixed_bfloat16_backbone_float32_processor"
            if config.bf16 and device.type == "cuda"
            else "float32"
        ),
    }
    if overfit_diagnostic is not None:
        train_state["overfit_diagnostic"] = overfit_diagnostic
    publication_checkpoint = (
        output_dir / ".final-overfit-staging"
        if overfit_diagnostic is not None
        else final_checkpoint
    )
    train_state = _publish_checkpoint(publication_checkpoint, model, config.model, train_state)
    if overfit_diagnostic is not None and diagnostic_examples is not None:
        del optimizer
        del decode_session
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            train_state = _verify_overfit_checkpoint(
                publication_checkpoint,
                config,
                device,
                distill,
                select_all,
                tokenizer.get_vocab_size(),
                [example for _, example in diagnostic_examples],
                overfit_diagnostic,
                train_state,
            )
            os.rename(publication_checkpoint, final_checkpoint)
            _fsync_directory(output_dir)
        except Exception:
            shutil.rmtree(publication_checkpoint, ignore_errors=True)
            raise
    _atomic_write_json(output_dir / "summary.json", train_state)
    print(json.dumps(train_state, sort_keys=True))


def _verify_overfit_checkpoint(
    checkpoint: Path,
    config: TrainConfig,
    device: torch.device,
    distill: DistillationConfig,
    select_all: bool,
    valid_vocab_size: int,
    locked: list[DistillationExample],
    overfit_diagnostic: dict[str, Any],
    train_state: dict[str, Any],
) -> dict[str, Any]:
    reloaded, reloaded_config, _ = load_checkpoint(checkpoint, device=device)
    if not isinstance(reloaded, BGT) or reloaded_config != config.model:
        raise RuntimeError("overfit diagnostic checkpoint reload changed the model contract")
    reloaded.backbone.eval()
    reloaded.cache_processor.train()
    reloaded_session = BGTDecodeSession(
        reloaded,
        process_every=distill.response_process_every,
        select_all=select_all,
        valid_vocab_size=valid_vocab_size,
        record_displacements=True,
    )
    post_load_metrics = _evaluate_locked_examples(
        reloaded,
        locked,
        reloaded_session,
        config,
        device,
    )
    pre_save_metrics = overfit_diagnostic["pre_save"]
    if post_load_metrics != pre_save_metrics:
        raise RuntimeError(
            "locked-corpus diagnostics changed across checkpoint reload: "
            f"{pre_save_metrics!r} -> {post_load_metrics!r}"
        )
    overfit_diagnostic["post_load"] = post_load_metrics
    post_load_probe, post_probe_tokens, post_probe_events = _student_logit_probe_sha256(
        reloaded,
        locked,
        reloaded_session,
        config,
        device,
    )
    if (
        post_probe_tokens != overfit_diagnostic["logit_probe_response_tokens"]
        or post_probe_events != overfit_diagnostic["logit_probe_event_cache_lengths"]
    ):
        raise RuntimeError("logit probe coverage changed after checkpoint reload")
    pre_save_probe = str(overfit_diagnostic["pre_save_logit_probe_sha256"])
    if post_load_probe != pre_save_probe:
        raise RuntimeError(
            "student logits changed across checkpoint reload: "
            f"{pre_save_probe} -> {post_load_probe}"
        )
    overfit_diagnostic["post_load_logit_probe_sha256"] = post_load_probe
    overfit_diagnostic["post_load_logits_verified"] = True
    verified_state = {**train_state, "overfit_diagnostic": overfit_diagnostic}
    _atomic_write_json(checkpoint / "train_state.json", verified_state)
    _fsync_directory(checkpoint)
    return verified_state


def _epoch_training_items(
    config: TrainConfig,
    locked_examples: list[tuple[PromptRecord, DistillationExample]] | None,
) -> Iterable[tuple[PromptRecord, DistillationExample | None]]:
    if locked_examples is not None:
        return locked_examples
    records = iter_openmath_prompts(
        config.data.data_dir or "",
        skip_rows=config.data.sft_skip_rows,
        take_rows=config.data.sft_take_rows,
    )
    return ((record, None) for record in records)


def _prepare_locked_examples(
    model: BGT,
    tokenizer: TransformersTokenizer,
    config: TrainConfig,
    distill: DistillationConfig,
    trace_store: TeacherTraceStore,
    close_ids: tuple[int, ...],
    decode_session: BGTDecodeSession,
    device: torch.device,
) -> tuple[list[tuple[PromptRecord, DistillationExample]], dict[str, Any]]:
    locked: list[tuple[PromptRecord, DistillationExample]] = []
    attempts = 0
    skips: list[dict[str, Any]] = []
    records = iter_openmath_prompts(
        config.data.data_dir or "",
        skip_rows=config.data.sft_skip_rows,
        take_rows=config.data.sft_take_rows,
    )
    for record in records:
        attempts += 1
        try:
            with _autocast_context(config, device):
                example, _ = _build_example(
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
        except InvalidDistillationExample as exc:
            if not distill.skip_invalid_traces:
                raise
            skips.append({"source_index": record.source_index, "reason": str(exc)})
            continue
        locked.append((record, example))
    if not locked:
        raise RuntimeError("overfit diagnostic produced no valid locked examples")
    expected = distill.expected_locked_examples
    if expected is None or len(locked) != expected:
        raise RuntimeError(
            f"overfit diagnostic expected exactly {expected} locked examples but got {len(locked)} "
            f"from {attempts} attempts with {len(skips)} skips"
        )
    trace_store.sync()
    return locked, {
        "lock_attempts": attempts,
        "lock_skips": skips,
        "locked_source_indices": [record.source_index for record, _ in locked],
    }


def _backward_distillation_example(
    model: BGT,
    example: DistillationExample,
    decode_session: BGTDecodeSession,
    config: TrainConfig,
    device: torch.device,
) -> DistillationLossResult:
    distill = config.distillation
    if distill is None:
        raise ValueError("distillation training requires distillation configuration")
    modes = (
        ("student", "teacher")
        if distill.visible_trajectory_mode == "both"
        else (distill.visible_trajectory_mode,)
    )
    detached_results: list[DistillationLossResult] = []
    for mode in modes:
        with _autocast_context(config, device):
            result = distillation_example_loss(
                model,
                example,
                decode_session,
                temperature=distill.temperature,
                token_clip=distill.token_clip,
                sequence_chunk_size=distill.kl_sequence_chunk_size,
                visible_trajectory_mode=mode,
            )
        if not bool(torch.isfinite(result.loss)):
            raise FloatingPointError(f"{mode} trajectory distillation loss is non-finite")
        (result.loss / len(modes)).backward()
        detached_results.append(_detach_loss_result(result))
        del result
    if len(detached_results) == 1:
        return detached_results[0]
    return _combine_serial_trajectory_results(*detached_results)


def _detach_loss_result(result: DistillationLossResult) -> DistillationLossResult:
    return DistillationLossResult(
        loss=result.loss.detach(),
        unclipped_kl=result.unclipped_kl.detach(),
        clipping_fraction=result.clipping_fraction.detach(),
        student_entropy=result.student_entropy.detach(),
        teacher_entropy=result.teacher_entropy.detach(),
        answer_tokens=result.answer_tokens,
        student_trajectory_tokens=result.student_trajectory_tokens,
        teacher_trajectory_tokens=result.teacher_trajectory_tokens,
        event_cache_lengths=result.event_cache_lengths,
        event_relative_l2_displacements=result.event_relative_l2_displacements,
        teacher_trajectory_event_cache_lengths=(
            result.teacher_trajectory_event_cache_lengths
        ),
        teacher_trajectory_event_relative_l2_displacements=(
            result.teacher_trajectory_event_relative_l2_displacements
        ),
    )


def _combine_serial_trajectory_results(
    student: DistillationLossResult, teacher: DistillationLossResult
) -> DistillationLossResult:
    def mean_scalars(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.stack((left, right)).mean()

    return DistillationLossResult(
        loss=mean_scalars(student.loss, teacher.loss),
        unclipped_kl=mean_scalars(student.unclipped_kl, teacher.unclipped_kl),
        clipping_fraction=mean_scalars(
            student.clipping_fraction, teacher.clipping_fraction
        ),
        student_entropy=mean_scalars(student.student_entropy, teacher.student_entropy),
        teacher_entropy=mean_scalars(student.teacher_entropy, teacher.teacher_entropy),
        answer_tokens=student.answer_tokens + teacher.answer_tokens,
        student_trajectory_tokens=student.student_trajectory_tokens,
        teacher_trajectory_tokens=teacher.teacher_trajectory_tokens,
        event_cache_lengths=student.event_cache_lengths,
        event_relative_l2_displacements=student.event_relative_l2_displacements,
        teacher_trajectory_event_cache_lengths=teacher.event_cache_lengths,
        teacher_trajectory_event_relative_l2_displacements=(
            teacher.event_relative_l2_displacements
        ),
    )


@torch.no_grad()
def _evaluate_locked_examples(
    model: BGT,
    examples: list[DistillationExample],
    decode_session: BGTDecodeSession,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    distill = config.distillation
    if distill is None:
        raise ValueError("locked-example evaluation requires distillation configuration")
    aggregate_loss = 0.0
    aggregate_kl = 0.0
    answer_tokens = 0
    student_displacements: list[float] = []
    teacher_displacements: list[float] = []
    for example in examples:
        with _autocast_context(config, device):
            result = distillation_example_loss(
                model,
                example,
                decode_session,
                temperature=distill.temperature,
                token_clip=distill.token_clip,
                sequence_chunk_size=distill.kl_sequence_chunk_size,
                visible_trajectory_mode=distill.visible_trajectory_mode,
            )
        weight = 1 if distill.visible_trajectory_mode == "both" else result.answer_tokens
        aggregate_loss += float(result.loss.detach().float().item()) * weight
        aggregate_kl += float(result.unclipped_kl.detach().float().item()) * weight
        answer_tokens += result.answer_tokens
        student_displacements.extend(result.event_relative_l2_displacements)
        teacher_displacements.extend(
            result.teacher_trajectory_event_relative_l2_displacements
        )
    if answer_tokens == 0:
        raise RuntimeError("locked-example evaluation produced no answer tokens")
    metrics: dict[str, Any] = {
        "examples": len(examples),
        "answer_tokens": answer_tokens,
        "processor_gate": _processor_gate(model),
    }
    if distill.visible_trajectory_mode == "both":
        metrics.update(
            {
                "objective_aggregation": "equal_example_mean_of_equal_trajectory_means",
                "objective_mean_loss": aggregate_loss / len(examples),
                "objective_mean_unclipped_kl": aggregate_kl / len(examples),
                "student_cache_events": len(student_displacements),
                "teacher_cache_events": len(teacher_displacements),
                "mean_student_cache_relative_l2_displacement": (
                    sum(student_displacements) / len(student_displacements)
                    if student_displacements
                    else 0.0
                ),
                "mean_teacher_cache_relative_l2_displacement": (
                    sum(teacher_displacements) / len(teacher_displacements)
                    if teacher_displacements
                    else 0.0
                ),
            }
        )
    else:
        metrics.update(
            {
                "objective_aggregation": "token_weighted_visible_positions",
                "token_weighted_loss": aggregate_loss / answer_tokens,
                "token_weighted_unclipped_kl": aggregate_kl / answer_tokens,
                "cache_events": len(student_displacements),
                "mean_cache_relative_l2_displacement": (
                    sum(student_displacements) / len(student_displacements)
                    if student_displacements
                    else 0.0
                ),
            }
        )
    return metrics


@torch.no_grad()
def _student_logit_probe_sha256(
    model: BGT,
    examples: list[DistillationExample],
    decode_session: BGTDecodeSession,
    config: TrainConfig,
    device: torch.device,
) -> tuple[str, int, list[int]]:
    if not examples:
        raise ValueError("student logit probe requires at least one locked example")
    example = max(examples, key=lambda item: len(item.response_ids))
    minimum = decode_session.process_every + 1
    if len(example.response_ids) < minimum:
        raise RuntimeError(
            "student logit probe requires a response long enough to cross one recurrent event: "
            f"need {minimum}, got {len(example.response_ids)}"
        )
    probe_count = min(len(example.response_ids), 2 * decode_session.process_every + 1)
    probe_ids = example.response_ids[:probe_count]
    with _autocast_context(config, device):
        replay = decode_session.replay(example.student_prompt, probe_ids)
    if len(replay.event_cache_lengths) < 2:
        raise RuntimeError("student logit probe did not exercise a recurrent processor event")
    logits = replay.logits.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = json.dumps(
        {"dtype": str(logits.dtype), "shape": list(logits.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(metadata)
    digest.update(logits.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest(), probe_count, list(replay.event_cache_lengths)


def _locked_examples_payload(
    examples: list[tuple[PromptRecord, DistillationExample]],
) -> dict[str, Any]:
    return {
        "schema": "gcw.opsd-overfit-locked-examples.v2",
        "examples": [
            {
                "source_index": record.source_index,
                "prompt": record.prompt,
                "student_base_ids": list(example.student_prompt.base_ids),
                "student_trigger_ids": list(example.student_prompt.trigger_ids),
                "teacher_base_ids": list(example.teacher_prompt.base_ids),
                "teacher_trigger_ids": list(example.teacher_prompt.trigger_ids),
                "teacher_reasoning_ids": list(example.teacher_reasoning_ids),
                "teacher_response_ids": (
                    list(example.teacher_response_ids)
                    if example.teacher_response_ids is not None
                    else None
                ),
                "response_ids": list(example.response_ids),
            }
            for record, example in examples
        ],
    }


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
        teacher_prompt = format_teacher_prompt(
            tokenizer,
            record.prompt,
            reasoning_open_text=distill.reasoning_open_text,
        )
    except ValueError as exc:
        raise InvalidDistillationExample(str(exc)) from exc
    teacher_response_ids: tuple[int, ...] | None = None
    open_ids = tuple(tokenizer.encode(distill.reasoning_open_text))
    if distill.teacher_trajectory_mode == "native_full":
        native = trace_store.get_native_trajectory(
            source_index=record.source_index, prompt=record.prompt
        )
        trace_cached = native is not None
        if native is None:
            teacher_generator = _generator(device, seed + 2 * record.source_index)
            try:
                generated = sample_backbone_completion(
                    model.backbone,
                    teacher_prompt.full_ids,
                    max_new_tokens=distill.max_teacher_completion_tokens,
                    eos_token_id=tokenizer.eot_id,
                    stop_ids=None,
                    temperature=distill.temperature,
                    top_p=distill.top_p,
                    top_k=distill.top_k,
                    valid_vocab_size=tokenizer.get_vocab_size(),
                    generator=teacher_generator,
                )
                native = parse_native_teacher_trajectory(
                    generated,
                    teacher_trigger_ids=teacher_prompt.trigger_ids,
                    open_ids=open_ids,
                    close_ids=close_ids,
                    eos_token_id=tokenizer.eot_id,
                )
            except ValueError as exc:
                raise InvalidDistillationExample(str(exc)) from exc
            _validate_native_trajectory_caps(native, distill)
            trace_store.put_native_trajectory(
                source_index=record.source_index,
                prompt=record.prompt,
                reasoning_ids=native.reasoning_ids,
                visible_response_ids=native.visible_response_ids,
                full_completion_ids=native.full_completion_ids,
            )
        else:
            native = parse_native_teacher_trajectory(
                native.full_completion_ids,
                teacher_trigger_ids=teacher_prompt.trigger_ids,
                open_ids=open_ids,
                close_ids=close_ids,
                eos_token_id=tokenizer.eot_id,
            )
            _validate_native_trajectory_caps(native, distill)
        reasoning_ids = native.reasoning_ids
        teacher_response_ids = native.visible_response_ids
    else:
        cached_reasoning = trace_store.get(
            source_index=record.source_index, prompt=record.prompt
        )
        trace_cached = cached_reasoning is not None
        if cached_reasoning is None:
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
                reasoning_ids = split_teacher_reasoning(
                    generated,
                    close_ids=close_ids,
                    eos_token_id=tokenizer.eot_id,
                    close_at_eos=distill.close_reasoning_at_eos,
                )
                if len(reasoning_ids) > distill.max_teacher_reasoning_tokens:
                    raise ValueError("closed teacher reasoning exceeds max_teacher_reasoning_tokens")
            except ValueError as exc:
                raise InvalidDistillationExample(str(exc)) from exc
            trace_store.put(
                source_index=record.source_index,
                prompt=record.prompt,
                reasoning_ids=reasoning_ids,
            )
        else:
            reasoning_ids = cached_reasoning
    if any(token_id >= tokenizer.get_vocab_size() for token_id in reasoning_ids):
        raise ValueError(
            f"cached teacher trace contains out-of-vocabulary IDs for source_index={record.source_index}"
        )
    if teacher_response_ids is not None and any(
        token_id >= tokenizer.get_vocab_size() for token_id in teacher_response_ids
    ):
        raise ValueError(
            f"cached teacher visible response contains out-of-vocabulary IDs for source_index={record.source_index}"
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
    if (
        distill.teacher_trajectory_mode == "native_full"
        and rollout.response_ids[-1] != tokenizer.eot_id
    ):
        raise InvalidDistillationExample("native full student response must end with EOT")
    if distill.teacher_trajectory_mode == "native_full" and (
        _contains_token_sequence(rollout.response_ids, open_ids)
        or _contains_token_sequence(rollout.response_ids, close_ids)
    ):
        raise InvalidDistillationExample(
            "native full student response contains thinking delimiters"
        )
    for trajectory_name, response_ids in (
        ("student", rollout.response_ids),
        ("teacher", teacher_response_ids),
    ):
        if response_ids is None:
            continue
        teacher_scoring_tokens = len(teacher_prompt.full_ids) + len(reasoning_ids) + len(response_ids)
        if teacher_scoring_tokens > model.config.sequence_len:
            raise InvalidDistillationExample(
                f"teacher context and {trajectory_name} response require {teacher_scoring_tokens} tokens but "
                f"model sequence_len is {model.config.sequence_len}"
            )
    return (
        DistillationExample(
            student_prompt=student_prompt,
            teacher_prompt=teacher_prompt,
            teacher_reasoning_ids=reasoning_ids,
            response_ids=rollout.response_ids,
            teacher_response_ids=teacher_response_ids,
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
    open_ids: tuple[int, ...],
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
        "schema": "gcw.teacher-trace.v5",
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
            "max_teacher_completion_tokens": distill.max_teacher_completion_tokens,
            "teacher_trajectory_mode": distill.teacher_trajectory_mode,
            "eos_token_id": tokenizer.eot_id,
            "reasoning_open_text": distill.reasoning_open_text,
            "open_ids": list(open_ids),
            "reasoning_close_text": distill.reasoning_close_text,
            "close_ids": list(close_ids),
            "close_reasoning_at_eos": distill.close_reasoning_at_eos,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_native_trajectory_caps(
    trajectory: NativeTeacherTrajectory, distill: DistillationConfig
) -> None:
    if len(trajectory.reasoning_ids) > distill.max_teacher_reasoning_tokens:
        raise InvalidDistillationExample(
            "native teacher reasoning exceeds max_teacher_reasoning_tokens"
        )
    if len(trajectory.full_completion_ids) > distill.max_teacher_completion_tokens:
        raise InvalidDistillationExample(
            "native teacher completion exceeds max_teacher_completion_tokens"
        )


def _contains_token_sequence(
    sequence: tuple[int, ...], pattern: tuple[int, ...]
) -> bool:
    return any(
        sequence[start : start + len(pattern)] == pattern
        for start in range(len(sequence) - len(pattern) + 1)
    )


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_state() -> tuple[str | None, bool]:
    repository = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None, False
    return revision, not bool(dirty)


def _processor_state_sha256(model: BGT) -> str:
    return tensor_state_sha256(dict(model.cache_processor.state_dict()))


def _processor_gate(model: BGT) -> float:
    gate = getattr(getattr(model.cache_processor, "block", None), "gate", None)
    if not isinstance(gate, torch.Tensor) or gate.numel() != 1:
        raise ValueError("BGT cache processor must expose one scalar block.gate")
    gate_tensor = cast(torch.Tensor, gate)
    return float(gate_tensor.detach().float().cpu().item())


def _publish_checkpoint(
    destination: Path,
    model: BGT,
    model_config: ModelConfig,
    train_state: dict[str, Any],
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    expected_hash = _processor_state_sha256(model)
    expected_gate = _processor_gate(model)
    published_state = {
        **train_state,
        "checkpoint_fidelity_verified": False,
        "pre_save_processor_sha256": expected_hash,
        "pre_save_processor_gate": expected_gate,
    }
    temporary = Path(tempfile.mkdtemp(prefix=".final-", dir=destination.parent))
    try:
        save_checkpoint(temporary, model, model_config, published_state)
        reloaded, reloaded_config, _ = load_checkpoint(temporary, device="cpu")
        if not isinstance(reloaded, BGT) or reloaded_config != model_config:
            raise RuntimeError("published checkpoint did not reload as the expected BGT model")
        observed_hash = _processor_state_sha256(reloaded)
        observed_gate = _processor_gate(reloaded)
        if observed_hash != expected_hash or observed_gate != expected_gate:
            raise RuntimeError(
                "cache processor changed across checkpoint publication: "
                f"hash {expected_hash} -> {observed_hash}, gate {expected_gate} -> {observed_gate}"
            )
        published_state.update(
            {
                "checkpoint_fidelity_verified": True,
                "post_load_processor_sha256": observed_hash,
                "post_load_processor_gate": observed_gate,
            }
        )
        del reloaded
        gc.collect()
        _atomic_write_json(temporary / "train_state.json", published_state)
        for child in temporary.iterdir():
            if child.is_file():
                with child.open("rb") as handle:
                    os.fsync(handle.fileno())
        os.rename(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return published_state


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
