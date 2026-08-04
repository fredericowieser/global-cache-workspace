from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from gcw.checkpoint import load_checkpoint
from gcw.config import TrainConfig, load_train_config, load_yaml, to_dict
from gcw.data.sft import _iter_openmath_rows, _prompt_answer
from gcw.data.tokenizers import TransformersTokenizer, get_tokenizer
from gcw.distill.objective import exact_forward_kl_with_metrics
from gcw.distill.pipeline import DistillationExample
from gcw.distill.prompt import format_chat_prompt, format_teacher_prompt
from gcw.distill.session import BGTDecodeSession
from gcw.distill.teacher import sample_backbone_completion, teacher_answer_logits
from gcw.distill.traces import TeacherTraceStore
from gcw.distill.trainer import (
    InvalidDistillationExample,
    _autocast_context,
    _generator,
    _processor_gate,
    _processor_state_sha256,
    _teacher_fingerprint,
)
from gcw.models.bgt import BGT
from gcw.opsd_eval import aggregate_opsd_three_way, audit_opsd_prompt_overlap
from gcw.prompt_identity import normalized_prompt_text_sha256, prompt_text_sha256
from gcw.state_hash import tensor_state_sha256


@dataclass(frozen=True)
class HeldoutSourceRow:
    source_index: int
    prompt: str
    dataset_answer: str


def iter_heldout_source_rows(
    source: str | Path,
    *,
    skip_rows: int,
    max_attempts: int,
) -> Iterator[HeldoutSourceRow]:
    if skip_rows < 0:
        raise ValueError("skip_rows must be non-negative")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    attempted = 0
    for source_index, raw in enumerate(_iter_openmath_rows(source)):
        if source_index < skip_rows:
            continue
        if attempted >= max_attempts:
            return
        attempted += 1
        prompt, dataset_answer = _prompt_answer(raw)
        yield HeldoutSourceRow(
            source_index=source_index,
            prompt=prompt,
            dataset_answer=dataset_answer,
        )


def classify_teacher_completion(
    generated_ids: tuple[int, ...],
    *,
    close_ids: tuple[int, ...],
    eos_token_id: int | None,
    close_at_eos: bool,
) -> tuple[tuple[int, ...], str]:
    if not close_ids:
        raise ValueError("teacher close IDs must not be empty")
    last_start = len(generated_ids) - len(close_ids)
    for start in range(last_start + 1):
        if generated_ids[start : start + len(close_ids)] == close_ids:
            return generated_ids[: start + len(close_ids)], "native_close"
    if (
        close_at_eos
        and eos_token_id is not None
        and generated_ids
        and generated_ids[-1] == eos_token_id
    ):
        return generated_ids[:-1] + close_ids, "eot_rewritten"
    raise ValueError("budget_exhausted: teacher completion has no close or terminal EOT")


def dataset_solution_token_nll(
    aligned_logits: torch.Tensor,
    dataset_solution_ids: tuple[int, ...],
) -> float:
    if aligned_logits.ndim != 3 or aligned_logits.shape[0] != 1:
        raise ValueError("dataset solution logits must have shape [1, sequence, vocabulary]")
    if not dataset_solution_ids:
        raise ValueError("dataset solution IDs must not be empty")
    if aligned_logits.shape[1] != len(dataset_solution_ids):
        raise ValueError("dataset solution logits and target IDs must align")
    targets = torch.tensor(
        dataset_solution_ids,
        dtype=torch.long,
        device=aligned_logits.device,
    )
    loss = F.cross_entropy(
        aligned_logits.float().reshape(-1, aligned_logits.shape[-1]),
        targets,
    )
    value = float(loss.detach().cpu().item())
    if not math.isfinite(value) or value < 0:
        raise RuntimeError("dataset solution NLL must be finite and non-negative")
    return value


@dataclass(frozen=True)
class _LockedAuditExample:
    source: HeldoutSourceRow
    example: DistillationExample
    dataset_solution_ids: tuple[int, ...]
    raw_teacher_completion_ids: tuple[int, ...]
    teacher_trace_termination: str
    identity: dict[str, Any]


@dataclass(frozen=True)
class _AuditScore:
    exact_forward_kl: float
    student_entropy: float
    teacher_entropy: float
    answer_tokens: int
    event_cache_lengths: tuple[int, ...]
    event_relative_l2_displacements: tuple[float, ...]
    teacher_logits_sha256: str


@dataclass(frozen=True)
class OpsdMatchedAuditConfig:
    train_config: str
    trained_checkpoint: str
    untrained_checkpoint: str
    training_metrics: str
    evaluation_inputs: list[str]
    output: str
    expected_source_revision: str
    source_skip_rows: int = 128
    valid_examples: int = 32
    max_attempts: int = 64
    student_seed: int = 24
    bootstrap_seed: int = 240806
    bootstrap_samples: int = 10_000

    def __post_init__(self) -> None:
        required_paths = (
            self.train_config,
            self.trained_checkpoint,
            self.untrained_checkpoint,
            self.training_metrics,
            self.output,
        )
        if any(not isinstance(path, str) or not path for path in required_paths):
            raise ValueError("matched OPSD audit paths must be non-empty")
        if len(self.evaluation_inputs) != 3 or any(
            not isinstance(path, str) or not path for path in self.evaluation_inputs
        ):
            raise ValueError("matched OPSD audit requires exactly three evaluation inputs")
        if not isinstance(self.expected_source_revision, str) or len(
            self.expected_source_revision
        ) not in (40, 64):
            raise ValueError("matched OPSD audit requires a full expected source revision")
        production_values = {
            "source_skip_rows": (self.source_skip_rows, 128),
            "valid_examples": (self.valid_examples, 32),
            "max_attempts": (self.max_attempts, 64),
            "student_seed": (self.student_seed, 24),
            "bootstrap_samples": (self.bootstrap_samples, 10_000),
        }
        drift = [name for name, (actual, expected) in production_values.items() if actual != expected]
        if drift:
            raise ValueError(
                "matched OPSD audit production protocol drift in " + ", ".join(drift)
            )
        if self.bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be non-negative")


def load_opsd_matched_audit_config(path: str | Path) -> OpsdMatchedAuditConfig:
    return OpsdMatchedAuditConfig(**load_yaml(path))


def run_matched_opsd_audit(config: OpsdMatchedAuditConfig) -> dict[str, Any]:
    output = Path(config.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite matched OPSD audit output: {output}")
    result = _execute_matched_opsd_audit(config)
    _atomic_write_json_new(output, result)
    return result


def _execute_matched_opsd_audit(config: OpsdMatchedAuditConfig) -> dict[str, Any]:
    train_config, static_provenance, forbidden_hashes = _validate_static_provenance(config)
    distill = train_config.distillation
    if distill is None or train_config.data.data_dir is None:
        raise ValueError("matched OPSD audit requires distillation data configuration")
    tokenizer = get_tokenizer(train_config.data.tokenizer)
    if not isinstance(tokenizer, TransformersTokenizer):
        raise TypeError("matched OPSD audit requires a TransformersTokenizer")
    expected_tokenizer_hash = static_provenance["tokenizer_identity_sha256"]
    if tokenizer.identity_sha256() != expected_tokenizer_hash:
        raise ValueError("runtime tokenizer identity differs from checkpoint provenance")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("matched OPSD audit requires CUDA")
    output_parent = Path(config.output).parent
    output_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_parent,
        prefix=".opsd-kl-audit-",
    ) as scratch_name:
        scratch = Path(scratch_name)
        trace_path = scratch / "teacher-traces.jsonl"
        open_ids = tuple(tokenizer.encode(distill.reasoning_open_text))
        close_ids = tuple(tokenizer.encode(distill.reasoning_close_text))
        trace_store = TeacherTraceStore(
            trace_path,
            fingerprint=_teacher_fingerprint(train_config, open_ids, close_ids, tokenizer),
        )

        untrained_model, untrained_model_config, untrained_state = load_checkpoint(
            config.untrained_checkpoint,
            device="cpu",
        )
        if not isinstance(untrained_model, BGT):
            raise TypeError("matched OPSD audit requires a BGT untrained checkpoint")
        if to_dict(untrained_model_config) != to_dict(train_config.model):
            raise ValueError("untrained checkpoint model config differs from audit train config")
        _validate_live_mixed_dtype(untrained_model, label="untrained")
        untrained_backbone_sha = tensor_state_sha256(
            dict(untrained_model.backbone.state_dict())
        )
        if untrained_backbone_sha != untrained_state.get("control_backbone_sha256"):
            raise ValueError("untrained checkpoint backbone hash differs from provenance")
        untrained_model.to(device)
        untrained_model.eval()
        untrained_session = _audit_session(
            untrained_model,
            train_config,
            tokenizer,
        )
        locked, untrained_rows, selection = _lock_and_score_untrained(
            untrained_model,
            untrained_session,
            train_config,
            tokenizer,
            trace_store,
            close_ids,
            config,
            forbidden_hashes,
            device,
        )
        trace_store.sync()
        trace_sha256 = _sha256_file(trace_path)
        untrained_processor_sha = _processor_state_sha256(untrained_model)
        untrained_gate = _processor_gate(untrained_model)
        if (
            untrained_processor_sha != untrained_state.get("initial_processor_sha256")
            or untrained_gate != untrained_state.get("initial_gate_float32")
        ):
            raise ValueError("live untrained processor differs from control provenance")
        del untrained_session, untrained_model
        gc.collect()
        torch.cuda.empty_cache()

        trained_model, trained_model_config, trained_state = load_checkpoint(
            config.trained_checkpoint,
            device="cpu",
        )
        if not isinstance(trained_model, BGT):
            raise TypeError("matched OPSD audit requires a BGT trained checkpoint")
        _validate_live_mixed_dtype(trained_model, label="trained")
        if to_dict(trained_model_config) != to_dict(untrained_model_config):
            raise ValueError("trained and untrained audit model configurations differ")
        trained_backbone_sha = tensor_state_sha256(dict(trained_model.backbone.state_dict()))
        if trained_backbone_sha != untrained_backbone_sha:
            raise ValueError("trained and untrained audit backbones differ")
        trained_model.to(device)
        trained_model.eval()
        trained_session = _audit_session(trained_model, train_config, tokenizer)
        trained_rows = _score_locked_examples(
            trained_model,
            trained_session,
            train_config,
            locked,
            untrained_rows,
            device,
        )
        trained_processor_sha = _processor_state_sha256(trained_model)
        trained_gate = _processor_gate(trained_model)
        if (
            trained_processor_sha != trained_state.get("post_load_processor_sha256")
            or trained_gate != trained_state.get("post_load_processor_gate")
        ):
            raise ValueError("live trained processor differs from checkpoint provenance")
        del trained_session, trained_model
        gc.collect()
        torch.cuda.empty_cache()

    aggregate = aggregate_matched_opsd_audit(
        untrained_rows,
        trained_rows,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_samples=config.bootstrap_samples,
        expected_examples=config.valid_examples,
    )
    return {
        **aggregate,
        "audit_config": to_dict(config),
        "protocol": {
            "source_skip_rows": config.source_skip_rows,
            "valid_examples": config.valid_examples,
            "max_attempts": config.max_attempts,
            "student_seed": config.student_seed,
            "fixed_student_prefixes": True,
            "teacher_consistency_target": "unverified normalized dataset solution tokens",
            "dataset_solution_reference_provenance": "unverified_normalized_answer_field",
            "teacher_contexts": ["no_privilege", "self_generated_reasoning"],
            "temperature": distill.temperature,
            "response_process_every": distill.response_process_every,
        },
        "selection": selection,
        "provenance": {
            **static_provenance,
            "teacher_trace_sha256": trace_sha256,
            "untrained_backbone_sha256": untrained_backbone_sha,
            "trained_backbone_sha256": trained_backbone_sha,
            "untrained_processor_sha256": untrained_processor_sha,
            "trained_processor_sha256": trained_processor_sha,
            "untrained_processor_gate": untrained_gate,
            "trained_processor_gate": trained_gate,
            "untrained_train_state": untrained_state,
            "trained_train_state": trained_state,
            "model_config": to_dict(trained_model_config),
            "torch_version": torch.__version__,
        },
        "untrained_rows": untrained_rows,
        "trained_rows": trained_rows,
        "locked_examples": [_locked_payload(item) for item in locked],
    }


def _build_audit_example(
    model: BGT,
    tokenizer: TransformersTokenizer,
    source: HeldoutSourceRow,
    train_config: TrainConfig,
    trace_store: TeacherTraceStore,
    close_ids: tuple[int, ...],
    session: BGTDecodeSession,
    device: torch.device,
) -> tuple[DistillationExample, tuple[int, ...], str]:
    distill = train_config.distillation
    if distill is None:
        raise ValueError("audit example building requires distillation configuration")
    try:
        student_prompt = format_chat_prompt(
            tokenizer,
            source.prompt,
            enable_thinking=False,
        )
        teacher_prompt = format_teacher_prompt(
            tokenizer,
            source.prompt,
            reasoning_open_text=distill.reasoning_open_text,
        )
        raw_completion = sample_backbone_completion(
            model.backbone,
            teacher_prompt.full_ids,
            max_new_tokens=distill.max_teacher_reasoning_tokens,
            eos_token_id=tokenizer.eot_id,
            stop_ids=close_ids,
            temperature=distill.temperature,
            top_p=distill.top_p,
            top_k=distill.top_k,
            valid_vocab_size=tokenizer.get_vocab_size(),
            generator=_generator(device, train_config.seed + 2 * source.source_index),
        )
        reasoning_ids, termination = classify_teacher_completion(
            raw_completion,
            close_ids=close_ids,
            eos_token_id=tokenizer.eot_id,
            close_at_eos=distill.close_reasoning_at_eos,
        )
        if len(reasoning_ids) > distill.max_teacher_reasoning_tokens:
            raise ValueError("closed teacher reasoning exceeds max_teacher_reasoning_tokens")
        trace_store.put(
            source_index=source.source_index,
            prompt=source.prompt,
            reasoning_ids=reasoning_ids,
        )
        rollout = session.rollout(
            student_prompt,
            max_response_tokens=distill.max_response_tokens,
            eos_token_id=tokenizer.eot_id,
            temperature=distill.temperature,
            top_p=distill.top_p,
            top_k=distill.top_k,
            generator=_generator(
                device,
                train_config.seed + 2 * source.source_index + 1,
            ),
        )
    except ValueError as exc:
        raise InvalidDistillationExample(str(exc)) from exc
    teacher_scoring_tokens = (
        len(teacher_prompt.full_ids) + len(reasoning_ids) + len(rollout.response_ids)
    )
    if teacher_scoring_tokens > model.config.sequence_len:
        raise InvalidDistillationExample(
            f"teacher context and student response require {teacher_scoring_tokens} tokens "
            f"but model sequence_len is {model.config.sequence_len}"
        )
    return (
        DistillationExample(
            student_prompt=student_prompt,
            teacher_prompt=teacher_prompt,
            teacher_reasoning_ids=reasoning_ids,
            response_ids=rollout.response_ids,
        ),
        raw_completion,
        termination,
    )


def _score_audit_example(
    model: BGT,
    session: BGTDecodeSession,
    example: DistillationExample,
    train_config: TrainConfig,
    device: torch.device,
) -> _AuditScore:
    distill = train_config.distillation
    if distill is None:
        raise ValueError("audit scoring requires distillation configuration")
    with torch.no_grad(), _autocast_context(train_config, device):
        student = session.replay(example.student_prompt, example.response_ids)
        teacher = teacher_answer_logits(
            model.backbone,
            example.teacher_prompt.full_ids + example.teacher_reasoning_ids,
            example.response_ids,
        )
        if teacher.device != student.logits.device:
            teacher = teacher.to(student.logits.device)
        teacher_logits_sha256 = _tensor_sha256(teacher)
        answer_mask = torch.ones(
            student.logits.shape[:2],
            dtype=torch.bool,
            device=student.logits.device,
        )
        objective = exact_forward_kl_with_metrics(
            student.logits,
            teacher,
            answer_mask,
            temperature=distill.temperature,
            token_clip=distill.token_clip,
            sequence_chunk_size=distill.kl_sequence_chunk_size,
        )
    return _AuditScore(
        exact_forward_kl=float(objective.unclipped_kl.detach().float().item()),
        student_entropy=float(objective.student_entropy.detach().float().item()),
        teacher_entropy=float(objective.teacher_entropy.detach().float().item()),
        answer_tokens=len(example.response_ids),
        event_cache_lengths=student.event_cache_lengths,
        event_relative_l2_displacements=student.event_relative_l2_displacements,
        teacher_logits_sha256=teacher_logits_sha256,
    )


def _lock_and_score_untrained(
    model: BGT,
    session: BGTDecodeSession,
    train_config: TrainConfig,
    tokenizer: TransformersTokenizer,
    trace_store: TeacherTraceStore,
    close_ids: tuple[int, ...],
    audit_config: OpsdMatchedAuditConfig,
    forbidden_hashes: dict[str, set[str]],
    device: torch.device,
) -> tuple[list[_LockedAuditExample], list[dict[str, Any]], dict[str, Any]]:
    distill = train_config.distillation
    if distill is None or train_config.data.data_dir is None:
        raise ValueError("held-out locking requires distillation data configuration")
    locked: list[_LockedAuditExample] = []
    rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    skip_counts: dict[str, int] = {}
    locked_raw_hashes: set[str] = set()
    locked_normalized_hashes: set[str] = set()
    for source in iter_heldout_source_rows(
        train_config.data.data_dir,
        skip_rows=audit_config.source_skip_rows,
        max_attempts=audit_config.max_attempts,
    ):
        raw_hash = prompt_text_sha256(source.prompt)
        normalized_hash = normalized_prompt_text_sha256(source.prompt)
        attempt = {
            "source_index": source.source_index,
            "prompt_text_sha256": raw_hash,
            "normalized_prompt_text_sha256": normalized_hash,
        }
        reason: str | None = None
        if not source.prompt:
            reason = "empty prompt"
        elif not source.dataset_answer:
            reason = "empty dataset answer"
        elif (
            raw_hash in locked_raw_hashes
            or normalized_hash in locked_normalized_hashes
        ):
            reason = "duplicate held-out prompt"
        elif (
            raw_hash in forbidden_hashes["prompt_text_sha256"]
            or normalized_hash
            in forbidden_hashes["normalized_prompt_text_sha256"]
        ):
            reason = "training/evaluation prompt overlap"
        if reason is not None:
            attempts.append({**attempt, "status": "skipped", "reason": reason})
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            continue
        try:
            with _autocast_context(train_config, device):
                (
                    example,
                    raw_teacher_completion,
                    trace_termination,
                ) = _build_audit_example(
                    model,
                    tokenizer,
                    source,
                    train_config,
                    trace_store,
                    close_ids,
                    session,
                    device,
                )
        except InvalidDistillationExample as exc:
            reason = str(exc)
            attempts.append({**attempt, "status": "skipped", "reason": reason})
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            continue

        dataset_ids = tuple(int(token) for token in tokenizer.encode(source.dataset_answer))
        if not dataset_ids or dataset_ids[-1] != tokenizer.eot_id:
            dataset_ids = dataset_ids + (tokenizer.eot_id,)
        no_privilege_length = len(example.student_prompt.full_ids) + len(dataset_ids)
        self_reasoning_length = (
            len(example.teacher_prompt.full_ids)
            + len(example.teacher_reasoning_ids)
            + len(dataset_ids)
        )
        if max(no_privilege_length, self_reasoning_length) > model.config.sequence_len:
            reason = "dataset solution context exceeds model sequence length"
            attempts.append({**attempt, "status": "skipped", "reason": reason})
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            continue

        result = _score_audit_example(
            model,
            session,
            example,
            train_config,
            device,
        )
        with torch.no_grad(), _autocast_context(train_config, device):
            no_privilege_logits = teacher_answer_logits(
                model.backbone,
                example.student_prompt.full_ids,
                dataset_ids,
            )
        no_privilege_nll = dataset_solution_token_nll(
            no_privilege_logits,
            dataset_ids,
        )
        del no_privilege_logits
        with torch.no_grad(), _autocast_context(train_config, device):
            self_reasoning_logits = teacher_answer_logits(
                model.backbone,
                example.teacher_prompt.full_ids + example.teacher_reasoning_ids,
                dataset_ids,
            )
        self_reasoning_nll = dataset_solution_token_nll(
            self_reasoning_logits,
            dataset_ids,
        )
        del self_reasoning_logits
        identity = {
            "source_index": source.source_index,
            "prompt_text_sha256": raw_hash,
            "normalized_prompt_text_sha256": normalized_hash,
            "dataset_answer_text_sha256": prompt_text_sha256(source.dataset_answer),
            "dataset_solution_ids_sha256": _token_ids_sha256(dataset_ids),
            "student_response_ids_sha256": _token_ids_sha256(example.response_ids),
            "teacher_reasoning_ids_sha256": _token_ids_sha256(
                example.teacher_reasoning_ids
            ),
            "raw_teacher_completion_ids_sha256": _token_ids_sha256(
                raw_teacher_completion
            ),
            "teacher_trace_termination": trace_termination,
            "raw_teacher_completion_tokens": len(raw_teacher_completion),
            "teacher_reasoning_tokens": len(example.teacher_reasoning_ids),
            "response_anchor": "seed_identical_untrained_cp_on",
            "student_prompt_ids_sha256": _token_ids_sha256(
                example.student_prompt.full_ids
            ),
            "teacher_prompt_ids_sha256": _token_ids_sha256(
                example.teacher_prompt.full_ids
            ),
            "teacher_logits_sha256": result.teacher_logits_sha256,
            "event_cache_lengths": list(result.event_cache_lengths),
            "answer_tokens": result.answer_tokens,
            "dataset_solution_tokens": len(dataset_ids),
            "no_privilege_dataset_solution_nll": no_privilege_nll,
            "self_reasoning_dataset_solution_nll": self_reasoning_nll,
        }
        row = {
            **identity,
            "exact_forward_kl": result.exact_forward_kl,
            "student_entropy": result.student_entropy,
            "teacher_entropy": result.teacher_entropy,
            "cache_event_count": len(result.event_cache_lengths),
            "mean_cache_relative_l2_displacement": _mean(
                result.event_relative_l2_displacements
            ),
        }
        locked.append(
            _LockedAuditExample(
                source=source,
                example=example,
                dataset_solution_ids=dataset_ids,
                raw_teacher_completion_ids=raw_teacher_completion,
                teacher_trace_termination=trace_termination,
                identity=identity,
            )
        )
        locked_raw_hashes.add(raw_hash)
        locked_normalized_hashes.add(normalized_hash)
        rows.append(row)
        attempts.append(
            {
                **attempt,
                "status": "locked",
                "teacher_trace_termination": trace_termination,
                "raw_teacher_completion_ids_sha256": identity[
                    "raw_teacher_completion_ids_sha256"
                ],
                "student_response_ids_sha256": identity[
                    "student_response_ids_sha256"
                ],
            }
        )
        if len(locked) == audit_config.valid_examples:
            break
    if len(locked) != audit_config.valid_examples:
        raise RuntimeError(
            f"matched OPSD audit locked {len(locked)} valid examples; "
            f"expected {audit_config.valid_examples} within {audit_config.max_attempts} attempts"
        )
    return (
        locked,
        rows,
        {
            "attempted_rows": len(attempts),
            "locked_examples": len(locked),
            "skipped_examples": len(attempts) - len(locked),
            "skip_counts": skip_counts,
            "attempts": attempts,
        },
    )


def _score_locked_examples(
    model: BGT,
    session: BGTDecodeSession,
    train_config: TrainConfig,
    locked: list[_LockedAuditExample],
    untrained_rows: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    if train_config.distillation is None:
        raise ValueError("locked scoring requires distillation configuration")
    output: list[dict[str, Any]] = []
    for item, untrained in zip(locked, untrained_rows, strict=True):
        result = _score_audit_example(
            model,
            session,
            item.example,
            train_config,
            device,
        )
        if result.answer_tokens != item.identity["answer_tokens"]:
            raise RuntimeError("trained replay changed the locked answer-token count")
        if result.teacher_logits_sha256 != item.identity["teacher_logits_sha256"]:
            raise RuntimeError("teacher logits differ across sequential audit passes")
        if list(result.event_cache_lengths) != item.identity["event_cache_lengths"]:
            raise RuntimeError("cache event schedule differs across audit passes")
        output.append(
            {
                **item.identity,
                "exact_forward_kl": result.exact_forward_kl,
                "student_entropy": result.student_entropy,
                "teacher_entropy": result.teacher_entropy,
                "cache_event_count": len(result.event_cache_lengths),
                "mean_cache_relative_l2_displacement": _mean(
                    result.event_relative_l2_displacements
                ),
                "untrained_row_sha256": _json_sha256(untrained),
            }
        )
    return output


def _locked_payload(item: _LockedAuditExample) -> dict[str, Any]:
    example = item.example
    return {
        **item.identity,
        "student_base_ids": list(example.student_prompt.base_ids),
        "student_trigger_ids": list(example.student_prompt.trigger_ids),
        "teacher_base_ids": list(example.teacher_prompt.base_ids),
        "teacher_trigger_ids": list(example.teacher_prompt.trigger_ids),
        "teacher_reasoning_ids": list(example.teacher_reasoning_ids),
        "raw_teacher_completion_ids": list(item.raw_teacher_completion_ids),
        "teacher_trace_termination": item.teacher_trace_termination,
        "student_response_ids": list(example.response_ids),
        "dataset_solution_ids": list(item.dataset_solution_ids),
    }


def _validate_static_provenance(
    config: OpsdMatchedAuditConfig,
) -> tuple[TrainConfig, dict[str, Any], dict[str, set[str]]]:
    revision, clean = _repository_state()
    if revision is None or not clean:
        raise ValueError("matched OPSD audit requires a clean Git source revision")
    train_config = load_train_config(config.train_config)
    distill = train_config.distillation
    if (
        train_config.mode != "cot_distillation"
        or distill is None
        or train_config.seed != config.student_seed
        or train_config.data.sft_skip_rows != 0
        or train_config.data.sft_take_rows != 128
        or distill.epochs != 1
        or distill.token_clip is not None
        or not distill.record_cache_displacements
    ):
        raise ValueError("matched OPSD audit train config differs from the approved pilot")
    if train_config.data.data_dir is None:
        raise ValueError("matched OPSD audit train config lacks a data source")

    trained_checkpoint = Path(config.trained_checkpoint).resolve(strict=True)
    untrained_checkpoint = Path(config.untrained_checkpoint).resolve(strict=True)
    training_metrics_path = Path(config.training_metrics).resolve(strict=True)
    if training_metrics_path.parent != trained_checkpoint.parent:
        raise ValueError("training metrics do not belong to the trained checkpoint run")
    trained_state = _read_json(trained_checkpoint / "train_state.json")
    untrained_state = _read_json(untrained_checkpoint / "train_state.json")
    run_config_payload = json.dumps(
        to_dict(train_config),
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    run_config_sha256 = hashlib.sha256(run_config_payload).hexdigest()
    if (
        trained_state.get("source_revision") != config.expected_source_revision
        or trained_state.get("source_clean") is not True
        or trained_state.get("training_seed") != config.student_seed
        or trained_state.get("run_config_sha256") != run_config_sha256
        or trained_state.get("checkpoint_fidelity_verified") is not True
        or trained_state.get("dtype")
        != "mixed_bfloat16_backbone_float32_processor"
    ):
        raise ValueError("trained checkpoint provenance differs from the approved pilot")
    if (
        untrained_state.get("kind")
        != "derived_untrained_fp32_cache_processor_control"
        or untrained_state.get("source_revision") != config.expected_source_revision
        or untrained_state.get("processor_initialization_seed") != config.student_seed
        or untrained_state.get("source_run_config_sha256") != run_config_sha256
        or untrained_state.get("source_checkpoint")
        != str(trained_checkpoint)
        or untrained_state.get("trained_processor_sha256")
        != trained_state.get("post_load_processor_sha256")
        or untrained_state.get("initial_processor_sha256")
        != trained_state.get("initial_processor_sha256")
        or untrained_state.get("dtype")
        != "mixed_bfloat16_backbone_float32_processor"
        or untrained_state.get("source_backbone_sha256")
        != untrained_state.get("control_backbone_sha256")
    ):
        raise ValueError("untrained checkpoint provenance differs from the approved control")
    trained_model_sha256 = _sha256_file(trained_checkpoint / "model.safetensors")
    if untrained_state.get("source_model_sha256") != trained_model_sha256:
        raise ValueError("untrained control source model hash differs from trained checkpoint")
    tokenizer_hash = trained_state.get("tokenizer_identity_sha256")
    if (
        not isinstance(tokenizer_hash, str)
        or untrained_state.get("tokenizer_identity_sha256") != tokenizer_hash
    ):
        raise ValueError("trained and untrained tokenizer provenance differs")

    training_rows = _read_jsonl(training_metrics_path)
    evaluation_paths = {
        label: Path(path).resolve(strict=True)
        for label, path in zip(
            ("off", "untrained", "trained"),
            config.evaluation_inputs,
            strict=True,
        )
    }
    evaluation_documents = {
        label: _read_json(path) for label, path in evaluation_paths.items()
    }
    for label, document in evaluation_documents.items():
        if document.get("source_revision") != config.expected_source_revision:
            raise ValueError(f"{label} evaluation source revision differs from audit")
        if document.get("model_config") != to_dict(train_config.model):
            raise ValueError(f"{label} evaluation model config differs from audit")
    trained_checkpoint_manifest = _checkpoint_manifest(trained_checkpoint)
    untrained_checkpoint_manifest = _checkpoint_manifest(untrained_checkpoint)
    expected_checkpoint_files = {
        "off": trained_checkpoint_manifest["files"],
        "untrained": untrained_checkpoint_manifest["files"],
        "trained": trained_checkpoint_manifest["files"],
    }
    for label, document in evaluation_documents.items():
        if document.get("checkpoint_manifest", {}).get("files") != expected_checkpoint_files[
            label
        ]:
            raise ValueError(f"{label} evaluation checkpoint differs from live artifact")
    prior_three_way = aggregate_opsd_three_way(evaluation_documents)
    overlap = audit_opsd_prompt_overlap(training_rows, evaluation_documents)
    if overlap["held_out_prompt_text_verified"] is not True:
        raise ValueError("approved training and evaluation manifests overlap")
    forbidden_hashes = {
        key: {str(row[key]) for row in training_rows}
        | {
            str(record[key])
            for record in evaluation_documents["off"]["generation_records"]
        }
        for key in ("prompt_text_sha256", "normalized_prompt_text_sha256")
    }
    return (
        train_config,
        {
            "audit_source_revision": revision,
            "audit_source_clean": clean,
            "pilot_source_revision": config.expected_source_revision,
            "train_config_file_sha256": _sha256_file(Path(config.train_config)),
            "train_run_config_sha256": run_config_sha256,
            "training_metrics_sha256": _sha256_file(training_metrics_path),
            "training_manifest_sha256": overlap["training_manifest_sha256"],
            "evaluation_manifest_sha256": overlap["evaluation_manifest_sha256"],
            "evaluation_input_sha256": {
                label: _sha256_file(path) for label, path in evaluation_paths.items()
            },
            "prior_three_way_aggregate_sha256": _json_sha256(prior_three_way),
            "source_dataset_sha256": _sha256_file(Path(train_config.data.data_dir)),
            "trained_checkpoint_manifest": trained_checkpoint_manifest,
            "untrained_checkpoint_manifest": untrained_checkpoint_manifest,
            "tokenizer_identity_sha256": tokenizer_hash,
        },
        forbidden_hashes,
    )


def _audit_session(
    model: BGT,
    train_config: TrainConfig,
    tokenizer: TransformersTokenizer,
) -> BGTDecodeSession:
    distill = train_config.distillation
    if distill is None:
        raise ValueError("audit session requires distillation configuration")
    return BGTDecodeSession(
        model,
        process_every=distill.response_process_every,
        select_all=distill.boundary_selection == "all",
        valid_vocab_size=tokenizer.get_vocab_size(),
        record_displacements=True,
    )


def _atomic_write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite matched OPSD audit output: {path}"
            ) from exc
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required audit file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_manifest(path: Path) -> dict[str, Any]:
    files = {}
    for name in ("model.safetensors", "config.json", "train_state.json"):
        candidate = path / name
        files[name] = {
            "sha256": _sha256_file(candidate),
            "size_bytes": candidate.stat().st_size,
        }
    return {"path": str(path), "files": files}


def _repository_state() -> tuple[str | None, bool]:
    repository = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repository,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, False
    return revision, not bool(dirty)


def _validate_live_mixed_dtype(model: Any, *, label: str) -> None:
    def floating_dtypes(module: torch.nn.Module) -> set[torch.dtype]:
        tensors = [*module.parameters(), *module.buffers()]
        return {tensor.dtype for tensor in tensors if tensor.is_floating_point()}

    backbone_dtypes = floating_dtypes(model.backbone)
    processor_dtypes = floating_dtypes(model.cache_processor)
    if backbone_dtypes != {torch.bfloat16}:
        raise ValueError(
            f"{label} checkpoint does not have a pure BF16 backbone: {backbone_dtypes}"
        )
    if processor_dtypes != {torch.float32}:
        raise ValueError(
            f"{label} checkpoint does not have a pure FP32 processor: {processor_dtypes}"
        )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    detached = tensor.detach()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(detached.dtype), "shape": list(detached.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    sequence = detached.shape[1] if detached.ndim >= 2 else 1
    for start in range(0, sequence, 16):
        if detached.ndim >= 2:
            chunk = detached[:, start : start + 16]
        else:
            chunk = detached
        digest.update(chunk.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    return hashlib.sha256(
        json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _json_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    result = sum(float(value) for value in values) / len(values)
    if not math.isfinite(result) or result < 0:
        raise RuntimeError("cache displacement mean must be finite and non-negative")
    return result


_IDENTITY_KEYS = (
    "source_index",
    "prompt_text_sha256",
    "normalized_prompt_text_sha256",
    "dataset_answer_text_sha256",
    "dataset_solution_ids_sha256",
    "student_response_ids_sha256",
    "teacher_reasoning_ids_sha256",
    "raw_teacher_completion_ids_sha256",
    "teacher_trace_termination",
    "raw_teacher_completion_tokens",
    "teacher_reasoning_tokens",
    "response_anchor",
    "student_prompt_ids_sha256",
    "teacher_prompt_ids_sha256",
    "teacher_logits_sha256",
    "event_cache_lengths",
    "answer_tokens",
    "dataset_solution_tokens",
    "no_privilege_dataset_solution_nll",
    "self_reasoning_dataset_solution_nll",
)


def aggregate_matched_opsd_audit(
    untrained_rows: list[dict[str, Any]],
    trained_rows: list[dict[str, Any]],
    *,
    bootstrap_seed: int = 240806,
    bootstrap_samples: int = 10_000,
    expected_examples: int = 32,
) -> dict[str, Any]:
    if len(untrained_rows) != expected_examples or len(trained_rows) != expected_examples:
        raise ValueError(f"matched OPSD audit requires exactly {expected_examples} examples")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    seen: set[int] = set()
    seen_prompt_hashes: dict[str, set[str]] = {
        "prompt_text_sha256": set(),
        "normalized_prompt_text_sha256": set(),
    }
    for index, (untrained, trained) in enumerate(zip(untrained_rows, trained_rows, strict=True)):
        for key in _IDENTITY_KEYS:
            if untrained.get(key) != trained.get(key):
                raise ValueError(f"matched OPSD audit rows differ in {key} at index {index}")
        source_index = _positive_int(untrained, "source_index", allow_zero=True)
        if source_index in seen:
            raise ValueError(f"duplicate source_index in matched OPSD audit: {source_index}")
        seen.add(source_index)
        for key in (
            "prompt_text_sha256",
            "normalized_prompt_text_sha256",
            "dataset_answer_text_sha256",
            "dataset_solution_ids_sha256",
            "student_response_ids_sha256",
            "teacher_reasoning_ids_sha256",
            "raw_teacher_completion_ids_sha256",
            "student_prompt_ids_sha256",
            "teacher_prompt_ids_sha256",
            "teacher_logits_sha256",
        ):
            value = untrained.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"invalid {key} at index {index}")
        for key, values in seen_prompt_hashes.items():
            prompt_hash = str(untrained[key])
            if prompt_hash in values:
                raise ValueError(f"duplicate {key} in matched OPSD audit")
            values.add(prompt_hash)
        if untrained["teacher_trace_termination"] not in (
            "native_close",
            "eot_rewritten",
        ):
            raise ValueError("invalid teacher_trace_termination")
        if untrained["response_anchor"] != "seed_identical_untrained_cp_on":
            raise ValueError("invalid matched OPSD response anchor")
        events = untrained["event_cache_lengths"]
        if (
            not isinstance(events, list)
            or not events
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in events)
            or events != sorted(set(events))
        ):
            raise ValueError("invalid matched OPSD event cache lengths")
        _positive_int(untrained, "raw_teacher_completion_tokens")
        _positive_int(untrained, "teacher_reasoning_tokens")
        _positive_int(untrained, "answer_tokens")
        _positive_int(untrained, "dataset_solution_tokens")
        for row in (untrained, trained):
            _nonnegative_finite(row, "exact_forward_kl")
        _nonnegative_finite(untrained, "no_privilege_dataset_solution_nll")
        _nonnegative_finite(untrained, "self_reasoning_dataset_solution_nll")

    recurrent_coverage_examples = sum(
        len(row["event_cache_lengths"]) >= 2 for row in untrained_rows
    )
    if recurrent_coverage_examples == 0:
        raise ValueError("matched OPSD audit has no recurrent-event coverage")
    trace_termination_counts = {
        mode: sum(row["teacher_trace_termination"] == mode for row in untrained_rows)
        for mode in ("native_close", "eot_rewritten")
    }
    answer_tokens = np.asarray([row["answer_tokens"] for row in untrained_rows], dtype=np.float64)
    dataset_tokens = np.asarray(
        [row["dataset_solution_tokens"] for row in untrained_rows], dtype=np.float64
    )
    untrained_kl = np.asarray(
        [row["exact_forward_kl"] for row in untrained_rows], dtype=np.float64
    )
    trained_kl = np.asarray([row["exact_forward_kl"] for row in trained_rows], dtype=np.float64)
    no_privilege_nll = np.asarray(
        [row["no_privilege_dataset_solution_nll"] for row in untrained_rows],
        dtype=np.float64,
    )
    self_reasoning_nll = np.asarray(
        [row["self_reasoning_dataset_solution_nll"] for row in untrained_rows],
        dtype=np.float64,
    )

    untrained_weighted = _weighted_mean(untrained_kl, answer_tokens)
    trained_weighted = _weighted_mean(trained_kl, answer_tokens)
    if untrained_weighted <= 0:
        raise ValueError("untrained token-weighted KL must be positive")
    no_privilege_weighted = _weighted_mean(no_privilege_nll, dataset_tokens)
    self_reasoning_weighted = _weighted_mean(self_reasoning_nll, dataset_tokens)

    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, expected_examples, size=(bootstrap_samples, expected_examples))
    sampled_answer_tokens = answer_tokens[indices]
    sampled_dataset_tokens = dataset_tokens[indices]
    sampled_untrained = _row_weighted_means(untrained_kl[indices], sampled_answer_tokens)
    sampled_trained = _row_weighted_means(trained_kl[indices], sampled_answer_tokens)
    sampled_no_privilege = _row_weighted_means(
        no_privilege_nll[indices], sampled_dataset_tokens
    )
    sampled_self_reasoning = _row_weighted_means(
        self_reasoning_nll[indices], sampled_dataset_tokens
    )

    matrix_digest = hashlib.sha256()
    matrix_digest.update(
        json.dumps(
            {
                "dtype": str(indices.dtype),
                "shape": list(indices.shape),
                "seed": bootstrap_seed,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    matrix_digest.update(indices.tobytes())

    relative_reduction = 1.0 - trained_weighted / untrained_weighted
    per_example_delta = trained_kl - untrained_kl
    tied = np.isclose(per_example_delta, 0.0, rtol=1e-9, atol=1e-12)
    sampled_relative_reduction = 1.0 - sampled_trained / sampled_untrained
    return {
        "kind": "opsd_matched_heldout_kl_teacher_consistency_audit",
        "schema_version": 1,
        "examples": expected_examples,
        "source_indices": [int(row["source_index"]) for row in untrained_rows],
        "student_response_tokens": int(answer_tokens.sum()),
        "dataset_solution_tokens": int(dataset_tokens.sum()),
        "recurrent_coverage_examples": recurrent_coverage_examples,
        "teacher_trace_termination_counts": trace_termination_counts,
        "exact_forward_kl": {
            "untrained_token_weighted": untrained_weighted,
            "trained_token_weighted": trained_weighted,
            "trained_minus_untrained": trained_weighted - untrained_weighted,
            "mean_paired_per_example_delta": float(np.mean(per_example_delta)),
            "improved_examples": int(np.sum((per_example_delta < 0) & ~tied)),
            "worsened_examples": int(np.sum((per_example_delta > 0) & ~tied)),
            "tied_examples": int(np.sum(tied)),
            "trained_relative_reduction": relative_reduction,
            "trained_minus_untrained_bootstrap_95_ci": _quantile_interval(
                sampled_trained - sampled_untrained
            ),
            "trained_relative_reduction_bootstrap_95_ci": _quantile_interval(
                sampled_relative_reduction
            ),
        },
        "teacher_context_consistency": {
            "target": "unverified_dataset_solution_tokens",
            "no_privilege_dataset_solution_nll": no_privilege_weighted,
            "self_reasoning_dataset_solution_nll": self_reasoning_weighted,
            "self_reasoning_minus_no_privilege_nll": (
                self_reasoning_weighted - no_privilege_weighted
            ),
            "self_reasoning_minus_no_privilege_bootstrap_95_ci": _quantile_interval(
                sampled_self_reasoning - sampled_no_privilege
            ),
        },
        "bootstrap": {
            "seed": bootstrap_seed,
            "samples": bootstrap_samples,
            "unit": "prompt",
            "numpy_version": np.__version__,
            "quantile_method": "linear",
            "index_matrix_sha256": matrix_digest.hexdigest(),
        },
    }


def _positive_int(row: dict[str, Any], key: str, *, allow_zero: bool = False) -> int:
    value = row.get(key)
    lower_bound = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lower_bound:
        raise ValueError(f"{key} must be an integer >= {lower_bound}")
    return value


def _nonnegative_finite(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{key} must be finite and non-negative")
    return number


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _row_weighted_means(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(values * weights, axis=1) / np.sum(weights, axis=1)


def _quantile_interval(values: np.ndarray) -> list[float]:
    return [
        float(np.quantile(values, 0.025, method="linear")),
        float(np.quantile(values, 0.975, method="linear")),
    ]
