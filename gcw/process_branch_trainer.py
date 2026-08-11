from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time
from typing import Any, Iterator

import torch
from safetensors.torch import save_file

from gcw.checkpoint import (
    load_backbone_checkpoint,
    load_checkpoint,
    publish_final_checkpoint_manifest,
    save_checkpoint,
)
from gcw.config import (
    ProcessBranchConfig,
    TrainConfig,
    process_branch_training_contract,
    to_dict,
)
from gcw.data.tokenizers import get_tokenizer
from gcw.data.training import ProcessBranchCursor
from gcw.models import build_model
from gcw.models.longshot import Longshot
from gcw.process_branch import (
    AnswerOnlyTokenExample,
    InvalidProcessBranchExample,
    ProcessBranchTokenExample,
    answer_only_example_loss,
    branch_answer_loss,
    build_process_branch_state,
    process_branch_example_loss,
    tokenize_answer_only_record,
    tokenize_process_record,
)
from gcw.process_data import (
    AnswerOnlyRecord,
    ProcessBranchRecord,
    authenticate_answer_only_artifact,
    read_answer_only_records_bytes,
    read_process_branch_records_bytes,
)
from gcw.state_hash import tensor_state_sha256
from gcw.training_checkpoint import ResumeState, TrainingCheckpoint, save_training_checkpoint
from gcw.trajectory_distillation import (
    authenticate_trajectory_artifact,
    tokenize_trajectory_record,
    trajectory_replay_example_loss,
)


def _sample_cycle_count(
    process: ProcessBranchConfig,
    random_generator: random.Random,
) -> int:
    return random_generator.choice(process.cycle_choices)


@dataclass(frozen=True)
class ProcessCursor:
    epoch: int
    order: tuple[int, ...]
    offset: int
    order_rng_state: tuple[int, tuple[int, ...], float | None]
    depth_rng_state: tuple[int, tuple[int, ...], float | None]

    def __post_init__(self) -> None:
        epoch = _strict_int(self.epoch, "epoch")
        offset = _strict_int(self.offset, "offset")
        if not isinstance(self.order, tuple):
            raise ValueError("process cursor order must be a tuple")
        order = tuple(_strict_int(value, "order item") for value in self.order)
        if not order:
            if epoch != -1 or offset != 0:
                raise ValueError("empty process cursor must be at the initial position")
        elif epoch < 0 or sorted(order) != list(range(len(order))):
            raise ValueError("populated process cursor must contain a full permutation")
        if order and not 0 <= offset <= len(order):
            raise ValueError("process cursor offset is outside its order")
        for name, state in (
            ("order", self.order_rng_state),
            ("depth", self.depth_rng_state),
        ):
            if (
                not isinstance(state, tuple)
                or len(state) != 3
                or (
                    state[2] is not None
                    and (
                        isinstance(state[2], bool)
                        or not isinstance(state[2], int | float)
                        or not math.isfinite(state[2])
                    )
                )
            ):
                raise ValueError(f"process cursor {name} RNG state is invalid")
            try:
                random.Random().setstate(state)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"process cursor {name} RNG state is invalid") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "order": list(self.order),
            "offset": self.offset,
            "order_rng_state": _random_state_to_dict(self.order_rng_state),
            "depth_rng_state": _random_state_to_dict(self.depth_rng_state),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ProcessCursor:
        expected = {
            "epoch",
            "order",
            "offset",
            "order_rng_state",
            "depth_rng_state",
        }
        if set(payload) != expected:
            raise ValueError("process cursor fields differ from the required schema")
        epoch = _strict_int(payload["epoch"], "epoch")
        offset = _strict_int(payload["offset"], "offset")
        order_payload = payload["order"]
        if not isinstance(order_payload, list):
            raise ValueError("process cursor order must be a list")
        order = tuple(_strict_int(value, "order item") for value in order_payload)
        if epoch < -1 or offset < 0 or offset > len(order):
            raise ValueError("process cursor position is invalid")
        if len(set(order)) != len(order) or any(index < 0 for index in order):
            raise ValueError("process cursor order is not a non-negative permutation")
        return cls(
            epoch=epoch,
            order=order,
            offset=offset,
            order_rng_state=_random_state_from_dict(payload["order_rng_state"]),
            depth_rng_state=_random_state_from_dict(payload["depth_rng_state"]),
        )


def _strict_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"process cursor {field} must be an integer")
    return value


def _random_state_to_dict(
    state: tuple[int, tuple[int, ...], float | None],
) -> dict[str, object]:
    version, internal_state, gaussian = state
    return {
        "version": version,
        "state": list(internal_state),
        "gauss_next": gaussian,
    }


def _random_state_from_dict(
    payload: object,
) -> tuple[int, tuple[int, ...], float | None]:
    if not isinstance(payload, Mapping) or set(payload) != {"version", "state", "gauss_next"}:
        raise ValueError("process cursor RNG state fields differ from the required schema")
    version = _strict_int(payload["version"], "RNG version")
    state_payload = payload["state"]
    if not isinstance(state_payload, list):
        raise ValueError("process cursor RNG state must be a list")
    internal_state = tuple(_strict_int(value, "RNG state item") for value in state_payload)
    gaussian = payload["gauss_next"]
    if gaussian is not None and (isinstance(gaussian, bool) or not isinstance(gaussian, (int, float))):
        raise ValueError("process cursor RNG gaussian cache must be numeric or null")
    return version, internal_state, None if gaussian is None else float(gaussian)


class ProcessSchedule:
    def __init__(
        self,
        *,
        record_count: int,
        cycle_choices: tuple[int, ...],
        seed: int,
        cursor: ProcessCursor | None = None,
    ) -> None:
        if record_count <= 0:
            raise ValueError("process schedule requires at least one record")
        if not cycle_choices:
            raise ValueError("process schedule requires at least one cycle choice")
        self._record_count = record_count
        self._cycle_choices = cycle_choices
        self._order_generator = random.Random(seed)
        self._depth_generator = random.Random(seed)
        self._order: list[int] = []
        self._offset = 0
        self._epoch = -1
        if cursor is not None:
            if cursor.order and sorted(cursor.order) != list(range(record_count)):
                raise ValueError("process cursor order does not match the record population")
            self._order_generator.setstate(cursor.order_rng_state)
            self._depth_generator.setstate(cursor.depth_rng_state)
            self._order = list(cursor.order)
            self._offset = cursor.offset
            self._epoch = cursor.epoch

    @property
    def cursor(self) -> ProcessCursor:
        return ProcessCursor(
            epoch=self._epoch,
            order=tuple(self._order),
            offset=self._offset,
            order_rng_state=self._order_generator.getstate(),
            depth_rng_state=self._depth_generator.getstate(),
        )

    def next_example(self) -> tuple[int, int]:
        if self._offset >= len(self._order):
            self._order = list(range(self._record_count))
            self._order_generator.shuffle(self._order)
            self._offset = 0
            self._epoch += 1
        record_index = self._order[self._offset]
        self._offset += 1
        cycle_count = self._depth_generator.choice(self._cycle_choices)
        return record_index, cycle_count


TrainingRecord = AnswerOnlyRecord | ProcessBranchRecord
TrainingExample = AnswerOnlyTokenExample | ProcessBranchTokenExample


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selected_record_payload(record: TrainingRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_id": record.source_id,
        "prompt": record.prompt,
        "final_answer": record.final_answer,
    }
    if isinstance(record, ProcessBranchRecord):
        payload["reasoning_steps"] = list(record.reasoning_steps)
    return payload


def _split_data_identity(
    *,
    skip_rows: int,
    take_rows: int | None,
    source_records: Sequence[TrainingRecord],
    eligible_records: Sequence[TrainingRecord],
    rejection_reasons: Counter[str],
) -> dict[str, Any]:
    ordered_source_ids = [record.source_id for record in eligible_records]
    selected_records = [_selected_record_payload(record) for record in eligible_records]
    return {
        "bounds": {"skip_rows": skip_rows, "take_rows": take_rows},
        "source_records": len(source_records),
        "eligible_records": len(eligible_records),
        "rejected_records": len(source_records) - len(eligible_records),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "ordered_source_ids": ordered_source_ids,
        "ordered_source_ids_sha256": _canonical_json_sha256(ordered_source_ids),
        "selected_records_sha256": _canonical_json_sha256(selected_records),
    }


def _build_training_data_identity(
    config: TrainConfig,
    source_bytes: bytes,
    records: Sequence[TrainingRecord],
    valid_records: Sequence[TrainingRecord],
    heldout_records: Sequence[TrainingRecord],
    valid_heldout: Sequence[TrainingRecord],
    rejection_reasons: Counter[str],
    heldout_rejection_reasons: Counter[str],
) -> dict[str, Any]:
    process = config.process_branch
    if process is None:
        raise ValueError("process_branch configuration is missing")
    return {
        "kind": "gcw.process_branch.training_data_identity.v1",
        "objective": process.objective,
        "source": {
            "size_bytes": len(source_bytes),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "splits": {
            "train": _split_data_identity(
                skip_rows=config.data.sft_skip_rows,
                take_rows=config.data.sft_take_rows,
                source_records=records,
                eligible_records=valid_records,
                rejection_reasons=rejection_reasons,
            ),
            "heldout": _split_data_identity(
                skip_rows=process.heldout_skip_rows,
                take_rows=process.heldout_take_rows,
                source_records=heldout_records,
                eligible_records=valid_heldout,
                rejection_reasons=heldout_rejection_reasons,
            ),
        },
    }


def _write_training_data_audit(
    output_dir: Path,
    source_path: Path,
    data_identity: dict[str, Any],
) -> str:
    payload = {
        "kind": "gcw.process_branch.training_data_audit.v3",
        "data_identity": data_identity,
        "source": {
            "basename": source_path.name,
            "role": "informational",
        },
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    audit_path = output_dir / "training-data-audit.json"
    with audit_path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(encoded).hexdigest()




def _path_tree_sha256(path: Path) -> str:
    if not path.is_dir():
        raise ValueError(f"tokenizer artifact directory is not visible: {path}")
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"tokenizer artifact directory is empty: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = candidate.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _authenticated_tokenizer_snapshot(
    source: Path,
) -> Iterator[tuple[Path, str]]:
    if not source.is_dir():
        raise ValueError(f"tokenizer artifact directory is not visible: {source}")
    captured: list[tuple[str, bytes]] = []
    for candidate in sorted(path for path in source.rglob("*") if path.is_file()):
        captured.append((candidate.relative_to(source).as_posix(), candidate.read_bytes()))
    if not captured:
        raise ValueError(f"tokenizer artifact directory is empty: {source}")
    digest = hashlib.sha256()
    for relative_text, payload in captured:
        relative = relative_text.encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    with tempfile.TemporaryDirectory(prefix="gcw-train-tokenizer-") as temporary:
        snapshot = Path(temporary)
        for relative_text, payload in captured:
            destination = snapshot / relative_text
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        yield snapshot, digest.hexdigest()


def _local_hf_tokenizer_path(config: TrainConfig) -> Path:
    tokenizer_name = config.data.tokenizer
    if tokenizer_name == "hf" and config.data.tokenizer_source:
        return Path(config.data.tokenizer_source)
    if tokenizer_name.startswith("hf:") and tokenizer_name.removeprefix("hf:"):
        return Path(tokenizer_name.removeprefix("hf:"))
    raise ValueError("canonical process_branch training requires a local HF tokenizer artifact")


def _validate_frozen_training_identities(
    config: TrainConfig,
    *,
    data_identity: dict[str, Any],
    initial_backbone_sha256: str,
    observed_tokenizer_sha256: str | None = None,
) -> dict[str, Any] | None:
    process = config.process_branch
    if process is None or not process.require_frozen_identities:
        return None
    expected = {
        "train_records": process.expected_train_eligible_records,
        "train_source_ids_sha256": process.expected_train_eligible_source_ids_sha256,
        "heldout_records": process.expected_heldout_eligible_records,
        "heldout_source_ids_sha256": process.expected_heldout_eligible_source_ids_sha256,
        "initial_backbone_sha256": process.expected_initial_backbone_sha256,
        "tokenizer_sha256": process.expected_tokenizer_sha256,
    }
    missing = sorted(name for name, value in expected.items() if value is None)
    if missing:
        raise ValueError(
            "process_branch frozen training identities are incomplete: " + ", ".join(missing)
        )
    train_identity = data_identity["splits"]["train"]
    heldout_identity = data_identity["splits"]["heldout"]
    observed = {
        "train_records": train_identity["eligible_records"],
        "train_source_ids_sha256": train_identity["ordered_source_ids_sha256"],
        "heldout_records": heldout_identity["eligible_records"],
        "heldout_source_ids_sha256": heldout_identity["ordered_source_ids_sha256"],
        "initial_backbone_sha256": initial_backbone_sha256,
    }
    tokenizer_name = config.data.tokenizer
    if tokenizer_name == "hf":
        tokenizer_path = config.data.tokenizer_source
    elif tokenizer_name.startswith("hf:"):
        tokenizer_path = tokenizer_name.removeprefix("hf:")
    else:
        tokenizer_path = None
    if not tokenizer_path:
        raise ValueError("frozen process_branch identities require a local HF tokenizer artifact")
    observed["tokenizer_sha256"] = (
        observed_tokenizer_sha256
        if observed_tokenizer_sha256 is not None
        else _path_tree_sha256(Path(tokenizer_path))
    )
    mismatched = sorted(name for name, value in observed.items() if value != expected[name])
    if mismatched:
        raise ValueError(
            "process_branch frozen training identity mismatch: " + ", ".join(mismatched)
        )
    return observed


def _tokenize_training_record(
    record: TrainingRecord,
    tokenizer,
    config: TrainConfig,
    *,
    cycles: int,
    device: torch.device | str = "cpu",
) -> TrainingExample:
    process = config.process_branch
    if process is None:
        raise ValueError("process_branch configuration is missing")
    if process.objective == "answer_only":
        if not isinstance(record, AnswerOnlyRecord):
            raise TypeError("answer_only objective requires AnswerOnlyRecord")
        return tokenize_answer_only_record(
            record,
            tokenizer,
            answer_response_prefix=config.data.response_prefix,
            prompt_suffix=config.data.prompt_suffix,
            max_prompt_tokens=process.max_prompt_tokens,
            max_answer_tokens=process.max_answer_tokens,
            device=device,
        )
    if process.objective == "trajectory_replay":
        if not isinstance(record, ProcessBranchRecord):
            raise TypeError("trajectory_replay objective requires ProcessBranchRecord")
        return tokenize_trajectory_record(
            record,
            tokenizer,
            prompt_suffix=config.data.prompt_suffix,
            rationale_joiner=process.rationale_joiner,
            reasoning_close_text=process.reasoning_close_text,
            max_prompt_tokens=process.max_prompt_tokens,
            max_trajectory_tokens=process.max_step_tokens,
            max_answer_tokens=process.max_answer_tokens,
            device=device,
        )
    if not isinstance(record, ProcessBranchRecord):
        raise TypeError("process_branch objective requires ProcessBranchRecord")
    return tokenize_process_record(
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




def _source_code_identity(*, require_git: bool = False) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    source_files = sorted((repository / "gcw").rglob("*.py"))
    source_files.extend(
        candidate
        for candidate in (
            repository / "scripts" / "train.py",
            repository / "scripts" / "prepare_answer_only_gsm8k.py",
            repository / "scripts" / "prepare_gsm8k_trajectory_replay.py",
            repository / "pyproject.toml",
            repository / "uv.lock",
        )
        if candidate.is_file()
    )
    digest = hashlib.sha256()
    for candidate in sorted(set(source_files)):
        relative = candidate.relative_to(repository).as_posix().encode("utf-8")
        payload = candidate.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    git = shutil.which("git")
    if git is None and Path("/run/current-system/sw/bin/git").is_file():
        git = "/run/current-system/sw/bin/git"
    commit: str | None = None
    status_bytes: bytes | None = None
    if git is not None:
        try:
            commit = subprocess.check_output(
                [git, "rev-parse", "HEAD"], cwd=repository, stderr=subprocess.DEVNULL
            ).decode("ascii").strip()
            status_bytes = subprocess.check_output(
                [
                    git,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--",
                    "gcw",
                    "scripts/train.py",
                    "scripts/prepare_answer_only_gsm8k.py",
                    "pyproject.toml",
                    "uv.lock",
                ],
                cwd=repository,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as error:
            if require_git:
                raise RuntimeError("canonical process_branch training requires Git provenance") from error
            commit = None
            status_bytes = None
    elif require_git:
        raise RuntimeError("canonical process_branch training requires Git provenance")
    if require_git and (commit is None or status_bytes is None):
        raise RuntimeError("canonical process_branch training requires Git provenance")
    return {
        "schema": "gcw.source_code_identity.v1",
        "source_tree_sha256": digest.hexdigest(),
        "git_commit": commit,
        "git_dirty": None if status_bytes is None else bool(status_bytes),
        "git_status_sha256": (
            None if status_bytes is None else hashlib.sha256(status_bytes).hexdigest()
        ),
    }


def _process_training_fingerprint(
    config: TrainConfig,
    *,
    data_identity: dict[str, Any],
    initial_processor_sha256: str,
    observed_tokenizer_sha256: str | None = None,
) -> dict[str, Any]:
    process = config.process_branch
    if process is None:
        raise ValueError("process_branch configuration is missing")
    if observed_tokenizer_sha256 is None:
        observed_tokenizer_sha256 = hashlib.sha256(
            config.data.tokenizer.encode("utf-8")
        ).hexdigest()
    return {
        "kind": "gcw.process_branch.training_fingerprint.v2",
        "code_identity": _source_code_identity(require_git=config.data.canonical_execution),
        "distributed": config.distributed,
        "bf16": config.bf16,
        "objective": {
            "kind": process.objective,
            "process_weight": process.process_weight,
            "answer_weight": process.answer_weight,
        },
        "cycle_choices": list(process.cycle_choices),
        "data_identity": data_identity,
        "seed": config.seed,
        "model": to_dict(config.model),
        "initial_processor_sha256": initial_processor_sha256,
        "observed_tokenizer_sha256": observed_tokenizer_sha256,
        "optimizer": {
            "kind": "AdamW",
            "learning_rate": config.learning_rate,
            "betas": [config.adam_beta1, config.adam_beta2],
            "eps": config.adam_eps,
            "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip,
            "gradient_accumulation_steps": process.gradient_accumulation_steps,
            "max_optimizer_steps": process.max_optimizer_steps,
        },
        "frozen_identities": {
            "required": process.require_frozen_identities,
            "expected_train_eligible_records": process.expected_train_eligible_records,
            "expected_train_eligible_source_ids_sha256": (
                process.expected_train_eligible_source_ids_sha256
            ),
            "expected_heldout_eligible_records": process.expected_heldout_eligible_records,
            "expected_heldout_eligible_source_ids_sha256": (
                process.expected_heldout_eligible_source_ids_sha256
            ),
            "expected_initial_backbone_sha256": process.expected_initial_backbone_sha256,
            "expected_tokenizer_sha256": process.expected_tokenizer_sha256,
        },
        "tokenization": {
            "data_kind": config.data.kind,
            "tokenizer_role": "authenticated_local_hf",
            "prompt_suffix": config.data.prompt_suffix,
            "answer_response_prefix": config.data.response_prefix,
            "step_response_prefix": process.step_response_prefix,
            "rationale_joiner": process.rationale_joiner,
            "reasoning_close_text": process.reasoning_close_text,
            "max_prompt_tokens": process.max_prompt_tokens,
            "max_step_tokens": process.max_step_tokens,
            "max_answer_tokens": process.max_answer_tokens,
        },
    }


def _validate_resume_schedule_consistency(
    *,
    step: int,
    seen_examples: int,
    cursor: ProcessCursor,
    record_count: int,
    cycle_choices: tuple[int, ...],
    seed: int,
    gradient_accumulation_steps: int,
    max_optimizer_steps: int,
) -> None:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("process_branch resume step must be a non-negative integer")
    if step > max_optimizer_steps:
        raise ValueError("process_branch resume step exceeds max_optimizer_steps")
    expected_seen_examples = step * gradient_accumulation_steps
    if seen_examples != expected_seen_examples:
        raise ValueError(
            "process_branch resume seen_examples does not match optimizer step "
            "and gradient accumulation"
        )
    replay = ProcessSchedule(
        record_count=record_count,
        cycle_choices=cycle_choices,
        seed=seed,
    )
    for _ in range(seen_examples):
        replay.next_example()
    if replay.cursor != cursor:
        raise ValueError(
            "process_branch resume cursor/RNG state differs from deterministic replay"
        )


def _train_process_branch_impl(
    config: TrainConfig,
    *,
    tokenizer_name_override: str | None = None,
    observed_tokenizer_sha256: str | None = None,
) -> None:
    process = config.process_branch
    if config.mode != "process_branch" or process is None:
        raise ValueError("train_process_branch requires process_branch mode and configuration")
    if config.data.data_dir is None:
        raise ValueError("process_branch requires data.data_dir")
    source_path = Path(config.data.data_dir)
    source_bytes = source_path.read_bytes()
    if config.data.canonical_execution:
        artifact_sha256 = config.data.expected_artifact_sha256
        source_ids_sha256 = config.data.expected_source_ids_sha256
        if artifact_sha256 is None or source_ids_sha256 is None:
            raise ValueError("canonical process-data hashes are missing")
        if process.objective == "answer_only":
            source_records: list[TrainingRecord] = list(
                authenticate_answer_only_artifact(
                    source_bytes,
                    expected_artifact_sha256=artifact_sha256,
                    expected_source_ids_sha256=source_ids_sha256,
                )
            )
        elif process.objective == "trajectory_replay":
            source_records = list(
                authenticate_trajectory_artifact(
                    source_bytes,
                    expected_artifact_sha256=artifact_sha256,
                    expected_source_ids_sha256=source_ids_sha256,
                )
            )
        else:
            raise ValueError(
                "canonical process data authentication supports answer_only and "
                "trajectory_replay only"
            )
    elif process.objective == "answer_only":
        source_records = list(read_answer_only_records_bytes(source_bytes))
    else:
        source_records = list(read_process_branch_records_bytes(source_bytes))

    train_end = (
        None
        if config.data.sft_take_rows is None
        else config.data.sft_skip_rows + config.data.sft_take_rows
    )
    records = source_records[config.data.sft_skip_rows:train_end]
    if not records:
        raise ValueError("process_branch training slice is empty")
    heldout_end = (
        process.heldout_skip_rows + process.heldout_take_rows
        if process.heldout_take_rows
        else process.heldout_skip_rows
    )
    heldout_records = source_records[process.heldout_skip_rows:heldout_end]
    overlap = {record.source_id for record in records} & {
        record.source_id for record in heldout_records
    }
    if overlap:
        raise ValueError(f"process_branch train/held-out overlap: {sorted(overlap)[:4]}")

    output_dir = Path(config.output_dir)

    tokenizer_name = tokenizer_name_override or config.data.tokenizer
    if tokenizer_name == "hf":
        if not config.data.tokenizer_source:
            raise ValueError("process_branch tokenizer='hf' requires data.tokenizer_source")
        tokenizer_name = f"hf:{config.data.tokenizer_source}"
    tokenizer = get_tokenizer(tokenizer_name)
    if tokenizer.get_vocab_size() > config.model.vocab_size:
        raise ValueError("process_branch tokenizer vocabulary exceeds model vocabulary")
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model)
    if not isinstance(model, Longshot):
        raise TypeError("process_branch training requires a Longshot model")
    model.init_weights()
    if config.resume is None:
        load_backbone_checkpoint(model, config.backbone_checkpoint or "")
    model.backbone.requires_grad_(False)
    model.backbone.eval()
    model.cache_processor.train()
    model.to(device)
    if config.bf16 and device.type == "cuda":
        model.backbone.to(dtype=torch.bfloat16)
    initial_backbone_sha256 = (
        tensor_state_sha256(model.backbone.state_dict())
        if config.resume is None
        else None
    )

    valid_records, rejection_reasons = _valid_records(records, tokenizer, config)
    if not valid_records:
        raise ValueError("process_branch has no valid token-bounded records")
    valid_heldout, heldout_rejection_reasons = _valid_records(heldout_records, tokenizer, config)
    if heldout_records and len(valid_heldout) < 2:
        raise ValueError("process_branch requires at least two valid held-out records")
    data_identity = _build_training_data_identity(
        config,
        source_bytes,
        records,
        valid_records,
        heldout_records,
        valid_heldout,
        rejection_reasons,
        heldout_rejection_reasons,
    )
    parameters = [parameter for parameter in model.cache_processor.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    initial_processor_state = {
        key: value.detach().cpu().contiguous().clone()
        for key, value in model.cache_processor.state_dict().items()
    }
    initial_processor_sha256 = tensor_state_sha256(initial_processor_state)
    training_fingerprint = _process_training_fingerprint(
        config,
        data_identity=data_identity,
        initial_processor_sha256=initial_processor_sha256,
        observed_tokenizer_sha256=(
            observed_tokenizer_sha256
            if observed_tokenizer_sha256 is not None
            else hashlib.sha256(config.data.tokenizer.encode("utf-8")).hexdigest()
        ),
    )
    started = time.monotonic()
    optimizer_steps = 0
    accumulated = 0
    seen_examples = 0
    last_loss = 0.0
    metric_sums = {
        "total": 0.0,
        "process": 0.0,
        "answer": 0.0,
        "direct_answer": 0.0,
        "thinking_answer": 0.0,
        "cycles": 0.0,
    }
    metric_source_ids: list[str] = []
    metric_cycle_counts: list[int] = []
    schedule_cursor: ProcessCursor | None = None
    checkpoint: TrainingCheckpoint | None = None
    if config.resume is not None:
        checkpoint = TrainingCheckpoint.open(
            config.resume.checkpoint,
            expected_fingerprint=training_fingerprint,
            topology=config.resume.topology,
            current_world_size=1,
            expected_manifest_sha256=config.resume.expected_manifest_sha256,
        )
        loader_cursor = checkpoint.loader_cursor(rank=0)
        if not isinstance(loader_cursor, ProcessBranchCursor):
            raise TypeError("process_branch resume requires an authenticated process cursor")
        schedule_cursor = ProcessCursor.from_dict(loader_cursor.process_cursor)
        saved_initial_hash = checkpoint.resume_state.metadata.get("initial_processor_sha256")
        if saved_initial_hash != initial_processor_sha256:
            raise ValueError("process_branch resume changed the immutable initial processor hash")
        initial_processor_state = checkpoint.load_initial_processor_state(device="cpu")
        if tensor_state_sha256(initial_processor_state) != saved_initial_hash:
            raise ValueError("process_branch resume initial processor artifact hash mismatch")
        saved_backbone_hash = checkpoint.resume_state.metadata.get("initial_backbone_sha256")
        if not isinstance(saved_backbone_hash, str) or len(saved_backbone_hash) != 64:
            raise ValueError("process_branch resume initial backbone hash is invalid")
        saved_seen_examples = checkpoint.resume_state.metadata.get("seen_examples")
        if isinstance(saved_seen_examples, bool) or not isinstance(saved_seen_examples, int):
            raise ValueError("process_branch resume seen_examples must be an integer")
        if saved_seen_examples < 0:
            raise ValueError("process_branch resume seen_examples must be non-negative")
        optimizer_steps = checkpoint.resume_state.step
        seen_examples = saved_seen_examples
        _validate_resume_schedule_consistency(
            step=optimizer_steps,
            seen_examples=seen_examples,
            cursor=schedule_cursor,
            record_count=len(valid_records),
            cycle_choices=process.cycle_choices,
            seed=config.seed,
            gradient_accumulation_steps=process.gradient_accumulation_steps,
            max_optimizer_steps=process.max_optimizer_steps,
        )
        last_loss = checkpoint.resume_state.smooth_loss
        checkpoint.restore_model(model)
        restored_backbone_sha256 = tensor_state_sha256(model.backbone.state_dict())
        if restored_backbone_sha256 != saved_backbone_hash:
            raise ValueError("process_branch resume changed the immutable backbone tensors")
        initial_backbone_sha256 = saved_backbone_hash
        checkpoint.restore_optimizer(optimizer, device=device)
        checkpoint.restore_rng(rank=0)
    if initial_backbone_sha256 is None:
        raise RuntimeError("process_branch initial backbone hash was not established")
    frozen_identities = _validate_frozen_training_identities(
        config,
        data_identity=data_identity,
        initial_backbone_sha256=initial_backbone_sha256,
        observed_tokenizer_sha256=observed_tokenizer_sha256,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "metrics.jsonl"
    final_checkpoint = output_dir / "final"
    run_config_bytes = (
        json.dumps(process_branch_training_contract(config), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with (output_dir / "run_config.json").open("xb") as handle:
        handle.write(run_config_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    training_data_audit_sha256 = _write_training_data_audit(
        output_dir,
        source_path,
        data_identity,
    )
    schedule = ProcessSchedule(
        record_count=len(valid_records),
        cycle_choices=process.cycle_choices,
        seed=config.seed,
        cursor=schedule_cursor,
    )
    optimizer.zero_grad(set_to_none=True)
    training_checkpoint_path = output_dir / "training-checkpoint"
    last_saved_step = -1

    def save_exact_resume_checkpoint() -> None:
        nonlocal last_saved_step
        if accumulated != 0:
            raise RuntimeError("process_branch checkpoints are allowed only at optimizer boundaries")
        save_training_checkpoint(
            training_checkpoint_path,
            model,
            config.model,
            optimizer,
            ResumeState(
                step=optimizer_steps,
                smooth_loss=last_loss,
                loader_cursor=ProcessBranchCursor(process_cursor=schedule.cursor.to_dict()),
                saved_world_size=1,
                exact_topology=True,
                metadata={
                    "kind": "gcw.process_branch.resume_state.v1",
                    "seen_examples": seen_examples,
                    "initial_processor_sha256": initial_processor_sha256,
                    "initial_backbone_sha256": initial_backbone_sha256,
                    "data_identity": data_identity,
                },
            ),
            training_fingerprint,
            initial_processor_state=initial_processor_state,
        )
        last_saved_step = optimizer_steps

    with metrics_path.open("x", encoding="utf-8") as metrics_handle:
        while optimizer_steps < process.max_optimizer_steps:
            for _ in range(len(valid_records)):
                record_index, cycle_count = schedule.next_example()
                example = _tokenize_training_record(
                    valid_records[record_index],
                    tokenizer,
                    config,
                    cycles=cycle_count,
                    device=device,
                )
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if config.bf16 and device.type == "cuda"
                    else nullcontext()
                )
                direct_answer_metric: torch.Tensor | None = None
                thinking_answer_metric: torch.Tensor | None = None
                with context:
                    if process.objective == "answer_only":
                        if not isinstance(example, AnswerOnlyTokenExample):
                            raise TypeError("answer_only objective produced the wrong token example")
                        answer_only_loss = answer_only_example_loss(
                            model,
                            example,
                            cycles=cycle_count,
                        )
                        answer_loss = answer_only_loss.answer_loss
                        total_loss = process.answer_weight * answer_loss
                        process_loss = None
                        realised_cycles = answer_only_loss.cycles
                    elif process.objective == "trajectory_replay":
                        if not isinstance(example, ProcessBranchTokenExample):
                            raise TypeError(
                                "trajectory_replay objective produced the wrong token example"
                            )
                        replay_losses = trajectory_replay_example_loss(
                            model,
                            example,
                            cycles=cycle_count,
                            rationale_weight=process.process_weight,
                            answer_weight=process.answer_weight,
                        )
                        answer_loss = replay_losses.answer_loss
                        direct_answer_metric = replay_losses.direct_answer_loss
                        thinking_answer_metric = replay_losses.thinking_answer_loss
                        total_loss = replay_losses.total_loss
                        process_loss = replay_losses.rationale_loss
                        realised_cycles = replay_losses.cycles
                    else:
                        if not isinstance(example, ProcessBranchTokenExample):
                            raise TypeError("process_branch objective produced the wrong token example")
                        branch_losses = process_branch_example_loss(
                            model,
                            example,
                            process_weight=process.process_weight,
                            answer_weight=process.answer_weight,
                        )
                        answer_loss = branch_losses.answer_loss
                        total_loss = branch_losses.total_loss
                        process_loss = branch_losses.process_loss
                        realised_cycles = branch_losses.cycles
                if not bool(torch.isfinite(total_loss)):
                    raise FloatingPointError("process_branch loss is non-finite")
                (total_loss / process.gradient_accumulation_steps).backward()
                accumulated += 1
                seen_examples += 1
                metric_source_ids.append(valid_records[record_index].source_id)
                metric_cycle_counts.append(cycle_count)
                metric_sums["total"] += float(total_loss.detach().float())
                if process_loss is not None:
                    metric_sums["process"] += float(process_loss.detach().float())
                metric_sums["answer"] += float(answer_loss.detach().float())
                if direct_answer_metric is not None and thinking_answer_metric is not None:
                    metric_sums["direct_answer"] += float(
                        direct_answer_metric.detach().float()
                    )
                    metric_sums["thinking_answer"] += float(
                        thinking_answer_metric.detach().float()
                    )
                metric_sums["cycles"] += realised_cycles
                if accumulated < process.gradient_accumulation_steps:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                step_loss = metric_sums["total"] / accumulated
                payload = {
                    "train/step": optimizer_steps,
                    "train/loss": step_loss,
                    "train/answer_loss": metric_sums["answer"] / accumulated,
                    "train/cycles": metric_sums["cycles"] / accumulated,
                    "train/source_ids": metric_source_ids,
                    "train/cycle_counts": metric_cycle_counts,
                    "train/grad_norm": float(grad_norm.detach().float()),
                    "train/examples": seen_examples,
                    "train/elapsed_seconds": time.monotonic() - started,
                }
                if process.objective != "answer_only":
                    payload["train/process_loss"] = metric_sums["process"] / accumulated
                if process.objective == "trajectory_replay":
                    payload["train/direct_answer_loss"] = (
                        metric_sums["direct_answer"] / accumulated
                    )
                    payload["train/thinking_answer_loss"] = (
                        metric_sums["thinking_answer"] / accumulated
                    )
                metrics_handle.write(json.dumps(payload, sort_keys=True) + "\n")
                metrics_handle.flush()
                last_loss = step_loss
                accumulated = 0
                metric_sums = {
                    "total": 0.0,
                    "process": 0.0,
                    "answer": 0.0,
                    "direct_answer": 0.0,
                    "thinking_answer": 0.0,
                    "cycles": 0.0,
                }
                metric_source_ids = []
                metric_cycle_counts = []
                if config.save_every > 0 and optimizer_steps % config.save_every == 0:
                    save_exact_resume_checkpoint()
                if optimizer_steps >= process.max_optimizer_steps:
                    break

    if last_saved_step != optimizer_steps:
        save_exact_resume_checkpoint()

    final_processor_sha256 = tensor_state_sha256(model.cache_processor.state_dict())
    heldout_controls = (
        _evaluate_heldout_controls(model, valid_heldout, tokenizer, config, device)
        if valid_heldout
        else None
    )
    final_backbone_sha256 = tensor_state_sha256(model.backbone.state_dict())
    if final_backbone_sha256 != initial_backbone_sha256:
        raise RuntimeError("process_branch training changed frozen backbone tensors")
    train_state: dict[str, Any] = {
        "kind": "gcw.process_branch.train.v1",
        "code_identity": training_fingerprint["code_identity"],
        "training_fingerprint": training_fingerprint,
        "objective": process.objective,
        "optimizer_steps": optimizer_steps,
        "seen_examples": seen_examples,
        "process_cursor": schedule.cursor.to_dict(),
        "source_records": len(records),
        "valid_records": len(valid_records),
        "rejected_records": len(records) - len(valid_records),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "heldout_source_records": len(heldout_records),
        "heldout_valid_records": len(valid_heldout),
        "heldout_rejection_reasons": dict(sorted(heldout_rejection_reasons.items())),
        "data_identity": data_identity,
        "frozen_identities": frozen_identities,
        "training_data_audit": {
            "path": "training-data-audit.json",
            "sha256": training_data_audit_sha256,
        },
        "initial_processor_sha256": initial_processor_sha256,
        "final_processor_sha256": final_processor_sha256,
        "initial_backbone_sha256": initial_backbone_sha256,
        "final_backbone_sha256": final_backbone_sha256,
        "dtype": (
            "mixed_bfloat16_backbone_float32_processor"
            if config.bf16 and device.type == "cuda"
            else "float32"
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    if heldout_controls is not None:
        train_state["heldout_controls"] = heldout_controls
    train_state["checkpoint_reload_verified"] = True
    save_checkpoint(final_checkpoint, model, config.model, train_state)
    save_file(initial_processor_state, final_checkpoint / "initial_processor.safetensors")
    with (final_checkpoint / "run_config.json").open("xb") as handle:
        handle.write(run_config_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    reloaded, reloaded_config, reloaded_train_state = load_checkpoint(
        final_checkpoint, device="cpu"
    )
    if not isinstance(reloaded, Longshot) or reloaded_config != config.model:
        raise RuntimeError("process_branch checkpoint reload changed the model contract")
    reloaded_hash = tensor_state_sha256(reloaded.cache_processor.state_dict())
    if reloaded_hash != final_processor_sha256:
        raise RuntimeError("process_branch checkpoint reload changed processor tensors")
    reloaded_backbone_hash = tensor_state_sha256(reloaded.backbone.state_dict())
    if reloaded_backbone_hash != final_backbone_sha256:
        raise RuntimeError("process_branch checkpoint reload changed backbone tensors")
    if reloaded_train_state.get("checkpoint_reload_verified") is not True:
        raise RuntimeError("process_branch checkpoint omitted reload verification intent")
    train_state["final_checkpoint_manifest_sha256"] = publish_final_checkpoint_manifest(
        final_checkpoint,
        final_processor_sha256,
        initial_processor_sha256,
    )
    with (output_dir / "summary.json").open("x", encoding="utf-8") as summary_handle:
        json.dump(train_state, summary_handle, indent=2, sort_keys=True)
        summary_handle.write("\n")
    print(json.dumps(train_state, sort_keys=True))


def train_process_branch(config: TrainConfig) -> None:
    tokenizer_source: Path | None
    try:
        tokenizer_source = _local_hf_tokenizer_path(config)
    except ValueError:
        if config.data.canonical_execution:
            raise
        tokenizer_source = None
    if tokenizer_source is not None and not tokenizer_source.is_dir():
        if config.data.canonical_execution or config.resume is not None:
            raise ValueError(f"tokenizer artifact directory is not visible: {tokenizer_source}")
        tokenizer_source = None
    if tokenizer_source is not None:
        with _authenticated_tokenizer_snapshot(tokenizer_source) as (
            tokenizer_snapshot,
            tokenizer_sha256,
        ):
            _train_process_branch_impl(
                config,
                tokenizer_name_override=f"hf:{tokenizer_snapshot}",
                observed_tokenizer_sha256=tokenizer_sha256,
            )
        return
    tokenizer_role_sha256 = hashlib.sha256(config.data.tokenizer.encode("utf-8")).hexdigest()
    _train_process_branch_impl(
        config,
        observed_tokenizer_sha256=tokenizer_role_sha256,
    )


@torch.no_grad()
def _evaluate_heldout_controls(
    model: Longshot,
    records: Sequence[TrainingRecord],
    tokenizer,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float | int]:
    process = config.process_branch
    if process is None:
        raise ValueError("process_branch configuration is missing")
    cycles = max(process.cycle_choices)
    examples: list[TrainingExample] = [
        _tokenize_training_record(
            record,
            tokenizer,
            config,
            cycles=cycles,
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
    records: Sequence[TrainingRecord],
    tokenizer,
    config: TrainConfig,
) -> tuple[list[TrainingRecord], Counter[str]]:
    process = config.process_branch
    if process is None:
        raise ValueError("process_branch configuration is missing")
    maximum_cycles = max(process.cycle_choices)
    valid: list[TrainingRecord] = []
    reasons: Counter[str] = Counter()
    for record in records:
        try:
            _tokenize_training_record(
                record,
                tokenizer,
                config,
                cycles=maximum_cycles,
            )
        except InvalidProcessBranchExample as exc:
            reasons[str(exc)] += 1
            continue
        valid.append(record)
    return valid, reasons
