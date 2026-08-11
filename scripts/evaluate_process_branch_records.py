from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import sys
from typing import Any, Iterator, Literal, Mapping, Sequence, cast

import torch
from safetensors.torch import load as load_safetensors

from gcw.checkpoint import load_checkpoint, load_checkpoint_bytes
from gcw.config import (
    TrainConfig,
    load_train_config_bytes,
    process_branch_training_contract,
    to_dict,
)
from gcw.data.tokenizers import get_tokenizer
from gcw.models.longshot import GlobalCacheProcessor, Longshot
from gcw.models.windowed_cache import WindowedCache
from gcw.process_branch import generate_process_branch
from gcw.process_data import (
    AnswerOnlyRecord,
    authenticate_answer_only_artifact,
    read_process_branch_records,
)
from gcw.process_eval import extract_numeric_answer, numeric_answers_match
from gcw.process_branch_trainer import (
    ProcessSchedule,
    _build_training_data_identity,
    _valid_records,
)
from gcw.state_hash import tensor_state_sha256


CANONICAL_EVALUATOR_VERSION = "gcw.answer_only_eval.v1"
CANONICAL_ROW_SCHEMA = "gcw.answer_only_eval.row.v1"
CANONICAL_MANIFEST_SCHEMA = "gcw.answer_only_eval.manifest.v1"
CanonicalCondition = Literal["OFF", "INIT", "K1", "K2", "K4", "K8"]
_CANONICAL_CONDITIONS = ["OFF", "INIT", "K1", "K2", "K4", "K8"]
_CANONICAL_HELDOUT_INTERVAL = (6000, 256)
_CANONICAL_FINAL_INTERVAL = (6256, 256)


def _evaluator_source_tree_sha256(
    source_files: Sequence[Path] | None = None,
) -> str:
    repository = Path(__file__).resolve().parents[1]
    files = (
        tuple(sorted((repository / "gcw").rglob("*.py"))) + (Path(__file__).resolve(),)
        if source_files is None
        else tuple(source_files)
    )
    if not files:
        raise ValueError("evaluator source tree must contain at least one file")
    common = Path(os.path.commonpath([str(path.resolve()) for path in files]))
    base = common if common.is_dir() else common.parent
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda candidate: str(candidate.resolve())):
        resolved = path.resolve()
        relative = resolved.relative_to(base).as_posix().encode("utf-8")
        payload = resolved.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


CANONICAL_EVALUATOR_SOURCE_SHA256 = _evaluator_source_tree_sha256()
_CONDITION_CYCLES: dict[str, int | None] = {
    "OFF": None,
    "INIT": 0,
    "K1": 1,
    "K2": 2,
    "K4": 4,
    "K8": 8,
}



@dataclass(frozen=True)
class EvaluationIdentities:
    run_id: str
    run_sha256: str
    arm: str
    arm_sha256: str
    checkpoint_sha256: str
    config_sha256: str
    data_sha256: str

    def __post_init__(self) -> None:
        if not self.run_id or not self.arm:
            raise ValueError("run_id and arm must not be empty")
        for name in (
            "run_sha256",
            "arm_sha256",
            "checkpoint_sha256",
            "config_sha256",
            "data_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase sha256 digest")


@dataclass(frozen=True)
class DeterministicDecode:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise ValueError("canonical max_new_tokens must be an integer")
        if self.max_new_tokens <= 0:
            raise ValueError("canonical max_new_tokens must be positive")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise ValueError("canonical top_k must be an integer")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, int | float)
            or isinstance(self.top_p, bool)
            or not isinstance(self.top_p, int | float)
        ):
            raise ValueError("canonical temperature and top_p must be numeric")
        if self.temperature != 0.0 or self.top_p != 1.0 or self.top_k != 0:
            raise ValueError("canonical answer-only evaluation requires deterministic greedy decoding")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


@dataclass(frozen=True)
class AuthenticatedCheckpointSnapshot:
    manifest: dict[str, Any]
    files: dict[str, bytes]


@dataclass(frozen=True)
class CanonicalExecutionPlan:
    protocol_path: Path
    protocol_sha256: str
    arm: str
    arm_sha256: str
    run_id: str
    split: str
    train_config: Path
    checkpoint: Path
    expected_checkpoint_sha256: str
    expected_checkpoint_manifest_sha256: str
    expected_config_sha256: str
    data_path: Path
    expected_data_sha256: str
    expected_source_ids_sha256: str
    conditions: tuple[str, ...]
    decode: DeterministicDecode
    prompt_suffix: str
    answer_prefix: str
    skip_rows: int
    take_rows: int


def load_canonical_protocol(protocol_path: Path) -> dict[str, Any]:
    return load_canonical_protocol_bytes(protocol_path.read_bytes())


def load_canonical_protocol_bytes(protocol_bytes: bytes) -> dict[str, Any]:
    try:
        protocol_text = protocol_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("answer-only protocol must be UTF-8 JSON") from exc
    payload = _strict_json_loads(protocol_text)
    required_root = {
        "schema", "evaluator_version", "conditions", "dataset",
        "generation", "splits", "arms",
    }
    if not isinstance(payload, dict) or set(payload) != required_root:
        raise ValueError("answer-only protocol fields differ from required schema")
    if payload["schema"] != "gcw.answer_only_eval.protocol.v1":
        raise ValueError("invalid answer-only evaluation protocol schema")
    if payload["evaluator_version"] != CANONICAL_EVALUATOR_VERSION:
        raise ValueError("answer-only protocol evaluator version mismatch")
    if payload["conditions"] != _CANONICAL_CONDITIONS:
        raise ValueError("canonical protocol requires OFF/INIT/K1/K2/K4/K8")
    dataset = payload["dataset"]
    if not isinstance(dataset, dict) or set(dataset) != {
        "path", "sha256", "ordered_source_ids_sha256"
    }:
        raise ValueError("canonical protocol dataset fields differ from required schema")
    if not isinstance(dataset["path"], str) or not dataset["path"]:
        raise ValueError("canonical protocol dataset path is invalid")
    _require_sha256(dataset["sha256"], "canonical dataset sha256")
    _require_sha256(
        dataset["ordered_source_ids_sha256"], "canonical ordered source IDs sha256"
    )
    generation = payload["generation"]
    if not isinstance(generation, dict) or set(generation) != {
        "answer_prefix", "prompt_suffix", "max_new_tokens",
        "temperature", "top_p", "top_k",
    }:
        raise ValueError("canonical generation fields differ from required schema")
    if (
        not isinstance(generation["answer_prefix"], str)
        or not generation["answer_prefix"]
        or not isinstance(generation["prompt_suffix"], str)
        or not generation["prompt_suffix"]
    ):
        raise ValueError("canonical generation prompt fields are invalid")
    DeterministicDecode(
        max_new_tokens=generation["max_new_tokens"],
        temperature=generation["temperature"],
        top_p=generation["top_p"],
        top_k=generation["top_k"],
    )
    splits = payload["splits"]
    if not isinstance(splits, dict) or set(splits) != {"heldout", "final"}:
        raise ValueError("canonical protocol requires ordered heldout/final splits")
    for name, split_payload in splits.items():
        if not isinstance(split_payload, dict) or set(split_payload) != {
            "skip_rows", "take_rows", "execution_allowed"
        }:
            raise ValueError(f"canonical {name} split fields differ from required schema")
        skip_rows = split_payload["skip_rows"]
        take_rows = split_payload["take_rows"]
        allowed = split_payload["execution_allowed"]
        if (
            isinstance(skip_rows, bool) or not isinstance(skip_rows, int) or skip_rows < 0
            or isinstance(take_rows, bool) or not isinstance(take_rows, int) or take_rows <= 0
            or not isinstance(allowed, bool)
        ):
            raise ValueError(f"canonical {name} split bounds are invalid")
    if splits["heldout"]["execution_allowed"] is not True:
        raise ValueError("canonical heldout split must be execution-enabled")
    if splits["final"]["execution_allowed"] is not False:
        raise ValueError("canonical final split must remain execution-blocked")
    if (
        splits["heldout"]["skip_rows"],
        splits["heldout"]["take_rows"],
    ) != _CANONICAL_HELDOUT_INTERVAL:
        raise ValueError("canonical heldout interval must be exactly rows 6000:6256")
    if (
        splits["final"]["skip_rows"],
        splits["final"]["take_rows"],
    ) != _CANONICAL_FINAL_INTERVAL:
        raise ValueError("canonical protected final interval must be exactly rows 6256:6512")
    arms = payload["arms"]
    if not isinstance(arms, dict) or set(arms) != {"p0", "p1"}:
        raise ValueError("canonical protocol requires ordered p0/p1 arms")
    required_arm = {
        "run_id", "train_config", "checkpoint", "checkpoint_sha256",
        "checkpoint_manifest_sha256", "config_sha256",
    }
    for name, arm_payload in arms.items():
        if not isinstance(arm_payload, dict) or set(arm_payload) != required_arm:
            raise ValueError(f"canonical arm {name} fields differ from required schema")
        for path_name in ("run_id", "train_config", "checkpoint"):
            if not isinstance(arm_payload[path_name], str) or not arm_payload[path_name]:
                raise ValueError(f"canonical arm {name} {path_name} is invalid")
        _require_sha256(arm_payload["config_sha256"], f"canonical arm {name} config sha256")
        for digest_name in ("checkpoint_sha256", "checkpoint_manifest_sha256"):
            value = arm_payload[digest_name]
            if value is not None:
                _require_sha256(value, f"canonical arm {name} {digest_name}")
    return payload


def resolve_canonical_execution_plan(
    protocol_path: Path,
    *,
    arm: str,
    split: str,
) -> CanonicalExecutionPlan:
    if split == "final":
        raise ValueError("canonical final split execution is unconditionally forbidden")
    protocol_bytes = protocol_path.read_bytes()
    payload = load_canonical_protocol_bytes(protocol_bytes)
    if payload.get("schema") != "gcw.answer_only_eval.protocol.v1":
        raise ValueError("invalid answer-only evaluation protocol schema")
    if payload.get("evaluator_version") != CANONICAL_EVALUATOR_VERSION:
        raise ValueError("answer-only protocol evaluator version mismatch")
    conditions = payload.get("conditions")
    if conditions != _CANONICAL_CONDITIONS:
        raise ValueError("canonical protocol requires OFF/INIT/K1/K2/K4/K8")
    splits = payload.get("splits")
    if not isinstance(splits, dict) or split not in splits or not isinstance(splits[split], dict):
        raise ValueError(f"unknown canonical split: {split}")
    split_payload = splits[split]
    if split_payload.get("execution_allowed") is not True:
        raise ValueError(f"canonical split {split} forbids execution")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or arm not in arms or not isinstance(arms[arm], dict):
        raise ValueError(f"unknown canonical arm: {arm}")
    arm_payload = arms[arm]
    checkpoint_sha256 = _require_sha256(
        arm_payload.get("checkpoint_sha256"), f"canonical arm {arm} checkpoint sha256"
    )
    checkpoint_manifest_sha256 = _require_sha256(
        arm_payload.get("checkpoint_manifest_sha256"),
        f"canonical arm {arm} checkpoint manifest sha256",
    )
    config_sha256 = _require_sha256(
        arm_payload.get("config_sha256"), f"canonical arm {arm} config sha256"
    )
    dataset = payload.get("dataset")
    generation = payload.get("generation")
    if not isinstance(dataset, dict) or not isinstance(generation, dict):
        raise ValueError("canonical protocol dataset/generation must be objects")
    max_new_tokens = generation.get("max_new_tokens")
    temperature = generation.get("temperature")
    top_p = generation.get("top_p")
    top_k = generation.get("top_k")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("canonical max_new_tokens must be an integer")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("canonical top_k must be an integer")
    if isinstance(temperature, bool) or not isinstance(temperature, int | float):
        raise ValueError("canonical temperature must be numeric")
    if isinstance(top_p, bool) or not isinstance(top_p, int | float):
        raise ValueError("canonical top_p must be numeric")
    decode = DeterministicDecode(
        max_new_tokens=max_new_tokens,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=top_k,
    )
    skip_rows = split_payload.get("skip_rows")
    take_rows = split_payload.get("take_rows")
    if (
        isinstance(skip_rows, bool)
        or not isinstance(skip_rows, int)
        or skip_rows < 0
        or isinstance(take_rows, bool)
        or not isinstance(take_rows, int)
        or take_rows <= 0
    ):
        raise ValueError("canonical split bounds are invalid")
    string_fields = {
        "train_config": arm_payload.get("train_config"),
        "checkpoint": arm_payload.get("checkpoint"),
        "data_path": dataset.get("path"),
        "data_sha256": dataset.get("sha256"),
        "source_ids_sha256": dataset.get("ordered_source_ids_sha256"),
        "prompt_suffix": generation.get("prompt_suffix"),
        "answer_prefix": generation.get("answer_prefix"),
    }
    if any(not isinstance(value, str) or not value for value in string_fields.values()):
        raise ValueError("canonical protocol contains empty path, digest, or prompt fields")
    return CanonicalExecutionPlan(
        protocol_path=protocol_path,
        protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
        arm=arm,
        arm_sha256=_canonical_json_sha256(arm_payload),
        run_id=cast(str, arm_payload["run_id"]),
        split=split,
        train_config=Path(cast(str, string_fields["train_config"])),
        checkpoint=Path(cast(str, string_fields["checkpoint"])),
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        expected_config_sha256=config_sha256,
        data_path=Path(cast(str, string_fields["data_path"])),
        expected_data_sha256=cast(str, string_fields["data_sha256"]),
        expected_source_ids_sha256=cast(str, string_fields["source_ids_sha256"]),
        conditions=tuple(conditions),
        decode=decode,
        prompt_suffix=cast(str, string_fields["prompt_suffix"]),
        answer_prefix=cast(str, string_fields["answer_prefix"]),
        skip_rows=skip_rows,
        take_rows=take_rows,
    )


@dataclass(frozen=True)
class AnswerOnlyConditionGeneration:
    generated_ids: torch.Tensor
    condition: str
    cycles: int | None
    answer_nll_sum: float | None
    answer_nll_per_token: float | None
    target_token_count: int | None
    termination_reason: Literal["eos", "max_tokens"]
    cap_hit: bool
    delta_value_norm: float | None
    cache_head_coverage: float | None
    update_concentration: float | None
    update_entropy: float | None


@torch.no_grad()
def generate_answer_only_condition(
    model: Longshot,
    *,
    prompt_ids: torch.Tensor,
    answer_prefix_ids: torch.Tensor,
    condition: str,
    max_new_tokens: int,
    eos_token_id: int | None,
    valid_vocab_size: int | None = None,
    target_answer_ids: torch.Tensor | None = None,
    processor_override: GlobalCacheProcessor | None = None,
) -> AnswerOnlyConditionGeneration:
    if condition not in _CONDITION_CYCLES:
        raise ValueError(f"unsupported canonical condition: {condition}")
    if prompt_ids.ndim != 1 or prompt_ids.numel() == 0:
        raise ValueError("canonical generation requires one non-empty prompt")
    if answer_prefix_ids.ndim != 1 or answer_prefix_ids.numel() == 0:
        raise ValueError("canonical generation requires a non-empty answer prefix")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    _, prompt_cache = model.backbone.forward_chunk(
        prompt_ids.unsqueeze(0),
        0,
        None,
        logits=False,
        cache_format="windowed",
    )
    if not isinstance(prompt_cache, WindowedCache):
        raise TypeError("canonical generation prefill requires a WindowedCache")

    cycles = _CONDITION_CYCLES[condition]
    delta_value_norm = None
    cache_head_coverage = None
    update_concentration = None
    update_entropy = None
    if cycles is None:
        branch_cache = prompt_cache
    else:
        processor = processor_override or cast(GlobalCacheProcessor, model.cache_processor)
        source = processor.encode_source(prompt_cache, select_all=True)
        initial_workspace = processor.initialise_workspace(source)
        workspace = initial_workspace
        for _ in range(cycles):
            workspace = processor.deliberate(workspace, source)
        branch_cache = processor.write_for_readout(prompt_cache, source, workspace)
        delta = (workspace - initial_workspace).float()
        delta_value_norm = float(delta.norm().cpu())
        selected = torch.unique(source.selected_heads)
        cache_head_coverage = float(selected.numel() / processor.num_cache_heads)
        head_updates = delta.flatten(start_dim=2).norm(dim=-1)
        total = head_updates.sum()
        if bool(total > 0):
            probabilities = head_updates / total
            update_concentration = float(probabilities.max().cpu())
            positive = probabilities[probabilities > 0]
            update_entropy = float((-(positive * positive.log()).sum()).cpu())
        else:
            update_concentration = 0.0
            update_entropy = 0.0

    answer_nll_sum = None
    answer_nll_per_token = None
    target_token_count = None
    if target_answer_ids is not None:
        if target_answer_ids.ndim != 1 or target_answer_ids.numel() == 0:
            raise ValueError("canonical target answer must be one non-empty token sequence")
        target_token_count = int(target_answer_ids.numel())
        teacher_inputs = torch.cat([answer_prefix_ids, target_answer_ids[:-1]])
        masked_prefix = target_answer_ids.new_full((answer_prefix_ids.numel() - 1,), -1)
        teacher_targets = torch.cat([masked_prefix, target_answer_ids])
        target_hidden, _ = model.backbone.forward_chunk(
            teacher_inputs.unsqueeze(0),
            branch_cache.seq_len,
            branch_cache,
            logits=False,
            cache_format="windowed",
        )
        mean_nll = model.backbone.loss_from_hidden(
            target_hidden,
            teacher_targets.unsqueeze(0),
            "mean",
        )
        answer_nll_per_token = float(mean_nll.float().cpu())
        answer_nll_sum = answer_nll_per_token * target_token_count

    if max_new_tokens == 0:
        generated_ids = prompt_ids.new_empty((0,))
    else:
        hidden, decode_cache = model.backbone.forward_chunk(
            answer_prefix_ids.unsqueeze(0),
            branch_cache.seq_len,
            branch_cache,
            logits=False,
            cache_format="windowed",
        )
        if not isinstance(decode_cache, WindowedCache):
            raise TypeError("canonical generation requires a WindowedCache")
        next_logits = model.backbone.logits_from_hidden(hidden[:, -1:])[:, -1]
        vocab_size = valid_vocab_size or next_logits.shape[-1]
        if vocab_size <= 0 or vocab_size > next_logits.shape[-1]:
            raise ValueError("valid_vocab_size must be within the model vocabulary")
        generated = []
        for token_index in range(max_new_tokens):
            next_token = torch.argmax(next_logits[:, :vocab_size], dim=-1)
            generated.append(next_token)
            if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
                break
            if token_index + 1 >= max_new_tokens:
                break
            hidden, next_cache = model.backbone.forward_chunk(
                next_token[:, None],
                decode_cache.seq_len,
                decode_cache,
                logits=False,
                cache_format="windowed",
            )
            if not isinstance(next_cache, WindowedCache):
                raise TypeError("canonical generation requires a WindowedCache")
            decode_cache = next_cache
            next_logits = model.backbone.logits_from_hidden(hidden[:, -1:])[:, -1]
        generated_ids = torch.cat(generated, dim=0)

    eos_emitted = bool(
        eos_token_id is not None
        and generated_ids.numel() > 0
        and int(generated_ids[-1]) == eos_token_id
    )
    termination_reason: Literal["eos", "max_tokens"] = (
        "eos" if eos_emitted else "max_tokens"
    )
    return AnswerOnlyConditionGeneration(
        generated_ids=generated_ids,
        condition=condition,
        cycles=cycles,
        answer_nll_sum=answer_nll_sum,
        answer_nll_per_token=answer_nll_per_token,
        target_token_count=target_token_count,
        termination_reason=termination_reason,
        cap_hit=not eos_emitted,
        delta_value_norm=delta_value_norm,
        cache_head_coverage=cache_head_coverage,
        update_concentration=update_concentration,
        update_entropy=update_entropy,
    )


def _source_index(source_id: str) -> int:
    try:
        index = int(source_id.rsplit(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"canonical source_id lacks an integer index: {source_id}") from exc
    if index < 0:
        raise ValueError(f"canonical source_id has a negative index: {source_id}")
    return index


def evaluate_canonical_records(
    model: Longshot,
    tokenizer: Any,
    records: Sequence[AnswerOnlyRecord],
    *,
    conditions: Sequence[str],
    identities: EvaluationIdentities,
    decode: DeterministicDecode,
    prompt_suffix: str,
    answer_prefix: str,
    initial_processor_state: Mapping[str, torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("canonical evaluation requires at least one record")
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("canonical conditions must be non-empty and unique")
    unsupported = [condition for condition in conditions if condition not in _CONDITION_CYCLES]
    if unsupported:
        raise ValueError(f"unsupported canonical conditions: {unsupported}")
    answer_prefix_ids = torch.tensor(
        tokenizer.encode(answer_prefix),
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    if answer_prefix_ids.numel() == 0:
        raise ValueError("answer prefix tokenised to an empty sequence")

    initial_processor: GlobalCacheProcessor | None = None
    if "INIT" in conditions:
        if initial_processor_state is None:
            raise ValueError("canonical INIT requires an authenticated initial processor state")
        initial_processor = copy.deepcopy(
            cast(GlobalCacheProcessor, model.cache_processor)
        )
        initial_processor.load_state_dict(dict(initial_processor_state), strict=True)
        initial_processor.to(next(model.parameters()).device)
        initial_processor.eval()

    rows: list[dict[str, Any]] = []
    decode_payload = decode.as_dict()
    for condition in conditions:
        for evaluation_index, record in enumerate(records):
            prompt = record.prompt
            if prompt_suffix.strip():
                prompt = f"{prompt}\n\n{prompt_suffix.strip()}"
            prompt_ids = torch.tensor(
                tokenizer.encode(prompt),
                dtype=torch.long,
                device=answer_prefix_ids.device,
            )
            start = time.perf_counter()
            target_answer_ids = torch.tensor(
                tokenizer.encode(record.final_answer) + [int(tokenizer.eot_id)],
                dtype=torch.long,
                device=answer_prefix_ids.device,
            )
            generation = generate_answer_only_condition(
                model,
                prompt_ids=prompt_ids,
                answer_prefix_ids=answer_prefix_ids,
                condition=condition,
                max_new_tokens=decode.max_new_tokens,
                eos_token_id=int(tokenizer.eot_id),
                valid_vocab_size=int(tokenizer.get_vocab_size()),
                target_answer_ids=target_answer_ids,
                processor_override=initial_processor if condition == "INIT" else None,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            raw_text = tokenizer.decode(generation.generated_ids.tolist())
            parsed = extract_numeric_answer(raw_text)
            target_value = extract_numeric_answer(record.final_answer)
            if target_value is None:
                raise ValueError(f"canonical target is not numeric: {record.source_id}")
            rows.append(
                {
                    "schema": CANONICAL_ROW_SCHEMA,
                    "evaluator_version": CANONICAL_EVALUATOR_VERSION,
                    "evaluator_source_sha256": CANONICAL_EVALUATOR_SOURCE_SHA256,
                    "run_id": identities.run_id,
                    "run_sha256": identities.run_sha256,
                    "arm": identities.arm,
                    "arm_sha256": identities.arm_sha256,
                    "checkpoint_sha256": identities.checkpoint_sha256,
                    "config_sha256": identities.config_sha256,
                    "data_sha256": identities.data_sha256,
                    "source_id": record.source_id,
                    "source_index": _source_index(record.source_id),
                    "evaluation_index": evaluation_index,
                    "condition": condition,
                    "K": generation.cycles,
                    "decode": dict(decode_payload),
                    "prompt_token_count": int(prompt_ids.numel()),
                    "answer_token_count": int(generation.generated_ids.numel()),
                    "generated_token_ids": generation.generated_ids.tolist(),
                    "target_token_count": generation.target_token_count,
                    "answer_nll_sum": generation.answer_nll_sum,
                    "answer_nll_per_token": generation.answer_nll_per_token,
                    "termination_reason": generation.termination_reason,
                    "cap_hit": generation.cap_hit,
                    "raw_generated_text": raw_text,
                    "parsed_prediction": str(parsed) if parsed is not None else None,
                    "target": record.final_answer,
                    "correctness": parsed == target_value if parsed is not None else False,
                    "extraction_status": "ok" if parsed is not None else "no_numeric_prediction",
                    "latency_ms": latency_ms,
                    "diagnostics": {
                        "delta_value_norm": generation.delta_value_norm,
                        "cache_head_coverage": generation.cache_head_coverage,
                        "update_concentration": generation.update_concentration,
                        "update_entropy": generation.update_entropy,
                    },
                }
            )
    return rows



_CANONICAL_MANIFEST_BASE_FIELDS = {
    "schema",
    "evaluator_version",
    "evaluator_source_sha256",
    "row_schema",
    "identities",
    "conditions",
    "ordered_source_ids",
    "generation_settings",
    "eos_token_id",
    "row_count",
    "selection",
    "checkpoint_path",
    "config_path",
    "data_path",
    "protocol_sha256",
    "runtime",
}
_CANONICAL_MANIFEST_PUBLICATION_FIELDS = {
    "rows_path",
    "rows_sha256",
    "ordered_source_ids_sha256",
}
_CANONICAL_RUNTIME_FIELDS = {
    "device",
    "torch_version",
    "cuda_version",
    "gpu_name",
    "evaluation_precision",
    "elapsed_seconds",
}


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase sha256 digest")
    return value


_CANONICAL_ROW_REQUIRED_FIELDS = {
    "schema",
    "evaluator_version",
    "evaluator_source_sha256",
    "run_id",
    "run_sha256",
    "arm",
    "arm_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "data_sha256",
    "source_id",
    "source_index",
    "evaluation_index",
    "condition",
    "K",
    "decode",
    "prompt_token_count",
    "answer_token_count",
    "generated_token_ids",
    "target_token_count",
    "answer_nll_sum",
    "answer_nll_per_token",
    "termination_reason",
    "cap_hit",
    "raw_generated_text",
    "parsed_prediction",
    "target",
    "correctness",
    "extraction_status",
    "latency_ms",
    "diagnostics",
}


def _finite_number(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def validate_canonical_ledger(rows: Sequence[dict[str, Any]], manifest: dict[str, Any]) -> None:
    fields = set(manifest)
    if fields == _CANONICAL_MANIFEST_BASE_FIELDS:
        published = False
    elif fields == _CANONICAL_MANIFEST_BASE_FIELDS | _CANONICAL_MANIFEST_PUBLICATION_FIELDS:
        published = True
    else:
        missing_manifest = sorted(_CANONICAL_MANIFEST_BASE_FIELDS - fields)
        extra = sorted(fields - (_CANONICAL_MANIFEST_BASE_FIELDS | _CANONICAL_MANIFEST_PUBLICATION_FIELDS))
        partial_publication = sorted(fields & _CANONICAL_MANIFEST_PUBLICATION_FIELDS)
        raise ValueError(
            "canonical manifest fields differ from required schema: "
            f"missing={missing_manifest}, extra={extra}, partial_published={partial_publication}"
        )
    if manifest.get("schema") != CANONICAL_MANIFEST_SCHEMA:
        raise ValueError("invalid canonical manifest schema")
    if manifest.get("evaluator_version") != CANONICAL_EVALUATOR_VERSION:
        raise ValueError("invalid canonical evaluator version")
    evaluator_source_sha256 = _require_sha256(
        manifest.get("evaluator_source_sha256"), "evaluator_source_sha256"
    )
    if manifest.get("row_schema") != CANONICAL_ROW_SCHEMA:
        raise ValueError("invalid canonical row schema")
    conditions = manifest.get("conditions")
    source_ids = manifest.get("ordered_source_ids")
    if conditions != _CANONICAL_CONDITIONS:
        raise ValueError("canonical manifest requires exact OFF/INIT/K1/K2/K4/K8 conditions")
    if (
        not isinstance(source_ids, list)
        or not source_ids
        or any(not isinstance(source_id, str) or not source_id for source_id in source_ids)
        or len(set(source_ids)) != len(source_ids)
    ):
        raise ValueError("canonical manifest ordered_source_ids are invalid")
    if published:
        if manifest.get("rows_path") != "rows.jsonl":
            raise ValueError("canonical manifest rows_path must equal fixed artifact role 'rows.jsonl'")
        supplied_rows_sha256 = _require_sha256(manifest.get("rows_sha256"), "rows_sha256")
        actual_rows_sha256 = hashlib.sha256(_canonical_jsonl_bytes(rows)).hexdigest()
        if supplied_rows_sha256 != actual_rows_sha256:
            raise ValueError("canonical published rows_sha256 mismatch")
    expected_row_count = len(conditions) * len(source_ids)
    row_count = manifest.get("row_count")
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count != expected_row_count
        or row_count != len(rows)
    ):
        raise ValueError("canonical ledger row count mismatch")
    generation_settings = manifest["generation_settings"]
    eos_token_id = manifest["eos_token_id"]
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise ValueError("canonical eos_token_id must be a non-negative integer")
    if not isinstance(generation_settings, dict):
        raise ValueError("canonical generation settings must be an object")
    DeterministicDecode(**generation_settings)
    identities = manifest.get("identities")
    if not isinstance(identities, dict):
        raise ValueError("canonical manifest identities must be an object")
    identity = EvaluationIdentities(**identities)
    _require_sha256(manifest.get("protocol_sha256"), "protocol_sha256")
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or set(selection) != {"split", "skip_rows", "take_rows"}:
        raise ValueError("canonical manifest selection fields differ from required schema")
    if selection.get("split") != "heldout":
        raise ValueError("canonical manifest permits heldout rows only; final rows are forbidden")
    skip_rows = selection.get("skip_rows")
    take_rows = selection.get("take_rows")
    if (skip_rows, take_rows) != _CANONICAL_HELDOUT_INTERVAL:
        raise ValueError("canonical manifest heldout interval must be exactly rows 6000:6256")
    if (
        isinstance(skip_rows, bool)
        or not isinstance(skip_rows, int)
        or skip_rows < 0
        or isinstance(take_rows, bool)
        or not isinstance(take_rows, int)
        or take_rows != len(source_ids)
    ):
        raise ValueError("canonical manifest selection bounds are invalid")
    required_artifact_labels = {
        "checkpoint_path": "checkpoint",
        "config_path": "train-config",
        "data_path": "answer-only-data",
    }
    for path_name, required_label in required_artifact_labels.items():
        if manifest.get(path_name) != required_label:
            raise ValueError(
                f"canonical manifest {path_name} must equal fixed artifact role {required_label!r}"
            )
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != _CANONICAL_RUNTIME_FIELDS:
        raise ValueError("canonical manifest runtime fields differ from required schema")
    if not isinstance(runtime["device"], str) or not runtime["device"]:
        raise ValueError("canonical runtime device must be non-empty")
    if not isinstance(runtime["torch_version"], str) or not runtime["torch_version"]:
        raise ValueError("canonical runtime torch_version must be non-empty")
    for optional_name in ("cuda_version", "gpu_name"):
        if runtime[optional_name] is not None and not isinstance(runtime[optional_name], str):
            raise ValueError(f"canonical runtime {optional_name} must be a string or null")
    if runtime["evaluation_precision"] not in {
        "float32",
        "bfloat16",
        "mixed_bfloat16_float32",
        "bfloat16_autocast",
    }:
        raise ValueError("canonical runtime evaluation_precision is invalid")
    _finite_number(runtime["elapsed_seconds"], name="runtime elapsed_seconds", minimum=0.0)
    expected_run_sha256 = _canonical_json_sha256(
        {
            "run_id": identity.run_id,
            "arm": identity.arm,
            "arm_contract_sha256": identity.arm_sha256,
            "checkpoint_sha256": identity.checkpoint_sha256,
            "training_contract_sha256": identity.config_sha256,
            "data_sha256": identity.data_sha256,
            "evaluator_source_sha256": evaluator_source_sha256,
            "evaluation_precision": runtime["evaluation_precision"],
            "split": selection["split"],
            "conditions": conditions,
            "decode": generation_settings,
            "eos_token_id": manifest["eos_token_id"],
        }
    )
    if identity.run_sha256 != expected_run_sha256:
        raise ValueError("canonical manifest run_sha256 does not match its authenticated bindings")
    if published:
        if not isinstance(manifest["rows_path"], str) or not manifest["rows_path"]:
            raise ValueError("published canonical rows_path must be non-empty")
        _require_sha256(manifest["rows_sha256"], "rows_sha256")
        expected_source_hash = _canonical_json_sha256(source_ids)
        if manifest["ordered_source_ids_sha256"] != expected_source_hash:
            raise ValueError("published canonical ordered source IDs hash mismatch")

    seen_pairs: set[tuple[object, object]] = set()
    expected_pairs = [
        (condition, source_id, evaluation_index)
        for condition in conditions
        for evaluation_index, source_id in enumerate(source_ids)
    ]
    for row_index, (row, expected) in enumerate(zip(rows, expected_pairs, strict=True)):
        row_fields = set(row)
        if row_fields != _CANONICAL_ROW_REQUIRED_FIELDS:
            missing = sorted(_CANONICAL_ROW_REQUIRED_FIELDS - row_fields)
            extra = sorted(row_fields - _CANONICAL_ROW_REQUIRED_FIELDS)
            raise ValueError(
                f"row fields differ at row {row_index}: missing={missing}, extra={extra}"
            )
        if row["schema"] != CANONICAL_ROW_SCHEMA or row["evaluator_version"] != CANONICAL_EVALUATOR_VERSION:
            raise ValueError(f"row {row_index} schema/version mismatch")
        if row["evaluator_source_sha256"] != evaluator_source_sha256:
            raise ValueError(f"evaluator source sha256 mismatch at row {row_index}")
        pair = (row["condition"], row["source_id"])
        if pair in seen_pairs:
            raise ValueError(f"duplicate condition/source row: {pair}")
        seen_pairs.add(pair)
        expected_condition, expected_source_id, expected_evaluation_index = expected
        if (
            row["condition"] != expected_condition
            or row["source_id"] != expected_source_id
            or row["evaluation_index"] != expected_evaluation_index
        ):
            raise ValueError(f"ordered source coverage mismatch at row {row_index}")
        if row["source_index"] != _source_index(expected_source_id):
            raise ValueError(f"source_index mismatch at row {row_index}")
        if row["K"] != _CONDITION_CYCLES[expected_condition]:
            raise ValueError(f"condition/K mismatch at row {row_index}")
        if row["decode"] != generation_settings:
            raise ValueError(f"decode settings mismatch at row {row_index}")
        for identity_name in (
            "run_id",
            "run_sha256",
            "arm",
            "arm_sha256",
            "checkpoint_sha256",
            "config_sha256",
            "data_sha256",
        ):
            if row[identity_name] != identities.get(identity_name):
                raise ValueError(f"{identity_name} mismatch at row {row_index}")
        for count_name in ("prompt_token_count", "answer_token_count", "target_token_count"):
            count = row[count_name]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{count_name} must be a non-negative integer at row {row_index}")
        if row["prompt_token_count"] == 0 or row["target_token_count"] == 0:
            raise ValueError(f"prompt and target token counts must be positive at row {row_index}")
        generated_token_ids = row["generated_token_ids"]
        if (
            not isinstance(generated_token_ids, list)
            or len(generated_token_ids) != row["answer_token_count"]
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
                for token_id in generated_token_ids
            )
        ):
            raise ValueError(f"generated token IDs/count mismatch at row {row_index}")
        nll_sum = _finite_number(
            row["answer_nll_sum"], name=f"answer_nll_sum at row {row_index}", minimum=0.0
        )
        nll_per_token = _finite_number(
            row["answer_nll_per_token"],
            name=f"answer_nll_per_token at row {row_index}",
            minimum=0.0,
        )
        if not math.isclose(
            nll_sum,
            nll_per_token * row["target_token_count"],
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(f"answer NLL total/per-token mismatch at row {row_index}")
        if row["termination_reason"] not in {"eos", "max_tokens"}:
            raise ValueError(f"invalid termination_reason at row {row_index}")
        if not isinstance(row["cap_hit"], bool) or row["cap_hit"] is not (
            row["termination_reason"] == "max_tokens"
        ):
            raise ValueError(f"cap_hit/termination mismatch at row {row_index}")
        if (
            row["answer_token_count"] < 1
            or row["answer_token_count"] > generation_settings["max_new_tokens"]
        ):
            raise ValueError(f"answer token count exceeds decode bounds at row {row_index}")
        if row["termination_reason"] == "eos":
            if (
                not generated_token_ids
                or generated_token_ids[-1] != eos_token_id
                or eos_token_id in generated_token_ids[:-1]
            ):
                raise ValueError(f"EOS termination lacks unique terminal EOS evidence at row {row_index}")
        elif (
            row["answer_token_count"] != generation_settings["max_new_tokens"]
            or eos_token_id in generated_token_ids
        ):
            raise ValueError(f"max-token termination evidence mismatch at row {row_index}")
        if not isinstance(row["raw_generated_text"], str) or not isinstance(row["target"], str):
            raise ValueError(f"text fields must be strings at row {row_index}")
        if row["parsed_prediction"] is not None and not isinstance(row["parsed_prediction"], str):
            raise ValueError(f"parsed_prediction must be a string or null at row {row_index}")
        if not isinstance(row["correctness"], bool):
            raise ValueError(f"correctness must be boolean at row {row_index}")
        if row["extraction_status"] not in {"ok", "no_numeric_prediction"}:
            raise ValueError(f"invalid extraction_status at row {row_index}")
        reparsed_prediction = extract_numeric_answer(row["raw_generated_text"])
        reparsed_target = extract_numeric_answer(row["target"])
        if reparsed_target is None:
            raise ValueError(f"target extraction failed at row {row_index}")
        expected_prediction = str(reparsed_prediction) if reparsed_prediction is not None else None
        if row["parsed_prediction"] != expected_prediction:
            raise ValueError(f"parsed_prediction mismatch at row {row_index}")
        expected_status = "ok" if reparsed_prediction is not None else "no_numeric_prediction"
        if row["extraction_status"] != expected_status:
            raise ValueError(f"extraction_status mismatch at row {row_index}")
        expected_correctness = reparsed_prediction == reparsed_target if reparsed_prediction is not None else False
        if row["correctness"] is not expected_correctness:
            raise ValueError(f"correctness mismatch at row {row_index}")
        _finite_number(row["latency_ms"], name=f"latency_ms at row {row_index}", minimum=0.0)
        diagnostics = row["diagnostics"]
        if not isinstance(diagnostics, dict) or set(diagnostics) != {
            "delta_value_norm",
            "cache_head_coverage",
            "update_concentration",
            "update_entropy",
        }:
            raise ValueError(f"invalid diagnostics at row {row_index}")
        if expected_condition == "OFF":
            if any(value is not None for value in diagnostics.values()):
                raise ValueError(f"OFF diagnostics must be unavailable at row {row_index}")
        else:
            norm = _finite_number(diagnostics["delta_value_norm"], name="delta_value_norm", minimum=0.0)
            coverage = _finite_number(diagnostics["cache_head_coverage"], name="cache_head_coverage", minimum=0.0)
            concentration = _finite_number(diagnostics["update_concentration"], name="update_concentration", minimum=0.0)
            _finite_number(diagnostics["update_entropy"], name="update_entropy", minimum=0.0)
            if coverage > 1.0 or concentration > 1.0 or norm < 0.0:
                raise ValueError(f"bounded diagnostics invalid at row {row_index}")




def build_canonical_manifest(
    rows: Sequence[dict[str, Any]],
    *,
    identities: EvaluationIdentities,
    conditions: Sequence[str],
    decode: DeterministicDecode,
    eos_token_id: int,
    checkpoint_path: str,
    config_path: str,
    data_path: str,
    protocol_sha256: str,
    split: str,
    skip_rows: int,
    take_rows: int,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    if not conditions or len(rows) % len(conditions) != 0:
        raise ValueError("canonical rows do not form complete condition blocks")
    source_count = len(rows) // len(conditions)
    ordered_source_ids = [str(row["source_id"]) for row in rows[:source_count]]
    if skip_rows < 0 or take_rows <= 0 or source_count != take_rows:
        raise ValueError("canonical selection bounds do not match row coverage")
    if len(protocol_sha256) != 64 or any(character not in "0123456789abcdef" for character in protocol_sha256):
        raise ValueError("protocol_sha256 must be a lowercase sha256 digest")
    manifest = {
        "schema": CANONICAL_MANIFEST_SCHEMA,
        "evaluator_version": CANONICAL_EVALUATOR_VERSION,
        "evaluator_source_sha256": CANONICAL_EVALUATOR_SOURCE_SHA256,
        "row_schema": CANONICAL_ROW_SCHEMA,
        "identities": {
            "run_id": identities.run_id,
            "run_sha256": identities.run_sha256,
            "arm": identities.arm,
            "arm_sha256": identities.arm_sha256,
            "checkpoint_sha256": identities.checkpoint_sha256,
            "config_sha256": identities.config_sha256,
            "data_sha256": identities.data_sha256,
        },
        "conditions": list(conditions),
        "ordered_source_ids": ordered_source_ids,
        "generation_settings": decode.as_dict(),
        "eos_token_id": eos_token_id,
        "row_count": len(rows),
        "selection": {"split": split, "skip_rows": skip_rows, "take_rows": take_rows},
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "data_path": data_path,
        "protocol_sha256": protocol_sha256,
        "runtime": runtime,
    }
    validate_canonical_ledger(rows, manifest)
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate_canonical_checkpoint_snapshot(
    checkpoint: Path,
    *,
    expected_manifest_sha256: str,
    expected_model_sha256: str,
) -> AuthenticatedCheckpointSnapshot:
    _require_sha256(expected_manifest_sha256, "checkpoint manifest sha256")
    _require_sha256(expected_model_sha256, "checkpoint model sha256")
    manifest_bytes = (checkpoint / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise ValueError("canonical checkpoint manifest sha256 mismatch")
    try:
        manifest_text = manifest_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("canonical checkpoint manifest must be UTF-8 JSON") from exc
    payload = _strict_json_loads(manifest_text)
    required_manifest_fields = {
        "schema",
        "files",
        "initial_processor_sha256",
        "final_processor_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required_manifest_fields:
        raise ValueError("canonical checkpoint manifest fields differ from required schema")
    if payload["schema"] != "gcw.final_checkpoint.manifest.v2":
        raise ValueError("canonical checkpoint manifest schema mismatch")
    files = payload["files"]
    required_files = {
        "config.json",
        "initial_processor.safetensors",
        "model.safetensors",
        "run_config.json",
        "train_state.json",
    }
    if not isinstance(files, dict) or set(files) != required_files:
        raise ValueError("canonical checkpoint manifest file roles differ from required schema")
    _require_sha256(
        payload["initial_processor_sha256"], "checkpoint initial_processor_sha256"
    )
    _require_sha256(
        payload["final_processor_sha256"], "checkpoint final_processor_sha256"
    )
    file_snapshots: dict[str, bytes] = {}
    for name in sorted(required_files):
        expected = _require_sha256(files[name], f"checkpoint {name} sha256")
        candidate = checkpoint / name
        try:
            snapshot = candidate.read_bytes()
        except OSError as exc:
            raise ValueError(f"canonical checkpoint {name} is missing") from exc
        if hashlib.sha256(snapshot).hexdigest() != expected:
            raise ValueError(f"canonical checkpoint {name} sha256 mismatch")
        file_snapshots[name] = snapshot
    if files["model.safetensors"] != expected_model_sha256:
        raise ValueError("canonical checkpoint model sha256 differs from protocol")
    return AuthenticatedCheckpointSnapshot(manifest=payload, files=file_snapshots)


def authenticate_canonical_checkpoint(
    checkpoint: Path,
    *,
    expected_manifest_sha256: str,
    expected_model_sha256: str,
) -> dict[str, Any]:
    return authenticate_canonical_checkpoint_snapshot(
        checkpoint,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_model_sha256=expected_model_sha256,
    ).manifest


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_jsonl_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _write_fsynced_bytes_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def publish_canonical_ledger(
    bundle_path: Path,
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> Path:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.exists():
        raise FileExistsError(bundle_path)
    validate_canonical_ledger(rows, manifest)
    rows_bytes = _canonical_jsonl_bytes(rows)
    bound_manifest = {
        **manifest,
        "rows_path": "rows.jsonl",
        "rows_sha256": hashlib.sha256(rows_bytes).hexdigest(),
        "ordered_source_ids_sha256": _canonical_json_sha256(manifest["ordered_source_ids"]),
    }
    validate_canonical_ledger(rows, bound_manifest)
    manifest_bytes = (
        json.dumps(
            bound_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{bundle_path.name}.", dir=bundle_path.parent)
    )
    try:
        _write_fsynced_bytes_new(staging / "rows.jsonl", rows_bytes)
        _write_fsynced_bytes_new(staging / "manifest.json", manifest_bytes)
        _fsync_directory(staging)
        os.replace(staging, bundle_path)
        _fsync_directory(bundle_path.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return bundle_path


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        parse_constant=_reject_nonfinite_json,
        object_pairs_hook=_reject_duplicate_json_keys,
    )


def load_canonical_ledger(
    bundle_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_protocol_sha256: str,
    expected_evaluator_source_sha256: str = CANONICAL_EVALUATOR_SOURCE_SHA256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require_sha256(expected_manifest_sha256, "expected canonical manifest sha256")
    _require_sha256(expected_protocol_sha256, "expected canonical protocol sha256")
    _require_sha256(
        expected_evaluator_source_sha256,
        "expected canonical evaluator source sha256",
    )
    manifest_bytes = (bundle_path / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise ValueError("canonical manifest sha256 mismatch")
    rows_bytes = (bundle_path / "rows.jsonl").read_bytes()
    try:
        manifest_text = manifest_bytes.decode("utf-8")
        rows_text = rows_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("canonical ledger bundle must be UTF-8") from exc
    manifest_payload = _strict_json_loads(manifest_text)
    if not isinstance(manifest_payload, dict):
        raise ValueError("canonical manifest must be a JSON object")
    if manifest_payload.get("protocol_sha256") != expected_protocol_sha256:
        raise ValueError("canonical protocol sha256 mismatch")
    if (
        manifest_payload.get("evaluator_source_sha256")
        != expected_evaluator_source_sha256
        or expected_evaluator_source_sha256 != CANONICAL_EVALUATOR_SOURCE_SHA256
    ):
        raise ValueError("canonical evaluator source sha256 mismatch")
    rows: list[dict[str, Any]] = []
    lines = rows_text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for row_index, line in enumerate(lines):
        try:
            row = _strict_json_loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"malformed canonical JSONL row {row_index}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"canonical JSONL row {row_index} must be an object")
        rows.append(row)
    if manifest_payload.get("rows_path") != "rows.jsonl":
        raise ValueError("canonical rows path mismatch")
    if manifest_payload.get("rows_sha256") != hashlib.sha256(rows_bytes).hexdigest():
        raise ValueError("canonical rows sha256 mismatch")
    if manifest_payload.get("ordered_source_ids_sha256") != _canonical_json_sha256(
        manifest_payload.get("ordered_source_ids")
    ):
        raise ValueError("canonical ordered source IDs sha256 mismatch")
    validate_canonical_ledger(rows, manifest_payload)
    return rows, manifest_payload


def _atomic_write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _legacy_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate answer-only process-branch generation on process records.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--skip-rows", required=True, type=int)
    parser.add_argument("--take-rows", required=True, type=int)
    parser.add_argument("--prompt-suffix", required=True)
    parser.add_argument("--answer-prefix", default="\n\n")
    parser.add_argument("--cycles", required=True, type=int)
    parser.add_argument("--max-new-tokens", default=32, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.out.exists():
        raise FileExistsError(args.out)
    if args.take_rows <= 0:
        raise ValueError("take-rows must be positive")
    if args.cycles < 0:
        raise ValueError("cycles must be non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, train_state = load_checkpoint(args.checkpoint, device=device)
    if not isinstance(model, Longshot):
        raise TypeError("process-branch evaluation requires a Longshot checkpoint")
    model.eval()
    tokenizer = get_tokenizer(args.tokenizer)
    records = read_process_branch_records(
        args.records,
        skip_rows=args.skip_rows,
        take_rows=args.take_rows,
    )
    if len(records) != args.take_rows:
        raise ValueError(f"requested {args.take_rows} records but loaded {len(records)}")
    answer_prefix_ids = torch.tensor(
        tokenizer.encode(args.answer_prefix),
        dtype=torch.long,
        device=device,
    )
    if answer_prefix_ids.numel() == 0:
        raise ValueError("answer prefix tokenised to an empty sequence")

    rows = []
    correct = 0
    generated_tokens = 0
    truncated = 0
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with context:
        for record in records:
            prompt = record.prompt
            if args.prompt_suffix.strip():
                prompt = f"{prompt}\n\n{args.prompt_suffix.strip()}"
            prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
            generated = generate_process_branch(
                model,
                prompt_ids=prompt_ids,
                answer_prefix_ids=answer_prefix_ids,
                cycles=args.cycles,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=int(tokenizer.eot_id),
                valid_vocab_size=tokenizer.get_vocab_size(),
            )
            prediction = tokenizer.decode(generated.tolist())
            is_correct = numeric_answers_match(prediction, record.final_answer)
            is_truncated = (
                generated.numel() == args.max_new_tokens
                and int(generated[-1]) != int(tokenizer.eot_id)
            )
            correct += int(is_correct)
            generated_tokens += generated.numel()
            truncated += int(is_truncated)
            rows.append(
                {
                    "source_id": record.source_id,
                    "target": record.final_answer,
                    "prediction": prediction,
                    "correct": is_correct,
                    "generated_tokens": generated.numel(),
                    "truncated": is_truncated,
                }
            )

    payload = {
        "schema": "gcw.process_branch.record_eval.v1",
        "checkpoint": str(args.checkpoint),
        "processor_sha256": train_state.get("final_processor_sha256"),
        "records": str(args.records),
        "skip_rows": args.skip_rows,
        "take_rows": args.take_rows,
        "cycles": args.cycles,
        "max_new_tokens": args.max_new_tokens,
        "correct": correct,
        "accuracy": correct / len(records),
        "mean_generated_tokens": generated_tokens / len(records),
        "truncated": truncated,
        "rows": rows,
    }
    _atomic_write_json_new(args.out, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, sort_keys=True))
    return 0






def _realized_evaluation_precision(
    model: torch.nn.Module,
    *,
    device: torch.device,
    autocast_bf16: bool,
) -> str:
    if device.type == "cuda" and autocast_bf16:
        return "bfloat16_autocast"
    floating_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }
    if floating_dtypes == {torch.float32}:
        return "float32"
    if floating_dtypes == {torch.bfloat16}:
        return "bfloat16"
    if floating_dtypes == {torch.float32, torch.bfloat16}:
        return "mixed_bfloat16_float32"
    raise ValueError(f"unsupported realized evaluation dtypes: {sorted(map(str, floating_dtypes))}")


def _canonical_tokenizer_source(
    tokenizer_name: str,
    tokenizer_source: str | None,
) -> Path:
    if tokenizer_name == "hf" and tokenizer_source:
        return Path(tokenizer_source)
    if tokenizer_name.startswith("hf:"):
        source = tokenizer_name.removeprefix("hf:")
        if source:
            return Path(source)
    raise ValueError("canonical evaluation requires a local HF tokenizer artifact")


@contextmanager
def authenticated_tokenizer_snapshot(
    source: Path,
    *,
    expected_sha256: str,
) -> Iterator[Path]:
    _require_sha256(expected_sha256, "tokenizer sha256")
    if not source.is_dir():
        raise ValueError("tokenizer artifact directory is not visible")
    captured: list[tuple[Path, bytes]] = []
    digest = hashlib.sha256()
    for candidate in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = candidate.relative_to(source)
        relative_bytes = relative.as_posix().encode("utf-8")
        payload = candidate.read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        captured.append((relative, payload))
    if not captured:
        raise ValueError("tokenizer artifact directory is empty")
    if digest.hexdigest() != expected_sha256:
        raise ValueError("tokenizer artifact sha256 mismatch")

    with tempfile.TemporaryDirectory(prefix="gcw-tokenizer-snapshot-") as temporary:
        snapshot = Path(temporary)
        for relative, payload in captured:
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        yield snapshot


def _validate_completed_training_provenance(
    *,
    train_state: dict[str, Any],
    train_config: TrainConfig,
    checkpoint_manifest: dict[str, Any],
    observed_data_identity: dict[str, Any],
) -> None:
    process = train_config.process_branch
    if process is None:
        raise ValueError("canonical checkpoint has no process_branch config")
    data_identity = train_state.get("data_identity")
    code_identity = train_state.get("code_identity")
    fingerprint = train_state.get("training_fingerprint")
    if (
        not isinstance(data_identity, dict)
        or not isinstance(code_identity, dict)
        or not isinstance(fingerprint, dict)
    ):
        raise ValueError("canonical checkpoint training identity is missing")
    fingerprint_payload = cast(dict[str, Any], fingerprint)
    if data_identity != observed_data_identity:
        raise ValueError("canonical checkpoint data identity differs from authenticated recomputation")
    expected_fingerprint = {
        "kind": "gcw.process_branch.training_fingerprint.v2",
        "code_identity": code_identity,
        "distributed": train_config.distributed,
        "bf16": train_config.bf16,
        "objective": {
            "kind": process.objective,
            "process_weight": process.process_weight,
            "answer_weight": process.answer_weight,
        },
        "cycle_choices": list(process.cycle_choices),
        "data_identity": data_identity,
        "seed": train_config.seed,
        "model": to_dict(train_config.model),
        "initial_processor_sha256": checkpoint_manifest["initial_processor_sha256"],
        "observed_tokenizer_sha256": process.expected_tokenizer_sha256,
        "optimizer": {
            "kind": "AdamW",
            "learning_rate": train_config.learning_rate,
            "betas": [train_config.adam_beta1, train_config.adam_beta2],
            "eps": train_config.adam_eps,
            "weight_decay": train_config.weight_decay,
            "grad_clip": train_config.grad_clip,
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
            "data_kind": train_config.data.kind,
            "tokenizer_role": "authenticated_local_hf",
            "prompt_suffix": train_config.data.prompt_suffix,
            "answer_response_prefix": train_config.data.response_prefix,
            "step_response_prefix": process.step_response_prefix,
            "max_prompt_tokens": process.max_prompt_tokens,
            "max_step_tokens": process.max_step_tokens,
            "max_answer_tokens": process.max_answer_tokens,
        },
    }
    if fingerprint_payload != expected_fingerprint:
        raise ValueError("canonical checkpoint training fingerprint mismatch")
    if code_identity.get("schema") != "gcw.source_code_identity.v1":
        raise ValueError("canonical checkpoint code identity schema mismatch")
    for digest_field in ("source_tree_sha256", "git_status_sha256"):
        _require_sha256(code_identity.get(digest_field), f"training {digest_field}")
    commit = code_identity.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("canonical checkpoint Git commit is invalid")
    if not isinstance(code_identity.get("git_dirty"), bool):
        raise ValueError("canonical checkpoint Git dirty flag is invalid")
    if train_state.get("optimizer_steps") != process.max_optimizer_steps:
        raise ValueError("canonical checkpoint training schedule is incomplete")
    expected_seen = process.max_optimizer_steps * process.gradient_accumulation_steps
    if train_state.get("seen_examples") != expected_seen:
        raise ValueError("canonical checkpoint seen_examples is incomplete")
    data_splits = cast(dict[str, Any], fingerprint_payload["data_identity"]).get("splits")
    train_identity = data_splits.get("train") if isinstance(data_splits, dict) else None
    heldout_identity = data_splits.get("heldout") if isinstance(data_splits, dict) else None
    frozen_population = (
        (
            "train",
            train_identity,
            process.expected_train_eligible_records,
            process.expected_train_eligible_source_ids_sha256,
            "valid_records",
        ),
        (
            "heldout",
            heldout_identity,
            process.expected_heldout_eligible_records,
            process.expected_heldout_eligible_source_ids_sha256,
            "heldout_valid_records",
        ),
    )
    for split_name, identity, expected_count, expected_ids_sha256, state_count_field in frozen_population:
        if (
            not isinstance(identity, dict)
            or isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count <= 0
            or not isinstance(expected_ids_sha256, str)
            or identity.get("eligible_records") != expected_count
            or identity.get("ordered_source_ids_sha256") != expected_ids_sha256
            or train_state.get(state_count_field) != expected_count
        ):
            raise ValueError(
                f"canonical checkpoint {split_name} population differs from frozen identity"
            )
    record_count = cast(int, process.expected_train_eligible_records)
    expected_schedule = ProcessSchedule(
        record_count=record_count,
        cycle_choices=process.cycle_choices,
        seed=train_config.seed,
    )
    for _ in range(expected_seen):
        expected_schedule.next_example()
    if train_state.get("process_cursor") != expected_schedule.cursor.to_dict():
        raise ValueError("canonical checkpoint process cursor does not prove exact schedule replay")


def _canonical_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Run canonical paired answer-only OFF/INIT/K evaluation.")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--bundle-out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.bundle_out.exists():
        raise FileExistsError(args.bundle_out)

    plan = resolve_canonical_execution_plan(args.protocol, arm=args.arm, split=args.split)
    config_bytes = plan.train_config.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if config_sha256 != plan.expected_config_sha256:
        raise ValueError("canonical train config sha256 mismatch")
    checkpoint_snapshot = authenticate_canonical_checkpoint_snapshot(
        plan.checkpoint,
        expected_manifest_sha256=plan.expected_checkpoint_manifest_sha256,
        expected_model_sha256=plan.expected_checkpoint_sha256,
    )
    checkpoint_manifest = checkpoint_snapshot.manifest
    checkpoint_sha256 = cast(dict[str, str], checkpoint_manifest["files"])["model.safetensors"]
    data_bytes = plan.data_path.read_bytes()
    data_sha256 = hashlib.sha256(data_bytes).hexdigest()
    if data_sha256 != plan.expected_data_sha256:
        raise ValueError("canonical data sha256 mismatch")
    source_records = cast(
        list[AnswerOnlyRecord],
        authenticate_answer_only_artifact(
            data_bytes,
            expected_artifact_sha256=plan.expected_data_sha256,
            expected_source_ids_sha256=plan.expected_source_ids_sha256,
        ),
    )

    train_config = load_train_config_bytes(config_bytes)
    try:
        checkpoint_run_config = _strict_json_loads(
            checkpoint_snapshot.files["run_config.json"].decode("utf-8")
        )
    except UnicodeDecodeError as exc:
        raise ValueError("canonical checkpoint run_config.json is not UTF-8") from exc
    if checkpoint_run_config != process_branch_training_contract(train_config):
        raise ValueError("canonical checkpoint run_config differs from protocol-pinned training contract")
    process_config = train_config.process_branch
    if process_config is None or process_config.objective != "answer_only":
        raise ValueError("canonical evaluation requires an answer-only training config")
    frozen_expected = {
        "train_records": process_config.expected_train_eligible_records,
        "train_source_ids_sha256": (
            process_config.expected_train_eligible_source_ids_sha256
        ),
        "heldout_records": process_config.expected_heldout_eligible_records,
        "heldout_source_ids_sha256": (
            process_config.expected_heldout_eligible_source_ids_sha256
        ),
        "initial_backbone_sha256": process_config.expected_initial_backbone_sha256,
        "tokenizer_sha256": process_config.expected_tokenizer_sha256,
    }
    if (
        not process_config.require_frozen_identities
        or any(value is None for value in frozen_expected.values())
    ):
        raise ValueError("canonical training identities are not frozen")
    if (
        Path(cast(str, train_config.data.data_dir)) != plan.data_path
        or train_config.data.expected_artifact_sha256 != plan.expected_data_sha256
        or train_config.data.expected_source_ids_sha256 != plan.expected_source_ids_sha256
    ):
        raise ValueError("canonical data role/config mismatch")
    if (
        train_config.data.prompt_suffix != plan.prompt_suffix
        or train_config.data.response_prefix != plan.answer_prefix
    ):
        raise ValueError("canonical generation/config prompt mismatch")
    if plan.split == "heldout" and (
        process_config.heldout_skip_rows != plan.skip_rows
        or process_config.heldout_take_rows != plan.take_rows
    ):
        raise ValueError("canonical heldout bounds/config mismatch")

    records = source_records[plan.skip_rows : plan.skip_rows + plan.take_rows]
    if len(records) != plan.take_rows:
        raise ValueError(f"canonical split requested {plan.take_rows} rows but loaded {len(records)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config, train_state = load_checkpoint_bytes(
        config_bytes=checkpoint_snapshot.files["config.json"],
        train_state_bytes=checkpoint_snapshot.files["train_state.json"],
        model_bytes=checkpoint_snapshot.files["model.safetensors"],
        device=device,
    )
    if not isinstance(model, Longshot) or model_config != train_config.model:
        raise ValueError("canonical checkpoint/model config mismatch")
    if train_state.get("objective") != "answer_only":
        raise ValueError("canonical checkpoint objective mismatch")
    data_identity = train_state.get("data_identity")
    if (
        not isinstance(data_identity, dict)
        or data_identity.get("source", {}).get("sha256") != plan.expected_data_sha256
    ):
        raise ValueError("canonical checkpoint/data hash mismatch")
    if train_state.get("frozen_identities") != frozen_expected:
        raise ValueError("canonical checkpoint frozen training identities mismatch")
    expected_backbone_sha256 = frozen_expected["initial_backbone_sha256"]
    loaded_backbone_sha256 = tensor_state_sha256(model.backbone.state_dict())
    if (
        train_state.get("initial_backbone_sha256") != expected_backbone_sha256
        or train_state.get("final_backbone_sha256") != expected_backbone_sha256
        or loaded_backbone_sha256 != expected_backbone_sha256
    ):
        raise ValueError("canonical checkpoint loaded backbone hash mismatch")
    recorded_processor_sha256 = train_state.get("final_processor_sha256")
    if (
        not isinstance(recorded_processor_sha256, str)
        or recorded_processor_sha256 != checkpoint_manifest["final_processor_sha256"]
        or tensor_state_sha256(cast(torch.nn.Module, model.cache_processor).state_dict())
        != recorded_processor_sha256
    ):
        raise ValueError("canonical checkpoint processor hash mismatch")
    recorded_initial_processor_sha256 = train_state.get("initial_processor_sha256")
    initial_processor_state = load_safetensors(
        checkpoint_snapshot.files["initial_processor.safetensors"]
    )
    if (
        not isinstance(recorded_initial_processor_sha256, str)
        or recorded_initial_processor_sha256
        != checkpoint_manifest["initial_processor_sha256"]
        or tensor_state_sha256(initial_processor_state)
        != recorded_initial_processor_sha256
    ):
        raise ValueError("canonical initial processor artifact hash mismatch")
    model.eval()
    tokenizer_source = _canonical_tokenizer_source(
        train_config.data.tokenizer,
        train_config.data.tokenizer_source,
    )
    expected_tokenizer_sha256 = cast(str, frozen_expected["tokenizer_sha256"])
    evaluation_precision = _realized_evaluation_precision(
        model,
        device=device,
        autocast_bf16=train_config.bf16,
    )
    training_contract_sha256 = _canonical_json_sha256(
        process_branch_training_contract(train_config)
    )
    arm_contract_sha256 = _canonical_json_sha256(
        {
            "arm": plan.arm,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_manifest_sha256": plan.expected_checkpoint_manifest_sha256,
            "training_contract_sha256": training_contract_sha256,
        }
    )

    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if evaluation_precision == "bfloat16_autocast"
        else nullcontext()
    )
    started = time.perf_counter()
    with authenticated_tokenizer_snapshot(
        tokenizer_source,
        expected_sha256=expected_tokenizer_sha256,
    ) as tokenizer_snapshot:
        tokenizer = get_tokenizer(f"hf:{tokenizer_snapshot}")
        train_end = (
            None
            if train_config.data.sft_take_rows is None
            else train_config.data.sft_skip_rows + train_config.data.sft_take_rows
        )
        training_records = source_records[train_config.data.sft_skip_rows:train_end]
        heldout_end = process_config.heldout_skip_rows + process_config.heldout_take_rows
        heldout_records = source_records[process_config.heldout_skip_rows:heldout_end]
        valid_training, training_rejections = _valid_records(
            training_records,
            tokenizer,
            train_config,
        )
        valid_heldout, heldout_rejections = _valid_records(
            heldout_records,
            tokenizer,
            train_config,
        )
        observed_data_identity = _build_training_data_identity(
            train_config,
            data_bytes,
            training_records,
            valid_training,
            heldout_records,
            valid_heldout,
            training_rejections,
            heldout_rejections,
        )
        _validate_completed_training_provenance(
            train_state=train_state,
            train_config=train_config,
            checkpoint_manifest=checkpoint_manifest,
            observed_data_identity=observed_data_identity,
        )
        observed_heldout_ids = observed_data_identity["splits"]["heldout"]["ordered_source_ids"]
        if [record.source_id for record in records] != observed_heldout_ids:
            raise ValueError("canonical evaluated heldout IDs differ from frozen eligible population")
        eos_token_id = int(tokenizer.eot_id)
        arm_sha256 = arm_contract_sha256
        run_id = plan.run_id
        run_sha256 = _canonical_json_sha256(
            {
                "run_id": run_id,
                "arm": plan.arm,
                "arm_contract_sha256": arm_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "training_contract_sha256": training_contract_sha256,
                "data_sha256": data_sha256,
                "evaluator_source_sha256": CANONICAL_EVALUATOR_SOURCE_SHA256,
                "evaluation_precision": evaluation_precision,
                "split": plan.split,
                "conditions": plan.conditions,
                "decode": plan.decode.as_dict(),
                "eos_token_id": eos_token_id,
            }
        )
        identities = EvaluationIdentities(
            run_id=run_id,
            run_sha256=run_sha256,
            arm=plan.arm,
            arm_sha256=arm_sha256,
            checkpoint_sha256=checkpoint_sha256,
            config_sha256=training_contract_sha256,
            data_sha256=data_sha256,
        )
        with context:
            rows = evaluate_canonical_records(
                model,
                tokenizer,
                records,
                conditions=plan.conditions,
                identities=identities,
                decode=plan.decode,
                prompt_suffix=plan.prompt_suffix,
                answer_prefix=plan.answer_prefix,
                initial_processor_state=initial_processor_state,
            )
    runtime = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "evaluation_precision": evaluation_precision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest = build_canonical_manifest(
        rows,
        identities=identities,
        conditions=plan.conditions,
        decode=plan.decode,
        eos_token_id=eos_token_id,
        checkpoint_path="checkpoint",
        config_path="train-config",
        data_path="answer-only-data",
        protocol_sha256=plan.protocol_sha256,
        split=plan.split,
        skip_rows=plan.skip_rows,
        take_rows=plan.take_rows,
        runtime=runtime,
    )
    publish_canonical_ledger(args.bundle_out, rows, manifest)
    print(json.dumps({**manifest, "ordered_source_ids": "<bound in manifest>"}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argument_list = list(argv) if argv is not None else sys.argv[1:]
    if argument_list and argument_list[0] == "canonical":
        return _canonical_main(argument_list[1:])
    raise ValueError(
        "evaluate_process_branch_records.py is canonical-only; "
        "legacy arbitrary record slices are forbidden"
    )


if __name__ == "__main__":
    raise SystemExit(main())
