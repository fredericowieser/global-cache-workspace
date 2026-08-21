from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Literal, Mapping, Protocol, Sequence, cast

import torch

from gcw.evaluation.scoring import extract_numeric_answer

DATASET_REPO_ID = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
EXPECTED_TEST_ROWS = 1_319
MAX_NEW_TOKENS = 2_048
DEFAULT_BATCH_SIZE = 64
RESULT_SCHEMA = "gcw.benchmark_matrix.v2"
PromptMode = Literal["thinking_off", "thinking_on"]
ScoringProtocol = Literal["numeric_last_answer", "lm_eval_flexible_exact"]
TargetKind = Literal["hf", "gcw"]
_FLEXIBLE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


@dataclass(frozen=True)
class BenchmarkConfig:
    model_path: str
    output_dir: str
    target_kind: TargetKind = "hf"
    tokenizer_path: str | None = None
    processor_enabled: bool = True
    scoring_protocol: ScoringProtocol = "lm_eval_flexible_exact"
    batch_size: int = DEFAULT_BATCH_SIZE
    max_new_tokens: int = MAX_NEW_TOKENS
    limit: int | None = None
    modes: tuple[PromptMode, ...] = ("thinking_off", "thinking_on")

    def __post_init__(self) -> None:
        if self.target_kind not in {"hf", "gcw"}:
            raise ValueError("target_kind must be 'hf' or 'gcw'")
        if self.target_kind == "hf" and not self.processor_enabled:
            raise ValueError("HF targets have no GCW processor to bypass")
        if self.scoring_protocol not in {
            "numeric_last_answer",
            "lm_eval_flexible_exact",
        }:
            raise ValueError("unknown scoring_protocol")
        if isinstance(self.batch_size, bool) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.max_new_tokens != MAX_NEW_TOKENS:
            raise ValueError(
                f"Qwen3 GSM8K reproduction requires max_new_tokens={MAX_NEW_TOKENS}"
            )
        if self.limit is not None and (
            isinstance(self.limit, bool) or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer")
        if not self.modes or any(
            mode not in {"thinking_off", "thinking_on"} for mode in self.modes
        ):
            raise ValueError(
                "modes must contain thinking_off and/or thinking_on"
            )
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("modes must not contain duplicates")


@dataclass(frozen=True)
class PreparedExample:
    index: int
    question: str
    target_text: str
    target_answer: Fraction
    prompt_ids: tuple[int, ...]


class BenchmarkBackend(Protocol):
    identity: dict[str, Any]
    eos_ids: tuple[int, ...]
    pad_id: int
    device: torch.device
    supports_left_padding: bool

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor: ...


class HFBenchmarkBackend:
    def __init__(self, source: Path, device: torch.device):
        from transformers import AutoModelForCausalLM

        self.model = (
            cast(
                Any,
                AutoModelForCausalLM.from_pretrained(
                    source,
                    local_files_only=True,
                    trust_remote_code=False,
                    dtype=torch.bfloat16,
                ),
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.supports_left_padding = True
        self.eos_ids = _eos_ids(self.model, None)
        self.pad_id = int(
            getattr(self.model.generation_config, "pad_token_id", None)
            or self.eos_ids[0]
        )
        self.identity = _directory_identity(source, kind="hf")

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=(
                self.eos_ids[0]
                if len(self.eos_ids) == 1
                else list(self.eos_ids)
            ),
            pad_token_id=self.pad_id,
            use_cache=True,
        )


class GCWBenchmarkBackend:
    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        *,
        processor_enabled: bool,
        generation_process_every: int | None = None,
    ):
        from gcw.checkpoint import load_checkpoint
        from gcw.evaluation.core import ProcessorBypass
        from gcw.evaluation.lm_eval_adapter import (
            _apply_generation_process_every,
        )

        model, config, state = load_checkpoint(checkpoint, device=device)
        if generation_process_every is not None and not processor_enabled:
            raise ValueError(
                "generation_process_every requires processor_enabled=True"
            )
        _apply_generation_process_every(config, generation_process_every)
        if not processor_enabled:
            model = ProcessorBypass(model)
        self.model = model.eval()
        self.device = device
        self.supports_left_padding = False
        self.eos_ids: tuple[int, ...] = (
            int(state.get("eos_token_id", 151645)),
        )
        self.pad_id = int(state.get("pad_token_id", self.eos_ids[0]))
        self.identity = {
            **_directory_identity(checkpoint, kind="gcw"),
            "architecture": config.architecture_name,
            "processor_enabled": processor_enabled,
            "generation_process_every": generation_process_every,
            "checkpoint_state": state,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        if not bool(attention_mask.all()):
            raise ValueError(
                "native GCW generation does not support padded prompts"
            )
        return cast(Any, self.model).generate_fast(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.eos_ids[0],
            pad_token_id=self.pad_id,
        )


def apply_qwen_chat_template(
    tokenizer: Any, question: str, *, enable_thinking: bool
) -> tuple[int, ...]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if isinstance(rendered, torch.Tensor):
        rendered = rendered.tolist()
    if isinstance(rendered, Mapping):
        rendered = rendered.get("input_ids")
    if (
        isinstance(rendered, list)
        and len(rendered) == 1
        and isinstance(rendered[0], list)
    ):
        rendered = rendered[0]
    if not isinstance(rendered, (list, tuple)) or not rendered:
        raise ValueError("Qwen chat template produced no token IDs")
    return tuple(int(token) for token in rendered)


def extract_gsm8k_target(answer: str) -> Fraction:
    marker = answer.rsplit("####", 1)
    candidate = marker[-1] if len(marker) == 2 else answer
    parsed = extract_numeric_answer(candidate)
    if parsed is None:
        raise ValueError(f"GSM8K target has no numeric answer: {answer!r}")
    return parsed


def score_numeric_response(response: str, target: Fraction) -> bool:
    predicted = extract_numeric_answer(response)
    return predicted is not None and predicted == target


def extract_lm_eval_flexible(response: str) -> str | None:
    matches = list(_FLEXIBLE_RE.finditer(response))
    if not matches:
        return None
    groups = matches[-1].groups()
    return next((value for value in groups if value is not None), None)


def score_lm_eval_flexible(response: str, target_text: str) -> bool:
    prediction = extract_lm_eval_flexible(response)
    if prediction is None:
        return False
    target = target_text.rsplit("####", 1)[-1]
    return _lm_eval_normalize(prediction) == _lm_eval_normalize(target)


def _lm_eval_normalize(value: str) -> str:
    value = value.lower()
    for pattern in (r",", r"\$", r"(?s).*#### ", r"\.$", r"\s+"):
        value = re.sub(pattern, "", value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_identity(path: Path, *, kind: str) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"evaluation target does not exist: {path}")
    required = ["config.json", "model.safetensors"]
    if kind == "gcw":
        required.append("train_state.json")
    files: dict[str, dict[str, Any]] = {}
    for name in required:
        candidate = path / name
        if not candidate.is_file():
            raise FileNotFoundError(f"evaluation target is missing {name}")
        files[name] = {
            "bytes": candidate.stat().st_size,
            "sha256": _sha256_file(candidate),
        }
    return {"kind": kind, "path": str(path), "files": files}


def _source_commit(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def _load_tokenizer(source: Path) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("benchmark tokenizer has no official chat template")
    files: dict[str, dict[str, Any]] = {}
    identity: dict[str, Any] = {"path": str(source), "files": files}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"tokenizer source is missing {name}")
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return tokenizer, identity


def _load_test_rows(
    limit: int | None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_REPO_ID, "main", split="test", revision=DATASET_REVISION
    )
    if len(dataset) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"expected {EXPECTED_TEST_ROWS} GSM8K test rows, found {len(dataset)}"
        )
    count = len(dataset) if limit is None else min(limit, len(dataset))
    rows = [
        {
            "question": str(dataset[index]["question"]),
            "answer": str(dataset[index]["answer"]),
        }
        for index in range(count)
    ]
    identity = {
        "repository": DATASET_REPO_ID,
        "revision": DATASET_REVISION,
        "config": "main",
        "split": "test",
        "full_rows": EXPECTED_TEST_ROWS,
        "evaluated_rows": count,
        "ordered_rows_sha256": hashlib.sha256(
            json.dumps(
                rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest(),
    }
    return rows, identity


def _prepare(
    rows: list[dict[str, str]], tokenizer: Any, enable_thinking: bool
) -> list[PreparedExample]:
    return [
        PreparedExample(
            index=index,
            question=row["question"],
            target_text=row["answer"],
            target_answer=extract_gsm8k_target(row["answer"]),
            prompt_ids=apply_qwen_chat_template(
                tokenizer, row["question"], enable_thinking=enable_thinking
            ),
        )
        for index, row in enumerate(rows)
    ]


def _eos_ids(model: Any, tokenizer: Any | None) -> tuple[int, ...]:
    value = getattr(model.generation_config, "eos_token_id", None)
    if value is None and tokenizer is not None:
        value = tokenizer.eos_token_id
    values = value if isinstance(value, (list, tuple)) else [value]
    result = tuple(
        dict.fromkeys(int(token) for token in values if token is not None)
    )
    if not result:
        raise ValueError("model has no EOS token")
    return result


def _trim_at_eos(
    tokens: Sequence[int], eos_ids: tuple[int, ...]
) -> tuple[list[int], str]:
    eos = set(eos_ids)
    for index, token in enumerate(tokens):
        if token in eos:
            return list(tokens[:index]), "eos"
    return list(tokens), "max_new_tokens"


def evaluate_mode(
    backend: BenchmarkBackend,
    tokenizer: Any,
    rows: list[dict[str, str]],
    *,
    mode: PromptMode,
    batch_size: int,
    max_new_tokens: int,
    scoring_protocol: ScoringProtocol,
) -> dict[str, Any]:
    examples = _prepare(rows, tokenizer, mode == "thinking_on")
    if backend.supports_left_padding:
        batches = [
            sorted(examples, key=lambda example: len(example.prompt_ids))[
                offset : offset + batch_size
            ]
            for offset in range(0, len(examples), batch_size)
        ]
        grouping = "length_sorted_left_padding"
    else:
        groups: dict[int, list[PreparedExample]] = defaultdict(list)
        for example in examples:
            groups[len(example.prompt_ids)].append(example)
        batches = [
            group[offset : offset + batch_size]
            for _, group in sorted(groups.items())
            for offset in range(0, len(group), batch_size)
        ]
        grouping = "exact_prompt_token_length"
    generated_rows: list[dict[str, Any] | None] = [None] * len(examples)
    started = time.monotonic()
    for chunk in batches:
        prompt_width = max(len(example.prompt_ids) for example in chunk)
        input_ids = torch.full(
            (len(chunk), prompt_width),
            backend.pad_id,
            dtype=torch.long,
            device=backend.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, example in enumerate(chunk):
            prompt = torch.tensor(
                example.prompt_ids, dtype=torch.long, device=backend.device
            )
            input_ids[row_index, -len(prompt) :] = prompt
            attention_mask[row_index, -len(prompt) :] = 1
        output = backend.generate(input_ids, attention_mask, max_new_tokens)
        for example, token_row in zip(
            chunk, output[:, prompt_width:].detach().cpu().tolist(), strict=True
        ):
            response_ids, termination = _trim_at_eos(token_row, backend.eos_ids)
            response = tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            numeric_correct = score_numeric_response(
                response, example.target_answer
            )
            flexible_correct = score_lm_eval_flexible(
                response, example.target_text
            )
            primary_correct = (
                numeric_correct
                if scoring_protocol == "numeric_last_answer"
                else flexible_correct
            )
            generated_rows[example.index] = {
                "index": example.index,
                "question": example.question,
                "target_text": example.target_text,
                "target_answer": str(example.target_answer),
                "prompt_tokens": len(example.prompt_ids),
                "response_tokens": len(response_ids),
                "termination": termination,
                "response": response,
                "numeric_prediction": (
                    None
                    if extract_numeric_answer(response) is None
                    else str(extract_numeric_answer(response))
                ),
                "flexible_prediction": extract_lm_eval_flexible(response),
                "numeric_correct": numeric_correct,
                "flexible_exact_correct": flexible_correct,
                "correct": primary_correct,
            }
    if any(row is None for row in generated_rows):
        raise RuntimeError("generation did not produce every GSM8K row")
    final_rows = [row for row in generated_rows if row is not None]
    correct = sum(bool(row["correct"]) for row in final_rows)
    return {
        "mode": mode,
        "enable_thinking": mode == "thinking_on",
        "scoring_protocol": scoring_protocol,
        "correct": correct,
        "total": len(final_rows),
        "accuracy": correct / len(final_rows),
        "numeric_correct": sum(
            bool(row["numeric_correct"]) for row in final_rows
        ),
        "flexible_exact_correct": sum(
            bool(row["flexible_exact_correct"]) for row in final_rows
        ),
        "elapsed_seconds": time.monotonic() - started,
        "generation": {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "grouping": grouping,
            "eos_token_ids": list(backend.eos_ids),
            "pad_token_id": backend.pad_id,
        },
        "rows": final_rows,
    }


def _publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload, handle, indent=2, sort_keys=True, allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite result: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def run_benchmark_matrix(
    config: BenchmarkConfig, repository: str | Path = "."
) -> dict[str, Any]:
    repository_path = Path(repository).resolve()
    if subprocess.check_output(
        ["git", "-C", str(repository_path), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("benchmark evaluation requires a clean Git checkout")
    output_dir = Path(config.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse output directory: {output_dir}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3 GSM8K evaluation requires CUDA")
    device = torch.device("cuda")
    target = Path(config.model_path).expanduser().resolve()
    backend: BenchmarkBackend
    if config.target_kind == "hf":
        backend = HFBenchmarkBackend(target, device)
    else:
        backend = GCWBenchmarkBackend(
            target, device, processor_enabled=config.processor_enabled
        )
    tokenizer_source = (
        Path(config.tokenizer_path or config.model_path).expanduser().resolve()
    )
    tokenizer, tokenizer_identity = _load_tokenizer(tokenizer_source)
    rows, dataset_identity = _load_test_rows(config.limit)
    cells = {
        mode: evaluate_mode(
            backend,
            tokenizer,
            rows,
            mode=mode,
            batch_size=config.batch_size,
            max_new_tokens=config.max_new_tokens,
            scoring_protocol=config.scoring_protocol,
        )
        for mode in config.modes
    }
    result = {
        "schema": RESULT_SCHEMA,
        "source_commit": _source_commit(repository_path),
        "config": asdict(config),
        "target": backend.identity,
        "tokenizer": tokenizer_identity,
        "dataset": dataset_identity,
        "runtime": {
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "cells": cells,
    }
    payload = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    document = {**result, "payload_sha256": hashlib.sha256(payload).hexdigest()}
    _publish_json(output_dir / "result.json", document)
    return document
