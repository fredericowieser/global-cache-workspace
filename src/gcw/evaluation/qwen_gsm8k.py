from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence, cast

import torch

from gcw.evaluation.scoring import extract_numeric_answer

MODEL_REPO_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DATASET_REPO_ID = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
EXPECTED_TEST_ROWS = 1_319
MAX_NEW_TOKENS = 2_048
DEFAULT_BATCH_SIZE = 16
RESULT_SCHEMA = "gcw.qwen3_gsm8k_baseline.v1"
MODEL_FILE_SHA256 = {
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}


@dataclass(frozen=True)
class QwenGSM8KConfig:
    model_path: str
    output_dir: str
    batch_size: int = DEFAULT_BATCH_SIZE
    max_new_tokens: int = MAX_NEW_TOKENS
    limit: int | None = None
    modes: tuple[str, ...] = ("thinking_off", "thinking_on")

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.max_new_tokens != MAX_NEW_TOKENS:
            raise ValueError(f"Qwen3 GSM8K reproduction requires max_new_tokens={MAX_NEW_TOKENS}")
        if self.limit is not None and (isinstance(self.limit, bool) or self.limit <= 0):
            raise ValueError("limit must be a positive integer")
        if not self.modes or any(mode not in {"thinking_off", "thinking_on"} for mode in self.modes):
            raise ValueError("modes must contain thinking_off and/or thinking_on")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("modes must not contain duplicates")


@dataclass(frozen=True)
class PreparedExample:
    index: int
    question: str
    target_text: str
    target_answer: Fraction
    prompt_ids: tuple[int, ...]


def apply_qwen_chat_template(tokenizer: Any, question: str, *, enable_thinking: bool) -> tuple[int, ...]:
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
    if isinstance(rendered, list) and len(rendered) == 1 and isinstance(rendered[0], list):
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit(repository: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()


def _load_model(model_path: Path, device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    from huggingface_hub import scan_cache_dir
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not model_path.is_dir():
        raise FileNotFoundError(f"model snapshot does not exist: {model_path}")
    files = {}
    for name, expected_sha256 in MODEL_FILE_SHA256.items():
        path = model_path / name
        if not path.is_file():
            raise FileNotFoundError(f"model snapshot is missing {name}")
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise ValueError(f"model snapshot hash mismatch: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": observed_sha256}
    cached_revisions = {
        revision.commit_hash
        for repository in scan_cache_dir().repos
        if repository.repo_id == MODEL_REPO_ID
        for revision in repository.revisions
        if Path(revision.snapshot_path).resolve() == model_path
    }
    if cached_revisions and cached_revisions != {MODEL_REVISION}:
        raise ValueError(f"model cache revision mismatch: {sorted(cached_revisions)}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
        ),
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer, {
        "repository": MODEL_REPO_ID,
        "revision": MODEL_REVISION,
        "path": str(model_path),
        "files": files,
    }


def _load_test_rows(limit: int | None) -> tuple[list[dict[str, str]], dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_REPO_ID, "main", split="test", revision=DATASET_REVISION)
    if len(dataset) != EXPECTED_TEST_ROWS:
        raise ValueError(f"expected {EXPECTED_TEST_ROWS} GSM8K test rows, found {len(dataset)}")
    count = len(dataset) if limit is None else min(limit, len(dataset))
    rows = [{"question": str(dataset[index]["question"]), "answer": str(dataset[index]["answer"])} for index in range(count)]
    identity = {
        "repository": DATASET_REPO_ID,
        "revision": DATASET_REVISION,
        "config": "main",
        "split": "test",
        "full_rows": EXPECTED_TEST_ROWS,
        "evaluated_rows": count,
        "ordered_rows_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest(),
    }
    return rows, identity


def _prepare(rows: list[dict[str, str]], tokenizer: Any, enable_thinking: bool) -> list[PreparedExample]:
    return [
        PreparedExample(
            index=index,
            question=row["question"],
            target_text=row["answer"],
            target_answer=extract_gsm8k_target(row["answer"]),
            prompt_ids=apply_qwen_chat_template(tokenizer, row["question"], enable_thinking=enable_thinking),
        )
        for index, row in enumerate(rows)
    ]


def _eos_ids(model: Any, tokenizer: Any) -> tuple[int, ...]:
    value = getattr(model.generation_config, "eos_token_id", None)
    if value is None:
        value = tokenizer.eos_token_id
    values = value if isinstance(value, (list, tuple)) else [value]
    result = tuple(dict.fromkeys(int(token) for token in values if token is not None))
    if not result:
        raise ValueError("Qwen model has no EOS token")
    return result


def _trim_at_eos(tokens: Sequence[int], eos_ids: tuple[int, ...]) -> tuple[list[int], str]:
    eos = set(eos_ids)
    for index, token in enumerate(tokens):
        if token in eos:
            return list(tokens[:index]), "eos"
    return list(tokens), "max_new_tokens"


def evaluate_mode(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, str]],
    *,
    mode: str,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    enable_thinking = mode == "thinking_on"
    examples = _prepare(rows, tokenizer, enable_thinking)
    groups: dict[int, list[PreparedExample]] = defaultdict(list)
    for example in examples:
        groups[len(example.prompt_ids)].append(example)
    eos_ids = _eos_ids(model, tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]
    generated_rows: list[dict[str, Any] | None] = [None] * len(examples)
    started = time.monotonic()
    for prompt_length, group in sorted(groups.items()):
        for offset in range(0, len(group), batch_size):
            chunk = group[offset : offset + batch_size]
            input_ids = torch.tensor([example.prompt_ids for example in chunk], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_ids[0] if len(eos_ids) == 1 else list(eos_ids),
                pad_token_id=pad_id,
                use_cache=True,
            )
            for example, token_row in zip(chunk, output[:, prompt_length:].detach().cpu().tolist(), strict=True):
                response_ids, termination = _trim_at_eos(token_row, eos_ids)
                response = tokenizer.decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                generated_rows[example.index] = {
                    "index": example.index,
                    "question": example.question,
                    "target_text": example.target_text,
                    "target_answer": str(example.target_answer),
                    "prompt_tokens": len(example.prompt_ids),
                    "response_tokens": len(response_ids),
                    "termination": termination,
                    "response": response,
                    "predicted_answer": (
                        None if extract_numeric_answer(response) is None else str(extract_numeric_answer(response))
                    ),
                    "correct": score_numeric_response(response, example.target_answer),
                }
    if any(row is None for row in generated_rows):
        raise RuntimeError("generation did not produce every GSM8K row")
    final_rows = [row for row in generated_rows if row is not None]
    correct = sum(bool(row["correct"]) for row in final_rows)
    return {
        "mode": mode,
        "enable_thinking": enable_thinking,
        "correct": correct,
        "total": len(final_rows),
        "accuracy": correct / len(final_rows),
        "elapsed_seconds": time.monotonic() - started,
        "generation": {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "grouping": "exact_prompt_token_length",
            "eos_token_ids": list(eos_ids),
            "pad_token_id": pad_id,
        },
        "rows": final_rows,
    }


def _publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite result: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def run_qwen_gsm8k(config: QwenGSM8KConfig, repository: str | Path = ".") -> dict[str, Any]:
    repository_path = Path(repository).resolve()
    if subprocess.check_output(["git", "-C", str(repository_path), "status", "--porcelain"], text=True).strip():
        raise ValueError("Qwen GSM8K evaluation requires a clean Git checkout")
    output_dir = Path(config.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen GSM8K evaluation requires CUDA")
    model, tokenizer, model_identity = _load_model(Path(config.model_path).expanduser().resolve(), device)
    rows, dataset_identity = _load_test_rows(config.limit)
    cells = {}
    for mode in config.modes:
        cells[mode] = evaluate_mode(
            model,
            tokenizer,
            rows,
            mode=mode,
            batch_size=config.batch_size,
            max_new_tokens=config.max_new_tokens,
            device=device,
        )
    result = {
        "schema": RESULT_SCHEMA,
        "source_commit": _source_commit(repository_path),
        "config": asdict(config),
        "model": model_identity,
        "dataset": dataset_identity,
        "runtime": {
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "cells": cells,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    document = {**result, "payload_sha256": hashlib.sha256(payload).hexdigest()}
    _publish_json(output_dir / "result.json", document)
    return document
