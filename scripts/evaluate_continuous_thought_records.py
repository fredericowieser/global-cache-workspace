#!/usr/bin/env python3
"""Evaluate continuous-thought emitter checkpoints by exact-answer generation."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcw.checkpoint import load_checkpoint  # noqa: E402
from gcw.continuous_thought import (  # noqa: E402
    ContinuousThoughtEmitter,
    continuous_thought_answer_loss,
    frozen_qwen_backbone,
    generate_continuous_thought_answer,
    load_emitter_checkpoint,
)
from gcw.data.tokenizers import get_tokenizer  # noqa: E402
from gcw.process_branch import InvalidProcessBranchExample, tokenize_answer_only_record  # noqa: E402
from gcw.process_data import AnswerOnlyRecord, read_answer_only_records_bytes  # noqa: E402
from gcw.process_eval import extract_numeric_answer  # noqa: E402


HELDOUT_START = 6000
HELDOUT_STOP = 6256
HELDOUT_ROWS = HELDOUT_STOP - HELDOUT_START
DEFAULT_BACKBONE = Path("artifacts/checkpoints/base_backbones/qwen3_06b/tf-v1")
DEFAULT_EMITTER = Path(
    "artifacts/experiments/explore/continuous-thought/epoch-k2-l2-lr3e4/emitter-checkpoint"
)
DEFAULT_TOKENIZER = Path("artifacts/models/Qwen3-0.6B-base-v1")
DEFAULT_DATA = Path("data/process_branch/gsm8k-answer-only-v1.jsonl")
DEFAULT_MANIFEST = Path("data/process_branch/gsm8k-answer-only-v1.manifest.json")
CONDITION_ORDER = ("k0", "matched", "shuffled")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_identity(path: Path, required: Sequence[str] | None = None) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    if required is not None:
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise ValueError(f"artifact directory {path} is missing {missing}")
        candidates = [path / name for name in required]
    else:
        candidates = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    files = {
        candidate.relative_to(path).as_posix(): {
            "sha256": _sha256_file(candidate),
            "size_bytes": candidate.stat().st_size,
        }
        for candidate in candidates
    }
    if not files:
        raise ValueError(f"artifact directory {path} is empty")
    return {
        "path": str(path.resolve()),
        "files": files,
        "manifest_sha256": _sha256_bytes(_canonical_json_bytes(files)),
    }


def extract_canonical_answer(text: str) -> str | None:
    """Extract the final balanced boxed number, or final visible number exactly."""
    value = extract_numeric_answer(text)
    if value is None:
        return None
    if not isinstance(value, Fraction):
        raise TypeError("numeric answer parser returned a non-Fraction")
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def exact_answers_match(prediction: str, target: str) -> bool:
    predicted = extract_canonical_answer(prediction)
    expected = extract_canonical_answer(target)
    return predicted is not None and expected is not None and predicted == expected


def canonical_jsonl_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)


def write_canonical_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    payload = canonical_jsonl_bytes(rows)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(payload)


def _authenticate_records(data: Path, manifest_path: Path) -> tuple[list[AnswerOnlyRecord], dict[str, Any]]:
    source_bytes = data.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("kind") != "gcw.answer_only.gsm8k.v1":
        raise ValueError("answer-only manifest schema is invalid")
    observed_sha256 = _sha256_bytes(source_bytes)
    if observed_sha256 != manifest.get("transformed_sha256"):
        raise ValueError("answer-only data sha256 differs from its manifest")
    records = read_answer_only_records_bytes(source_bytes)
    ordered_ids = [record.source_id for record in records]
    ids_sha256 = _sha256_bytes(_canonical_json_bytes(ordered_ids))
    if (
        len(records) != manifest.get("accepted_rows")
        or ids_sha256 != manifest.get("ordered_source_ids_sha256")
    ):
        raise ValueError("answer-only records differ from the authenticated manifest population")
    selected = records[HELDOUT_START:HELDOUT_STOP]
    if len(selected) != HELDOUT_ROWS:
        raise ValueError(f"authenticated data does not contain rows {HELDOUT_START}:{HELDOUT_STOP}")
    expected_ids = [f"gsm8k-train:{index}" for index in range(HELDOUT_START, HELDOUT_STOP)]
    if [record.source_id for record in selected] != expected_ids:
        raise ValueError("heldout interval source IDs are not the canonical GSM8K row interval")
    selected_payload = [
        {"source_id": row.source_id, "prompt": row.prompt, "final_answer": row.final_answer}
        for row in selected
    ]
    return selected, {
        "path": str(data.resolve()),
        "sha256": observed_sha256,
        "size_bytes": len(source_bytes),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "population_rows": len(records),
        "ordered_source_ids_sha256": ids_sha256,
        "heldout_start": HELDOUT_START,
        "heldout_stop": HELDOUT_STOP,
        "heldout_rows": len(selected),
        "heldout_source_ids_sha256": _sha256_bytes(_canonical_json_bytes(expected_ids)),
        "heldout_records_sha256": _sha256_bytes(_canonical_json_bytes(selected_payload)),
    }


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def _autocast_factory(device: torch.device, backbone: torch.nn.Module):
    use_bfloat16 = device.type == "cuda" and next(backbone.parameters()).dtype == torch.bfloat16

    def context():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()

    return context


@torch.no_grad()
def _prompt_source(backbone, prompt_ids: torch.Tensor) -> torch.Tensor:
    hidden, _ = backbone.forward_chunk(
        prompt_ids.unsqueeze(0), 0, None, logits=False, cache_format="windowed"
    )
    return hidden[:, -1].detach()


def _condition_result(
    backbone,
    emitter: ContinuousThoughtEmitter,
    tokenizer,
    example,
    *,
    cycles: int,
    source_override: torch.Tensor | None,
    source_id: str | None,
    target: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    generated = generate_continuous_thought_answer(
        backbone,
        emitter,
        prompt_ids=example.prompt_ids,
        response_prefix_ids=example.answer_response_prefix_ids,
        cycles=cycles,
        max_new_tokens=max_new_tokens,
        eos_token_id=int(tokenizer.eot_id),
        valid_vocab_size=tokenizer.get_vocab_size(),
        source_override=source_override,
    )
    loss = continuous_thought_answer_loss(
        backbone,
        emitter,
        prompt_ids=example.prompt_ids.unsqueeze(0),
        response_prefix_ids=example.answer_response_prefix_ids.unsqueeze(0),
        target_ids=example.final_answer_ids.unsqueeze(0),
        cycles=cycles,
        source_override=source_override,
    ).loss
    token_ids = [int(token) for token in generated.token_ids.tolist()]
    prediction = tokenizer.decode(token_ids)
    ended_by_eos = bool(token_ids and token_ids[-1] == int(tokenizer.eot_id))
    return {
        "answer_start_position": generated.answer_start_position,
        "continuous_positions": generated.trace.added_positions,
        "correct": exact_answers_match(prediction, target),
        "cycles": cycles,
        "emitter_source_id": source_id,
        "ended_by_eos": ended_by_eos,
        "extracted_answer": extract_canonical_answer(prediction),
        "generated_token_ids": token_ids,
        "generated_tokens": len(token_ids),
        "loss": float(loss.float()),
        "prediction": prediction,
        "truncated": len(token_ids) == max_new_tokens and not ended_by_eos,
    }


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in CONDITION_ORDER:
        values = [row["conditions"][condition] for row in rows]
        correct = sum(int(value["correct"]) for value in values)
        output[condition] = {
            "accuracy": correct / len(values),
            "correct": correct,
            "mean_generated_tokens": sum(value["generated_tokens"] for value in values) / len(values),
            "mean_loss": sum(value["loss"] for value in values) / len(values),
            "rows": len(values),
            "truncated": sum(int(value["truncated"]) for value in values),
        }
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Evaluate exact answers on authenticated rows {HELDOUT_START}:{HELDOUT_STOP}."
    )
    parser.add_argument("--backbone-checkpoint", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--emitter-checkpoint", type=Path, default=DEFAULT_EMITTER)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=None, help="must match the checkpoint-recorded cycle count")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-answer-tokens", type=int, default=64)
    parser.add_argument("--prompt-suffix", default="Return only the final answer in a box.")
    parser.add_argument("--answer-prefix", default="\n\nAnswer: ")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("checkpoint", "float32", "bfloat16"), default="checkpoint")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_new_tokens <= 0 or args.max_prompt_tokens <= 0 or args.max_answer_tokens <= 0:
        raise ValueError("token limits must be positive")
    for path in (
        args.backbone_checkpoint,
        args.emitter_checkpoint,
        args.tokenizer,
        args.data,
        args.data_manifest,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    started = time.perf_counter()
    records, data_identity = _authenticate_records(args.data, args.data_manifest)
    device = _device(args.device)
    loaded_model, model_config, backbone_train_state = load_checkpoint(
        args.backbone_checkpoint, device=device
    )
    backbone = frozen_qwen_backbone(loaded_model)
    if args.dtype == "float32":
        backbone.float()
    elif args.dtype == "bfloat16":
        backbone.to(dtype=torch.bfloat16)
    emitter, emitter_train_state = load_emitter_checkpoint(args.emitter_checkpoint, device=device)
    emitter.requires_grad_(False)
    emitter.eval()
    if emitter.config.model_width != backbone.config.hidden_size:
        raise ValueError("emitter and backbone model widths differ")
    checkpoint_k = emitter_train_state.get("k")
    if isinstance(checkpoint_k, bool) or not isinstance(checkpoint_k, int) or checkpoint_k < 0:
        raise ValueError("emitter checkpoint has no valid recorded k")
    cycles = checkpoint_k if args.k is None else args.k
    if cycles != checkpoint_k:
        raise ValueError("--k must match the emitter checkpoint's recorded k")
    if any(parameter.requires_grad for parameter in loaded_model.parameters()):
        raise RuntimeError("backbone model was not fully frozen")

    tokenizer = get_tokenizer(f"hf:{args.tokenizer}")
    if tokenizer.get_vocab_size() > backbone.config.vocab_size:
        raise ValueError("tokenizer vocabulary exceeds backbone vocabulary")
    examples = []
    for record in records:
        try:
            examples.append(
                tokenize_answer_only_record(
                    record,
                    tokenizer,
                    answer_response_prefix=args.answer_prefix,
                    prompt_suffix=args.prompt_suffix,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_answer_tokens=args.max_answer_tokens,
                    device=device,
                )
            )
        except InvalidProcessBranchExample as exc:
            raise ValueError(f"authenticated heldout row {record.source_id} is ineligible: {exc}") from exc
    context = _autocast_factory(device, backbone)
    with torch.inference_mode(), context():
        sources = [_prompt_source(backbone, example.prompt_ids) for example in examples]
        rows: list[dict[str, Any]] = []
        for row_offset, (record, example) in enumerate(zip(records, examples, strict=True)):
            donor_offset = (row_offset + 1) % len(records)
            donor = records[donor_offset]
            conditions = {
                "k0": _condition_result(
                    backbone,
                    emitter,
                    tokenizer,
                    example,
                    cycles=0,
                    source_override=None,
                    source_id=None,
                    target=record.final_answer,
                    max_new_tokens=args.max_new_tokens,
                ),
                "matched": _condition_result(
                    backbone,
                    emitter,
                    tokenizer,
                    example,
                    cycles=cycles,
                    source_override=None,
                    source_id=record.source_id,
                    target=record.final_answer,
                    max_new_tokens=args.max_new_tokens,
                ),
                "shuffled": _condition_result(
                    backbone,
                    emitter,
                    tokenizer,
                    example,
                    cycles=cycles,
                    source_override=sources[donor_offset],
                    source_id=donor.source_id,
                    target=record.final_answer,
                    max_new_tokens=args.max_new_tokens,
                ),
            }
            rows.append(
                {
                    "conditions": conditions,
                    "row_index": HELDOUT_START + row_offset,
                    "schema": "gcw.continuous_thought.record_eval.row.v1",
                    "source_id": record.source_id,
                    "target": record.final_answer,
                    "target_answer": extract_canonical_answer(record.final_answer),
                }
            )
            if (row_offset + 1) % 16 == 0:
                print(f"evaluated {row_offset + 1}/{len(records)} rows", file=sys.stderr, flush=True)

    elapsed = time.perf_counter() - started
    identities = {
        "backbone": _directory_identity(
            args.backbone_checkpoint, ("config.json", "model.safetensors", "train_state.json")
        ),
        "data": data_identity,
        "emitter": _directory_identity(
            args.emitter_checkpoint, ("config.json", "emitter.safetensors", "train_state.json")
        ),
        "tokenizer": _directory_identity(args.tokenizer),
    }
    summary: dict[str, Any] = {
        "aggregate": _aggregate(rows),
        "answer_prefix": args.answer_prefix,
        "backbone_config": asdict(model_config),
        "backbone_frozen_parameters": sum(
            parameter.numel() for parameter in backbone.parameters() if not parameter.requires_grad
        ),
        "backbone_train_state": backbone_train_state,
        "cycles": cycles,
        "identities": identities,
        "latent_tokens_per_cycle": emitter.config.latent_tokens_per_cycle,
        "max_new_tokens": args.max_new_tokens,
        "prompt_suffix": args.prompt_suffix,
        "row_count": len(rows),
        "runtime": {
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "elapsed_seconds": elapsed,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
        },
        "schema": "gcw.continuous_thought.record_eval.summary.v1",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    rows_sha256 = write_canonical_jsonl(args.output / "rows.jsonl", rows)
    summary["rows_jsonl"] = {
        "path": str((args.output / "rows.jsonl").resolve()),
        "sha256": rows_sha256,
        "size_bytes": (args.output / "rows.jsonl").stat().st_size,
    }
    summary_bytes = json.dumps(
        summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    with (args.output / "summary.json").open("xb") as handle:
        handle.write(summary_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"aggregate": summary["aggregate"], "output": str(args.output), "runtime": summary["runtime"]}, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
