from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, TypedDict

import torch

from gcw.analysis.ablation import HeadMeanAblator, UnigramBaseline, deterministic_masks
from gcw.analysis.head_synergy import (
    DEFAULT_FAMILIES,
    DEFAULT_SHARED_ROOT,
    DEFAULT_TASKS,
    DEFAULT_VARIANTS,
    AnalysisExample,
    CheckpointRecord,
    _flatten_examples,
    calibration_batches,
    fit_baseline,
    load_examples_for_tasks,
    load_examples_jsonl,
    prepare_continuation,
    score_prepared_continuation_nll,
    select_checkpoints,
)
from gcw.checkpoint import load_checkpoint
from gcw.eval_support import tokenizer_for_checkpoint


LOSS_COLUMNS = (
    "family",
    "variant",
    "checkpoint",
    "task",
    "example_id",
    "split_id",
    "token_index",
    "target_token_id",
    "target_token_text",
    "nll_full",
    "nll_base",
    "nll_a",
    "nll_b",
    "raw_token",
    "useful_token",
    "group_fraction",
    "seed",
    "num_splits",
    "max_length",
    "max_target_tokens",
)

EXAMPLE_COLUMNS = (
    "task",
    "example_id",
    "prompt_sha256",
    "prompt",
    "target",
    "context_len",
    "target_len",
    "target_token_ids",
)

SUMMARY_COLUMNS = (
    "family",
    "variant",
    "task",
    "n_examples",
    "n_rows",
    "n_tokens",
    "mean_raw_token",
    "mean_useful_token",
    "mean_full_nll",
    "mean_base_nll",
    "mean_a_nll",
    "mean_b_nll",
)


@dataclass(frozen=True)
class RawDumpSettings:
    num_splits: int
    group_fraction: float
    max_length: int
    max_target_tokens: int
    seed: int


@dataclass(frozen=True)
class ExampleMetadata:
    task: str
    example_id: int
    prompt_sha256: str
    prompt: str
    target: str
    context_len: int
    target_len: int
    target_token_ids: str


class PreviewEntry(TypedDict):
    examples: set[tuple[str, int]]
    n_rows: int
    n_tokens: int
    raw: float
    useful: float
    full: float
    base: float
    a: float
    b: float


def _new_preview_entry() -> PreviewEntry:
    return {
        "examples": set(),
        "n_rows": 0,
        "n_tokens": 0,
        "raw": 0.0,
        "useful": 0.0,
        "full": 0.0,
        "base": 0.0,
        "a": 0.0,
        "b": 0.0,
    }


class PreviewStats:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str], PreviewEntry] = defaultdict(_new_preview_entry)

    def update(self, row: dict[str, object]) -> None:
        key = (str(row["family"]), str(row["variant"]), str(row["task"]))
        stats = self.rows[key]
        stats["examples"].add((str(row["task"]), _integer_field(row, "example_id")))
        stats["n_rows"] += 1
        stats["n_tokens"] += 1
        stats["raw"] += _numeric_field(row, "raw_token")
        stats["useful"] += _numeric_field(row, "useful_token")
        stats["full"] += _numeric_field(row, "nll_full")
        stats["base"] += _numeric_field(row, "nll_base")
        stats["a"] += _numeric_field(row, "nll_a")
        stats["b"] += _numeric_field(row, "nll_b")

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            for (family, variant, task), stats in sorted(self.rows.items()):
                n_rows = stats["n_rows"]
                examples = stats["examples"]
                writer.writerow(
                    {
                        "family": family,
                        "variant": variant,
                        "task": task,
                        "n_examples": len(examples),
                        "n_rows": n_rows,
                        "n_tokens": stats["n_tokens"],
                        "mean_raw_token": _safe_mean(stats["raw"], n_rows),
                        "mean_useful_token": _safe_mean(stats["useful"], n_rows),
                        "mean_full_nll": _safe_mean(stats["full"], n_rows),
                        "mean_base_nll": _safe_mean(stats["base"], n_rows),
                        "mean_a_nll": _safe_mean(stats["a"], n_rows),
                        "mean_b_nll": _safe_mean(stats["b"], n_rows),
                    }
                )


def raw_loss_rows_for_example(
    *,
    record: CheckpointRecord,
    model: torch.nn.Module,
    tokenizer,
    baseline: UnigramBaseline,
    ablator: HeadMeanAblator,
    example: AnalysisExample,
    settings: RawDumpSettings,
) -> tuple[ExampleMetadata, list[dict[str, object]]]:
    prepared, cont = prepare_continuation(
        model,
        tokenizer,
        example,
        max_length=settings.max_length,
        max_target_tokens=settings.max_target_tokens,
    )
    full = score_prepared_continuation_nll(model, prepared)
    base = baseline.nll(cont)
    metadata = ExampleMetadata(
        task=example.task,
        example_id=example.index,
        prompt_sha256=hashlib.sha256(example.prompt.encode("utf-8")).hexdigest(),
        prompt=example.prompt,
        target=example.target,
        context_len=prepared.ctx_len,
        target_len=len(cont),
        target_token_ids=json.dumps([int(token_id) for token_id in cont]),
    )

    rows = []
    for split_id in range(settings.num_splits):
        effective_seed = settings.seed + example.index * 31
        mask_a, mask_b = deterministic_masks(
            ablator,
            split_id=split_id,
            seed=effective_seed,
            group_fraction=settings.group_fraction,
        )
        loss_a = score_prepared_continuation_nll(model, prepared, ablator=ablator, masks=mask_a)
        loss_b = score_prepared_continuation_nll(model, prepared, ablator=ablator, masks=mask_b)
        for token_index, token_id in enumerate(cont):
            nll_full = float(full[token_index].item())
            nll_base = float(base[token_index].item())
            nll_a = float(loss_a[token_index].item())
            nll_b = float(loss_b[token_index].item())
            rows.append(
                {
                    "family": record.family,
                    "variant": record.variant,
                    "checkpoint": record.path,
                    "task": example.task,
                    "example_id": example.index,
                    "split_id": split_id,
                    "token_index": token_index,
                    "target_token_id": int(token_id),
                    "target_token_text": _decode_token(tokenizer, int(token_id)),
                    "nll_full": nll_full,
                    "nll_base": nll_base,
                    "nll_a": nll_a,
                    "nll_b": nll_b,
                    "raw_token": (nll_a + nll_b) - (nll_full + nll_base),
                    "useful_token": nll_base - nll_full,
                    "group_fraction": settings.group_fraction,
                    "seed": effective_seed,
                    "num_splits": settings.num_splits,
                    "max_length": settings.max_length,
                    "max_target_tokens": settings.max_target_tokens,
                }
            )
    return metadata, rows


def dump_record(
    record: CheckpointRecord,
    examples_by_task: dict[str, list[AnalysisExample]],
    calibration_by_task: dict[str, list[AnalysisExample]],
    *,
    losses_writer: csv.DictWriter,
    examples_writer: csv.DictWriter,
    stats: PreviewStats,
    seen_examples: set[tuple[str, int]],
    family_dir: Path,
    settings: RawDumpSettings,
    num_calibration: int,
    device: str,
) -> None:
    model, cfg, state = load_checkpoint(record.path, device=device)
    model.eval()
    tokenizer = tokenizer_for_checkpoint(cfg, state)
    ablator = HeadMeanAblator(model)
    calibration = _flatten_examples(calibration_by_task, num_calibration)
    baseline = fit_baseline(tokenizer, calibration, cfg.vocab_size)
    ablator.capture(
        calibration_batches(
            model,
            tokenizer,
            calibration,
            max_length=settings.max_length,
            max_target_tokens=settings.max_target_tokens,
        )
    )

    try:
        for task, examples in examples_by_task.items():
            _write_progress(family_dir, record, task, "running")
            for example in examples:
                metadata, rows = raw_loss_rows_for_example(
                    record=record,
                    model=model,
                    tokenizer=tokenizer,
                    baseline=baseline,
                    ablator=ablator,
                    example=example,
                    settings=settings,
                )
                example_key = (metadata.task, metadata.example_id)
                if example_key not in seen_examples:
                    examples_writer.writerow(asdict(metadata))
                    seen_examples.add(example_key)
                for row in rows:
                    losses_writer.writerow(row)
                    stats.update(row)
            _write_progress(family_dir, record, task, "completed")
    finally:
        ablator.remove()
        del model
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def dump_family(
    family: str,
    records: list[CheckpointRecord],
    examples_by_task: dict[str, list[AnalysisExample]],
    calibration_by_task: dict[str, list[AnalysisExample]],
    *,
    out_dir: Path,
    settings: RawDumpSettings,
    num_calibration: int,
    device: str,
    root: str,
    tasks: list[str],
) -> None:
    family_dir = out_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": root,
        "family": family,
        "tasks": tasks,
        "checkpoints": [asdict(record) for record in records],
        "settings": asdict(settings) | {"num_calibration": num_calibration, "device": device},
    }
    (family_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    stats = PreviewStats()
    seen_examples: set[tuple[str, int]] = set()
    with gzip.open(family_dir / "losses.csv.gz", "wt", newline="", encoding="utf-8") as loss_f, (
        family_dir / "examples.csv"
    ).open("w", newline="", encoding="utf-8") as example_f:
        losses_writer = csv.DictWriter(loss_f, fieldnames=LOSS_COLUMNS)
        examples_writer = csv.DictWriter(example_f, fieldnames=EXAMPLE_COLUMNS)
        losses_writer.writeheader()
        examples_writer.writeheader()
        for record in records:
            _write_progress(family_dir, record, None, "running")
            dump_record(
                record,
                examples_by_task,
                calibration_by_task,
                losses_writer=losses_writer,
                examples_writer=examples_writer,
                stats=stats,
                seen_examples=seen_examples,
                family_dir=family_dir,
                settings=settings,
                num_calibration=num_calibration,
                device=device,
            )
            stats.write(family_dir / "summary_preview.csv")
            _write_progress(family_dir, record, None, "completed")

    stats.write(family_dir / "summary_preview.csv")
    _write_progress(family_dir, records[-1], None, "completed")


def run(args: argparse.Namespace) -> None:
    checkpoints = select_checkpoints(args.root, args.families, args.variants)
    if args.limit_checkpoints is not None:
        checkpoints = checkpoints[: args.limit_checkpoints]

    if args.dry_run:
        print(json.dumps(_dry_run_payload(args, checkpoints), indent=2))
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.examples_jsonl:
        examples_by_task, calibration_by_task = load_examples_jsonl(args.examples_jsonl, args.tasks, args.num_examples)
    else:
        examples_by_task, calibration_by_task = load_examples_for_tasks(
            args.tasks,
            limit=args.num_examples,
            calibration_limit=args.num_calibration,
        )

    settings = RawDumpSettings(
        num_splits=args.num_splits,
        group_fraction=args.group_fraction,
        max_length=args.max_length,
        max_target_tokens=args.max_target_tokens,
        seed=args.seed,
    )
    if len(args.families) > 1:
        root_manifest = {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "root": args.root,
            "tasks": args.tasks,
            "families": args.families,
            "variants": args.variants,
            "checkpoints": [asdict(record) for record in checkpoints],
            "settings": {key: value for key, value in vars(args).items() if key != "dry_run"},
        }
        (out_dir / "manifest.json").write_text(json.dumps(root_manifest, indent=2), encoding="utf-8")

    for family, records in _records_by_family(checkpoints).items():
        dump_family(
            family,
            records,
            examples_by_task,
            calibration_by_task,
            out_dir=out_dir,
            settings=settings,
            num_calibration=args.num_calibration,
            device=args.device,
            root=args.root,
            tasks=args.tasks,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump raw token NLLs for attention-head synergy analysis.")
    parser.add_argument("--root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--num-examples", type=int, default=256)
    parser.add_argument("--num-calibration", type=int, default=512)
    parser.add_argument("--num-splits", type=int, default=64)
    parser.add_argument("--group-fraction", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-target-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-checkpoints", type=int, default=None)
    parser.add_argument("--examples-jsonl", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.out_dir:
        run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        args.out_dir = str(DEFAULT_SHARED_ROOT / "results" / "head_loss_dump" / run_id)
    run(args)


def _records_by_family(records: Iterable[CheckpointRecord]) -> dict[str, list[CheckpointRecord]]:
    grouped: dict[str, list[CheckpointRecord]] = defaultdict(list)
    for record in records:
        grouped[record.family].append(record)
    return dict(grouped)


def _dry_run_payload(args: argparse.Namespace, checkpoints: list[CheckpointRecord]) -> dict[str, object]:
    jobs = [
        {
            "array_index": idx,
            "family": family,
            "checkpoints": [asdict(record) for record in records],
        }
        for idx, (family, records) in enumerate(_records_by_family(checkpoints).items())
    ]
    return {
        "out_dir": args.out_dir,
        "tasks": args.tasks,
        "family_jobs": jobs,
        "checkpoints": [asdict(record) for record in checkpoints],
    }


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return str(tokenizer.decode([token_id]))
    except Exception:
        return ""


def _write_progress(family_dir: Path, record: CheckpointRecord, task: str | None, status: str) -> None:
    payload = {
        "status": status,
        "family": record.family,
        "variant": record.variant,
        "checkpoint": record.path,
        "task": task,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (family_dir / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _numeric_field(row: dict[str, object], key: str) -> float:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric, got {type(value).__name__}")
    return float(value)


def _integer_field(row: dict[str, object], key: str) -> int:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer, got {type(value).__name__}")
    return value


def _safe_mean(total: float, count: int) -> float:
    return total / count if count else float("nan")


if __name__ == "__main__":
    main()
