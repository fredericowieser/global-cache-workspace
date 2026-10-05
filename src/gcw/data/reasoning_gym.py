from __future__ import annotations

import argparse
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import sys
import time

from gcw.data._staging import (
    PublishedDatasetCleanupError,
    dataset_staging,
    publish_dataset,
)
from gcw.data.reasoning_gym_traces import (
    AdmissionError,
    FAMILIES,
    MAX_TRANSITIONS,
    TASKS,
    create_dataset,
    render_trace,
)
from gcw.data.tokenizers import get_tokenizer

SCHEMA = "gcw.reasoning_gym.sft.v1"
DEFAULT_ROWS = 700032
SEED_STRIDE = 100_000_000
SPLITS = ("train", "validation", "test")
ADMISSION_POLICY = {
    "manipulate_matrix": {
        "min_final_rows": 2,
        "min_final_cols": 2,
        "min_final_cells": 6,
        "min_productive_fraction": 0.75,
        "require_nonconstant": True,
    },
    "gcd": {"min_divisions": 4},
}
VENDOR = Path(__file__).parent / "_vendor" / "reasoning_gym"
VENDOR_SOURCES_SHA256 = (
    "2fde036af72a453229953f387dab1dc0539e5178fff4c7d5ee051c050a820227"
)


def _generate_batch(job):
    task, split, seed, start, count, min_steps = job
    dataset = create_dataset(task, seed=seed, size=start + count)
    rows = []
    rejected = Counter()
    for index in range(start, start + count):
        try:
            entry = dataset[index]
            trace = render_trace(task, dataset, entry)
            if trace.steps < min_steps:
                raise AdmissionError("insufficient_trace_depth")
            if task == "manipulate_matrix":
                policy = ADMISSION_POLICY[task]
                shape = trace.difficulty
                if (
                    shape["final_rows"] < policy["min_final_rows"]
                    or shape["final_cols"] < policy["min_final_cols"]
                    or shape["final_rows"] * shape["final_cols"]
                    < policy["min_final_cells"]
                ):
                    raise AdmissionError("matrix_final_size")
                if shape["distinct_cells"] < 2:
                    raise AdmissionError("matrix_constant_answer")
                if (
                    trace.steps
                    < policy["min_productive_fraction"] * trace.rendered_steps
                ):
                    raise AdmissionError("matrix_unproductive_operations")
            elif (
                task == "gcd"
                and trace.steps < ADMISSION_POLICY[task]["min_divisions"]
            ):
                raise AdmissionError("gcd_insufficient_divisions")
        except AdmissionError as exc:
            rejected[str(exc)] += 1
            continue
        prompt_hash = hashlib.sha256(
            " ".join(trace.prompt.split()).encode()
        ).hexdigest()
        problem_hash = hashlib.sha256(
            json.dumps([task, trace.problem], sort_keys=True).encode()
        ).hexdigest()
        rows.append(
            {
                "prompt": trace.prompt,
                "answer": "Worked solution:\n"
                + trace.solution
                + "\n\nFinal answer:\n<answer>"
                + trace.final_answer
                + "</answer>",
                "final_answer": trace.final_answer,
                "task": task,
                "family": FAMILIES[task],
                "source_id": f"{task}:{seed}:{index}",
                "split": split,
                "generator_seed": seed,
                "source_index": index,
                "effective_seed": seed + index,
                "prompt_sha256": prompt_hash,
                "problem_sha256": problem_hash,
                "trace_steps": trace.steps,
                "rendered_steps": trace.rendered_steps,
                "termination": trace.termination,
                "difficulty": trace.difficulty,
                "enable_thinking": False,
            }
        )
    return rows, rejected


def _lengths(tokenizer, rows):
    encoder = getattr(tokenizer, "enc", None)
    if callable(encoder):
        prompts = encoder.apply_chat_template(
            [[{"role": "user", "content": row["prompt"]}] for row in rows],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
        )["input_ids"]
        answers = encoder(
            [row["answer"] for row in rows],
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
    else:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in rows
        ]
        answers = [tokenizer.encode(row["answer"]) for row in rows]
    return [(len(p), len(a) + 1) for p, a in zip(prompts, answers, strict=True)]


def _histogram_summary(histogram):
    count = sum(histogram.values())
    if not count:
        return {"count": 0}
    ordered = sorted(histogram.items())
    result = {
        "count": count,
        "min": ordered[0][0],
        "max": ordered[-1][0],
        "total": sum(k * v for k, v in ordered),
    }
    result["mean"] = result["total"] / count
    for name, quantile in [("p50", 0.5), ("p95", 0.95), ("p99", 0.99)]:
        cumulative = 0
        for value, frequency in ordered:
            cumulative += frequency
            if cumulative >= count * quantile:
                result[name] = value
                break
    return result


def prepare_dataset(
    output: Path,
    tokenizer,
    *,
    train_rows: int = DEFAULT_ROWS,
    validation_rows: int = 6000,
    test_rows: int = 6000,
    sequence_length: int = 4096,
    seed: int = 24,
    min_steps: int = 2,
    attempts_per_row: int = 20,
    workers: int = 1,
    batch_size: int = 256,
    tokenizer_source: str = "",
    replace: bool = False,
) -> dict:
    output = Path(output)
    if output.exists() and not replace:
        raise FileExistsError(f"Dataset destination already exists: {output}")
    counts = dict(zip(SPLITS, (train_rows, validation_rows, test_rows)))
    if train_rows <= 0 or min(counts.values()) < 0:
        raise ValueError(
            "Training rows must be positive and holdout rows nonnegative"
        )
    if (
        min(sequence_length, min_steps, attempts_per_row, workers, batch_size)
        <= 0
        or seed < 0
    ):
        raise ValueError("Limits must be positive and seed must be nonnegative")
    if max(counts.values()) * attempts_per_row >= SEED_STRIDE:
        raise ValueError(
            "Attempt budget exceeds the reserved effective seed range"
        )
    source_bytes = (VENDOR / "SOURCES.json").read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != VENDOR_SOURCES_SHA256:
        raise ValueError("Bundled SOURCES.json differs from its pinned hash")
    sources = json.loads(source_bytes)
    for name, expected in sources["supporting_files"].items():
        if hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"Bundled support file differs from pinned source: {name}"
            )
    for name, expected in sources["files"].items():
        if (
            hashlib.sha256((VENDOR / name).read_bytes()).hexdigest()
            != expected["adapted_sha256"]
        ):
            raise ValueError(
                f"Bundled generator differs from pinned source: {name}"
            )
    manifest = {
        "schema": SCHEMA,
        "source": sources,
        "source_manifest_sha256": VENDOR_SOURCES_SHA256,
        "implementation_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("reasoning_gym_traces.py"),
                Path(__file__).with_name("_staging.py"),
            )
        },
        "seed": seed,
        "requested_rows": counts,
        "sequence_length": sequence_length,
        "min_trace_steps": min_steps,
        "admission_policy": ADMISSION_POLICY,
        "max_string_transitions": MAX_TRANSITIONS,
        "attempts_per_row": attempts_per_row,
        "batch_size": batch_size,
        "tokenizer": tokenizer_source,
        "tokenizer_identity_sha256": tokenizer.identity_sha256(),
        "runtime": {
            "python": sys.version,
            **{
                name: version(name)
                for name in ("numpy", "transformers", "tokenizers")
            },
        },
        "format": {
            "sft_chat_template": True,
            "sft_enable_thinking": False,
            "sft_loss_scope": "answer_only",
            "response_prefix": "",
            "prompt_suffix": "",
            "final_answer_delimiter": "<answer>...</answer>",
            "truncation": "reject",
        },
        "split_policy": "Disjoint effective seed ranges; global prompt and task-input deduplication. Validation and test are fresh instances of the training mixture, not family or composition holdouts.",
        "shuffle": "Python Random(seed + split_index); deterministic permutation of accepted JSONL offsets",
        "tasks": {},
        "files": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_prompts, seen_problems = set(), set()
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        with dataset_staging(output) as temporary:
            staging = Path(temporary) / "dataset"
            staging.mkdir()
            for split_index, (split, requested) in enumerate(counts.items()):
                offsets = array("Q")
                raw_path = Path(temporary) / f"{split}.raw.jsonl"
                states = {}
                for task_index, task in enumerate(TASKS):
                    quota = requested // len(TASKS) + (
                        task_index < requested % len(TASKS)
                    )
                    task_seed = (
                        seed
                        + (task_index * len(SPLITS) + split_index) * SEED_STRIDE
                    )
                    dataset = create_dataset(
                        task,
                        seed=task_seed,
                        size=max(1, quota * attempts_per_row),
                    )
                    stats = {
                        "requested": quota,
                        "retained": 0,
                        "attempted": 0,
                        "rejections": Counter(),
                        "termination": Counter(),
                        "difficulty": Counter(),
                        "sequence_tokens": Counter(),
                        "target_tokens": Counter(),
                        "trace_steps": Counter(),
                        "rendered_steps": Counter(),
                        "generator_config": asdict(dataset.config),
                    }
                    states[task] = stats
                    manifest["tasks"].setdefault(task, {})[split] = stats
                last_progress = time.monotonic()
                with raw_path.open("wb") as raw:
                    while any(
                        s["retained"] < s["requested"] for s in states.values()
                    ):
                        jobs = []
                        for task, stats in states.items():
                            remaining = stats["requested"] - stats["retained"]
                            if remaining <= 0:
                                continue
                            budget = stats["requested"] * attempts_per_row
                            size = min(batch_size, budget - stats["attempted"])
                            if size <= 0:
                                raise RuntimeError(
                                    f'{split}/{task}: retained {stats["retained"]}/{stats["requested"]} after {budget} attempts; refusing to publish an undersized mixture'
                                )
                            jobs.append(
                                (
                                    task,
                                    split,
                                    stats["generator_config"]["seed"],
                                    stats["attempted"],
                                    size,
                                    min_steps,
                                )
                            )
                        results = (
                            executor.map(_generate_batch, jobs)
                            if executor
                            else map(_generate_batch, jobs)
                        )
                        for job, (candidates, rejected) in zip(
                            jobs, results, strict=True
                        ):
                            stats = states[job[0]]
                            stats["attempted"] += job[4]
                            stats["rejections"].update(rejected)
                            unique = []
                            for row in candidates:
                                if (
                                    row["prompt_sha256"] in seen_prompts
                                    or row["problem_sha256"] in seen_problems
                                ):
                                    stats["rejections"][
                                        "duplicate_prompt_or_input"
                                    ] += 1
                                else:
                                    unique.append(row)
                            lengths = (
                                _lengths(tokenizer, unique) if unique else []
                            )
                            for row, (prompt_tokens, target_tokens) in zip(
                                unique, lengths, strict=True
                            ):
                                if stats["retained"] == stats["requested"]:
                                    stats["rejections"]["quota_filled"] += 1
                                    continue
                                if (
                                    prompt_tokens + target_tokens
                                    > sequence_length
                                ):
                                    stats["rejections"]["over_context"] += 1
                                    continue
                                if (
                                    row["prompt_sha256"] in seen_prompts
                                    or row["problem_sha256"] in seen_problems
                                ):
                                    stats["rejections"][
                                        "duplicate_prompt_or_input"
                                    ] += 1
                                    continue
                                row.update(
                                    prompt_tokens=prompt_tokens,
                                    target_tokens=target_tokens,
                                    sequence_tokens=prompt_tokens
                                    + target_tokens,
                                )
                                seen_prompts.add(row["prompt_sha256"])
                                seen_problems.add(row["problem_sha256"])
                                offsets.append(raw.tell())
                                raw.write(
                                    (
                                        json.dumps(
                                            row,
                                            ensure_ascii=False,
                                            sort_keys=True,
                                        )
                                        + "\n"
                                    ).encode()
                                )
                                stats["retained"] += 1
                                stats["termination"][row["termination"]] += 1
                                stats["difficulty"][
                                    json.dumps(
                                        row["difficulty"], sort_keys=True
                                    )
                                ] += 1
                                for field in (
                                    "sequence_tokens",
                                    "target_tokens",
                                    "trace_steps",
                                    "rendered_steps",
                                ):
                                    stats[field][row[field]] += 1
                        if time.monotonic() - last_progress >= 20:
                            print(
                                f"{split}: {len(offsets):,}/{requested:,} admitted",
                                file=sys.stderr,
                                flush=True,
                            )
                            last_progress = time.monotonic()
                random.Random(seed + split_index).shuffle(offsets)
                digest, source_ids = hashlib.sha256(), hashlib.sha256()
                path = staging / f"{split}.jsonl"
                with raw_path.open("rb") as raw, path.open("wb") as handle:
                    for order, offset in enumerate(offsets):
                        raw.seek(offset)
                        row = json.loads(raw.readline())
                        row["experiment_order"] = order
                        payload = (
                            json.dumps(row, sort_keys=True, ensure_ascii=False)
                            + "\n"
                        ).encode()
                        handle.write(payload)
                        digest.update(payload)
                        source_ids.update(
                            ("\n" if order else "").encode()
                            + row["source_id"].encode()
                        )
                manifest["files"][split] = {
                    "file": path.name,
                    "rows": len(offsets),
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                    "source_ids_sha256": source_ids.hexdigest(),
                }
                for stats in states.values():
                    stats["effective_seed_range_attempted"] = [
                        stats["generator_config"]["seed"],
                        stats["generator_config"]["seed"] + stats["attempted"],
                    ]
                    for field in (
                        "sequence_tokens",
                        "target_tokens",
                        "trace_steps",
                        "rendered_steps",
                    ):
                        stats[field] = _histogram_summary(stats[field])
                    if stats["attempted"] != stats["retained"] + sum(
                        stats["rejections"].values()
                    ):
                        raise AssertionError("Admission accounting mismatch")
                print(
                    f"{split}: wrote {len(offsets):,} rows ({path.stat().st_size:,} bytes)",
                    file=sys.stderr,
                    flush=True,
                )
            manifest["rows"] = train_rows
            manifest["dataset_sha256"] = manifest["files"]["train"]["sha256"]
            manifest["source_ids_sha256"] = manifest["files"]["train"][
                "source_ids_sha256"
            ]
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            publish_dataset(staging, output, replace=replace)
    except PublishedDatasetCleanupError:
        raise
    except Exception as exc:
        failure = output.with_name(output.name + ".failed.json")
        failure.write_text(
            json.dumps(
                {"error": str(exc), "report": manifest},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        if executor:
            executor.shutdown(cancel_futures=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare frozen Reasoning Gym worked-solution SFT data"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/train-data/reasoning-gym")
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Local HF tokenizer directory or repository ID; use a pinned local snapshot for reproducibility",
    )
    parser.add_argument("--train-rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--validation-rows", type=int, default=6000)
    parser.add_argument("--test-rows", type=int, default=6000)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--min-steps", type=int, default=2)
    parser.add_argument("--attempts-per-row", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace a prepared dataset only after the complete build succeeds",
    )
    args = parser.parse_args()
    tokenizer = get_tokenizer("hf:" + args.tokenizer)
    manifest = prepare_dataset(
        args.out_dir,
        tokenizer,
        train_rows=args.train_rows,
        validation_rows=args.validation_rows,
        test_rows=args.test_rows,
        sequence_length=args.sequence_length,
        seed=args.seed,
        min_steps=args.min_steps,
        attempts_per_row=args.attempts_per_row,
        workers=args.workers,
        batch_size=args.batch_size,
        tokenizer_source=args.tokenizer,
        replace=args.replace,
    )
    print(
        json.dumps(
            {"output": str(args.out_dir), "files": manifest["files"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
