from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from gcw.data._staging import dataset_staging, publish_dataset


def prepare_subset(
    source: Path,
    destination: Path,
    rows: int,
    seed: int,
    sequence_len: int = 1536,
    *,
    replace: bool = False,
):
    if destination.exists() and not replace:
        raise FileExistsError(destination)
    if rows <= 0:
        raise ValueError("rows must be positive")
    if sequence_len <= 0:
        raise ValueError("sequence_len must be positive")
    source_manifest_bytes = (source / "manifest.json").read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    source_file = source / source_manifest["files"]["train"]["file"]
    offsets = defaultdict(list)
    digest = hashlib.sha256()
    with source_file.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            digest.update(line)
            row = json.loads(line)
            if row["split"] != "train":
                raise ValueError("Source contains a non-training row")
            offsets[row["task"]].append(offset)
    source_count = sum(map(len, offsets.values()))
    if (
        digest.hexdigest() != source_manifest["files"]["train"]["sha256"]
        or source_count != source_manifest["files"]["train"]["rows"]
    ):
        raise ValueError("Training source does not match its manifest")
    if rows > source_count:
        raise ValueError("Cannot sample more rows than the source")
    rng = random.Random(seed)
    tasks = sorted(offsets)
    rng.shuffle(tasks)
    quotas = {task: rows * len(offsets[task]) // source_count for task in tasks}
    remainder_order = sorted(
        tasks,
        key=lambda task: rows * len(offsets[task]) % source_count,
        reverse=True,
    )
    for task in remainder_order[: rows - sum(quotas.values())]:
        quotas[task] += 1
    selected = []
    for task in sorted(tasks):
        selected.extend(rng.sample(offsets[task], quotas[task]))
    rng.shuffle(selected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with dataset_staging(destination) as temporary:
        staging = Path(temporary) / "subset"
        staging.mkdir()
        output = staging / "train.jsonl"
        output_digest = hashlib.sha256()
        identity_digest = hashlib.sha256()
        lengths = []
        task_lengths = defaultdict(list)
        with source_file.open("rb") as original, output.open("wb") as handle:
            for offset in selected:
                original.seek(offset)
                line = original.readline()
                row = json.loads(line)
                handle.write(line)
                output_digest.update(line)
                identity_digest.update((row["source_id"] + "\n").encode())
                lengths.append((row["prompt_tokens"], row["target_tokens"]))
                task_lengths[row["task"]].append(row["sequence_tokens"])
                if row["sequence_tokens"] != sum(lengths[-1]):
                    raise ValueError("Inconsistent stored token counts")
        counts = np.asarray(lengths, dtype=np.int64)
        totals = counts.sum(axis=1)
        cap = sequence_len
        truncated = totals > cap
        removed = totals[truncated] - cap
        lost_targets = counts[:, 1] - np.minimum(
            counts[:, 1], np.maximum(cap - counts[:, 0], 0)
        )
        report = {
            "schema": "gcw.reasoning_gym.subset.v1",
            "seed": seed,
            "selection": "Uniform without replacement within task; proportional largest-remainder quotas with seeded tie-breaking; final seeded shuffle",
            "source": str(source_file.as_posix()),
            "source_sha256": digest.hexdigest(),
            "source_manifest_sha256": hashlib.sha256(
                source_manifest_bytes
            ).hexdigest(),
            "source_rows": source_count,
            "rows": rows,
            "task_counts": quotas,
            "file": "train.jsonl",
            "bytes": output.stat().st_size,
            "sha256": output_digest.hexdigest(),
            "source_ids_sha256": identity_digest.hexdigest(),
            "row_identity": "Original source_id and experiment_order retained; physical file order is subset order",
            "tokenizer": source_manifest["tokenizer"],
            "tokenizer_identity_sha256": source_manifest[
                "tokenizer_identity_sha256"
            ],
            "source_format": source_manifest["format"],
            "holdouts": {
                split: source_manifest["files"][split]
                for split in ("validation", "test")
            },
            "audit": {
                "method": "Stored admission token counts from checksum-verified source; target includes EOS; population standard deviation conditional on truncation",
                "sequence_len": cap,
                "truncated_rows": int(truncated.sum()),
                "truncated_percent": float(100 * truncated.mean()),
                "original_tokens_mean_truncated": (
                    float(totals[truncated].mean()) if removed.size else None
                ),
                "removed_tokens_mean_truncated": (
                    float(removed.mean()) if removed.size else None
                ),
                "removed_tokens_std_truncated": (
                    float(removed.std()) if removed.size else None
                ),
                "prompt_only_rows": int((counts[:, 0] >= cap).sum()),
                "eos_only_truncated_rows": int((totals == cap + 1).sum()),
                "target_tokens_lost_percent": float(
                    100 * lost_targets.sum() / counts[:, 1].sum()
                ),
                "truncated_rows_by_task": {
                    task: sum(length > cap for length in values)
                    for task, values in task_lengths.items()
                },
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_dataset(staging, destination, replace=replace)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("data/train-data/reasoning-gym")
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/train-data/reasoning-gym-128k"),
    )
    parser.add_argument("--rows", type=int, default=128000)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--sequence-len", type=int, default=1536)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    report = prepare_subset(
        args.source,
        args.out_dir,
        args.rows,
        args.seed,
        args.sequence_len,
        replace=args.replace,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
