from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from gcw.process_data import ProcessDataError, parse_gsm8k_eval_record


_CALCULATOR_ANNOTATION = re.compile(r"<<[^<>]*>>")


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_trajectory_records(
    rows: Iterable[object],
    output: str | Path,
) -> dict[str, object]:
    """Convert canonical GSM8K rows to full-rationale replay records."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_rows = 0
    accepted_rows = 0
    rejection_reasons: Counter[str] = Counter()
    source_ids: list[str] = []
    created = False
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            created = True
            for source_index, row in enumerate(rows):
                source_rows += 1
                question = row.get("question") if isinstance(row, dict) else None
                answer = row.get("answer") if isinstance(row, dict) else None
                if not isinstance(question, str) or not isinstance(answer, str):
                    rejection_reasons["invalid_schema"] += 1
                    continue
                try:
                    record = parse_gsm8k_eval_record(
                        source_split="train",
                        source_index=source_index,
                        question=question,
                        answer=answer,
                    )
                except ProcessDataError as exc:
                    rejection_reasons[str(exc)] += 1
                    continue
                cleaned_steps = tuple(
                    cleaned
                    for step in record.reasoning_steps
                    if (cleaned := _CALCULATOR_ANNOTATION.sub("", step).strip())
                )
                if not cleaned_steps:
                    rejection_reasons["missing_reasoning_steps_after_cleaning"] += 1
                    continue
                payload = asdict(record)
                payload["reasoning_steps"] = list(cleaned_steps)
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                source_ids.append(record.source_id)
                accepted_rows += 1
    except Exception:
        if created:
            output_path.unlink(missing_ok=True)
        raise
    return {
        "source_rows": source_rows,
        "accepted_rows": accepted_rows,
        "rejected_rows": source_rows - accepted_rows,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "ordered_source_ids_sha256": _canonical_json_sha256(source_ids),
    }


def _read_jsonl(path: Path) -> list[object]:
    rows: list[object] = []
    for row_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid source JSON at row {row_index}") from exc
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare canonical GSM8K rationale trajectories for frozen-cache replay"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/exploratory/gsm8k-upstream-train.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/process_branch/gsm8k-trajectory-replay-v1.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/process_branch/gsm8k-trajectory-replay-v1.manifest.json"),
    )
    args = parser.parse_args(argv)
    source_bytes = args.source.read_bytes()
    rows = _read_jsonl(args.source)
    audit = prepare_trajectory_records(rows, args.output)
    output_bytes = args.output.read_bytes()
    manifest = {
        "schema": "gcw.gsm8k_trajectory_replay.v1",
        "source": str(args.source),
        "source_size_bytes": len(source_bytes),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output": str(args.output),
        "output_size_bytes": len(output_bytes),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "recommended_split": {
            "train": {"skip_rows": 0, "take_rows": 6000},
            "heldout": {"skip_rows": 6000, "take_rows": 256},
            "untouched": {"skip_rows": 6256, "take_rows": 1217},
        },
        **audit,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.manifest.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except Exception:
        args.output.unlink(missing_ok=True)
        raise
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
