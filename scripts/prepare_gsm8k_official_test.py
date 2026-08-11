#!/usr/bin/env python3
"""Prepare the pinned 1,319-row official GSM8K test artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from gcw.process_data import prepare_gsm8k_eval_records

REVISION = "740312add88f781978c0658806c59bc2815b9866"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.audit.exists():
        raise FileExistsError("output and audit paths must not already exist")
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split="test", revision=REVISION)
    if len(rows) != 1319:
        raise ValueError(f"pinned GSM8K test split must contain 1319 rows, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = prepare_gsm8k_eval_records(rows, args.output, source_split="test")
    if result["accepted_rows"] != 1319 or result["rejected_rows"] != 0:
        args.output.unlink(missing_ok=True)
        raise ValueError(f"all 1,319 official GSM8K test rows must be eligible: {result}")
    audit = {
        "dataset": "openai/gsm8k",
        "kind": "gcw.process_branch.gsm8k_eval",
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "revision": REVISION,
        "schema_version": 1,
        "source_split": "test",
        **result,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
