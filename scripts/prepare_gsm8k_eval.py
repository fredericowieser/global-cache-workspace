from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from gcw.process_data import prepare_gsm8k_eval_records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build authoritative GSM8K evaluation records from Arrow data.")
    parser.add_argument("--source-arrow", required=True, type=Path)
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args(argv)

    if not args.source_arrow.is_file():
        raise FileNotFoundError(args.source_arrow)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.audit.exists():
        raise FileExistsError(args.audit)

    from datasets import Dataset

    dataset = Dataset.from_file(str(args.source_arrow))
    audit = prepare_gsm8k_eval_records(
        dataset,
        args.output,
        source_split=args.source_split,
    )
    payload = {
        "schema_version": 1,
        "kind": "gcw.process_branch.gsm8k_eval",
        "source_split": args.source_split,
        "source_arrow": str(args.source_arrow),
        "source_sha256": _sha256(args.source_arrow),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        **audit,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
