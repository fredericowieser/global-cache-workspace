from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import tempfile
from typing import Sequence

from gcw.process_data import prepare_gsm8k_answer_records, read_answer_only_records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_no_clobber(staged: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(staged, destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build answer-only records directly from upstream GSM8K Arrow rows."
    )
    parser.add_argument("--source-arrow", required=True, type=Path)
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)

    if not args.source_arrow.is_file():
        raise FileNotFoundError(args.source_arrow)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.manifest.exists():
        raise FileExistsError(args.manifest)

    from datasets import Dataset

    token = secrets.token_hex(12)
    staged_output = args.output.parent / f".{args.output.name}.{token}.tmp"
    staged_manifest = args.manifest.parent / f".{args.manifest.name}.{token}.tmp"
    published_output = False
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="gcw-answer-only-arrow-") as snapshot_dir:
            source_snapshot = Path(snapshot_dir) / "source.arrow"
            shutil.copyfile(args.source_arrow, source_snapshot)
            source_sha256 = _sha256(source_snapshot)
            rows = Dataset.from_file(str(source_snapshot))
            audit = prepare_gsm8k_answer_records(
                rows,
                staged_output,
                source_split=args.source_split,
            )
        ordered_source_ids = [
            record.source_id for record in read_answer_only_records(staged_output)
        ]
        ordered_source_ids_sha256 = hashlib.sha256(
            json.dumps(
                ordered_source_ids,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        payload = {
            "schema_version": 1,
            "kind": "gcw.answer_only.gsm8k.v1",
            "source_split": args.source_split,
            "source_arrow": str(args.source_arrow),
            "source_sha256": source_sha256,
            "output": str(args.output),
            "transformed_sha256": _sha256(staged_output),
            "ordered_source_ids_sha256": ordered_source_ids_sha256,
            **audit,
        }
        with staged_manifest.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _publish_no_clobber(staged_output, args.output)
        published_output = True
        _publish_no_clobber(staged_manifest, args.manifest)
    except Exception:
        if published_output:
            args.output.unlink(missing_ok=True)
        raise
    finally:
        staged_output.unlink(missing_ok=True)
        staged_manifest.unlink(missing_ok=True)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
