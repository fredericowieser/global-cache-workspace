#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gcw.distill.precompute import precompute_teacher_traces_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override gcw_static batch size; hf_dynamic uses teacher_trace_batch_size",
    )
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    result = precompute_teacher_traces_from_config(
        args.config,
        batch_size=args.batch_size,
        source_revision=args.source_revision,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
