from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gcw.evaluation.core import run_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one experiment target in declared prompt modes"
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--checkpoint")
    target.add_argument("--hf-model")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--processor", choices=("on", "off"), default="on")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("thinking_off", "thinking_on"),
        default=("thinking_off", "thinking_on"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    summary = {}
    for mode in args.modes:
        mode_dir = output / mode
        mode_dir.mkdir()
        os.environ["GCW_EVAL_PARTIAL_DIR"] = str(mode_dir)
        result = run_all(
            checkpoint=args.checkpoint,
            hf_model=args.hf_model,
            processor_enabled=args.processor == "on",
            prompt_mode=mode,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        (mode_dir / "result.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n"
        )
        summary[mode] = result
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
