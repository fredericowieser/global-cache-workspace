from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gcw.config import EvalConfig
from gcw.evaluation.core import run_all
from gcw.evaluation.lm_eval_adapter import (
    _distributed_eval_state,
    _eval_device,
    _load_eval_target,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one experiment target with thinking disabled and enabled"
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--checkpoint")
    target.add_argument("--hf-model")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--processor", choices=("on", "off"), default="on")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    _state, local_rank, _world_size = _distributed_eval_state()
    loaded_target = _load_eval_target(
        checkpoint=args.checkpoint,
        hf_model=args.hf_model,
        processor_enabled=args.processor == "on",
        device=_eval_device(local_rank),
        dtype="auto",
    )
    summary = {}
    for enable_thinking in (False, True):
        label = "thinking-enabled" if enable_thinking else "thinking-disabled"
        mode_dir = output / label
        mode_dir.mkdir()
        os.environ["GCW_EVAL_PARTIAL_DIR"] = str(mode_dir)
        result = run_all(
            checkpoint=args.checkpoint,
            hf_model=args.hf_model,
            processor_enabled=args.processor == "on",
            config=EvalConfig(
                batch_size=args.batch_size,
                limit=args.limit,
                apply_chat_template=True,
                enable_thinking=enable_thinking,
            ),
            loaded_target=loaded_target,
        )
        (mode_dir / "result.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n"
        )
        summary[label] = result
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
