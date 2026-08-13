from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any


VAL_PATTERN = re.compile(r"val_loss=([0-9]+(?:\.[0-9]+)?)")


def select(root: Path) -> dict[str, Any]:
    records: dict[str, list[dict[str, Any]]] = {}
    for log in sorted(root.glob("*/*/train.log")):
        arm, learning_rate = log.relative_to(root).parts[:2]
        values = [float(value) for value in VAL_PATTERN.findall(log.read_text())]
        if len(values) < 2 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"missing finite start/final validation losses: {log}")
        records.setdefault(arm, []).append(
            {
                "learning_rate": float(learning_rate),
                "start_val_loss": values[0],
                "final_val_loss": values[-1],
                "improvement": values[0] - values[-1],
                "log": str(log.resolve()),
            }
        )
    if not records:
        raise ValueError(f"no LR screen logs under {root}")
    selected: dict[str, float] = {}
    for arm, candidates in records.items():
        winner = min(candidates, key=lambda item: (item["final_val_loss"], item["learning_rate"]))
        selected[arm] = winner["learning_rate"]
    return {"schema": "gcw.math_experiment_lr_selection.v1", "selected": selected, "candidates": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Select experiment learning rates by held-out SFT loss")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = select(Path(args.root))
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
