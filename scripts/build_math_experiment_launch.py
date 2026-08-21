from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from gcw.config import load_train_config
from gcw.flops import total_training_flops
from gcw.params import processor_params


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the immutable math experiment launch manifest"
    )
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--gates", required=True)
    parser.add_argument("--lr-selection", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config_dir = Path(args.config_dir).resolve()
    arms: dict[str, Any] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        config = load_train_config(path)
        maximum_tokens = (
            config.global_batch_rows * config.sequence_len * config.max_steps
        )
        arms[path.stem] = {
            "config": str(path),
            "config_sha256": _sha256(path),
            "learning_rate": config.learning_rate,
            "max_steps": config.max_steps,
            "maximum_tokens": maximum_tokens,
            "processor_params": processor_params(config.model),
            "total_training_flops": total_training_flops(
                config.model, maximum_tokens, config.sequence_len
            ),
            "output_dir": config.output_dir,
        }
    evidence = {}
    for name, value in {
        "data_manifest": args.data_manifest,
        "gates": args.gates,
        "lr_selection": args.lr_selection,
        "experiment_contract": config_dir / "experiment.json",
    }.items():
        path = Path(value).resolve()
        evidence[name] = {"path": str(path), "sha256": _sha256(path)}
    payload = {
        "schema": "gcw.math_experiment_launch.v1",
        "source_commit": args.source_commit,
        "status": "ready_requires_human_launch_approval",
        "evidence": evidence,
        "arms": arms,
    }
    Path(args.out).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
