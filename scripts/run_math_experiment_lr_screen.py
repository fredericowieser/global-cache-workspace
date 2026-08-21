from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

DEFAULT_GRIDS = {
    "backbone": (5e-6, 1e-5, 2e-5),
    "lora": (5e-5, 1e-4, 2e-4),
    "processor": (5e-5, 1e-4, 2e-4),
}


def _kind(payload: dict) -> str:
    trainable = payload["trainable"]
    if trainable == "backbone":
        return "backbone"
    if trainable == "lora":
        return "lora"
    return "processor"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bounded held-out-loss learning-rate screens"
    )
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--arms", nargs="*")
    args = parser.parse_args()
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected_arms = None if not args.arms else set(args.arms)
    env = dict(os.environ)
    env["GCW_DISABLE_TORCH_COMPILE"] = "1"
    env["GCW_DATA_ROOT"] = str(output / "records")
    for source in sorted(Path(args.config_dir).glob("*.yaml")):
        arm = source.stem
        if selected_arms is not None and arm not in selected_arms:
            continue
        base = yaml.safe_load(source.read_text())
        for rate in DEFAULT_GRIDS[_kind(base)]:
            run = output / arm / format(rate, ".8g")
            run.mkdir(parents=True)
            payload = deepcopy(base)
            payload["learning_rate"] = rate
            payload["max_steps"] = args.steps
            payload["warmup_steps"] = max(1, round(0.03 * args.steps))
            payload["save_every"] = 0
            payload["save_final"] = False
            payload["eval_every"] = args.steps
            payload["eval_batches"] = args.eval_batches
            payload["eval_on_start"] = True
            payload["output_dir"] = str(run / "state")
            payload["wandb"]["enabled"] = False
            contract = deepcopy(
                json.loads(
                    Path(payload["data"]["experiment_manifest"]).read_text()
                )
            )
            contract["max_steps"] = args.steps
            contract_path = run / "experiment.json"
            contract_path.write_text(
                json.dumps(contract, indent=2, sort_keys=True) + "\n"
            )
            payload["data"]["experiment_manifest"] = str(contract_path)
            config = run / "config.yaml"
            config.write_text(yaml.safe_dump(payload, sort_keys=False))
            with (run / "train.log").open("wb") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "gcw.cli",
                        "train",
                        "--config",
                        str(config),
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )


if __name__ == "__main__":
    main()
