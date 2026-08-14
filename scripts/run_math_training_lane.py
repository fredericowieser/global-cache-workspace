from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import torch

from gcw.config import load_train_config
from scripts.math_evaluation_bundle import create_bundle, verify_bundle
from scripts.run_math_experiments_full_single_seed import (
    EXPERIMENT,
    TRAIN_ORDER,
    _completed_training,
    _require_clean_repository,
    _resume_config,
    _run,
    _sha256,
    _source_commit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one math-campaign arm and publish an evaluation bundle")
    parser.add_argument("--arm", choices=TRAIN_ORDER, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--bundle-outbox", required=True)
    parser.add_argument("--record-root")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--minimum-free-gib", type=int, default=100)
    parser.add_argument("--expected-gpu", default="H200")
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def execute(args: argparse.Namespace) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser().resolve()
    root = Path(args.experiment_root).expanduser().resolve()
    bundle = Path(args.bundle_outbox).expanduser().resolve() / args.arm
    config = load_train_config(config_path)
    plan = {
        "schema": "gcw.math_training_lane_plan.v1",
        "experiment": args.experiment,
        "arm": args.arm,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str((Path(config.output_dir) / "checkpoint").resolve()),
        "bundle": str(bundle),
        "max_steps": config.max_steps,
        "global_batch_rows": config.global_batch_rows,
        "source_commit": _source_commit(repository),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    _require_clean_repository(repository)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("training lane requires exactly one visible CUDA GPU")
    device = torch.cuda.get_device_name(0)
    if args.expected_gpu not in device and not args.allow_other_gpu:
        raise RuntimeError(f"expected {args.expected_gpu}, found {device!r}")
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < args.minimum_free_gib * 1024**3:
        raise RuntimeError("insufficient free space for the training lane")
    env = dict(os.environ)
    record_root = (
        Path(args.record_root).expanduser().resolve()
        if args.record_root
        else (repository / "data").resolve()
    )
    env.update(
        {
            "GCW_DATA_ROOT": str(record_root),
            "GCW_WANDB_REQUIRED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if not _completed_training(config_path):
        command_config = _resume_config(config_path, root)
        _run(
            [sys.executable, "-m", "gcw.cli", "train", "--config", str(command_config)],
            root / "logs" / f"train-{args.arm}.log",
            env,
        )
    if not _completed_training(config_path):
        raise RuntimeError(f"training did not reach step {config.max_steps} for {args.arm}")
    checkpoint = Path(config.output_dir) / "checkpoint"
    if bundle.exists():
        manifest = verify_bundle(bundle, expected_arm=args.arm, expected_step=config.max_steps)
    else:
        manifest = create_bundle(
            checkpoint,
            bundle,
            arm=args.arm,
            expected_step=config.max_steps,
            repository=repository,
            training_config=config_path,
        )
    completion = {
        **plan,
        "schema": "gcw.math_training_lane_completion.v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "bundle_manifest_sha256": _sha256(bundle / "bundle-manifest.json"),
        "bundle_files": manifest["files"],
    }
    status = root / "training-lane" / f"{args.arm}.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")


def main() -> None:
    execute(build_parser().parse_args())


if __name__ == "__main__":
    main()
