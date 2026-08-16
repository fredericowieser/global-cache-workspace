from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from gcw.qualification.inputs import capture_file, capture_train_config
from gcw.qualification.resources import capture_resource_identity
from gcw.runtime import atomic_write_bytes, atomic_write_json
from gcw.training.checkpoint import verify_training_checkpoint_files
from scripts.math_evaluation_bundle import create_bundle, verify_bundle
from scripts.run_math_experiments_full_single_seed import (
    EXPERIMENT,
    TRAINABLE_ARMS,
    _completed_training,
    _reject_performance_overrides,
    _require_clean_repository,
    _resume_config,
    _run,
    _sha256,
    _source_commit,
    _validate_campaign_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one math-campaign arm and publish an evaluation bundle")
    parser.add_argument("--arm", choices=TRAINABLE_ARMS, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--bundle-outbox", required=True)
    parser.add_argument("--campaign-plan")
    parser.add_argument("--record-root")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--minimum-free-gib", type=int, default=100)
    parser.add_argument("--expected-gpu", default="H200")
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument("--stop-after-checkpoint-step", type=int)
    parser.add_argument("--stop-after-checkpoint-utc")
    parser.add_argument("--execute", action="store_true")
    return parser


def execute(args: argparse.Namespace) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser().resolve()
    root = Path(args.experiment_root).expanduser().resolve()
    bundle = Path(args.bundle_outbox).expanduser().resolve() / args.arm
    config_snapshot = capture_train_config(config_path)
    config = config_snapshot.load()
    source_commit = _source_commit(repository)
    plan = {
        "schema": "gcw.math_training_lane_plan.v1",
        "experiment": args.experiment,
        "arm": args.arm,
        "config": str(config_path),
        "config_sha256": config_snapshot.authored.identity.sha256,
        "resolved_config_sha256": config_snapshot.resolved_sha256,
        "checkpoint": str((Path(config.output_dir) / "checkpoint").resolve()),
        "bundle": str(bundle),
        "max_steps": config.max_steps,
        "global_batch_rows": config.global_batch_rows,
        "source_commit": source_commit,
        "stop_after_checkpoint_step": args.stop_after_checkpoint_step,
        "stop_after_checkpoint_utc": args.stop_after_checkpoint_utc,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    _require_clean_repository(repository)
    _reject_performance_overrides(dict(os.environ))
    for variable, supplied in (
        ("GCW_STOP_AFTER_CHECKPOINT_STEP", args.stop_after_checkpoint_step),
        ("GCW_STOP_AFTER_CHECKPOINT_UTC", args.stop_after_checkpoint_utc),
    ):
        if os.environ.get(variable) and supplied is None:
            raise RuntimeError(f"ambient {variable} is forbidden; use the explicit lane argument")
    if args.campaign_plan is None:
        raise ValueError("--campaign-plan is required for executable training lanes")
    campaign_plan_path = Path(args.campaign_plan).expanduser().resolve()
    captured_plan = capture_file(campaign_plan_path, role="math_campaign_plan")
    campaign_plan = json.loads(captured_plan.content)
    arm_plan = campaign_plan.get("arms", {}).get(args.arm, {})
    # The immutable matrix binds config/data digests below. The executable
    # source may advance over the preparation commit with a compatible fix;
    # both commits are recorded in the lane provenance rather than rejecting
    # a pure code fix that does not change training semantics.
    plan["plan_source_commit"] = campaign_plan.get("source_commit")
    plan["running_source_commit"] = source_commit
    if campaign_plan.get("experiment") != args.experiment:
        raise RuntimeError("campaign plan experiment does not match the lane")
    if arm_plan.get("config_sha256") != config_snapshot.authored.identity.sha256:
        raise RuntimeError("campaign plan does not bind the supplied arm config")
    if Path(str(arm_plan.get("config", ""))).resolve() != config_path:
        raise RuntimeError("campaign plan config path does not match the supplied arm config")
    _validate_campaign_config(config, arm=args.arm, root=root, experiment=args.experiment)
    plan["campaign_plan"] = str(campaign_plan_path)
    plan["campaign_plan_sha256"] = captured_plan.identity.sha256
    expected_accelerator = None if args.allow_other_gpu else args.expected_gpu.lower()
    if expected_accelerator not in {None, "h100", "h200"}:
        raise ValueError("--expected-gpu must be H100 or H200")
    resource = capture_resource_identity(
        expected_accelerator=expected_accelerator,
        require_single_cuda=True,
        require_gpu_authority=True,
    )
    plan["runtime_accelerator"] = asdict(resource)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "training-lane" / f".{args.arm}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lane_lock = lock_path.open("a+b")
    try:
        fcntl.flock(lane_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lane_lock.close()
        raise RuntimeError(f"another training lane owns arm {args.arm}") from error
    captured_config_path = root / "captured-configs" / f"{args.arm}-{config_snapshot.authored.identity.sha256}.yaml"
    captured_config_path.parent.mkdir(parents=True, exist_ok=True)
    if captured_config_path.exists():
        if capture_file(captured_config_path, role="captured_train_config").content != config_snapshot.authored.content:
            raise RuntimeError("captured training config path contains different bytes")
    else:
        atomic_write_bytes(captured_config_path, config_snapshot.authored.content, exclusive=True)
    if shutil.disk_usage(root.parent).free < args.minimum_free_gib * 1024**3:
        raise RuntimeError("insufficient free space for the training lane")
    env = dict(os.environ)
    record_root = Path(args.record_root).expanduser().resolve() if args.record_root else (repository / "data").resolve()
    env.update(
        {
            "GCW_DATA_ROOT": str(record_root),
            "GCW_WANDB_REQUIRED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if args.stop_after_checkpoint_step is not None:
        env["GCW_STOP_AFTER_CHECKPOINT_STEP"] = str(args.stop_after_checkpoint_step)
    if args.stop_after_checkpoint_utc is not None:
        env["GCW_STOP_AFTER_CHECKPOINT_UTC"] = args.stop_after_checkpoint_utc
    if not _completed_training(captured_config_path):
        command_config = _resume_config(captured_config_path, root)
        _run(
            [sys.executable, "-m", "gcw.cli", "train", "--config", str(command_config)],
            root / "logs" / f"train-{args.arm}.log",
            env,
        )
    checkpoint = Path(config.output_dir) / "checkpoint"
    if not _completed_training(captured_config_path):
        manifest_path = checkpoint / "manifest.json"
        state_path = checkpoint / "train_state.json"
        if not manifest_path.is_file() or not state_path.is_file():
            raise RuntimeError(f"training stopped without an authenticated checkpoint for {args.arm}")
        verified = verify_training_checkpoint_files(checkpoint)
        captured_manifest = capture_file(manifest_path, role="partial_checkpoint_manifest")
        captured_state = capture_file(state_path, role="partial_checkpoint_train_state")
        manifest = verified["manifest"]
        state = verified["train_state"]
        if manifest.get("complete") is not True or manifest.get("step") != state.get("step"):
            raise RuntimeError(f"partial checkpoint is incomplete for {args.arm}")
        if not 0 < int(state["step"]) < config.max_steps:
            raise RuntimeError(f"partial checkpoint step is outside the resumable range for {args.arm}")
        partial = {
            **plan,
            "schema": "gcw.math_training_lane_partial.v1",
            "stopped_at": datetime.now(timezone.utc).isoformat(),
            "completed_step": int(state["step"]),
            "checkpoint_manifest_sha256": captured_manifest.identity.sha256,
            "checkpoint_train_state_sha256": captured_state.identity.sha256,
        }
        status = root / "training-lane" / f"{args.arm}.partial.json"
        status.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(status, partial, indent=2, allow_nan=False)
        return
    verify_training_checkpoint_files(checkpoint)
    final_checkpoint_manifest = capture_file(checkpoint / "manifest.json", role="final_checkpoint_manifest")
    if bundle.exists():
        manifest = verify_bundle(bundle, expected_arm=args.arm, expected_step=config.max_steps)
    else:
        manifest = create_bundle(
            checkpoint,
            bundle,
            arm=args.arm,
            expected_step=config.max_steps,
            repository=repository,
            training_config=captured_config_path,
        )
    if manifest.get("source_commit") != source_commit:
        raise RuntimeError("evaluation bundle source commit does not match the lane")
    if manifest.get("training_config_sha256") != config_snapshot.authored.identity.sha256:
        raise RuntimeError("evaluation bundle training config does not match the lane")
    if manifest.get("source_checkpoint_manifest_sha256") != final_checkpoint_manifest.identity.sha256:
        raise RuntimeError("evaluation bundle checkpoint manifest does not match the lane")
    completion = {
        **plan,
        "schema": "gcw.math_training_lane_completion.v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "bundle_manifest_sha256": _sha256(bundle / "bundle-manifest.json"),
        "bundle_files": manifest["files"],
    }
    status = root / "training-lane" / f"{args.arm}.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(status, completion, indent=2, allow_nan=False)
    (status.parent / f"{args.arm}.partial.json").unlink(missing_ok=True)


def main() -> None:
    execute(build_parser().parse_args())


if __name__ == "__main__":
    main()
