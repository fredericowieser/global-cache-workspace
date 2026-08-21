from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

from scripts.math_evaluation_bundle import verify_bundle
from scripts.run_math_experiments_full_single_seed import (
    EVALUATION_ROOT,
    EXPERIMENT,
    SEED,
    _evaluate,
    _preflight,
    _sha256,
    _source_commit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one math-campaign arm on a dedicated GPU lane"
    )
    parser.add_argument("--arm", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--hf-model")
    target.add_argument("--checkpoint-bundle")
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--record-root")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument(
        "--evaluation-root", default=f"{EVALUATION_ROOT}-lrc-h100-v1"
    )
    parser.add_argument("--evaluation-lineage", default="lrc-h100-v1")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--expected-step", type=int, default=10_902)
    parser.add_argument("--deadline-hours", type=float, default=24.0)
    parser.add_argument("--minimum-free-gib", type=int, default=100)
    parser.add_argument("--expected-gpu", default="H100")
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def execute(args: argparse.Namespace) -> None:
    repository = Path(__file__).resolve().parents[1]
    root = Path(args.experiment_root).expanduser().resolve()
    hf_target = args.hf_model is not None
    if args.arm == "instruct-baseline" and not hf_target:
        raise ValueError("the instruct baseline requires --hf-model")
    if args.arm != "instruct-baseline" and hf_target:
        raise ValueError("trained arms require --checkpoint-bundle")
    if hf_target:
        target = Path(args.hf_model).expanduser().resolve()
        bundle_manifest = None
    else:
        target = Path(args.checkpoint_bundle).expanduser().resolve()
        bundle_manifest = verify_bundle(
            target,
            expected_arm=args.arm,
            expected_step=args.expected_step,
        )
    plan = {
        "schema": "gcw.math_evaluation_lane_plan.v1",
        "experiment": args.experiment,
        "arm": args.arm,
        "target": str(target),
        "target_kind": "hf" if hf_target else "authenticated_evaluation_bundle",
        "source_commit": _source_commit(repository),
        "evaluation_lineage": args.evaluation_lineage,
        "evaluation_root": args.evaluation_root,
        "batch_size": args.eval_batch_size,
        "seed": SEED,
        "bundle_manifest_sha256": (
            None
            if bundle_manifest is None
            else _sha256(target / "bundle-manifest.json")
        ),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    root.mkdir(parents=True, exist_ok=True)
    args.model = args.tokenizer_model
    env = dict(os.environ)
    record_root = (
        Path(args.record_root).expanduser().resolve()
        if args.record_root
        else (root / "run-records" / args.evaluation_lineage).resolve()
    )
    env.update(
        {
            "GCW_DATA_ROOT": str(record_root),
            "GCW_WANDB_REQUIRED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    _preflight(args, repository, root, env)
    deadline = datetime.now(timezone.utc) + timedelta(hours=args.deadline_hours)
    completed = _evaluate(
        args,
        root,
        args.arm,
        target,
        hf_target=hf_target,
        env=env,
        deadline=deadline,
    )
    if not completed:
        return
    completion = {
        **plan,
        "schema": "gcw.math_evaluation_lane_completion.v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    destination = root / args.evaluation_root / args.arm / "COMPLETED.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    execute(build_parser().parse_args())


if __name__ == "__main__":
    main()
