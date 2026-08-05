from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def flatten_finite_metrics(value: Any, *, prefix: str = "eval") -> dict[str, float]:
    metrics: dict[str, float] = {}

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, bool):
            metrics["/".join((prefix, *path))] = float(node)
        elif isinstance(node, (int, float)) and math.isfinite(float(node)):
            metrics["/".join((prefix, *path))] = float(node)
        elif isinstance(node, dict):
            for key, child in node.items():
                visit(child, (*path, str(key)))

    visit(value, ())
    return metrics


def main() -> None:
    import wandb

    out_dir = Path(os.environ["OUT_DIR"]).resolve(strict=True)
    aggregate_path = out_dir / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text())
    source_revision = os.environ["SOURCE_REVISION"]
    checkpoint = Path(os.environ["CHECKPOINT"]).resolve(strict=True)
    train_state = json.loads((checkpoint / "train_state.json").read_text())
    step = int(train_state["optimizer_steps"])
    if step != 1000:
        raise SystemExit(f"evaluation publisher requires a 1k checkpoint, got {step}")
    tasks = tuple(aggregate.get("tasks", ()))
    expected_tasks = (
        "gsm8k_openmath",
        "svamp",
        "gsm8k_hard",
        "logiqa_opsd_generate",
        "agieval_gaokao_mathqa_opsd_generate",
    )
    if tasks != expected_tasks:
        raise SystemExit(f"aggregate task contract mismatch: {tasks!r}")
    name = os.environ.get("WANDB_EVAL_NAME", f"opsd-native-v2-steps{step}-full-matrix")
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "global-cache-workspace"),
        name=name,
        group=os.environ.get("WANDB_GROUP", "opsd-native-full-v2"),
        job_type="evaluation",
        mode="online",
        tags=["opsd", "full-matrix", "five-benchmark", f"steps-{step}"],
        config={
            "source_revision": source_revision,
            "checkpoint": str(checkpoint),
            "training_step": step,
            "conditions": [
                "cot-off__cp-off",
                "cot-off__cp-on",
                "cot-on__cp-off",
                "cot-on__cp-on",
            ],
            "tasks": list(tasks),
        },
    )
    run.log({"train/step": step, **flatten_finite_metrics(aggregate)}, step=step)
    artifact = wandb.Artifact(
        name=f"{name}-results",
        type="evaluation",
        metadata={
            "source_revision": source_revision,
            "checkpoint": str(checkpoint),
            "aggregate_sha256": hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
        },
    )
    for name_part in (
        "cot-off__cp-off.json",
        "cot-off__cp-on.json",
        "cot-on__cp-off.json",
        "cot-on__cp-on.json",
        "aggregate.json",
    ):
        artifact.add_file(str(out_dir / name_part), name=name_part)
    run.log_artifact(artifact)
    run_id = run.id
    run_url = run.url
    run.finish()
    evidence = {
        "wandb_run_id": run_id,
        "wandb_run_url": run_url,
        "artifact_name": artifact.name,
        "source_revision": source_revision,
        "training_step": step,
        "tasks": list(tasks),
    }
    target = out_dir / "wandb-publication.json"
    temporary = out_dir / f".wandb-publication.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(json.dumps(evidence, indent=2, sort_keys=True).encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
