from __future__ import annotations

import argparse
from argparse import Namespace
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import torch
import yaml

from gcw.config import load_train_config
from gcw.flops import total_training_flops
from gcw.params import processor_params
from scripts.generate_math_experiment_configs import generate


EXPERIMENT = "qwen3-06b-instruct-math-full-single-seed-v1"
GENERAL_TASKS = (
    "lambada_openai",
    "hellaswag",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "mmlu",
)
MATH_TASKS = (
    "gsm8k_openmath",
    "svamp",
    "gsm8k_hard",
    "logiqa",
    "agieval_gaokao_mathqa",
)
TRAIN_ORDER = (
    "full-sft",
    "bt",
    "bgt",
    "diffusion-bgt-t4-sigma015",
    "longshot",
    "longshot-internal-r4",
    "longshot-external-r4",
    "lora-r32",
    "flow-bgt-t4",
)
PROCESSOR_ARMS = frozenset(
    {
        "bt",
        "bgt",
        "diffusion-bgt-t4-sigma015",
        "longshot",
        "longshot-internal-r4",
        "longshot-external-r4",
        "flow-bgt-t4",
    }
)
MICRO_BATCH_ROWS = {
    "full-sft": 8,
    "bt": 2,
    "bgt": 4,
    "diffusion-bgt-t4-sigma015": 4,
    "longshot": 2,
    "longshot-internal-r4": 2,
    "longshot-external-r4": 1,
    "lora-r32": 8,
    "flow-bgt-t4": 4,
}
PEAK_LEARNING_RATES = {
    "full-sft": 5e-6,
    "bt": 1e-4,
    "bgt": 1e-4,
    "diffusion-bgt-t4-sigma015": 1e-4,
    "longshot": 1e-4,
    "longshot-internal-r4": 1e-4,
    "longshot-external-r4": 1e-4,
    "lora-r32": 1e-4,
    "flow-bgt-t4": 1e-4,
}
SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_ROWS = 64
SEED = 24
VALIDATION_ROWS = 1024
CHECKPOINT_EVERY = 256
WANDB_ENTITY = "huawei-noahs-ark"
WANDB_PROJECT = "global-cache-workspace"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit(repository: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()


def _require_clean_repository(repository: Path) -> None:
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if status.strip():
        raise RuntimeError("tracked repository files must be clean before execution")


def _load_data_manifest(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(Path(args.data_manifest).expanduser().read_text())
    dataset = Path(args.dataset or payload["dataset_path"]).expanduser().resolve()
    validation = Path(args.validation_dataset or payload["validation_path"]).expanduser().resolve()
    for name, path, expected in (
        ("dataset", dataset, payload["dataset_sha256"]),
        ("validation", validation, payload["validation_sha256"]),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != expected:
            raise ValueError(f"{name} sha256 does not match the authenticated data manifest")
    payload["dataset_path"] = str(dataset)
    payload["validation_path"] = str(validation)
    return payload


def _plan(args: argparse.Namespace, data: dict[str, Any], source_commit: str) -> dict[str, Any]:
    rows = int(data["rows"])
    effective_rows = rows - rows % GLOBAL_BATCH_ROWS
    dropped_rows = rows - effective_rows
    max_steps = effective_rows // GLOBAL_BATCH_ROWS
    return {
        "schema": "gcw.math_experiment_series.v1",
        "experiment": args.experiment,
        "source_commit": source_commit,
        "status": "planned_not_started",
        "backbone": str(Path(args.model).expanduser().resolve()),
        "backbone_checkpoint": str(Path(args.backbone_checkpoint).expanduser().resolve()),
        "sequence_length": SEQUENCE_LENGTH,
        "global_batch_rows": GLOBAL_BATCH_ROWS,
        "seed": SEED,
        "dataset_rows": rows,
        "effective_epoch_rows": effective_rows,
        "drop_last_rows": dropped_rows,
        "max_steps": max_steps,
        "objective": "answer_only_sft",
        "scheduler": {
            "name": "warmup_stable_decay",
            "warmup_fraction": 0.03,
            "stable_fraction": 0.87,
            "warmdown_fraction": 0.10,
            "final_learning_rate_fraction": 0.0,
        },
        "learning_rates": PEAK_LEARNING_RATES,
        "micro_batch_rows": MICRO_BATCH_ROWS,
        "validation": {"rows": VALIDATION_ROWS, "approximately_every_steps": max_steps // 20},
        "training_order": list(TRAIN_ORDER),
        "evaluation_order": ["instruct-baseline", *TRAIN_ORDER],
        "tasks": {"general": list(GENERAL_TASKS), "math": list(MATH_TASKS)},
        "evaluation_cells": {
            "non_processor": ["thinking_off", "thinking_on"],
            "processor": [
                "processor_off__thinking_off",
                "processor_off__thinking_on",
                "processor_on__thinking_off",
                "processor_on__thinking_on",
            ],
        },
    }


def _wandb_id(experiment: str, name: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", experiment).strip("-")[:48]
    suffix = hashlib.sha256(f"{experiment}:{name}:{SEED}".encode()).hexdigest()[:12]
    return f"{prefix}-{suffix}"


def _prepare_configs(
    args: argparse.Namespace,
    root: Path,
    data: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Path]:
    config_dir = root / "configs"
    if config_dir.exists():
        paths = {name: config_dir / f"{name}.yaml" for name in TRAIN_ORDER}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete existing config directory: {missing}")
        return paths

    runtime_manifest = root / "data-manifest.json"
    runtime_manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    generated = generate(
        Namespace(
            model=args.model,
            backbone_checkpoint=args.backbone_checkpoint,
            data_manifest=str(runtime_manifest),
            out_dir=str(config_dir),
            run_root=str((root / "training-state").resolve()),
            experiment=args.experiment,
            sequence_length=SEQUENCE_LENGTH,
            global_batch_rows=GLOBAL_BATCH_ROWS,
            micro_batch=1,
            processor_width=1024,
            distributed="none",
            seed=SEED,
            max_steps=plan["max_steps"],
            disable_wandb=False,
            learning_rates=None,
            eval_batches=1,
            screen_mode=False,
        )
    )
    all_paths = {path.stem: path for path in generated}
    result: dict[str, Path] = {}
    for arm in TRAIN_ORDER:
        path = all_paths[arm]
        payload = yaml.safe_load(path.read_text())
        micro_batch = MICRO_BATCH_ROWS[arm]
        if GLOBAL_BATCH_ROWS % micro_batch:
            raise ValueError(f"microbatch for {arm} does not divide global batch")
        payload["batch_size"] = micro_batch
        payload["learning_rate"] = PEAK_LEARNING_RATES[arm]
        payload["warmup_steps"] = max(1, round(0.03 * plan["max_steps"]))
        payload["warmdown_ratio"] = 0.10
        payload["final_lr_frac"] = 0.0
        payload["save_every"] = CHECKPOINT_EVERY
        payload["save_final"] = True
        payload["eval_every"] = max(1, plan["max_steps"] // 20)
        payload["eval_batches"] = VALIDATION_ROWS // micro_batch
        payload["eval_on_start"] = True
        payload["data"]["sft_take_rows"] = plan["effective_epoch_rows"]
        payload["wandb"].update(
            {
                "entity": WANDB_ENTITY,
                "project": WANDB_PROJECT,
                "name": f"{args.experiment}-{arm}",
                "group": args.experiment,
                "id": _wandb_id(args.experiment, f"train-{arm}"),
                "resume": "allow",
                "tags": ["qwen3-0.6b", "math", "single-seed", "full-epoch", arm],
            }
        )
        payload["metadata"].update(
            {
                "series": args.experiment,
                "execution_order": TRAIN_ORDER.index(arm) + 1,
                "effective_epoch_rows": plan["effective_epoch_rows"],
                "drop_last_rows": plan["drop_last_rows"],
                "scheduler": "warmup_stable_decay",
            }
        )
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        result[arm] = path
    for name, path in all_paths.items():
        if name not in TRAIN_ORDER:
            path.unlink()
    return result


def _completed_training(config_path: Path) -> bool:
    config = load_train_config(config_path)
    state = Path(config.output_dir) / "checkpoint" / "train_state.json"
    if not state.is_file():
        return False
    return int(json.loads(state.read_text()).get("step", -1)) == config.max_steps


def _resume_config(config_path: Path, root: Path) -> Path:
    config = load_train_config(config_path)
    checkpoint = Path(config.output_dir) / "checkpoint"
    manifest = checkpoint / "manifest.json"
    if not manifest.is_file():
        return config_path
    payload = yaml.safe_load(config_path.read_text())
    payload["resume"] = {
        "checkpoint": str(checkpoint),
        "topology": "same",
        "expected_manifest_sha256": _sha256(manifest),
    }
    target = root / "resume-configs" / config_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False))
    return target


def _run(command: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        handle.write(("\n$ " + " ".join(command) + "\n").encode())
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, env=env)


def _deadline_reached(deadline: datetime) -> bool:
    return datetime.now(timezone.utc) >= deadline


def _evaluation_cells(arm: str) -> tuple[tuple[str, str], ...]:
    if arm not in PROCESSOR_ARMS:
        return (("on", "thinking_off"), ("on", "thinking_on"))
    return tuple(
        (processor, thinking)
        for processor in ("off", "on")
        for thinking in ("thinking_off", "thinking_on")
    )


def _flatten_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, bool):
        return output
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        output[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            output.update(_flatten_numbers(child, name))
    return output


def _mirror_evaluation(result: Path, args: argparse.Namespace, arm: str, processor: str, thinking: str) -> None:
    import wandb

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"{args.experiment}-eval-{arm}-{processor}-{thinking}",
        group=args.experiment,
        job_type="evaluation",
        id=_wandb_id(args.experiment, f"eval-{arm}-{processor}-{thinking}"),
        resume="allow",
        tags=["qwen3-0.6b", "math", "single-seed", "evaluation", arm, processor, thinking],
        config={
            "experiment": args.experiment,
            "arm": arm,
            "processor": processor,
            "thinking": thinking,
            "tasks": [*GENERAL_TASKS, *MATH_TASKS],
            "batch_size": args.eval_batch_size,
        },
    )
    if run is None:
        raise RuntimeError("wandb.init returned no evaluation run")
    payload = json.loads(result.read_text())
    run.log(_flatten_numbers(payload), step=0)
    artifact = wandb.Artifact(
        f"{args.experiment}-eval-{arm}-{processor}-{thinking}", type="benchmark-results"
    )
    artifact.add_file(str(result), name="result.json")
    run.log_artifact(artifact)
    run.finish()


def _evaluate(
    args: argparse.Namespace,
    root: Path,
    arm: str,
    target: Path,
    *,
    hf_target: bool,
    env: dict[str, str],
    deadline: datetime,
) -> bool:
    for processor, thinking in _evaluation_cells(arm):
        if _deadline_reached(deadline):
            return False
        cell = root / "evaluations" / arm / f"processor-{processor}" / thinking
        result = cell / "result.json"
        published = cell / "wandb-published.json"
        if not result.is_file():
            cell.mkdir(parents=True, exist_ok=True)
            cell_env = dict(env)
            cell_env["GCW_EVAL_PARTIAL_DIR"] = str(cell / "partial")
            command = [
                sys.executable,
                "-m",
                "gcw.cli",
                "eval",
                "all",
                "--batch-size",
                str(args.eval_batch_size),
                "--prompt-mode",
                thinking,
                "--processor",
                processor,
                "--out",
                str(result),
            ]
            command.extend(["--hf-model", str(target)] if hf_target else ["--checkpoint", str(target)])
            _run(command, cell / "eval.log", cell_env)
        if not published.is_file():
            _mirror_evaluation(result, args, arm, processor, thinking)
            published.write_text(
                json.dumps({"result_sha256": _sha256(result), "published": True}, indent=2) + "\n"
            )
    return True


def _preflight(args: argparse.Namespace, repository: Path, root: Path, env: dict[str, str]) -> None:
    _require_clean_repository(repository)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_name = torch.cuda.get_device_name(0)
    if "H200" not in device_name and not args.allow_other_gpu:
        raise RuntimeError(f"expected one H200, found {device_name!r}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
    usage = shutil.disk_usage(root.parent)
    if usage.free < args.minimum_free_gib * 1024**3:
        raise RuntimeError(f"less than {args.minimum_free_gib} GiB free under {root.parent}")
    hf_cli = Path(sys.executable).parent / "hf"
    if not hf_cli.is_file():
        raise FileNotFoundError("Hugging Face CLI is missing; sync the project environment first")
    subprocess.run([str(hf_cli), "auth", "whoami"], check=True, env=env)
    import wandb

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"{args.experiment}-preflight",
        group=args.experiment,
        job_type="preflight",
        id=_wandb_id(args.experiment, "preflight"),
        resume="allow",
        config={"source_commit": _source_commit(repository), "device": device_name},
    )
    if run is None:
        raise RuntimeError("W&B preflight returned no run")
    run.log({"preflight/ok": 1}, step=0)
    run.finish()


def _write_plan_with_costs(plan: dict[str, Any], configs: dict[str, Path], path: Path) -> None:
    payload = deepcopy(plan)
    payload["status"] = "ready_not_started"
    payload["arms"] = {}
    for arm, config_path in configs.items():
        config = load_train_config(config_path)
        maximum_tokens = config.global_batch_rows * config.sequence_len * config.max_steps
        payload["arms"][arm] = {
            "config": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "processor_params": processor_params(config.model),
            "maximum_tokens": maximum_tokens,
            "estimated_training_flops": total_training_flops(
                config.model, maximum_tokens, config.sequence_len
            ),
        }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def execute(args: argparse.Namespace) -> None:
    repository = Path(__file__).resolve().parents[1]
    source_commit = _source_commit(repository)
    data = _load_data_manifest(args)
    plan = _plan(args, data, source_commit)
    root = Path(args.experiment_root).expanduser().resolve()
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    record_root = (
        Path(args.record_root).expanduser().resolve()
        if args.record_root is not None
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
    _preflight(args, repository, root, env)
    configs = _prepare_configs(args, root, data, plan)
    _write_plan_with_costs(plan, configs, root / "plan.json")
    deadline = datetime.now(timezone.utc) + timedelta(hours=args.deadline_hours)

    if not _evaluate(
        args,
        root,
        "instruct-baseline",
        Path(args.model).expanduser().resolve(),
        hf_target=True,
        env=env,
        deadline=deadline,
    ):
        return

    for arm in TRAIN_ORDER:
        if _deadline_reached(deadline):
            return
        config_path = configs[arm]
        if not _completed_training(config_path):
            command_config = _resume_config(config_path, root)
            _run(
                [sys.executable, "-m", "gcw.cli", "train", "--config", str(command_config)],
                root / "logs" / f"train-{arm}.log",
                env,
            )
        checkpoint = Path(load_train_config(config_path).output_dir) / "checkpoint"
        if not _completed_training(config_path):
            raise RuntimeError(f"training did not reach the declared epoch for {arm}")
        if not _evaluate(
            args,
            root,
            arm,
            checkpoint,
            hf_target=False,
            env=env,
            deadline=deadline,
        ):
            return

    completion = {
        "schema": "gcw.math_experiment_series_completion.v1",
        "experiment": args.experiment,
        "source_commit": source_commit,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": _sha256(root / "plan.json"),
    }
    (root / "completed.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the single-seed full-epoch Qwen3 math training and evaluation series"
    )
    parser.add_argument("--model", required=True, help="Authenticated local Qwen3-0.6B Instruct HF directory")
    parser.add_argument("--backbone-checkpoint", required=True, help="Authenticated native GCW backbone")
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--dataset", help="Override the manifest training JSONL path without changing its hash")
    parser.add_argument("--validation-dataset", help="Override the manifest validation JSONL path")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--record-root")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--deadline-hours", type=float, default=72.0)
    parser.add_argument("--minimum-free-gib", type=int, default=200)
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Required to create artifacts or launch work")
    return parser


def main() -> None:
    execute(build_parser().parse_args())


if __name__ == "__main__":
    main()
