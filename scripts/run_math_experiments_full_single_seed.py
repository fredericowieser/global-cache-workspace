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
from gcw.runtime import atomic_write_bytes
from gcw.training.checkpoint import verify_training_checkpoint_files
from scripts.generate_math_experiment_configs import generate

EXPERIMENT = "qwen3-06b-instruct-math-full-single-seed-v2-matrix11"
CANONICAL_CORE_TASKS = (
    "hellaswag",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "mmlu",
)
CANONICAL_REASONING_TASKS = (
    "logiqa",
    "agieval_gaokao_mathqa",
)
CANONICAL_TASKS = (*CANONICAL_CORE_TASKS, *CANONICAL_REASONING_TASKS)
NUMERIC_MATH_TASKS = (
    "gsm8k_openmath",
    "svamp",
    "gsm8k_hard",
)
EVALUATION_ROOT = "evaluations-v3"
LEGACY_EVALUATION_ROOT = "evaluations-v2"
TRAIN_ORDER = (
    "full-sft",
    "lora-r32",
    "bt",
    "bgt",
    "icarus",
    "diffusion-bgt-t4-sigma015",
    "diffusion-bgt-t4-sigma0",
    "longshot",
    "longshot-internal-r4",
    "longshot-external-r4",
    "flow-bgt-t4",
)
IMPORTED_ARMS = frozenset({"full-sft"})
TRAINABLE_ARMS = tuple(arm for arm in TRAIN_ORDER if arm not in IMPORTED_ARMS)
PROCESSOR_ARMS = frozenset(
    {
        "bt",
        "bgt",
        "icarus",
        "diffusion-bgt-t4-sigma015",
        "diffusion-bgt-t4-sigma0",
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
    "icarus": 2,
    "diffusion-bgt-t4-sigma015": 4,
    "diffusion-bgt-t4-sigma0": 4,
    "longshot": 2,
    "longshot-internal-r4": 2,
    "longshot-external-r4": 1,
    "lora-r32": 8,
    "flow-bgt-t4": 4,
}
PEAK_LEARNING_RATES = {
    "full-sft": 5e-6,
    "bt": 1e-3,
    "bgt": 1e-3,
    "icarus": 1e-3,
    "diffusion-bgt-t4-sigma015": 1e-3,
    "diffusion-bgt-t4-sigma0": 1e-3,
    "longshot": 1e-3,
    "longshot-internal-r4": 1e-3,
    "longshot-external-r4": 1e-3,
    "lora-r32": 1e-4,
    "flow-bgt-t4": 1e-3,
}
SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_ROWS = 64
SEED = 24
DATASET_ROWS = 697_744
EFFECTIVE_EPOCH_ROWS = 697_728
MAX_STEPS = 10_902
VALIDATION_ROWS = 1024
CHECKPOINT_EVERY = 256
WANDB_ENTITY = "huawei-noahs-ark"
WANDB_PROJECT = "global-cache-workspace"
FULL_SFT_IMPORT_IDENTITIES = {
    "source_commit": "70e1de149e82cd95db5d247451c3490283f86ea9",
    "training_config_sha256": "18c574e36a30212f679fe4938aa2c7eb5c72cff39894625898a998c9a7f0ba78",
    "source_checkpoint_manifest_sha256": "73513e2c8e41b6d8a20cfada45f93e0198f92eb9a6a7b1c26199e9986cc1dfdc",
    "config.json": "ffdff0e6e3f305fe54c7432c905e50114ea3fc5f64313a90522278923c56e6cc",
    "model.safetensors": "2685068b048ac0d3e7adadc1b2961bd528a1ce07628a2e6d050e26f8f171b434",
    "train_state.json": "444cf9bad9e87d1181b74097205a9fbc671ce3ca405232289983dde32f1ee310",
}

_FORBIDDEN_PERFORMANCE_ENV = (
    "GCW_ALLOW_UNQUALIFIED_CUDA_GRAPH",
    "GCW_ALLOW_UNQUALIFIED_TRITON_CACHE",
    "GCW_ALLOW_REJECTED_ICARUS_COMPILE",
    "GCW_BT_PROCESSOR_BACKEND",
    "GCW_CACHE_PROCESSOR_BACKEND",
    "GCW_COMPILE",
    "GCW_ICARUS_PROCESSOR_BACKEND",
)


def _reject_performance_overrides(env: dict[str, str]) -> None:
    enabled = sorted(
        name for name in _FORBIDDEN_PERFORMANCE_ENV if env.get(name, "").strip()
    )
    if enabled:
        raise RuntimeError(
            f"canonical training forbids performance overrides: {enabled}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def _require_clean_repository(repository: Path) -> None:
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    if status.strip():
        raise RuntimeError(
            "repository must contain no tracked changes or untracked import shadows"
        )


def _capture_jsonl_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            rows += bool(line.strip())
    return digest.hexdigest(), rows


def _validate_campaign_config(
    config: Any, *, arm: str, root: Path, experiment: str
) -> None:
    if arm not in TRAIN_ORDER:
        raise ValueError(f"unknown campaign arm {arm}")
    if (
        config.max_steps != MAX_STEPS
        or config.global_batch_rows != GLOBAL_BATCH_ROWS
    ):
        raise ValueError(
            f"{arm} does not preserve the full-epoch optimizer contract"
        )
    if config.sequence_len != SEQUENCE_LENGTH or config.seed != SEED:
        raise ValueError(f"{arm} does not preserve sequence length and seed")
    if config.learning_rate != PEAK_LEARNING_RATES[arm]:
        raise ValueError(f"{arm} has the wrong peak learning rate")
    if config.batch_size != MICRO_BATCH_ROWS[arm]:
        raise ValueError(f"{arm} has the wrong physical microbatch")
    if (
        config.metadata.get("arm") != arm
        or config.metadata.get("experiment") != experiment
    ):
        raise ValueError(f"{arm} metadata does not bind the campaign identity")
    expected_output = (root / "training-state" / arm).resolve()
    if Path(config.output_dir).resolve() != expected_output:
        raise ValueError(f"{arm} output directory is outside the campaign root")
    if config.backbone_chunk_backend != "eager":
        raise ValueError(
            f"{arm} requests a rejected backbone execution backend"
        )
    performance = config.performance
    if (
        performance.adaptive_batching.enabled
        or performance.reference_loss.mode != "legacy"
        or performance.longshot_full_history.enabled
        or performance.kv_score_sidecar.enabled
        or performance.triton_cache_movement.enabled
        or performance.optimized_step.enabled
        or performance.noisy_diffusion.recurrence_backend != "eager"
    ):
        raise ValueError(f"{arm} requests a noncanonical performance policy")
    if arm in PROCESSOR_ARMS:
        if config.trainable != "cache_processor":
            raise ValueError(f"{arm} must train only its CacheProcessor")
        if config.activation_checkpoint_windows != 4:
            raise ValueError(
                f"{arm} must use the memory-safe four-window checkpoint oracle"
            )
        if config.activation_checkpoint_min_padded_sequence_length != 0:
            raise ValueError(
                f"{arm} must checkpoint every eligible backbone window"
            )
    elif config.activation_checkpoint_windows != 0:
        raise ValueError(
            f"{arm} must not enable CacheProcessor checkpoint execution"
        )


def _load_data_manifest(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(Path(args.data_manifest).expanduser().read_text())
    dataset = (
        Path(args.dataset or payload["dataset_path"]).expanduser().resolve()
    )
    validation = (
        Path(args.validation_dataset or payload["validation_path"])
        .expanduser()
        .resolve()
    )
    observed_rows: dict[str, int] = {}
    for name, path, expected in (
        ("dataset", dataset, payload["dataset_sha256"]),
        ("validation", validation, payload["validation_sha256"]),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_sha256, observed_rows[name] = _capture_jsonl_identity(path)
        if observed_sha256 != expected:
            raise ValueError(
                f"{name} sha256 does not match the authenticated data manifest"
            )
    if payload.get("schema") != "gcw.math_experiment_data.v1":
        raise ValueError(
            "math campaign requires the authenticated data-manifest schema"
        )
    if (
        int(payload.get("rows", -1)) != DATASET_ROWS
        or observed_rows["dataset"] != DATASET_ROWS
    ):
        raise ValueError(
            "math campaign training data must contain exactly 697,744 nonblank rows"
        )
    if (
        int(payload.get("validation_rows", -1)) != 4096
        or observed_rows["validation"] != 4096
    ):
        raise ValueError(
            "math campaign validation data must contain exactly 4,096 nonblank rows"
        )
    if (
        payload.get("seed") != SEED
        or payload.get("sequence_length") != SEQUENCE_LENGTH
    ):
        raise ValueError(
            "math campaign data manifest has the wrong seed or sequence length"
        )
    payload["dataset_path"] = str(dataset)
    payload["validation_path"] = str(validation)
    return payload


def _plan(
    args: argparse.Namespace, data: dict[str, Any], source_commit: str
) -> dict[str, Any]:
    rows = int(data["rows"])
    effective_rows = rows - rows % GLOBAL_BATCH_ROWS
    dropped_rows = rows - effective_rows
    max_steps = effective_rows // GLOBAL_BATCH_ROWS
    if effective_rows != EFFECTIVE_EPOCH_ROWS or max_steps != MAX_STEPS:
        raise ValueError(
            "math campaign epoch derivation differs from the approved contract"
        )
    return {
        "schema": "gcw.math_experiment_series.v2",
        "experiment": args.experiment,
        "source_commit": source_commit,
        "status": "planned_not_started",
        "backbone": str(Path(args.model).expanduser().resolve()),
        "backbone_checkpoint": str(
            Path(args.backbone_checkpoint).expanduser().resolve()
        ),
        "sequence_length": SEQUENCE_LENGTH,
        "global_batch_rows": GLOBAL_BATCH_ROWS,
        "seed": SEED,
        "data_identity": {
            "manifest": str(Path(args.data_manifest).expanduser().resolve()),
            "dataset": str(data["dataset_path"]),
            "dataset_sha256": data["dataset_sha256"],
            "validation": str(data["validation_path"]),
            "validation_sha256": data["validation_sha256"],
            "source_ids_sha256": data["source_ids_sha256"],
            "tokenizer_identity_sha256": data["tokenizer_identity_sha256"],
        },
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
        "validation": {
            "rows": VALIDATION_ROWS,
            "approximately_every_steps": max_steps // 20,
        },
        "training_order": list(TRAINABLE_ARMS),
        "imported_arms": sorted(IMPORTED_ARMS),
        "evaluation_order": ["instruct-baseline", *TRAIN_ORDER],
        "tasks": {
            "canonical_multiple_choice": list(CANONICAL_TASKS),
            "numeric_math": list(NUMERIC_MATH_TASKS),
        },
        "evaluation_protocols": {
            "canonical_likelihood": {
                "tasks": list(CANONICAL_TASKS),
                "prompt": "plain_no_chat_template",
                "thinking_dimension": False,
                "coverage": "full",
                "scoring": "official_lm_evaluation_harness_task_definition",
                "gaokao_multi_answer_rule": "argmax_option_in_gold_set",
            },
            "numeric_generation": {
                "tasks": list(NUMERIC_MATH_TASKS),
                "prompt": "official_chat_template",
                "thinking_modes": ["thinking_off", "thinking_on"],
                "max_new_tokens": 2048,
                "coverage": "full",
                "scoring": "numeric_last_answer",
            },
        },
        "evaluation_processor_states": {
            "non_processor": ["on"],
            "processor": ["off", "on"],
        },
    }


def _verify_full_sft_import(bundle: Path) -> dict[str, Any]:
    from scripts.math_evaluation_bundle import verify_bundle

    manifest = verify_bundle(
        bundle, expected_arm="full-sft", expected_step=MAX_STEPS
    )
    for field in (
        "source_commit",
        "training_config_sha256",
        "source_checkpoint_manifest_sha256",
    ):
        if manifest.get(field) != FULL_SFT_IMPORT_IDENTITIES[field]:
            raise ValueError(f"imported full-SFT bundle has the wrong {field}")
    for name in ("config.json", "model.safetensors", "train_state.json"):
        if (
            manifest.get("files", {}).get(name, {}).get("sha256")
            != FULL_SFT_IMPORT_IDENTITIES[name]
        ):
            raise ValueError(
                f"imported full-SFT bundle has the wrong {name} identity"
            )
    return manifest


def _write_import_manifest(
    args: argparse.Namespace, root: Path, source_commit: str
) -> None:
    if args.import_full_sft_bundle is None:
        raise ValueError(
            "the additive matrix requires --import-full-sft-bundle"
        )
    bundle = Path(args.import_full_sft_bundle).expanduser().resolve()
    manifest = _verify_full_sft_import(bundle)
    bundle_manifest = bundle / "bundle-manifest.json"
    payload = {
        "schema": "gcw.math_campaign_imports.v1",
        "experiment": args.experiment,
        "preparation_source_commit": source_commit,
        "imports": {
            "full-sft": {
                "bundle": str(bundle),
                "bundle_manifest_sha256": _sha256(bundle_manifest),
                "training_source_commit": manifest["source_commit"],
                "training_config_sha256": manifest["training_config_sha256"],
                "source_checkpoint_manifest_sha256": manifest[
                    "source_checkpoint_manifest_sha256"
                ],
                "model_sha256": FULL_SFT_IMPORT_IDENTITIES["model.safetensors"],
                "step": MAX_STEPS,
            }
        },
    }
    path = root / "imports.json"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(
                "refusing to replace a different immutable import manifest"
            )
        return
    atomic_write_bytes(path, encoded, exclusive=True)


def _wandb_id(experiment: str, name: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", experiment).strip("-")[:48]
    suffix = hashlib.sha256(f"{experiment}:{name}:{SEED}".encode()).hexdigest()[
        :12
    ]
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
            raise FileNotFoundError(
                f"incomplete existing config directory: {missing}"
            )
        existing_plan_path = root / "plan.json"
        if not existing_plan_path.is_file():
            raise RuntimeError(
                "existing campaign configs have no immutable launch plan"
            )
        existing_plan = json.loads(existing_plan_path.read_text())
        if existing_plan.get("source_commit") != plan["source_commit"]:
            raise RuntimeError(
                "existing campaign configs were prepared from another source commit"
            )
        if set(existing_plan.get("arms", {})) != set(TRAIN_ORDER):
            raise RuntimeError(
                "existing campaign plan has the wrong arm inventory"
            )
        for arm, path in paths.items():
            config = load_train_config(path)
            _validate_campaign_config(
                config, arm=arm, root=root, experiment=str(plan["experiment"])
            )
            if existing_plan["arms"][arm].get("config_sha256") != _sha256(path):
                raise RuntimeError(
                    f"existing campaign config digest differs for {arm}"
                )
        return paths

    runtime_manifest = root / "data-manifest.json"
    runtime_manifest.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n"
    )
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
            raise ValueError(
                f"microbatch for {arm} does not divide global batch"
            )
        payload["batch_size"] = micro_batch
        payload["learning_rate"] = PEAK_LEARNING_RATES[arm]
        payload["warmup_steps"] = max(1, round(0.03 * plan["max_steps"]))
        payload["warmdown_ratio"] = 0.10
        payload["final_lr_frac"] = 0.0
        payload["save_every"] = CHECKPOINT_EVERY
        payload["save_final"] = True
        if arm in PROCESSOR_ARMS:
            payload["activation_checkpoint_windows"] = 4
            payload["activation_checkpoint_min_padded_sequence_length"] = 0
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
                "tags": [
                    "qwen3-0.6b",
                    "math",
                    "single-seed",
                    "full-epoch",
                    arm,
                ],
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
        _validate_campaign_config(
            load_train_config(path),
            arm=arm,
            root=root,
            experiment=args.experiment,
        )
        result[arm] = path
    for name, path in all_paths.items():
        if name not in TRAIN_ORDER:
            path.unlink()
    return result


def _completed_training(config_path: Path) -> bool:
    config = load_train_config(config_path)
    checkpoint = Path(config.output_dir) / "checkpoint"
    state = checkpoint / "train_state.json"
    if (
        not state.is_file()
        or int(json.loads(state.read_text()).get("step", -1))
        != config.max_steps
    ):
        return False
    verified = verify_training_checkpoint_files(checkpoint)
    if int(verified["train_state"].get("step", -1)) != config.max_steps:
        raise RuntimeError(
            "verified checkpoint step differs from the configured final step"
        )
    return True


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
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )


def _deadline_reached(deadline: datetime) -> bool:
    return datetime.now(timezone.utc) >= deadline


def _processor_states(arm: str) -> tuple[str, ...]:
    return ("off", "on") if arm in PROCESSOR_ARMS else ("on",)


def _evaluation_cells(arm: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (processor, thinking)
        for processor in _processor_states(arm)
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


def _evaluation_base(args: argparse.Namespace, root: Path) -> Path:
    return root / getattr(args, "evaluation_root", EVALUATION_ROOT)


def _evaluation_lineage(args: argparse.Namespace) -> str:
    return str(getattr(args, "evaluation_lineage", "nebius-h200-v1"))


def _mirror_evaluation(
    result: Path,
    args: argparse.Namespace,
    arm: str,
    processor: str,
    protocol: str,
    *,
    thinking: str | None,
    tasks: tuple[str, ...],
    artifact_files: tuple[Path, ...] = (),
) -> None:
    import wandb

    cell_name = f"{protocol}-{thinking}" if thinking is not None else protocol
    lineage = _evaluation_lineage(args)
    run_name = f"{args.experiment}-eval-{lineage}-{arm}-{processor}-{cell_name}"
    tags = [
        "qwen3-0.6b",
        "math",
        "single-seed",
        "evaluation",
        arm,
        processor,
        protocol,
    ]
    if thinking is not None:
        tags.append(thinking)
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        group=args.experiment,
        job_type="evaluation",
        id=_wandb_id(
            args.experiment, f"eval-v3-{lineage}-{arm}-{processor}-{cell_name}"
        ),
        resume="allow",
        tags=[*tags, lineage],
        config={
            "experiment": args.experiment,
            "arm": arm,
            "processor": processor,
            "protocol": protocol,
            "thinking": thinking or "not_applicable",
            "tasks": list(tasks),
            "batch_size": args.eval_batch_size,
            "evaluation_lineage": lineage,
            "evaluator_host": os.uname().nodename,
            "evaluator_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cpu"
            ),
        },
    )
    if run is None:
        raise RuntimeError("wandb.init returned no evaluation run")
    payload = json.loads(result.read_text())
    run.log(_flatten_numbers(payload), step=0)
    artifact = wandb.Artifact(run_name, type="benchmark-results")
    artifact.add_file(str(result), name="result.json")
    for path in artifact_files:
        artifact.add_file(
            str(path), name=f"tasks/{path.parent.name}/result.json"
        )
    run.log_artifact(artifact)
    run.finish()


def _target_args(target: Path, *, hf_target: bool) -> list[str]:
    return (
        ["--hf-model", str(target)]
        if hf_target
        else ["--checkpoint", str(target)]
    )


def _reuse_result(source: Path, target: Path, *, protocol: str) -> bool:
    if not source.is_file() or target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if _sha256(source) != _sha256(target):
        target.unlink()
        raise RuntimeError(
            f"reused evaluation evidence changed while copying {source}"
        )
    marker = target.parent / "reused-evidence.json"
    marker.write_text(
        json.dumps(
            {
                "schema": "gcw.reused_evaluation_evidence.v1",
                "source": str(source.resolve()),
                "source_sha256": _sha256(source),
                "protocol": protocol,
                "bytes_preserved": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return True


def _validate_canonical_result(result: Path, tasks: tuple[str, ...]) -> None:
    payload = json.loads(result.read_text())
    protocol = payload.get("gcw_protocol", {})
    if (
        protocol.get("prompt_mode") != "plain"
        or protocol.get("apply_chat_template") is not False
    ):
        raise ValueError(
            f"canonical result does not use the plain no-chat protocol: {result}"
        )
    missing = sorted(set(tasks) - set(payload.get("results", {})))
    if missing:
        raise ValueError(
            f"canonical result is missing tasks {missing}: {result}"
        )


def _evaluate_canonical_group(
    args: argparse.Namespace,
    root: Path,
    arm: str,
    processor: str,
    target: Path,
    *,
    group: str,
    tasks: tuple[str, ...],
    hf_target: bool,
    env: dict[str, str],
) -> None:
    cell = (
        _evaluation_base(args, root)
        / arm
        / f"processor-{processor}"
        / "canonical-likelihood"
        / group
    )
    result = cell / "result.json"
    published = cell / "wandb-published.json"
    if (
        arm == "instruct-baseline"
        and processor == "on"
        and group == "core"
        and not result.exists()
    ):
        legacy = (
            root
            / LEGACY_EVALUATION_ROOT
            / arm
            / "processor-on"
            / "standard-general"
            / "result.json"
        )
        _reuse_result(
            legacy, result, protocol="canonical_plain_likelihood_core_v1"
        )
    if not result.is_file():
        cell.mkdir(parents=True, exist_ok=True)
        cell_env = dict(env)
        cell_env["GCW_EVAL_PARTIAL_DIR"] = str(cell / "partial")
        command = [
            sys.executable,
            "-m",
            "gcw.cli",
            "eval",
            "general" if group == "core" else "tasks",
        ]
        if group != "core":
            command.extend(["--tasks", *tasks])
        command.extend(
            [
                "--batch-size",
                str(args.eval_batch_size),
                "--prompt-mode",
                "plain",
                "--processor",
                processor,
                "--out",
                str(result),
                *_target_args(target, hf_target=hf_target),
            ]
        )
        _run(command, cell / "eval.log", cell_env)
    _validate_canonical_result(result, tasks)
    if not published.is_file():
        _mirror_evaluation(
            result,
            args,
            arm,
            processor,
            f"canonical-likelihood-{group}-v1",
            thinking=None,
            tasks=tasks,
        )
        published.write_text(
            json.dumps(
                {"result_sha256": _sha256(result), "published": True}, indent=2
            )
            + "\n"
        )


def _numeric_summary(
    task_results: list[Path], *, arm: str, processor: str, thinking: str
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for path in task_results:
        payload = json.loads(path.read_text())
        task = path.parent.name
        tasks[task] = {
            "result_sha256": _sha256(path),
            "dataset": payload["datasets"][task],
            "metrics": payload["tasks"][task],
            "elapsed_seconds": payload["elapsed_seconds"],
            "peak_memory_bytes": payload["peak_memory_bytes"],
            "payload_sha256": payload["payload_sha256"],
        }
    return {
        "schema": "gcw.numeric_math_benchmark_composite.v1",
        "arm": arm,
        "processor": processor,
        "thinking": thinking,
        "protocol": "official_chat_template_numeric_last_v1",
        "max_new_tokens": 2048,
        "tasks": tasks,
    }


def _validate_numeric_result(result: Path, task: str, thinking: str) -> None:
    payload = json.loads(result.read_text())
    if payload.get("mode") != thinking or payload.get("max_new_tokens") != 2048:
        raise ValueError(
            f"numeric result has the wrong generation contract: {result}"
        )
    if set(payload.get("tasks", {})) != {task} or set(
        payload.get("datasets", {})
    ) != {task}:
        raise ValueError(f"numeric result has the wrong task payload: {result}")
    rows = payload.get("rows", [])
    if not rows or any(row.get("target_kind") != "numeric" for row in rows):
        raise ValueError(
            f"numeric result contains non-numeric or missing rows: {result}"
        )


def _evaluate_numeric_mode(
    args: argparse.Namespace,
    root: Path,
    arm: str,
    processor: str,
    thinking: str,
    target: Path,
    *,
    hf_target: bool,
    env: dict[str, str],
    deadline: datetime,
) -> bool:
    cell = (
        _evaluation_base(args, root)
        / arm
        / f"processor-{processor}"
        / "numeric-generation"
        / thinking
    )
    task_results: list[Path] = []
    for task in NUMERIC_MATH_TASKS:
        if _deadline_reached(deadline):
            return False
        task_cell = cell / "tasks" / task
        task_result = task_cell / "result.json"
        task_results.append(task_result)
        if (
            arm == "instruct-baseline"
            and processor == "on"
            and not task_result.exists()
        ):
            legacy = (
                root
                / LEGACY_EVALUATION_ROOT
                / arm
                / "processor-on"
                / "chat-generation"
                / thinking
                / "tasks"
                / task
                / "result.json"
            )
            _reuse_result(
                legacy,
                task_result,
                protocol="official_chat_template_numeric_last_v1",
            )
        if not task_result.is_file():
            task_cell.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "gcw.cli",
                "eval",
                "chat",
                "--tasks",
                task,
                "--mode",
                thinking,
                "--batch-size",
                str(args.eval_batch_size),
                "--max-new-tokens",
                "2048",
                "--processor",
                processor,
                "--tokenizer-path",
                str(Path(args.model).expanduser().resolve()),
                "--out",
                str(task_result),
                *_target_args(target, hf_target=hf_target),
            ]
            _run(command, task_cell / "eval.log", env)
        _validate_numeric_result(task_result, task, thinking)
    result = cell / "result.json"
    published = cell / "wandb-published.json"
    if not result.is_file():
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(
            json.dumps(
                _numeric_summary(
                    task_results,
                    arm=arm,
                    processor=processor,
                    thinking=thinking,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    if not published.is_file():
        _mirror_evaluation(
            result,
            args,
            arm,
            processor,
            "numeric-generation-v1",
            thinking=thinking,
            tasks=NUMERIC_MATH_TASKS,
            artifact_files=tuple(task_results),
        )
        published.write_text(
            json.dumps(
                {"result_sha256": _sha256(result), "published": True}, indent=2
            )
            + "\n"
        )
    return True


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
    for processor in _processor_states(arm):
        if _deadline_reached(deadline):
            return False
        _evaluate_canonical_group(
            args,
            root,
            arm,
            processor,
            target,
            group="core",
            tasks=CANONICAL_CORE_TASKS,
            hf_target=hf_target,
            env=env,
        )
        _evaluate_canonical_group(
            args,
            root,
            arm,
            processor,
            target,
            group="reasoning",
            tasks=CANONICAL_REASONING_TASKS,
            hf_target=hf_target,
            env=env,
        )
        for thinking in ("thinking_off", "thinking_on"):
            if not _evaluate_numeric_mode(
                args,
                root,
                arm,
                processor,
                thinking,
                target,
                hf_target=hf_target,
                env=env,
                deadline=deadline,
            ):
                return False
    return True


def _preflight(
    args: argparse.Namespace, repository: Path, root: Path, env: dict[str, str]
) -> None:
    _require_clean_repository(repository)
    _reject_performance_overrides(env)
    ambient_stops = sorted(
        name
        for name in (
            "GCW_STOP_AFTER_CHECKPOINT_STEP",
            "GCW_STOP_AFTER_CHECKPOINT_UTC",
        )
        if env.get(name)
    )
    if ambient_stops:
        raise RuntimeError(
            f"monolithic campaign forbids ambient checkpoint stops: {ambient_stops}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_name = torch.cuda.get_device_name(0)
    expected_gpu = str(getattr(args, "expected_gpu", "H200"))
    if expected_gpu not in device_name and not args.allow_other_gpu:
        raise RuntimeError(
            f"expected one {expected_gpu}, found {device_name!r}"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one visible GPU, found {torch.cuda.device_count()}"
        )
    usage = shutil.disk_usage(root.parent)
    if usage.free < args.minimum_free_gib * 1024**3:
        raise RuntimeError(
            f"less than {args.minimum_free_gib} GiB free under {root.parent}"
        )
    hf_cli = Path(sys.executable).parent / "hf"
    if not hf_cli.is_file():
        raise FileNotFoundError(
            "Hugging Face CLI is missing; sync the project environment first"
        )
    subprocess.run([str(hf_cli), "auth", "whoami"], check=True, env=env)
    import wandb

    lineage = _evaluation_lineage(args)
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"{args.experiment}-preflight-{lineage}",
        group=args.experiment,
        job_type="preflight",
        id=_wandb_id(args.experiment, f"preflight-{lineage}"),
        resume="allow",
        tags=[lineage],
        config={
            "source_commit": _source_commit(repository),
            "device": device_name,
            "evaluation_lineage": lineage,
            "host": os.uname().nodename,
        },
    )
    if run is None:
        raise RuntimeError("W&B preflight returned no run")
    run.log({"preflight/ok": 1}, step=0)
    run.finish()


def _write_plan_with_costs(
    plan: dict[str, Any], configs: dict[str, Path], path: Path
) -> None:
    payload = deepcopy(plan)
    payload["status"] = "ready_not_started"
    payload["arms"] = {}
    for arm, config_path in configs.items():
        config = load_train_config(config_path)
        maximum_tokens = (
            config.global_batch_rows * config.sequence_len * config.max_steps
        )
        payload["arms"][arm] = {
            "config": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "processor_params": processor_params(config.model),
            "maximum_tokens": maximum_tokens,
            "estimated_training_flops": total_training_flops(
                config.model, maximum_tokens, config.sequence_len
            ),
        }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(
                "refusing to replace a different immutable campaign plan"
            )
        return
    atomic_write_bytes(path, encoded, exclusive=True)


def execute(args: argparse.Namespace) -> None:
    repository = Path(__file__).resolve().parents[1]
    source_commit = _source_commit(repository)
    data = _load_data_manifest(args)
    plan = _plan(args, data, source_commit)
    root = Path(args.experiment_root).expanduser().resolve()
    if args.execute and args.prepare_only:
        raise ValueError("--execute and --prepare-only are mutually exclusive")
    if not args.execute and not args.prepare_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    _require_clean_repository(repository)
    root.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        configs = _prepare_configs(args, root, data, plan)
        _write_plan_with_costs(plan, configs, root / "plan.json")
        _write_import_manifest(args, root, source_commit)
        print(
            json.dumps(
                {"root": str(root), "plan": str(root / "plan.json")},
                sort_keys=True,
            )
        )
        return
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
    _write_import_manifest(args, root, source_commit)
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

    full_sft_bundle = Path(args.import_full_sft_bundle).expanduser().resolve()
    if not _evaluate(
        args,
        root,
        "full-sft",
        full_sft_bundle,
        hf_target=False,
        env=env,
        deadline=deadline,
    ):
        return

    for arm in TRAINABLE_ARMS:
        if _deadline_reached(deadline):
            return
        config_path = configs[arm]
        if not _completed_training(config_path):
            command_config = _resume_config(config_path, root)
            _run(
                [
                    sys.executable,
                    "-m",
                    "gcw.cli",
                    "train",
                    "--config",
                    str(command_config),
                ],
                root / "logs" / f"train-{arm}.log",
                env,
            )
        checkpoint = (
            Path(load_train_config(config_path).output_dir) / "checkpoint"
        )
        if not _completed_training(config_path):
            raise RuntimeError(
                f"training did not reach the declared epoch for {arm}"
            )
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
        "schema": "gcw.math_experiment_series_completion.v2",
        "experiment": args.experiment,
        "source_commit": source_commit,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": _sha256(root / "plan.json"),
        "imports_sha256": _sha256(root / "imports.json"),
    }
    (root / "completed.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the single-seed full-epoch Qwen3 math training and evaluation series"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Authenticated local Qwen3-0.6B Instruct HF directory",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        required=True,
        help="Authenticated native GCW backbone",
    )
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument(
        "--dataset",
        help="Override the manifest training JSONL path without changing its hash",
    )
    parser.add_argument(
        "--validation-dataset",
        help="Override the manifest validation JSONL path",
    )
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--import-full-sft-bundle")
    parser.add_argument("--record-root")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--evaluation-root", default=EVALUATION_ROOT)
    parser.add_argument("--evaluation-lineage", default="nebius-h200-v1")
    parser.add_argument("--deadline-hours", type=float, default=72.0)
    parser.add_argument("--minimum-free-gib", type=int, default=200)
    parser.add_argument("--expected-gpu", default="H200")
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create immutable configs and plan without CUDA",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required to create artifacts or launch work",
    )
    return parser


def main() -> None:
    execute(build_parser().parse_args())


if __name__ == "__main__":
    main()
