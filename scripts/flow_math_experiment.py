#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from gcw.checkpoint import load_checkpoint
from gcw.config import load_train_config
from gcw.flow_math_experiment import (
    BACKBONE_PATH,
    BASELINE_DIR,
    CANONICAL_ARTIFACT_SPECS,
    CONFIG_DIR,
    build_experiment_manifest,
    checkpoint_metadata_sha256,
    build_directory_manifest,
    extract_pilot_metrics,
    extract_primary_metrics,
    load_json_strict,
    load_matrix_file,
    probe_flow_dynamics,
    sha256_file,
    summarise_metric_markers,
    verify_completed_flow_math_run,
    verify_directory_manifest,
    verify_execution_context,
    verify_run_marker,
    verify_training_audit,
)


EXPERIMENT_ROOT = Path("/mnt/jfs-01/shared/gcw/experiments/flow_math_llama32_1b_v1")
EVAL_PILOT_LIMIT = 16


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _verify_context(args: argparse.Namespace) -> None:
    manifest = verify_execution_context(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest,
        approved_manifest_sha256=args.approved_manifest_sha256,
        required_artifacts=CANONICAL_ARTIFACT_SPECS,
    )
    if manifest.get("source_commit") != args.source_commit:
        raise ValueError("SOURCE_COMMIT differs from the authenticated manifest")
    print(json.dumps({"source_commit": args.source_commit, "status": "verified"}, sort_keys=True))


def _matrix_row(args: argparse.Namespace) -> None:
    rows = load_matrix_file(args.matrix)
    if not 0 <= args.index < len(rows):
        raise ValueError(f"matrix index must be in [0, {len(rows) - 1}]")
    row = rows[args.index]
    print(f"{row['variant']}\t{row['seed']}\t{row['config']}")


def _build_manifest(args: argparse.Namespace) -> None:
    artifacts = {
        name: (Path(str(spec["path"])), str(spec["sha256"]))
        for name, spec in CANONICAL_ARTIFACT_SPECS.items()
    }
    payload = build_experiment_manifest(
        repo_root=args.repo_root.resolve(),
        source_commit=args.source_commit,
        artifacts=artifacts,
    )
    _write_json_exclusive(args.out, payload)
    print(args.out)


def _build_cache_manifest(args: argparse.Namespace) -> None:
    writable = [path for path in (args.cache_dir, *args.cache_dir.rglob("*")) if path.stat().st_mode & 0o222]
    if writable:
        raise ValueError(f"refusing to manifest writable evaluation cache: {writable[0]}")
    payload = build_directory_manifest(args.cache_dir)
    _write_json_exclusive(args.out, payload)
    print(args.out)


def _verify_cache_manifest(args: argparse.Namespace) -> None:
    payload = verify_directory_manifest(
        root=args.cache_dir,
        manifest_path=args.manifest,
        approved_manifest_sha256=args.approved_manifest_sha256,
        require_read_only=args.require_read_only,
    )
    print(json.dumps({"files": len(payload["files"]), "status": "verified"}, sort_keys=True))


def _verify_run(args: argparse.Namespace) -> None:
    payload = verify_completed_flow_math_run(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        backbone_model_path=BACKBONE_PATH,
    )
    payload["source_commit"] = args.source_commit
    payload["experiment_manifest_sha256"] = args.experiment_manifest_sha256
    _write_json_exclusive(args.out, payload)
    print(args.out)


def _check_run_marker(args: argparse.Namespace) -> None:
    payload = verify_run_marker(
        marker_path=args.marker,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        backbone_model_path=BACKBONE_PATH,
        source_commit=args.source_commit,
        experiment_manifest_sha256=args.experiment_manifest_sha256,
    )
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


def _verify_pilot_audit(args: argparse.Namespace) -> None:
    config = load_train_config(args.config)
    payload = verify_training_audit(
        args.audit,
        architecture=config.model.architecture_name,
        max_steps=config.max_steps,
        source_commit=args.source_commit,
        experiment_manifest_sha256=args.experiment_manifest_sha256,
    )
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


def _pilot_name(config_path: Path) -> str:
    names = {
        (CONFIG_DIR / "pilots/llama32_1b_flow_bgt_steps2.yaml").as_posix(): "bgt-steps2",
        (CONFIG_DIR / "pilots/llama32_1b_flow_icarus_steps2.yaml").as_posix(): "icarus-steps2",
        (CONFIG_DIR / "pilots/llama32_1b_flow_longshot_steps2.yaml").as_posix(): "longshot-steps2",
        (CONFIG_DIR / "pilots/llama32_1b_flow_bgt_steps100.yaml").as_posix(): "bgt-steps100",
    }
    try:
        relative = config_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        return names[relative]
    except (KeyError, ValueError) as error:
        raise ValueError(f"config is not an approved pilot: {config_path}") from error


def _validate_dynamics_payload(
    payload: Any,
    *,
    checkpoint: Path,
    checkpoint_payload: dict[str, Any],
    source_commit: str,
    experiment_manifest_sha256: str,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("dynamics payload must be an object")
    scalar_fields = ("max_abs_logit_diff", "max_flow_displacement_l2")
    scalars = []
    for field in scalar_fields:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"dynamics field is not finite: {field}")
        scalars.append(float(value))
    deltas = payload.get("flow_gate_delta_from_init")
    if not isinstance(deltas, list) or not deltas:
        raise ValueError("dynamics gate delta list is empty")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in deltas
    ):
        raise ValueError("dynamics gate deltas must be finite real numbers")
    if (
        payload.get("source_commit") != source_commit
        or payload.get("experiment_manifest_sha256") != experiment_manifest_sha256
        or payload.get("checkpoint") != str(checkpoint.resolve())
        or payload.get("checkpoint_model_sha256") != checkpoint_payload["checkpoint_model_sha256"]
        or payload.get("checkpoint_manifest_sha256") != checkpoint_payload["checkpoint_manifest_sha256"]
        or min(scalars) <= 0.0
        or max(abs(float(value)) for value in deltas) <= 0.0
    ):
        raise ValueError("dynamics payload failed checkpoint identity or non-triviality checks")


def _finalize_pilot(args: argparse.Namespace) -> None:
    config = load_train_config(args.config)
    name = _pilot_name(args.config)
    checkpoint = Path(config.output_dir) / "checkpoint"
    checkpoint_payload = verify_run_marker(
        marker_path=args.checkpoint_marker,
        config_path=args.config,
        checkpoint_path=checkpoint,
        backbone_model_path=BACKBONE_PATH,
        source_commit=args.source_commit,
        experiment_manifest_sha256=args.experiment_manifest_sha256,
    )
    audit_payload = verify_training_audit(
        args.audit,
        architecture=config.model.architecture_name,
        max_steps=config.max_steps,
        source_commit=args.source_commit,
        experiment_manifest_sha256=args.experiment_manifest_sha256,
    )
    dynamics_payload = None
    if config.max_steps == 100:
        if args.dynamics is None:
            raise ValueError("100-step pilot requires a dynamics record")
        dynamics_payload = load_json_strict(args.dynamics)
        _validate_dynamics_payload(
            dynamics_payload,
            checkpoint=checkpoint,
            checkpoint_payload=checkpoint_payload,
            source_commit=args.source_commit,
            experiment_manifest_sha256=args.experiment_manifest_sha256,
        )
    elif args.dynamics is not None:
        raise ValueError("two-step pilot must not claim a dynamics record")
    payload = {
        "passed": True,
        "pilot_name": name,
        "source_commit": args.source_commit,
        "experiment_manifest_sha256": args.experiment_manifest_sha256,
        "config": str(args.config),
        "config_sha256": checkpoint_payload["config_sha256"],
        "checkpoint_marker": str(args.checkpoint_marker),
        "checkpoint_marker_sha256": sha256_file(args.checkpoint_marker),
        "checkpoint_manifest_sha256": checkpoint_payload["checkpoint_manifest_sha256"],
        "checkpoint_model_sha256": checkpoint_payload["checkpoint_model_sha256"],
        "audit": str(args.audit),
        "audit_sha256": audit_payload["sha256"],
        "tokens_per_second": audit_payload["tokens_per_second"],
        "peak_memory_mib": audit_payload["peak_memory_mib"],
    }
    if args.dynamics is not None:
        payload.update(
            {
                "dynamics": str(args.dynamics),
                "dynamics_sha256": sha256_file(args.dynamics),
            }
        )
    _write_json_exclusive(args.out, payload)
    print(args.out)


def _check_pilot_gates(args: argparse.Namespace) -> None:
    config_paths = {
        "bgt-steps2": CONFIG_DIR / "pilots/llama32_1b_flow_bgt_steps2.yaml",
        "icarus-steps2": CONFIG_DIR / "pilots/llama32_1b_flow_icarus_steps2.yaml",
        "longshot-steps2": CONFIG_DIR / "pilots/llama32_1b_flow_longshot_steps2.yaml",
        "bgt-steps100": CONFIG_DIR / "pilots/llama32_1b_flow_bgt_steps100.yaml",
    }
    names = ["bgt-steps2", "icarus-steps2", "longshot-steps2"]
    if not args.short_only:
        names.append("bgt-steps100")
    experiment_root = args.markers_dir.parent
    for name in names:
        config_path = config_paths[name]
        config = load_train_config(config_path)
        checkpoint = Path(config.output_dir) / "checkpoint"
        checkpoint_marker = args.markers_dir / f"pilot-{name}-{args.source_commit[:12]}-checkpoint.json"
        audit = experiment_root / "pilots/audits" / f"{name}-{args.source_commit[:12]}.json"
        marker = args.markers_dir / f"pilot-{name}-{args.source_commit[:12]}.json"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        checkpoint_payload = verify_run_marker(
            marker_path=checkpoint_marker,
            config_path=config_path,
            checkpoint_path=checkpoint,
            backbone_model_path=BACKBONE_PATH,
            source_commit=args.source_commit,
            experiment_manifest_sha256=args.experiment_manifest_sha256,
        )
        audit_payload = verify_training_audit(
            audit,
            architecture=config.model.architecture_name,
            max_steps=config.max_steps,
            source_commit=args.source_commit,
            experiment_manifest_sha256=args.experiment_manifest_sha256,
        )
        if (
            payload.get("passed") is not True
            or payload.get("pilot_name") != name
            or payload.get("source_commit") != args.source_commit
            or payload.get("experiment_manifest_sha256") != args.experiment_manifest_sha256
            or payload.get("config") != str(config_path)
            or payload.get("config_sha256") != checkpoint_payload["config_sha256"]
            or payload.get("checkpoint_marker") != str(checkpoint_marker)
            or payload.get("checkpoint_marker_sha256") != sha256_file(checkpoint_marker)
            or payload.get("checkpoint_manifest_sha256") != checkpoint_payload["checkpoint_manifest_sha256"]
            or payload.get("checkpoint_model_sha256") != checkpoint_payload["checkpoint_model_sha256"]
            or payload.get("audit") != str(audit)
            or payload.get("audit_sha256") != audit_payload["sha256"]
        ):
            raise ValueError(f"pilot gate identity or semantic verification failed: {name}")
        if name == "bgt-steps100":
            dynamics = experiment_root / "pilots/dynamics" / f"{name}-{args.source_commit[:12]}.json"
            dynamics_payload = load_json_strict(dynamics)
            _validate_dynamics_payload(
                dynamics_payload,
                checkpoint=checkpoint,
                checkpoint_payload=checkpoint_payload,
                source_commit=args.source_commit,
                experiment_manifest_sha256=args.experiment_manifest_sha256,
            )
            if (
                payload.get("dynamics") != str(dynamics)
                or payload.get("dynamics_sha256") != sha256_file(dynamics)
            ):
                raise ValueError("100-step pilot dynamics is not bound to its canonical artifact")
    print(json.dumps({"pilots": len(names), "status": "verified"}, sort_keys=True))


def _dynamics(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, config, _ = load_checkpoint(args.checkpoint, device=device)
    input_ids = (torch.arange(args.input_length, device=device) % config.vocab_size).unsqueeze(0)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        payload = probe_flow_dynamics(model, input_ids)
    if max(abs(value) for value in payload["flow_gate_delta_from_init"]) <= 0.0:
        raise ValueError("trained flow gate did not move from its initial value")
    payload.update(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_model_sha256": sha256_file(args.checkpoint / "model.safetensors"),
            "checkpoint_manifest_sha256": sha256_file(args.checkpoint / "manifest.json"),
            "source_commit": args.source_commit,
            "experiment_manifest_sha256": args.experiment_manifest_sha256,
            "input_length": args.input_length,
            "device": str(device),
        }
    )
    _write_json_exclusive(args.out, payload)
    print(args.out)


def _extract_metrics(args: argparse.Namespace) -> None:
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    metrics = extract_primary_metrics(payload)
    output: Any = metrics
    if args.source_commit is not None:
        if (
            args.experiment_manifest_sha256 is None
            or args.eval_cache_manifest_sha256 is None
            or args.checkpoint is None
            or args.checkpoint_sha256 is None
            or args.checkpoint_metadata_sha256 is None
            or args.target is None
        ):
            raise ValueError(
                "authenticated metric output requires target, experiment/cache manifest hashes, and checkpoint hash"
            )
        checkpoint_model = args.checkpoint / "model.safetensors"
        actual_checkpoint_sha256 = sha256_file(checkpoint_model)
        if actual_checkpoint_sha256 != args.checkpoint_sha256:
            raise ValueError("evaluated checkpoint differs from the approved checkpoint hash")
        actual_metadata_sha256 = checkpoint_metadata_sha256(args.checkpoint)
        if actual_metadata_sha256 != args.checkpoint_metadata_sha256:
            raise ValueError("evaluated checkpoint metadata differs from the approved identity")
        output = {
            "target": args.target,
            "source_commit": args.source_commit,
            "experiment_manifest_sha256": args.experiment_manifest_sha256,
            "eval_cache_manifest_sha256": args.eval_cache_manifest_sha256,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": actual_checkpoint_sha256,
            "checkpoint_metadata_sha256": actual_metadata_sha256,
            "result": str(args.result),
            "result_sha256": sha256_file(args.result),
            "metrics": metrics,
        }
    if args.out is None:
        print(json.dumps(output, sort_keys=True, allow_nan=False))
    else:
        _write_json_exclusive(args.out, output)
        print(args.out)


def _eval_pilot_evidence(args: argparse.Namespace) -> dict[str, Any]:
    manifest = verify_execution_context(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.experiment_manifest,
        approved_manifest_sha256=args.experiment_manifest_sha256,
        required_artifacts=CANONICAL_ARTIFACT_SPECS,
    )
    if manifest.get("source_commit") != args.source_commit:
        raise ValueError("evaluation pilot source differs from the experiment manifest")
    verify_directory_manifest(
        root=args.eval_cache_dir,
        manifest_path=args.eval_cache_manifest,
        approved_manifest_sha256=args.eval_cache_manifest_sha256,
        require_read_only=True,
    )
    config_path = CONFIG_DIR / "llama32_1b_flow_bgt_seed24.yaml"
    config = load_train_config(config_path)
    flow_checkpoint = Path(config.output_dir) / "checkpoint"
    train_marker = EXPERIMENT_ROOT / "markers" / f"train-bgt-seed24-{args.source_commit[:12]}.json"
    flow_identity = verify_run_marker(
        marker_path=train_marker,
        config_path=config_path,
        checkpoint_path=flow_checkpoint,
        backbone_model_path=BACKBONE_PATH,
        source_commit=args.source_commit,
        experiment_manifest_sha256=args.experiment_manifest_sha256,
    )
    result_paths = {
        "tf-backbone": EXPERIMENT_ROOT / "eval/pilot" / f"tf-backbone-{args.source_commit[:12]}.json",
        "bgt-seed24": EXPERIMENT_ROOT / "eval/pilot" / f"bgt-seed24-{args.source_commit[:12]}.json",
    }
    metrics = {}
    result_hashes = {}
    for target, result_path in result_paths.items():
        payload = load_json_strict(result_path)
        if not isinstance(payload, dict):
            raise ValueError(f"evaluation pilot result is not an object: {target}")
        metrics[target] = extract_pilot_metrics(payload, limit=EVAL_PILOT_LIMIT)
        result_hashes[target] = sha256_file(result_path)
    return {
        "passed": True,
        "source_commit": args.source_commit,
        "experiment_manifest_sha256": args.experiment_manifest_sha256,
        "eval_cache_manifest_sha256": args.eval_cache_manifest_sha256,
        "limit": EVAL_PILOT_LIMIT,
        "world_size": 1,
        "batch_size": 16,
        "results": {target: str(path) for target, path in result_paths.items()},
        "result_sha256": result_hashes,
        "metrics": metrics,
        "checkpoints": {
            "tf-backbone": str(BASELINE_DIR),
            "bgt-seed24": str(flow_checkpoint),
        },
        "checkpoint_model_sha256": {
            "tf-backbone": sha256_file(BACKBONE_PATH),
            "bgt-seed24": flow_identity["checkpoint_model_sha256"],
        },
        "checkpoint_metadata_sha256": {
            "tf-backbone": checkpoint_metadata_sha256(BASELINE_DIR),
            "bgt-seed24": checkpoint_metadata_sha256(flow_checkpoint),
        },
    }


def _finalize_eval_pilot(args: argparse.Namespace) -> None:
    payload = _eval_pilot_evidence(args)
    _write_json_exclusive(args.out, payload)
    print(args.out)


def _check_eval_pilot(args: argparse.Namespace) -> None:
    expected = _eval_pilot_evidence(args)
    marker = load_json_strict(args.marker)
    if marker != expected:
        raise ValueError("evaluation pilot marker differs from live authenticated evidence")
    print(json.dumps({"status": "verified", "targets": 2}, sort_keys=True))


def _summarise(args: argparse.Namespace) -> None:
    manifest = verify_execution_context(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.experiment_manifest,
        approved_manifest_sha256=args.experiment_manifest_sha256,
        required_artifacts=CANONICAL_ARTIFACT_SPECS,
    )
    if manifest.get("source_commit") != args.source_commit:
        raise ValueError("summary source commit differs from the authenticated manifest")
    verify_directory_manifest(
        root=args.eval_cache_dir,
        manifest_path=args.eval_cache_manifest,
        approved_manifest_sha256=args.eval_cache_manifest_sha256,
        require_read_only=True,
    )
    targets = ["tf-backbone"] + [
        f"{variant}-seed{seed}"
        for variant in ("bgt", "icarus", "longshot")
        for seed in (24, 25, 26)
    ]
    rows = load_matrix_file(CONFIG_DIR / "matrix.tsv")
    configs_by_target = {f"{row['variant']}-seed{row['seed']}": Path(row["config"]) for row in rows}
    markers = []
    for target in targets:
        path = args.markers_dir / f"eval-{target}-{args.source_commit[:12]}.json"
        marker = load_json_strict(path)
        if not isinstance(marker, dict):
            raise ValueError(f"evaluation marker is not an object: {target}")
        if target == "tf-backbone":
            checkpoint = BASELINE_DIR
            model_sha = sha256_file(BACKBONE_PATH)
        else:
            config_path = configs_by_target[target]
            config = load_train_config(config_path)
            checkpoint = Path(config.output_dir) / "checkpoint"
            train_marker = args.markers_dir / f"train-{target}-{args.source_commit[:12]}.json"
            authenticated = verify_run_marker(
                marker_path=train_marker,
                config_path=config_path,
                checkpoint_path=checkpoint,
                backbone_model_path=BACKBONE_PATH,
                source_commit=args.source_commit,
                experiment_manifest_sha256=args.experiment_manifest_sha256,
            )
            model_sha = authenticated["checkpoint_model_sha256"]
        metadata_sha = checkpoint_metadata_sha256(checkpoint)
        if (
            Path(str(marker.get("checkpoint", ""))).resolve() != checkpoint.resolve()
            or marker.get("checkpoint_sha256") != model_sha
            or marker.get("checkpoint_metadata_sha256") != metadata_sha
        ):
            raise ValueError(f"evaluation target is not bound to its canonical checkpoint: {target}")
        markers.append(marker)
    summary = summarise_metric_markers(
        markers,
        source_commit=args.source_commit,
        experiment_manifest_sha256=args.experiment_manifest_sha256,
        eval_cache_manifest_sha256=args.eval_cache_manifest_sha256,
    )
    _write_json_exclusive(args.out, summary)
    print(args.out)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Authenticated Flow Math experiment operations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    context = subparsers.add_parser("verify-context")
    context.add_argument("--repo-root", type=Path, default=Path.cwd())
    context.add_argument("--manifest", type=Path, required=True)
    context.add_argument("--approved-manifest-sha256", required=True)
    context.add_argument("--source-commit", required=True)
    context.set_defaults(func=_verify_context)

    row = subparsers.add_parser("matrix-row")
    row.add_argument("--matrix", type=Path, required=True)
    row.add_argument("--index", type=int, required=True)
    row.set_defaults(func=_matrix_row)

    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("--repo-root", type=Path, default=Path.cwd())
    manifest.add_argument("--source-commit", required=True)
    manifest.add_argument("--out", type=Path, required=True)
    manifest.set_defaults(func=_build_manifest)

    cache_build = subparsers.add_parser("build-cache-manifest")
    cache_build.add_argument("--cache-dir", type=Path, required=True)
    cache_build.add_argument("--out", type=Path, required=True)
    cache_build.set_defaults(func=_build_cache_manifest)

    cache_verify = subparsers.add_parser("verify-cache-manifest")
    cache_verify.add_argument("--cache-dir", type=Path, required=True)
    cache_verify.add_argument("--manifest", type=Path, required=True)
    cache_verify.add_argument("--approved-manifest-sha256", required=True)
    cache_verify.add_argument("--require-read-only", action="store_true")
    cache_verify.set_defaults(func=_verify_cache_manifest)

    verify = subparsers.add_parser("verify-run")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify.add_argument("--source-commit", required=True)
    verify.add_argument("--experiment-manifest-sha256", required=True)
    verify.add_argument("--out", type=Path, required=True)
    verify.set_defaults(func=_verify_run)

    marker = subparsers.add_parser("check-run-marker")
    marker.add_argument("--marker", type=Path, required=True)
    marker.add_argument("--config", type=Path, required=True)
    marker.add_argument("--checkpoint", type=Path, required=True)
    marker.add_argument("--source-commit", required=True)
    marker.add_argument("--experiment-manifest-sha256", required=True)
    marker.set_defaults(func=_check_run_marker)

    audit = subparsers.add_parser("verify-pilot-audit")
    audit.add_argument("--audit", type=Path, required=True)
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--source-commit", required=True)
    audit.add_argument("--experiment-manifest-sha256", required=True)
    audit.set_defaults(func=_verify_pilot_audit)

    finalize = subparsers.add_parser("finalize-pilot")
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument("--checkpoint-marker", type=Path, required=True)
    finalize.add_argument("--audit", type=Path, required=True)
    finalize.add_argument("--dynamics", type=Path)
    finalize.add_argument("--source-commit", required=True)
    finalize.add_argument("--experiment-manifest-sha256", required=True)
    finalize.add_argument("--out", type=Path, required=True)
    finalize.set_defaults(func=_finalize_pilot)

    gates = subparsers.add_parser("check-pilot-gates")
    gates.add_argument("--markers-dir", type=Path, required=True)
    gates.add_argument("--source-commit", required=True)
    gates.add_argument("--experiment-manifest-sha256", required=True)
    gates.add_argument("--short-only", action="store_true")
    gates.set_defaults(func=_check_pilot_gates)

    dynamics = subparsers.add_parser("dynamics")
    dynamics.add_argument("--checkpoint", type=Path, required=True)
    dynamics.add_argument("--device", default="cuda")
    dynamics.add_argument("--input-length", type=int, default=128)
    dynamics.add_argument("--source-commit", required=True)
    dynamics.add_argument("--experiment-manifest-sha256", required=True)
    dynamics.add_argument("--out", type=Path, required=True)
    dynamics.set_defaults(func=_dynamics)

    metrics = subparsers.add_parser("extract-metrics")
    metrics.add_argument("--result", type=Path, required=True)
    metrics.add_argument("--target")
    metrics.add_argument("--source-commit")
    metrics.add_argument("--experiment-manifest-sha256")
    metrics.add_argument("--eval-cache-manifest-sha256")
    metrics.add_argument("--checkpoint", type=Path)
    metrics.add_argument("--checkpoint-sha256")
    metrics.add_argument("--checkpoint-metadata-sha256")
    metrics.add_argument("--out", type=Path)
    metrics.set_defaults(func=_extract_metrics)

    eval_pilot = subparsers.add_parser("finalize-eval-pilot")
    eval_pilot.add_argument("--repo-root", type=Path, default=Path.cwd())
    eval_pilot.add_argument("--experiment-manifest", type=Path, required=True)
    eval_pilot.add_argument("--experiment-manifest-sha256", required=True)
    eval_pilot.add_argument("--source-commit", required=True)
    eval_pilot.add_argument("--eval-cache-dir", type=Path, required=True)
    eval_pilot.add_argument("--eval-cache-manifest", type=Path, required=True)
    eval_pilot.add_argument("--eval-cache-manifest-sha256", required=True)
    eval_pilot.add_argument("--out", type=Path, required=True)
    eval_pilot.set_defaults(func=_finalize_eval_pilot)

    pilot_gate = subparsers.add_parser("check-eval-pilot")
    pilot_gate.add_argument("--repo-root", type=Path, default=Path.cwd())
    pilot_gate.add_argument("--experiment-manifest", type=Path, required=True)
    pilot_gate.add_argument("--experiment-manifest-sha256", required=True)
    pilot_gate.add_argument("--source-commit", required=True)
    pilot_gate.add_argument("--eval-cache-dir", type=Path, required=True)
    pilot_gate.add_argument("--eval-cache-manifest", type=Path, required=True)
    pilot_gate.add_argument("--eval-cache-manifest-sha256", required=True)
    pilot_gate.add_argument("--marker", type=Path, required=True)
    pilot_gate.set_defaults(func=_check_eval_pilot)

    summary = subparsers.add_parser("summarise")
    summary.add_argument("--markers-dir", type=Path, required=True)
    summary.add_argument("--repo-root", type=Path, default=Path.cwd())
    summary.add_argument("--experiment-manifest", type=Path, required=True)
    summary.add_argument("--source-commit", required=True)
    summary.add_argument("--experiment-manifest-sha256", required=True)
    summary.add_argument("--eval-cache-dir", type=Path, required=True)
    summary.add_argument("--eval-cache-manifest", type=Path, required=True)
    summary.add_argument("--eval-cache-manifest-sha256", required=True)
    summary.add_argument("--out", type=Path, required=True)
    summary.set_defaults(func=_summarise)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
