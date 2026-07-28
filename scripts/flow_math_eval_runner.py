from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_eval_command(
    *,
    python: Path,
    checkpoint: Path,
    batch_size: int,
    limit: int | None,
    out: Path,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "gcw.cli",
        "eval",
        "math",
        "--checkpoint",
        str(checkpoint),
        "--batch-size",
        str(batch_size),
        "--out",
        str(out),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


CANONICAL_ARRAY_TARGETS = (
    "tf-backbone",
    "bgt-seed24", "icarus-seed24", "longshot-seed24",
    "bgt-seed25", "icarus-seed25", "longshot-seed25",
    "bgt-seed26", "icarus-seed26", "longshot-seed26",
)


def _validate_array_target(target: str, expected_array_task_id: int | None) -> None:
    if expected_array_task_id is None:
        return
    if not 0 <= expected_array_task_id < len(CANONICAL_ARRAY_TARGETS):
        raise ValueError("evaluation array task ID is outside the canonical matrix")
    if CANONICAL_ARRAY_TARGETS[expected_array_task_id] != target:
        raise ValueError("evaluation array task ID does not match its canonical target")


def _integer_environment(environ: Mapping[str, str], name: str, default: int) -> int:
    value = environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def build_runtime_evidence(
    *,
    target: str,
    command: list[str],
    checkpoint: Path,
    output: Path,
    temporary_output: Path,
    partial_identity: Path | None,
    eval_cache_dir: Path,
    expected_array_task_id: int | None,
    batch_size: int,
    limit: int | None,
    cuda_device_count: int,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    _validate_array_target(target, expected_array_task_id)
    world_size = _integer_environment(environ, "WORLD_SIZE", 1)
    local_rank = _integer_environment(environ, "LOCAL_RANK", 0)
    rank = _integer_environment(environ, "RANK", 0)
    if cuda_device_count != 1:
        raise ValueError(f"evaluation requires exactly one visible CUDA device, got {cuda_device_count}")
    if world_size != 1:
        raise ValueError(f"evaluation requires world size 1, got {world_size}")
    if local_rank != 0 or rank != 0:
        raise ValueError(f"evaluation requires rank zero, got local_rank={local_rank}, rank={rank}")
    if batch_size != 16:
        raise ValueError(f"evaluation requires batch size 16, got {batch_size}")
    if limit not in (None, 16):
        raise ValueError(f"evaluation limit must be 16 for pilot or absent for full evaluation, got {limit}")
    if environ.get("GCW_STRICT_FLASH") != "1":
        raise ValueError("evaluation requires strict FlashAttention")
    if environ.get("WANDB_MODE") != "offline":
        raise ValueError("evaluation requires WANDB_MODE=offline")
    offline = {
        "hf_datasets_offline": environ.get("HF_DATASETS_OFFLINE"),
        "hf_hub_offline": environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": environ.get("TRANSFORMERS_OFFLINE"),
    }
    if set(offline.values()) != {"1"}:
        raise ValueError("evaluation requires fully offline evaluation environment")
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or "," in visible:
        raise ValueError("CUDA_VISIBLE_DEVICES must identify exactly one allocated device")
    job_id = environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise ValueError("evaluation runtime requires numeric Slurm job identity")
    if environ.get("SLURM_NTASKS") != "1" or environ.get("SLURM_CPUS_PER_TASK") != "16":
        raise ValueError("evaluation requires one Slurm task with 16 CPUs")
    array_job_id = environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = environ.get("SLURM_ARRAY_TASK_ID")
    if expected_array_task_id is None:
        if array_job_id is not None or array_task_id is not None:
            raise ValueError("non-array evaluation inherited Slurm array identity")
    elif (
        not array_job_id
        or not array_job_id.isdigit()
        or array_task_id != str(expected_array_task_id)
    ):
        raise ValueError("full evaluation Slurm array identity does not match its canonical target")
    cache_root = eval_cache_dir.resolve()
    required_paths = {
        "HF_HOME": cache_root / "huggingface",
        "HF_DATASETS_CACHE": cache_root / "huggingface" / "datasets",
        "HF_HUB_CACHE": cache_root / "huggingface" / "hub",
    }
    for name, expected_path in required_paths.items():
        if Path(environ.get(name, "")).resolve() != expected_path:
            raise ValueError(f"evaluation cache environment differs: {name}")
    expected_partial_dir = str(partial_identity.parent.resolve()) if partial_identity is not None else None
    actual_partial_dir = environ.get("GCW_EVAL_PARTIAL_DIR")
    if (actual_partial_dir or None) != expected_partial_dir:
        raise ValueError("GCW_EVAL_PARTIAL_DIR differs from authenticated partial identity")
    if environ.get("GCW_EVAL_PROGRESS_PATH") not in (None, ""):
        raise ValueError("GCW_EVAL_PROGRESS_PATH must be unset")
    if environ.get("GCW_EVAL_SKIP_TASKS") != "theoremqa":
        raise ValueError("evaluation must exclude exactly TheoremQA")
    if environ.get("GCW_EVAL_REFERENCE_GENERATE") != "0":
        raise ValueError("evaluation reference-generation control is not canonical")
    if environ.get("PYTHONPATH", "") != "":
        raise ValueError("evaluation PYTHONPATH must be empty")
    if environ.get("PYTORCH_CUDA_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("evaluation allocator policy is not canonical")
    partial_path = partial_identity.resolve() if partial_identity is not None else None
    return {
        "schema_version": 2,
        "kind": "flow_math_eval_runtime",
        "target": target,
        "command": command,
        "python_executable": command[0],
        "checkpoint": str(checkpoint.resolve()),
        "output": str(output.resolve()),
        "temporary_output": str(temporary_output.resolve()),
        "partial_identity": str(partial_path) if partial_path is not None else None,
        "partial_identity_sha256": _sha256_file(partial_path) if partial_path is not None else None,
        "eval_cache_dir": str(cache_root),
        "gcw_eval_partial_dir": expected_partial_dir,
        "gcw_eval_progress_path": None,
        "effective_progress_path": str(partial_identity.parent.resolve() / "progress.json") if partial_identity is not None else None,
        "gcw_eval_skip_tasks": environ.get("GCW_EVAL_SKIP_TASKS"),
        "gcw_eval_reference_generate": environ.get("GCW_EVAL_REFERENCE_GENERATE"),
        "pythonpath": environ.get("PYTHONPATH", ""),
        "pytorch_cuda_alloc_conf": environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "cuda_visible_devices": visible,
        "cuda_device_count": cuda_device_count,
        "world_size": world_size,
        "local_rank": local_rank,
        "rank": rank,
        "batch_size": batch_size,
        "limit": limit,
        "strict_flash": environ.get("GCW_STRICT_FLASH"),
        "wandb_mode": environ.get("WANDB_MODE"),
        **offline,
        "generation_policy": {
            "do_sample": False,
            "legacy_bottleneck_generate": False,
        },
        "slurm_job_id": job_id,
        "slurm_array_job_id": array_job_id,
        "slurm_array_task_id": array_task_id,
        "slurm_ntasks": environ.get("SLURM_NTASKS"),
        "slurm_cpus_per_task": environ.get("SLURM_CPUS_PER_TASK"),
        "exit_status": 0,
    }


def validate_runtime_evidence(
    payload: Mapping[str, Any],
    *,
    target: str,
    checkpoint: Path,
    output: Path,
    partial_identity: Path | None,
    eval_cache_dir: Path,
    expected_array_task_id: int | None,
    expected_python: Path,
    batch_size: int,
    limit: int | None,
    require_slurm: bool,
) -> None:
    _validate_array_target(target, expected_array_task_id)
    expected_keys = {
        "schema_version", "kind", "target", "command", "python_executable", "checkpoint",
        "output", "temporary_output", "partial_identity", "partial_identity_sha256",
        "eval_cache_dir", "gcw_eval_partial_dir", "gcw_eval_progress_path", "effective_progress_path", "gcw_eval_skip_tasks",
        "gcw_eval_reference_generate", "pythonpath", "pytorch_cuda_alloc_conf",
        "cuda_visible_devices", "cuda_device_count", "world_size", "local_rank", "rank",
        "batch_size", "limit", "strict_flash", "wandb_mode", "hf_datasets_offline",
        "hf_hub_offline", "transformers_offline", "generation_policy", "slurm_job_id",
        "slurm_array_job_id", "slurm_array_task_id", "slurm_ntasks", "slurm_cpus_per_task",
        "exit_status",
    }
    if set(payload) != expected_keys:
        raise ValueError("runtime evidence schema differs from the canonical schema")
    temporary = Path(str(payload["temporary_output"]))
    command = payload.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(value, str) for value in command):
        raise ValueError("runtime evidence command is invalid")
    python_executable = Path(str(payload["python_executable"]))
    if python_executable.resolve() != expected_python.resolve():
        raise ValueError("runtime evidence Python executable is not the evaluator interpreter")
    output_resolved = output.resolve()
    if temporary.parent.resolve() != output_resolved.parent or not temporary.name.startswith(f".{output.name}.") or not temporary.name.endswith(".tmp.json"):
        raise ValueError("runtime evidence temporary output path is not canonical")
    expected_command = build_eval_command(
        python=python_executable,
        checkpoint=checkpoint,
        batch_size=batch_size,
        limit=limit,
        out=temporary,
    )
    expected_partial = partial_identity.resolve() if partial_identity is not None else None
    required = {
        "schema_version": 2,
        "kind": "flow_math_eval_runtime",
        "target": target,
        "command": expected_command,
        "checkpoint": str(checkpoint.resolve()),
        "output": str(output.resolve()),
        "partial_identity": str(expected_partial) if expected_partial is not None else None,
        "partial_identity_sha256": _sha256_file(expected_partial) if expected_partial is not None else None,
        "eval_cache_dir": str(eval_cache_dir.resolve()),
        "gcw_eval_partial_dir": str(expected_partial.parent) if expected_partial is not None else None,
        "gcw_eval_progress_path": None,
        "effective_progress_path": str(expected_partial.parent / "progress.json") if expected_partial is not None else None,
        "gcw_eval_skip_tasks": "theoremqa",
        "gcw_eval_reference_generate": "0",
        "pythonpath": "",
        "pytorch_cuda_alloc_conf": "expandable_segments:True",
        "cuda_device_count": 1,
        "world_size": 1,
        "local_rank": 0,
        "rank": 0,
        "batch_size": batch_size,
        "limit": limit,
        "strict_flash": "1",
        "wandb_mode": "offline",
        "hf_datasets_offline": "1",
        "hf_hub_offline": "1",
        "transformers_offline": "1",
        "generation_policy": {"do_sample": False, "legacy_bottleneck_generate": False},
        "slurm_array_task_id": str(expected_array_task_id) if expected_array_task_id is not None else None,
        "slurm_ntasks": "1",
        "slurm_cpus_per_task": "16",
        "exit_status": 0,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"runtime evidence {key} differs from the canonical invocation")
    if not payload.get("cuda_visible_devices"):
        raise ValueError("runtime evidence lacks CUDA allocation")
    if require_slurm and (not str(payload.get("slurm_job_id", "")).isdigit()):
        raise ValueError("runtime evidence lacks numeric Slurm identity")
    if expected_array_task_id is None:
        if payload.get("slurm_array_job_id") is not None:
            raise ValueError("runtime evidence has unexpected Slurm array identity")
    elif not str(payload.get("slurm_array_job_id", "")).isdigit():
        raise ValueError("runtime evidence lacks Slurm array identity")


def publish_result_no_replace(temporary: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.link(temporary, output)
    output.chmod(0o444)
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    temporary.unlink()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one authenticated Flow Math evaluator process")
    parser.add_argument("--target", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime-evidence", type=Path, required=True)
    parser.add_argument("--partial-identity", type=Path)
    parser.add_argument("--eval-cache-dir", type=Path, required=True)
    parser.add_argument("--array-task-id", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(f"evaluation output already exists: {args.out}")
    if args.runtime_evidence.exists() or args.runtime_evidence.is_symlink():
        raise FileExistsError(f"runtime evidence already exists: {args.runtime_evidence}")
    if args.partial_identity is not None and not args.partial_identity.is_file():
        raise ValueError(f"partial identity is missing: {args.partial_identity}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.out.name}.", suffix=".tmp.json", dir=args.out.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    command = build_eval_command(
        python=Path(sys.executable),
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        limit=args.limit,
        out=temporary,
    )
    try:
        import torch

        evidence = build_runtime_evidence(
            target=args.target,
            command=command,
            checkpoint=args.checkpoint,
            output=args.out,
            temporary_output=temporary,
            partial_identity=args.partial_identity,
            eval_cache_dir=args.eval_cache_dir,
            expected_array_task_id=args.array_task_id,
            batch_size=args.batch_size,
            limit=args.limit,
            cuda_device_count=torch.cuda.device_count(),
            environ=os.environ,
        )
        subprocess.run(command, check=True)
        publish_result_no_replace(temporary, args.out)
        _write_json_exclusive(args.runtime_evidence, evidence)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
