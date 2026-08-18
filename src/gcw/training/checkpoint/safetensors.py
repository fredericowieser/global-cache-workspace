from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
import random
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file

from gcw.config import ModelConfig, to_dict
from gcw.data.training import cursor_to_dict
from gcw.provenance import file_sha256
from gcw.runtime import fsync_directory
from gcw.training.checkpoint.types import (
    FORMAT_VERSION,
    INITIAL_PROCESSOR_FILE,
    OPTIONAL_PAYLOAD_FILES,
    PAYLOAD_FILES,
    LegacyCheckpointError,
    ResumeState,
    TrainingCheckpointError,
)


def _capture_verified_payloads(
    path: Path,
    manifest: dict[str, Any],
) -> dict[str, bytes]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TrainingCheckpointError("checkpoint manifest files must be an object")
    expected_names = PAYLOAD_FILES | ({INITIAL_PROCESSOR_FILE} if INITIAL_PROCESSOR_FILE in files else set())
    if set(files) != expected_names:
        raise TrainingCheckpointError("checkpoint manifest payload roles differ from required schema")
    payloads: dict[str, bytes] = {}
    for name in sorted(expected_names):
        expected = files.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise TrainingCheckpointError(f"checkpoint manifest digest is invalid for {name}")
        try:
            payload = (path / name).read_bytes()
        except OSError as exc:
            raise TrainingCheckpointError(f"checkpoint payload is missing: {name}") from exc
        if hashlib.sha256(payload).hexdigest() != expected:
            raise TrainingCheckpointError(f"checkpoint payload digest differs for {name}")
        payloads[name] = payload
    return payloads


def verify_training_checkpoint_files(path: str | Path) -> dict[str, Any]:
    resolved = _resolve_checkpoint_path(Path(path))
    manifest = _read_manifest(resolved)
    _verify_files(resolved, manifest)
    try:
        train_state = json.loads((resolved / "train_state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingCheckpointError(f"invalid checkpoint train state in {resolved}: {exc}") from exc
    if train_state.get("step") != manifest.get("step"):
        raise TrainingCheckpointError("checkpoint manifest and train state steps differ")
    return {"path": str(resolved), "manifest": manifest, "train_state": train_state}


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_config: ModelConfig,
    optimizer: torch.optim.Optimizer,
    resume_state: ResumeState,
    training_fingerprint: dict[str, Any],
    *,
    initial_processor_state: dict[str, torch.Tensor] | None = None,
    cache_processor_only: bool = False,
) -> None:
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    local_runtime = {"rng": _capture_rng_state(), "loader_cursor": cursor_to_dict(resume_state.loader_cursor)}
    gathered_runtime: list[dict[str, Any]] | None = [None] * world_size if rank == 0 else None  # type: ignore[list-item]
    if distributed:
        dist.gather_object(local_runtime, gathered_runtime, dst=0)
    else:
        gathered_runtime = [local_runtime]

    status: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            import gcw.training.checkpoint as checkpoint_pkg
            write_fn = getattr(checkpoint_pkg, "_write_checkpoint", _write_checkpoint)
            write_fn(
                Path(path),
                model,
                model_config,
                optimizer,
                replace(resume_state, saved_world_size=world_size),
                training_fingerprint,
                gathered_runtime or [],
                initial_processor_state=initial_processor_state,
                cache_processor_only=cache_processor_only,
            )
            status[0] = {"ok": True}
        except Exception as exc:  # broadcast recoverable rank-zero failures before raising
            status[0] = {"ok": False, "type": type(exc).__name__, "message": str(exc)}
    if distributed:
        dist.broadcast_object_list(status, src=0)
    result = status[0]
    if not result or not result.get("ok"):
        error = result or {"type": "TrainingCheckpointError", "message": "rank zero did not report checkpoint status"}
        raise TrainingCheckpointError(f"{error['type']}: {error['message']}")

    if distributed:
        dist.barrier()
    cleanup_status: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            import gcw.training.checkpoint as checkpoint_pkg
            remove_fn = getattr(checkpoint_pkg, "_remove_previous", _remove_previous)
            remove_fn(Path(path))
            cleanup_status[0] = {"ok": True}
        except Exception as exc:
            cleanup_status[0] = {"ok": False, "type": type(exc).__name__, "message": str(exc)}
    if distributed:
        dist.broadcast_object_list(cleanup_status, src=0)
    cleanup_result = cleanup_status[0]
    if not cleanup_result or not cleanup_result.get("ok"):
        error = cleanup_result or {"type": "TrainingCheckpointError", "message": "rank zero did not report cleanup status"}
        raise TrainingCheckpointError(f"checkpoint committed but rollback cleanup failed: {error['type']}: {error['message']}")


def _write_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_config: ModelConfig,
    optimizer: torch.optim.Optimizer,
    resume_state: ResumeState,
    training_fingerprint: dict[str, Any],
    runtime_states: list[dict[str, Any]],
    *,
    initial_processor_state: dict[str, torch.Tensor] | None = None,
    cache_processor_only: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=path.parent))
    previous = path.with_name(f"{path.name}.previous")
    moved_previous = False
    try:
        model_state = {
            key: value.detach().cpu().contiguous()
            for key, value in model.state_dict().items()
            if not cache_processor_only or key.startswith("cache_processor.")
        }
        if cache_processor_only and not model_state:
            raise TrainingCheckpointError("cache_processor_only checkpoint contained no cache_processor state")
        save_file(model_state, temp_path / "model.safetensors")
        if initial_processor_state is not None:
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in initial_processor_state.items()},
                temp_path / INITIAL_PROCESSOR_FILE,
            )
        torch.save(optimizer.state_dict(), temp_path / "optimizer.pt")
        torch.save([state["rng"] for state in runtime_states], temp_path / "rng.pt")
        _write_json(temp_path / "config.json", {"model": to_dict(model_config)})
        train_state_payload = {
            **resume_state.metadata,
            "step": resume_state.step,
            "smooth_loss": resume_state.smooth_loss,
            "loader_cursor": cursor_to_dict(resume_state.loader_cursor),
            "rank_loader_cursors": [state["loader_cursor"] for state in runtime_states],
            "saved_world_size": resume_state.saved_world_size,
            "exact_topology": True,
        }
        _write_json(temp_path / "train_state.json", train_state_payload)
        required_names = list(PAYLOAD_FILES) + ([INITIAL_PROCESSOR_FILE] if initial_processor_state is not None else [])
        files = {name: file_sha256(temp_path / name) for name in sorted(required_names)}
        manifest = {
            "format_version": FORMAT_VERSION,
            "complete": True,
            "step": resume_state.step,
            "training_fingerprint": training_fingerprint,
            "files": files,
        }
        _write_json(temp_path / "manifest.json", manifest)
        _fsync_tree(temp_path)

        if path.exists():
            _remove_previous(path)
            os.replace(path, previous)
            moved_previous = True
            fsync_directory(path.parent)
        os.replace(temp_path, path)
        fsync_directory(path.parent)
    except Exception:
        if moved_previous and not path.exists() and previous.exists():
            os.replace(previous, path)
            fsync_directory(path.parent)
        raise
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)


def _resolve_checkpoint_path(path: Path) -> Path:
    previous = path.with_name(f"{path.name}.previous")
    primary_error = _checkpoint_error(path)
    if primary_error is None:
        return path
    previous_error = _checkpoint_error(previous)
    if previous_error is None:
        return previous
    if not (path / "manifest.json").exists() and (path / "model.safetensors").exists():
        raise LegacyCheckpointError(f"{path} is not a complete version-2 training checkpoint")
    if path.exists():
        raise TrainingCheckpointError(
            f"invalid training checkpoint at {path}: {primary_error}; no valid rollback checkpoint: {previous_error}"
        ) from primary_error
    raise TrainingCheckpointError(f"training checkpoint not found: {path}; rollback checkpoint: {previous_error}")


def _checkpoint_error(path: Path) -> TrainingCheckpointError | None:
    try:
        manifest = _read_manifest(path)
        _verify_files(path, manifest)
    except TrainingCheckpointError as exc:
        return exc
    return None


def _parse_manifest_bytes(payload: bytes, path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingCheckpointError(f"invalid checkpoint manifest in {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != FORMAT_VERSION:
        version = manifest.get("format_version") if isinstance(manifest, dict) else None
        raise TrainingCheckpointError(
            f"unsupported training checkpoint version {version!r}; expected {FORMAT_VERSION}"
        )
    if manifest.get("complete") is not True:
        raise TrainingCheckpointError(f"incomplete training checkpoint: {path}")
    return manifest


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = (path / "manifest.json").read_bytes()
    except OSError as exc:
        raise TrainingCheckpointError(f"invalid checkpoint manifest in {path}: {exc}") from exc
    return _parse_manifest_bytes(payload, path)


def _verify_files(path: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TrainingCheckpointError("checkpoint manifest has no file hashes")
    file_names = set(files)
    if not PAYLOAD_FILES <= file_names or not file_names <= PAYLOAD_FILES | OPTIONAL_PAYLOAD_FILES:
        missing = sorted(PAYLOAD_FILES - file_names)
        unexpected = sorted(file_names - PAYLOAD_FILES - OPTIONAL_PAYLOAD_FILES)
        raise TrainingCheckpointError(
            f"checkpoint manifest does not contain exactly the required file hashes; missing={missing}, unexpected={unexpected}"
        )
    for name, expected_hash in files.items():
        if not isinstance(expected_hash, str):
            raise TrainingCheckpointError(f"checkpoint hash for {name} is not a string")
        candidate = path / name
        if not candidate.is_file():
            raise TrainingCheckpointError(f"checkpoint file is missing: {name}")
        actual_hash = file_sha256(candidate)
        if actual_hash != expected_hash:
            raise TrainingCheckpointError(f"checkpoint hash mismatch for {name}")


def _remove_previous(path: Path) -> None:
    previous = path.with_name(f"{path.name}.previous")
    if previous.is_dir():
        shutil.rmtree(previous)
        fsync_directory(path.parent)
    elif previous.exists():
        previous.unlink()
        fsync_directory(path.parent)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _elastic_seed(seed: int, step: int, rank: int, world_size: int) -> int:
    digest = hashlib.sha256(f"{seed}:{step}:{rank}:{world_size}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _move_tensors(value: Any, device: torch.device) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, torch.Tensor):
                value[key] = item.to(device)
            else:
                _move_tensors(item, device)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, torch.Tensor):
                value[index] = item.to(device)
            else:
                _move_tensors(item, device)


def _write_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _fsync_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_file():
            with open(child, "rb") as handle:
                os.fsync(handle.fileno())
    fsync_directory(path)
