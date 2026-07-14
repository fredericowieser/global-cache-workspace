from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from gcw.config import ModelConfig, to_dict
from gcw.data.training import LoaderCursor, cursor_from_dict, cursor_to_dict


FORMAT_VERSION = 2


class TrainingCheckpointError(RuntimeError):
    pass


class LegacyCheckpointError(TrainingCheckpointError):
    pass


class CheckpointCompatibilityError(TrainingCheckpointError):
    pass


class UnsupportedTopologyError(CheckpointCompatibilityError):
    pass


@dataclass(frozen=True)
class ResumeState:
    step: int
    smooth_loss: float
    loader_cursor: LoaderCursor
    saved_world_size: int
    exact_topology: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingCheckpoint:
    def __init__(
        self,
        path: Path,
        resume_state: ResumeState,
        manifest: dict[str, Any],
        *,
        elastic_seed: int | None,
        rank_loader_cursors: list[LoaderCursor],
    ) -> None:
        self.path = path
        self.resume_state = resume_state
        self.manifest = manifest
        self._elastic_seed = elastic_seed
        self._rank_loader_cursors = rank_loader_cursors

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        expected_fingerprint: dict[str, Any],
        topology: Literal["same", "elastic_ddp"],
        current_world_size: int,
    ) -> TrainingCheckpoint:
        resolved = _resolve_checkpoint_path(Path(path))
        manifest = _read_manifest(resolved)
        _verify_files(resolved, manifest)
        saved_fingerprint = manifest.get("training_fingerprint")
        if saved_fingerprint != expected_fingerprint:
            differing = sorted(
                key
                for key in set(saved_fingerprint or {}) | set(expected_fingerprint)
                if (saved_fingerprint or {}).get(key) != expected_fingerprint.get(key)
            )
            raise CheckpointCompatibilityError(f"training fingerprint differs for: {', '.join(differing)}")
        train_state = json.loads((resolved / "train_state.json").read_text(encoding="utf-8"))
        reserved = {
            "step",
            "smooth_loss",
            "loader_cursor",
            "rank_loader_cursors",
            "saved_world_size",
            "exact_topology",
        }
        rank_loader_cursors = [
            cursor_from_dict(value) for value in train_state.get("rank_loader_cursors", [train_state["loader_cursor"]])
        ]
        resume_state = ResumeState(
            step=int(train_state["step"]),
            smooth_loss=float(train_state["smooth_loss"]),
            loader_cursor=cursor_from_dict(train_state["loader_cursor"]),
            saved_world_size=int(train_state["saved_world_size"]),
            exact_topology=True,
            metadata={key: value for key, value in train_state.items() if key not in reserved},
        )
        if expected_fingerprint.get("distributed") == "fsdp":
            raise UnsupportedTopologyError("FSDP training resume is not supported")
        elastic_seed = None
        if resume_state.saved_world_size != current_world_size:
            if topology != "elastic_ddp":
                raise CheckpointCompatibilityError(
                    f"checkpoint world size {resume_state.saved_world_size} does not match current world size {current_world_size}"
                )
            if expected_fingerprint.get("distributed") != "ddp" or expected_fingerprint.get("data_kind") != "fineweb":
                raise UnsupportedTopologyError("elastic DDP resume is supported only for FineWeb DDP training")
            elastic_seed = _elastic_seed(
                int(expected_fingerprint["seed"]),
                resume_state.step,
                current_world_size,
            )
            resume_state = replace(resume_state, exact_topology=False)
        return cls(
            resolved,
            resume_state,
            manifest,
            elastic_seed=elastic_seed,
            rank_loader_cursors=rank_loader_cursors,
        )

    def loader_cursor(self, *, rank: int) -> LoaderCursor:
        if not self.resume_state.exact_topology:
            return self.resume_state.loader_cursor
        if rank >= len(self._rank_loader_cursors):
            raise CheckpointCompatibilityError(f"checkpoint has no loader cursor for rank {rank}")
        return self._rank_loader_cursors[rank]

    def restore_model(self, model: torch.nn.Module) -> None:
        device = next(model.parameters()).device
        weights = load_file(self.path / "model.safetensors", device=str(device))
        model.load_state_dict(weights, strict=True)

    def restore_optimizer(self, optimizer: torch.optim.Optimizer, *, device: torch.device | str) -> None:
        state = torch.load(self.path / "optimizer.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state)
        target = torch.device(device)
        for value in optimizer.state.values():
            _move_tensors(value, target)

    def restore_rng(self, *, rank: int) -> None:
        if self._elastic_seed is not None:
            seed = self._elastic_seed + rank
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            return
        states = torch.load(self.path / "rng.pt", map_location="cpu", weights_only=False)
        if rank >= len(states):
            raise CheckpointCompatibilityError(f"checkpoint has no RNG state for rank {rank}")
        _restore_rng_state(states[rank])


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_config: ModelConfig,
    optimizer: torch.optim.Optimizer,
    resume_state: ResumeState,
    training_fingerprint: dict[str, Any],
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
            _write_checkpoint(
                Path(path),
                model,
                model_config,
                optimizer,
                replace(resume_state, saved_world_size=world_size),
                training_fingerprint,
                gathered_runtime or [],
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


def _write_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_config: ModelConfig,
    optimizer: torch.optim.Optimizer,
    resume_state: ResumeState,
    training_fingerprint: dict[str, Any],
    runtime_states: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=path.parent))
    previous = path.with_name(f"{path.name}.previous")
    moved_previous = False
    try:
        model_state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
        save_file(model_state, temp_path / "model.safetensors")
        torch.save(optimizer.state_dict(), temp_path / "optimizer.pt")
        torch.save([state["rng"] for state in runtime_states], temp_path / "rng.pt")
        _write_json(temp_path / "config.json", {"model": to_dict(model_config)})
        _write_json(
            temp_path / "train_state.json",
            {
                **resume_state.metadata,
                "step": resume_state.step,
                "smooth_loss": resume_state.smooth_loss,
                "loader_cursor": cursor_to_dict(resume_state.loader_cursor),
                "rank_loader_cursors": [state["loader_cursor"] for state in runtime_states],
                "saved_world_size": resume_state.saved_world_size,
                "exact_topology": resume_state.exact_topology,
            },
        )
        payload_files = ["model.safetensors", "optimizer.pt", "rng.pt", "config.json", "train_state.json"]
        manifest = {
            "format_version": FORMAT_VERSION,
            "complete": True,
            "step": resume_state.step,
            "saved_world_size": resume_state.saved_world_size,
            "training_fingerprint": training_fingerprint,
            "files": {name: _sha256(temp_path / name) for name in payload_files},
        }
        _write_json(temp_path / "manifest.json", manifest)
        _fsync_tree(temp_path)
        if previous.exists():
            shutil.rmtree(previous)
        if path.exists():
            os.replace(path, previous)
            moved_previous = True
            _fsync_directory(path.parent)
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
        if previous.exists():
            shutil.rmtree(previous)
            _fsync_directory(path.parent)
    except Exception:
        if moved_previous and not path.exists() and previous.exists():
            os.replace(previous, path)
            _fsync_directory(path.parent)
        raise
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)


def _resolve_checkpoint_path(path: Path) -> Path:
    manifest = path / "manifest.json"
    if manifest.exists():
        return path
    previous = path.with_name(f"{path.name}.previous")
    if (previous / "manifest.json").exists():
        return previous
    if (path / "model.safetensors").exists():
        raise LegacyCheckpointError(f"{path} is not a complete version-2 training checkpoint")
    raise TrainingCheckpointError(f"training checkpoint not found: {path}")


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingCheckpointError(f"invalid checkpoint manifest in {path}: {exc}") from exc
    if manifest.get("format_version") != FORMAT_VERSION:
        raise TrainingCheckpointError(
            f"unsupported training checkpoint version {manifest.get('format_version')!r}; expected {FORMAT_VERSION}"
        )
    if manifest.get("complete") is not True:
        raise TrainingCheckpointError(f"incomplete training checkpoint: {path}")
    return manifest


def _verify_files(path: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TrainingCheckpointError("checkpoint manifest has no file hashes")
    for name, expected_hash in files.items():
        candidate = path / name
        if not candidate.is_file():
            raise TrainingCheckpointError(f"checkpoint file is missing: {name}")
        actual_hash = _sha256(candidate)
        if actual_hash != expected_hash:
            raise TrainingCheckpointError(f"checkpoint hash mismatch for {name}")


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


def _elastic_seed(seed: int, step: int, world_size: int) -> int:
    digest = hashlib.sha256(f"{seed}:{step}:{world_size}".encode()).digest()
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _fsync_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_file():
            with open(child, "rb") as handle:
                os.fsync(handle.fileno())
    _fsync_directory(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
