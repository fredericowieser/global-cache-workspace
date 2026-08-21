from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Literal, Mapping
from uuid import uuid4

import torch
import yaml

from gcw.config import to_dict
from gcw.provenance import repository_source_identity
from gcw.runtime import atomic_write_json, atomic_write_text, fsync_directory

RunKind = Literal["training", "evaluation"]
_RUN_SCHEMA = "gcw.run_record.v1"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class RunPaths:
    """Paths owned by one local execution record."""

    root: Path
    config: Path
    metrics_stream: Path
    metrics_summary: Path
    logs: Path
    status: Path
    provenance: Path
    environment: Path
    wandb: Path
    state: Path

    @classmethod
    def under(cls, root: Path) -> RunPaths:
        return cls(
            root=root,
            config=root / "config.yaml",
            metrics_stream=root / "metrics.jsonl",
            metrics_summary=root / "metrics.json",
            logs=root / "logs.txt",
            status=root / "status.json",
            provenance=root / "provenance.json",
            environment=root / "environment.json",
            wandb=root / "wandb.json",
            state=root / "state.json",
        )


class RunRecord:
    """Append-only local record for one training or evaluation execution.

    The record contains configuration, metrics, logs, environment, and
    provenance. Model weights and optimizer state stay in the separately
    declared mutable state path.
    """

    def __init__(self, paths: RunPaths, *, kind: RunKind, execution_id: str):
        self.paths = paths
        self.kind = kind
        self.execution_id = execution_id
        self._closed = False

    @classmethod
    def create(
        cls,
        *,
        kind: RunKind,
        experiment_name: str,
        config: Any,
        state_path: str | Path | None,
        data_root: str | Path | None = None,
        execution_id: str | None = None,
        repository: str | Path | None = None,
    ) -> RunRecord:
        """Claim and initialize a fresh run record.

        Args:
            kind: Training or evaluation record domain.
            experiment_name: Human-readable experiment name.
            config: Fully resolved typed config or JSON-compatible mapping.
            state_path: Mutable checkpoint state outside the run-log tree.
            data_root: Repository data root. Defaults to ``GCW_DATA_ROOT`` or ``data``.
            execution_id: Unique attempt ID. Defaults to ``GCW_EXECUTION_ID`` or a UUID prefix.
            repository: Checkout root used for source identity.

        Raises:
            FileExistsError: If the execution directory already exists.
            ValueError: If a name or identity cannot form a safe path.
        """
        clean_name = _safe_component(experiment_name, "experiment_name")
        attempt = _safe_component(
            execution_id
            or os.environ.get("GCW_EXECUTION_ID")
            or uuid4().hex[:12],
            "execution_id",
        )
        now = datetime.now(timezone.utc)
        domain = "training-runs" if kind == "training" else "eval-runs"
        base = Path(data_root or os.environ.get("GCW_DATA_ROOT", "data"))
        root = (
            base / domain / now.date().isoformat() / f"{clean_name}--{attempt}"
        )
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir(exist_ok=False)
        paths = RunPaths.under(root)
        record = cls(paths, kind=kind, execution_id=attempt)
        resolved = to_dict(config)
        paths.config.write_text(
            yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
        )
        atomic_write_text(paths.logs, "")
        atomic_write_json(
            paths.environment, _environment_payload(), indent=2, allow_nan=False
        )
        source = _source_identity(repository)
        commit = source.get("git_commit") if source is not None else None
        atomic_write_text(root / "git_hash.txt", f"{commit or 'unavailable'}\n")
        atomic_write_json(
            paths.provenance,
            {
                "schema": _RUN_SCHEMA,
                "kind": kind,
                "execution_id": attempt,
                "experiment_name": clean_name,
                "created_at": now.isoformat(),
                "source": source,
            },
            indent=2,
            allow_nan=False,
        )
        atomic_write_json(
            paths.state,
            {
                "schema": _RUN_SCHEMA,
                "mutable_state_path": (
                    None if state_path is None else str(Path(state_path))
                ),
            },
            indent=2,
            allow_nan=False,
        )
        atomic_write_json(
            paths.status,
            {
                "schema": _RUN_SCHEMA,
                "state": "running",
                "updated_at": now.isoformat(),
            },
            indent=2,
            allow_nan=False,
        )
        fsync_directory(root)
        return record

    def append_metrics(
        self, metrics: Mapping[str, Any], *, step: int | None = None
    ) -> None:
        """Append one finite JSON metric event and fsync it."""
        self._require_open()
        payload = (
            {"step": step, **dict(metrics)}
            if step is not None
            else dict(metrics)
        )
        encoded = (
            json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False)
            + "\n"
        )
        with self.paths.metrics_stream.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def append_log(self, message: str) -> None:
        """Append one local log line and fsync it."""
        self._require_open()
        line = message.rstrip("\n") + "\n"
        with self.paths.logs.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def bind_wandb(
        self, *, run_id: str | None, url: str | None, mode: str | None
    ) -> None:
        """Bind this local execution to its optional W&B mirror."""
        self._require_open()
        atomic_write_json(
            self.paths.wandb,
            {"schema": _RUN_SCHEMA, "run_id": run_id, "url": url, "mode": mode},
            indent=2,
            allow_nan=False,
        )

    def finish(
        self, *, summary: Mapping[str, Any], error: BaseException | None = None
    ) -> None:
        """Finalize metrics and status without deleting the event stream."""
        self._require_open()
        atomic_write_json(
            self.paths.metrics_summary,
            _json_safe(dict(summary)),
            indent=2,
            allow_nan=False,
        )
        now = datetime.now(timezone.utc).isoformat()
        status: dict[str, Any] = {
            "schema": _RUN_SCHEMA,
            "state": "failed" if error is not None else "succeeded",
            "updated_at": now,
        }
        if error is not None:
            status["error_type"] = type(error).__name__
            status["error"] = str(error)
        atomic_write_json(self.paths.status, status, indent=2, allow_nan=False)
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("run record is already finalized")


def _safe_component(value: str, field: str) -> str:
    cleaned = _SAFE_NAME.sub("-", value.strip()).strip("-.")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"{field} must contain a filesystem-safe name")
    return cleaned[:120]


def _environment_payload() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "schema": _RUN_SCHEMA,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cuda_devices": (
            [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if cuda_available
            else []
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


def _source_identity(repository: str | Path | None) -> dict[str, Any] | None:
    root = (
        Path(repository).resolve()
        if repository is not None
        else Path.cwd().resolve()
    )
    if (
        not (root / "pyproject.toml").is_file()
        or not (root / "src" / "gcw").is_dir()
    ):
        return None
    try:
        return repository_source_identity(root, require_git=False)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")
