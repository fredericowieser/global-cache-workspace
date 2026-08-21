from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import random
from typing import Any, Literal
import warnings

import numpy as np
import torch
from safetensors.torch import load as load_safetensors

from gcw.data.training import FineWebCursor, LoaderCursor, cursor_from_dict
from gcw.training.checkpoint.safetensors import (
    _capture_verified_payloads,
    _elastic_seed,
    _move_tensors,
    _parse_manifest_bytes,
    _resolve_checkpoint_path,
    _restore_rng_state,
)
from gcw.training.checkpoint.types import (
    INITIAL_PROCESSOR_FILE,
    CheckpointCompatibilityError,
    ResumeState,
    TrainingCheckpointError,
    UnsupportedTopologyError,
)

LEGACY_PRECISION_FINGERPRINT_KEY = "bf16"


def _fingerprint_with_legacy_precision(
    saved: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    """Accept manifests written before ``precision`` replaced ``bf16`` in the training fingerprint.

    The legacy flag was ``config.precision != "fp32"``, so it cannot separate ``bf16_mixed`` from ``bf16``.
    A legacy manifest therefore resumes into either mode when the flag agrees, which is exactly the
    behaviour those checkpoints were written under. The ambiguity is warned about, not hidden.
    """
    if "precision" in saved or LEGACY_PRECISION_FINGERPRINT_KEY not in saved:
        return saved
    legacy_bf16 = bool(saved[LEGACY_PRECISION_FINGERPRINT_KEY])
    expected_precision = expected.get("precision")
    upgraded = {
        key: value
        for key, value in saved.items()
        if key != LEGACY_PRECISION_FINGERPRINT_KEY
    }
    if legacy_bf16 != (expected_precision != "fp32"):
        upgraded["precision"] = "bf16_mixed" if legacy_bf16 else "fp32"
        return upgraded
    if legacy_bf16:
        warnings.warn(
            f"checkpoint predates precision fingerprinting and records only bf16=true; "
            f"resuming as precision={expected_precision!r} without being able to confirm it was not "
            f"{'bf16' if expected_precision == 'bf16_mixed' else 'bf16_mixed'}",
            RuntimeWarning,
            stacklevel=2,
        )
    upgraded["precision"] = expected_precision
    return upgraded


class TrainingCheckpoint:
    def __init__(
        self,
        path: Path,
        resume_state: ResumeState,
        manifest: dict[str, Any],
        *,
        elastic_seeds: list[int] | None,
        rank_loader_cursors: list[LoaderCursor],
        payloads: dict[str, bytes],
    ) -> None:
        self.path = path
        self.resume_state = resume_state
        self.manifest = manifest
        self._elastic_seeds = elastic_seeds
        self._rank_loader_cursors = rank_loader_cursors
        self._payloads = payloads

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        expected_fingerprint: dict[str, Any],
        topology: Literal["same", "elastic_ddp"],
        current_world_size: int,
        expected_manifest_sha256: str | None = None,
    ) -> TrainingCheckpoint:
        if expected_fingerprint.get("distributed") == "fsdp":
            raise UnsupportedTopologyError(
                "FSDP training resume is not supported"
            )
        resolved = _resolve_checkpoint_path(Path(path))
        try:
            manifest_bytes = (resolved / "manifest.json").read_bytes()
        except OSError as exc:
            raise TrainingCheckpointError(
                f"invalid checkpoint manifest in {resolved}: {exc}"
            ) from exc
        if expected_manifest_sha256 is not None:
            actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            if actual_manifest_sha256 != expected_manifest_sha256:
                raise TrainingCheckpointError(
                    "training checkpoint manifest sha256 mismatch"
                )
        manifest = _parse_manifest_bytes(manifest_bytes, resolved)
        payloads = _capture_verified_payloads(resolved, manifest)
        saved_fingerprint = manifest.get("training_fingerprint")
        if isinstance(saved_fingerprint, dict):
            saved_fingerprint = _fingerprint_with_legacy_precision(
                saved_fingerprint, expected_fingerprint
            )
        if saved_fingerprint != expected_fingerprint:
            differing = sorted(
                key
                for key in set(saved_fingerprint or {})
                | set(expected_fingerprint)
                if (saved_fingerprint or {}).get(key)
                != expected_fingerprint.get(key)
            )
            raise CheckpointCompatibilityError(
                f"training fingerprint differs for: {', '.join(differing)}"
            )
        train_state = json.loads(payloads["train_state.json"].decode("utf-8"))
        reserved = {
            "step",
            "smooth_loss",
            "loader_cursor",
            "rank_loader_cursors",
            "saved_world_size",
            "exact_topology",
        }
        rank_loader_cursors = [
            cursor_from_dict(value)
            for value in train_state.get(
                "rank_loader_cursors", [train_state["loader_cursor"]]
            )
        ]
        resume_state = ResumeState(
            step=int(train_state["step"]),
            smooth_loss=float(train_state["smooth_loss"]),
            loader_cursor=cursor_from_dict(train_state["loader_cursor"]),
            saved_world_size=int(train_state["saved_world_size"]),
            exact_topology=True,
            metadata={
                key: value
                for key, value in train_state.items()
                if key not in reserved
            },
        )
        if resume_state.saved_world_size <= 0:
            raise CheckpointCompatibilityError(
                "checkpoint saved_world_size must be positive"
            )
        if len(rank_loader_cursors) != resume_state.saved_world_size:
            raise CheckpointCompatibilityError(
                f"checkpoint has {len(rank_loader_cursors)} rank cursors for saved world size {resume_state.saved_world_size}"
            )
        elastic_seeds = None
        if resume_state.saved_world_size != current_world_size:
            if topology != "elastic_ddp":
                raise CheckpointCompatibilityError(
                    f"checkpoint world size {resume_state.saved_world_size} does not match current world size {current_world_size}"
                )
            if (
                expected_fingerprint.get("distributed") != "ddp"
                or expected_fingerprint.get("data_kind") != "fineweb"
            ):
                raise UnsupportedTopologyError(
                    "elastic DDP resume is supported only for FineWeb DDP training"
                )
            if not isinstance(resume_state.loader_cursor, FineWebCursor) or any(
                cursor != resume_state.loader_cursor
                for cursor in rank_loader_cursors
            ):
                raise CheckpointCompatibilityError(
                    "elastic FineWeb resume requires one shared global loader cursor"
                )
            elastic_seeds = [
                _elastic_seed(
                    int(expected_fingerprint["seed"]),
                    resume_state.step,
                    rank,
                    current_world_size,
                )
                for rank in range(current_world_size)
            ]
            from dataclasses import replace

            resume_state = replace(resume_state, exact_topology=False)
        return cls(
            resolved,
            resume_state,
            manifest,
            elastic_seeds=elastic_seeds,
            rank_loader_cursors=rank_loader_cursors,
            payloads=payloads,
        )

    def loader_cursor(self, *, rank: int) -> LoaderCursor:
        if not self.resume_state.exact_topology:
            return self.resume_state.loader_cursor
        if rank >= len(self._rank_loader_cursors):
            raise CheckpointCompatibilityError(
                f"checkpoint has no loader cursor for rank {rank}"
            )
        return self._rank_loader_cursors[rank]

    def restore_model(self, model: torch.nn.Module) -> None:
        device = next(model.parameters()).device
        weights = load_safetensors(self._payloads["model.safetensors"])
        from gcw.checkpoint import _upgrade_qwen35_gated_delta_keys

        weights = _upgrade_qwen35_gated_delta_keys(weights)
        weights = {name: tensor.to(device) for name, tensor in weights.items()}
        model_keys = set(model.state_dict())
        weight_keys = set(weights)
        missing = model_keys - weight_keys
        if missing:
            backbone_keys = {
                key for key in model_keys if key.startswith("backbone.")
            }
            if not missing.issubset(backbone_keys):
                raise KeyError(
                    f"missing checkpoint keys: {sorted(missing)[:8]}"
                )
        model.load_state_dict(weights, strict=not missing)

    def load_initial_processor_state(
        self, *, device: torch.device | str = "cpu"
    ) -> dict[str, torch.Tensor]:
        payload = self._payloads.get(INITIAL_PROCESSOR_FILE)
        if payload is None:
            raise CheckpointCompatibilityError(
                "training checkpoint has no authenticated initial processor state"
            )
        return {
            name: tensor.to(device)
            for name, tensor in load_safetensors(payload).items()
        }

    def restore_optimizer(
        self, optimizer: torch.optim.Optimizer, *, device: torch.device | str
    ) -> None:
        state = torch.load(
            io.BytesIO(self._payloads["optimizer.pt"]),
            map_location="cpu",
            weights_only=True,
        )
        for group in state.get("param_groups", []):
            group["lr"] = optimizer.param_groups[0]["lr"]
        missing_states = set(
            range(len(optimizer.param_groups[0]["params"]))
        ) - set(state.get("state", {}).keys())
        if missing_states:
            raise CheckpointCompatibilityError(
                f"checkpoint optimizer state missing parameters: {sorted(missing_states)[:8]}"
            )
        optimizer.load_state_dict(state)
        target = torch.device(device)
        for value in optimizer.state.values():
            _move_tensors(value, target)

    def restore_rng(self, *, rank: int) -> None:
        if self._elastic_seeds is not None:
            if rank >= len(self._elastic_seeds):
                raise CheckpointCompatibilityError(
                    f"elastic checkpoint has no deterministic seed for rank {rank}"
                )
            seed = self._elastic_seeds[rank]
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            return
        states = torch.load(
            io.BytesIO(self._payloads["rng.pt"]),
            map_location="cpu",
            weights_only=False,
        )
        if rank >= len(states):
            raise CheckpointCompatibilityError(
                f"checkpoint has no RNG state for rank {rank}"
            )
        _restore_rng_state(states[rank])
