from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import torch
import torch.distributed as dist

from gcw.checkpoint import save_checkpoint
from gcw.config import TrainConfig
from gcw.data.training import TrainingBatchStream, cursor_to_dict
from gcw.runtime import dist_info
from gcw.training.audit import _checkpoint_state, _resume_state
from gcw.training.checkpoint import ResumeState, save_training_checkpoint
from gcw.training.distributed import _unwrap


def _save_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    resume_state: ResumeState,
    fingerprint: dict[str, Any],
    rank: int,
    initial_processor_state: dict[str, torch.Tensor] | None = None,
) -> None:
    cache_processor_only = bool(
        config.save_cache_processor_only or config.freeze_backbone
    )
    if config.distributed == "fsdp":
        if rank == 0:
            save_checkpoint(
                path,
                model,
                config.model,
                {**resume_state.metadata, "step": resume_state.step},
                cache_processor_only=cache_processor_only,
            )
        return
    checkpoint_kwargs: dict[str, Any] = (
        {"initial_processor_state": initial_processor_state}
        if initial_processor_state is not None
        else {}
    )
    if cache_processor_only:
        checkpoint_kwargs["cache_processor_only"] = True
    save_training_checkpoint(
        path,
        model,
        config.model,
        optimizer,
        resume_state,
        fingerprint,
        **checkpoint_kwargs,
    )


def _save_step_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    step: int,
    smooth_loss: float,
    stream: TrainingBatchStream,
    world_size: int,
    metadata: dict[str, Any],
    fingerprint: dict[str, Any],
    rank: int,
    *,
    use_fsdp: bool,
    initial_processor_state: dict[str, torch.Tensor] | None,
) -> None:
    if use_fsdp:
        state = _checkpoint_state(metadata, step, cursor_to_dict(stream.cursor))
        _save_training_checkpoint(
            path,
            model,
            config.model,
            state,
            use_fsdp=True,
            cache_processor_only=bool(
                config.save_cache_processor_only or config.freeze_backbone
            ),
        )
        return
    resume_state = _resume_state(
        step, smooth_loss, stream, world_size, metadata
    )
    if initial_processor_state is None:
        _save_training_state(
            path,
            _unwrap(model),
            optimizer,
            config,
            resume_state,
            fingerprint,
            rank,
        )
    else:
        _save_training_state(
            path,
            _unwrap(model),
            optimizer,
            config,
            resume_state,
            fingerprint,
            rank,
            initial_processor_state=initial_processor_state,
        )


def _json_safe_loader_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return None
    if isinstance(value, dict):
        return {
            k: safe
            for k, v in value.items()
            if (safe := _json_safe_loader_state(v)) is not None
        }
    if isinstance(value, list):
        return [
            safe
            for item in value
            if (safe := _json_safe_loader_state(item)) is not None
        ]
    if isinstance(value, tuple):
        return [
            safe
            for item in value
            if (safe := _json_safe_loader_state(item)) is not None
        ]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _retained_checkpoint_every() -> int | None:
    keep = os.environ.get("GCW_KEEP_STEP_CHECKPOINTS", "0").lower()
    if keep not in {"1", "true", "yes", "on"}:
        return None
    raw = os.environ.get("GCW_STEP_CHECKPOINT_EVERY", "1000")
    try:
        every = int(raw)
    except ValueError as exc:
        raise ValueError(
            "GCW_STEP_CHECKPOINT_EVERY must be an integer"
        ) from exc
    if every <= 0:
        raise ValueError("GCW_STEP_CHECKPOINT_EVERY must be positive")
    return every


def _save_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_config,
    checkpoint_state: dict[str, Any],
    *,
    use_fsdp: bool = False,
    cache_processor_only: bool = False,
) -> None:
    if use_fsdp:
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state = model.state_dict()
        if dist_info()[1] == 0:
            save_checkpoint(
                path,
                state,
                model_config,
                checkpoint_state,
                cache_processor_only=cache_processor_only,
            )
        return
    if dist.is_available() and dist.is_initialized() and dist_info()[1] != 0:
        return
    unwrapped = _unwrap(model)
    unwrapped = getattr(unwrapped, "_orig_mod", unwrapped)
    save_checkpoint(
        path,
        unwrapped,
        model_config,
        checkpoint_state,
        cache_processor_only=cache_processor_only,
    )
