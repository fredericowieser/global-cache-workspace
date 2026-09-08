from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from gcw.config.model import CacheProcessorConfig, DiffusionProcessorConfig
from gcw.config.training import TrainConfig
from gcw.performance.config import PerformanceConfig


def load_train_config_bytes(payload: bytes) -> TrainConfig:
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("training config must be UTF-8 YAML") from exc
    data = yaml.safe_load(decoded)
    if not isinstance(data, dict):
        raise ValueError("training config must decode to a mapping")
    return TrainConfig(**data)


def load_train_config(path: str | Path) -> TrainConfig:
    return load_train_config_bytes(Path(path).read_bytes())


def to_dict(value: Any) -> Any:
    if isinstance(value, PerformanceConfig):
        payload = {
            field.name: to_dict(getattr(value, field.name))
            for field in fields(value)
        }
        if not value.longshot_full_history.enabled:
            payload.pop("longshot_full_history")
        if not value.kv_score_sidecar.enabled:
            payload.pop("kv_score_sidecar")
        if not value.triton_cache_movement.enabled:
            payload.pop("triton_cache_movement")
        if not value.optimized_step.enabled:
            payload.pop("optimized_step")
        return payload
    if isinstance(value, TrainConfig):
        payload = {
            field.name: to_dict(getattr(value, field.name))
            for field in fields(value)
        }
        # Preserve pre-feature config identities for the disabled default while
        # retaining the field in serialized provenance when the policy is used.
        if not value.truncated_bptt_windows:
            payload.pop("truncated_bptt_windows")
        return payload
    if isinstance(value, CacheProcessorConfig):
        payload = {
            field.name: to_dict(getattr(value, field.name))
            for field in fields(value)
        }
        if not value.live_rank1_linear_writer:
            payload.pop("live_rank1_linear_writer")
        if value.output_scale_mode == "learned":
            payload.pop("output_scale_mode")
        return payload
    if isinstance(value, DiffusionProcessorConfig):
        payload = {
            field.name: to_dict(getattr(value, field.name))
            for field in fields(value)
        }
        # Unset controls retain pre-feature checkpoint/run identities.
        if value.training_method == "fixed":
            payload.pop("training_method")
        if value.perturbation_enabled:
            payload.pop("perturbation_enabled")
        for name in (
            "schedule_horizon",
            "training_solver_steps",
            "inference_solver_steps",
        ):
            if getattr(value, name) is None:
                payload.pop(name)
        return payload
    if is_dataclass(value):
        return {f.name: to_dict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [to_dict(x) for x in value]
    if isinstance(value, dict):
        return {k: to_dict(v) for k, v in value.items()}
    return value


def dump_yaml(path: str | Path, value: Any) -> None:
    data = to_dict(value)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
