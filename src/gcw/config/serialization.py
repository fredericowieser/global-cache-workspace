from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

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
        payload = {field.name: to_dict(getattr(value, field.name)) for field in fields(value)}
        if not value.longshot_full_history.enabled:
            payload.pop("longshot_full_history")
        if not value.kv_score_sidecar.enabled:
            payload.pop("kv_score_sidecar")
        if not value.triton_cache_movement.enabled:
            payload.pop("triton_cache_movement")
        if not value.optimized_step.enabled:
            payload.pop("optimized_step")
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
