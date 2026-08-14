from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from gcw.config import ModelConfig, to_dict
from gcw.models import build_model


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module | dict[str, torch.Tensor],
    model_config: ModelConfig,
    train_state: dict[str, Any] | None = None,
    *,
    cache_processor_only: bool = False,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = model if isinstance(model, dict) else model.state_dict()
    if cache_processor_only:
        state = {
            key: tensor
            for key, tensor in state.items()
            if key.startswith("cache_processor.")
        }
        if not state:
            raise ValueError(
                "cannot save a cache-processor-only checkpoint because the "
                "model has no cache_processor state"
            )
    state = {k: v.detach().cpu() for k, v in state.items()}
    save_file(state, path / "model.safetensors")
    with open(path / "config.json", "w", encoding="utf-8") as f:
        json.dump({"model": to_dict(model_config)}, f, indent=2)
    with open(path / "train_state.json", "w", encoding="utf-8") as f:
        json.dump(train_state or {}, f, indent=2)


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | str | None = "auto",
):
    path = Path(path)
    with open(path / "config.json", "r", encoding="utf-8") as f:
        cfg = ModelConfig(**json.load(f)["model"])
    state_path = path / "train_state.json"
    train_state = (
        json.loads(state_path.read_text()) if state_path.exists() else {}
    )
    weights = load_file(path / "model.safetensors", device=str(device))
    weights = _upgrade_qwen35_gated_delta_keys(weights)
    model = build_model(cfg)
    model_dtype = _resolve_dtype(dtype, weights)
    if model_dtype is not None:
        model.to(dtype=model_dtype)
    model_keys = set(model.state_dict())
    weight_keys = set(weights)
    unexpected = weight_keys - model_keys
    if unexpected:
        raise KeyError(f"unexpected checkpoint keys: {sorted(unexpected)[:8]}")
    missing = model_keys - weight_keys
    if missing:
        backbone_keys = {
            key for key in model_keys if key.startswith("backbone.")
        }
        if not missing.issubset(backbone_keys):
            raise KeyError(f"missing checkpoint keys: {sorted(missing)[:8]}")
        backbone_checkpoint = train_state.get("backbone_checkpoint")
        if not backbone_checkpoint:
            raise KeyError(
                "checkpoint contains only cache processor parameters but "
                "train_state.json does not specify backbone_checkpoint"
            )
        load_backbone_checkpoint(model, backbone_checkpoint)
    model.load_state_dict(weights, strict=not missing)
    model.to(device)
    return model, cfg, train_state


def load_backbone_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    path = Path(path)
    weights = load_file(path / "model.safetensors", device="cpu")
    weights = _upgrade_qwen35_gated_delta_keys(weights)
    target = model.backbone if hasattr(model, "backbone") else model
    target_state = target.state_dict()
    loadable = {}
    missing = []
    for key, expected in target_state.items():
        source_key = key if key in weights else f"backbone.{key}"
        if (
            source_key not in weights
            and key == "lm_head.weight"
            and "embed_tokens.weight" in weights
        ):
            source_key = "embed_tokens.weight"
        if (
            source_key not in weights
            and key == "lm_head.weight"
            and "backbone.embed_tokens.weight" in weights
        ):
            source_key = "backbone.embed_tokens.weight"
        if source_key not in weights:
            missing.append(key)
            continue
        tensor = weights[source_key]
        if tensor.shape != expected.shape:
            raise ValueError(
                f"shape mismatch for backbone key {key}: target {tuple(expected.shape)} "
                f"vs checkpoint {tuple(tensor.shape)} from {source_key}"
            )
        loadable[key] = tensor.to(device=expected.device, dtype=expected.dtype)
    if missing:
        raise KeyError(f"missing backbone checkpoint keys: {missing[:8]}")
    target.load_state_dict(loadable, strict=True)


def _upgrade_qwen35_gated_delta_keys(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    upgraded = dict(state)
    suffixes = {
        ".linear_attn.A_log": ".linear_attn.dynamics.A_log",
        ".linear_attn.dt_bias": ".linear_attn.dynamics.dt_bias",
    }
    for key in list(state):
        for legacy_suffix, current_suffix in suffixes.items():
            if not key.endswith(legacy_suffix):
                continue
            current_key = key[: -len(legacy_suffix)] + current_suffix
            if current_key not in upgraded:
                upgraded[current_key] = upgraded[key]
            del upgraded[key]
            break
    return upgraded


def _resolve_dtype(
    dtype: torch.dtype | str | None, state: dict[str, torch.Tensor]
) -> torch.dtype | None:
    if dtype is None or dtype == "auto":
        return _infer_floating_dtype(state)
    if isinstance(dtype, torch.dtype):
        return dtype
    choices = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return choices[dtype]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dtype {dtype!r}; use auto, bfloat16, float16, or float32"
        ) from exc


def _infer_floating_dtype(state: dict[str, torch.Tensor]) -> torch.dtype | None:
    for tensor in state.values():
        if tensor.is_floating_point():
            return tensor.dtype
    return None
