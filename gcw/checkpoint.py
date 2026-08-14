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
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = model if isinstance(model, dict) else model.state_dict()
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
    weights = load_file(path / "model.safetensors", device=str(device))
    model = build_model(cfg)
    model_dtype = _resolve_dtype(dtype, weights)
    if model_dtype is not None:
        model.to(dtype=model_dtype)
    model.load_state_dict(weights, strict=True)
    model.to(device)
    state_path = path / "train_state.json"
    train_state = (
        json.loads(state_path.read_text()) if state_path.exists() else {}
    )
    return model, cfg, train_state


def load_backbone_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    path = Path(path)
    weights = load_file(path / "model.safetensors", device="cpu")
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
