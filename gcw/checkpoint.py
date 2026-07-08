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
    model: torch.nn.Module,
    model_config: ModelConfig,
    train_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    save_file(state, path / "model.safetensors")
    with open(path / "config.json", "w", encoding="utf-8") as f:
        json.dump({"model": to_dict(model_config)}, f, indent=2)
    with open(path / "train_state.json", "w", encoding="utf-8") as f:
        json.dump(train_state or {}, f, indent=2)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu"):
    path = Path(path)
    with open(path / "config.json", "r", encoding="utf-8") as f:
        cfg = ModelConfig(**json.load(f)["model"])
    weights = load_file(path / "model.safetensors", device=str(device))
    model = build_model(cfg)
    dtype = _infer_floating_dtype(weights)
    if dtype is not None:
        model.to(dtype=dtype)
    model.load_state_dict(weights, strict=True)
    model.to(device)
    state_path = path / "train_state.json"
    train_state = json.loads(state_path.read_text()) if state_path.exists() else {}
    return model, cfg, train_state


def _infer_floating_dtype(state: dict[str, torch.Tensor]) -> torch.dtype | None:
    for tensor in state.values():
        if tensor.is_floating_point():
            return tensor.dtype
    return None
