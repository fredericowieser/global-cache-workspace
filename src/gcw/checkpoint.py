from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load as load_safetensors
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
    if not isinstance(model, dict):
        runtime_config = getattr(model, "config", None)
        if isinstance(runtime_config, ModelConfig) and to_dict(runtime_config) != to_dict(model_config):
            raise ValueError("checkpoint model_config does not match the model's runtime config")
    state = model if isinstance(model, dict) else model.state_dict()
    state = {k: v.detach().cpu() for k, v in state.items()}
    save_file(state, path / "model.safetensors")
    with open(path / "config.json", "w", encoding="utf-8") as f:
        json.dump({"model": to_dict(model_config)}, f, indent=2)
    with open(path / "train_state.json", "w", encoding="utf-8") as f:
        json.dump(train_state or {}, f, indent=2)


def publish_final_checkpoint_manifest(
    path: str | Path,
    final_processor_sha256: str,
    initial_processor_sha256: str,
) -> str:
    path = Path(path)
    for name, digest in (
        ("final_processor_sha256", final_processor_sha256),
        ("initial_processor_sha256", initial_processor_sha256),
    ):
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} must be a lowercase sha256 digest")
    required_files = (
        "config.json",
        "initial_processor.safetensors",
        "model.safetensors",
        "run_config.json",
        "train_state.json",
    )
    files: dict[str, str] = {}
    for name in required_files:
        candidate = path / name
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        files[name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    payload = {
        "schema": "gcw.final_checkpoint.manifest.v2",
        "files": files,
        "initial_processor_sha256": initial_processor_sha256,
        "final_processor_sha256": final_processor_sha256,
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with (path / "manifest.json").open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(encoded).hexdigest()




def load_checkpoint_bytes(
    *,
    config_bytes: bytes,
    train_state_bytes: bytes,
    model_bytes: bytes,
    device: torch.device | str = "cpu",
):
    try:
        config_payload = json.loads(config_bytes.decode("utf-8"))
        train_state = json.loads(train_state_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint JSON snapshot is invalid") from exc
    if not isinstance(config_payload, dict) or set(config_payload) != {"model"}:
        raise ValueError("checkpoint config snapshot fields are invalid")
    if not isinstance(train_state, dict):
        raise ValueError("checkpoint train-state snapshot must be an object")
    cfg = _model_config_from_checkpoint(config_payload["model"])
    weights = load_safetensors(model_bytes)
    model = build_model(cfg)
    dtype_name = train_state.get("dtype")
    if dtype_name == "mixed_bfloat16_backbone_float32_processor":
        backbone = getattr(model, "backbone", None)
        cache_processor = getattr(model, "cache_processor", None)
        if not isinstance(backbone, torch.nn.Module) or not isinstance(
            cache_processor, torch.nn.Module
        ):
            raise ValueError(
                "mixed BGT checkpoint requires backbone and cache_processor modules"
            )
        cast(torch.nn.Module, backbone).to(dtype=torch.bfloat16)
        cast(torch.nn.Module, cache_processor).to(dtype=torch.float32)
    else:
        recorded_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}.get(
            dtype_name if isinstance(dtype_name, str) else ""
        )
        dtype = recorded_dtype or _infer_floating_dtype(weights)
        if dtype is not None:
            model.to(dtype=dtype)
    model.load_state_dict(weights, strict=True)
    model.to(device)
    return model, cfg, train_state


def _model_config_from_checkpoint(payload: Any) -> ModelConfig:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint model configuration must be an object")
    model_payload = dict(payload)
    legacy_generate = model_payload.pop("legacy_bottleneck_generate", None)
    if legacy_generate is not None:
        if not isinstance(legacy_generate, bool):
            raise ValueError("legacy_bottleneck_generate must be boolean")
        model_payload.setdefault(
            "processor_generation_mode",
            "post_prefill_relative" if legacy_generate else "windowed",
        )
    return ModelConfig(**model_payload)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu"):
    path = Path(path)
    return load_checkpoint_bytes(
        config_bytes=(path / "config.json").read_bytes(),
        train_state_bytes=(path / "train_state.json").read_bytes(),
        model_bytes=(path / "model.safetensors").read_bytes(),
        device=device,
    )


def load_backbone_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    path = Path(path)
    weights = load_file(path / "model.safetensors", device="cpu")
    target = cast(torch.nn.Module, getattr(model, "backbone", model))
    target_state = target.state_dict()
    loadable = {}
    missing = []
    for key, expected in target_state.items():
        source_key = key if key in weights else f"backbone.{key}"
        if source_key not in weights and key == "lm_head.weight" and "embed_tokens.weight" in weights:
            source_key = "embed_tokens.weight"
        if source_key not in weights and key == "lm_head.weight" and "backbone.embed_tokens.weight" in weights:
            source_key = "backbone.embed_tokens.weight"
        if source_key not in weights:
            if key.endswith((".lora_a", ".lora_b")):
                continue
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


def _infer_floating_dtype(state: dict[str, torch.Tensor]) -> torch.dtype | None:
    for tensor in state.values():
        if tensor.is_floating_point():
            return tensor.dtype
    return None
