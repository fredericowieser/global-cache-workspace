from __future__ import annotations

from pathlib import Path
import torch

from gcw.checkpoint import (
    _model_config_from_checkpoint,
    load_backbone_checkpoint,
    load_checkpoint,
)
from gcw.config import ModelConfig, TrainConfig, to_dict
from gcw.models import build_model
from gcw.models.lora import LoRALinear


def _load_initial_checkpoint(
    model_config: ModelConfig | None, checkpoint_path: str | Path
) -> torch.nn.Module:
    source, source_config, _ = load_checkpoint(checkpoint_path, device="cpu")
    if not isinstance(model_config, type(source_config)) or to_dict(
        _model_config_from_checkpoint(to_dict(model_config))
    ) != to_dict(source_config):
        raise ValueError(
            "init_checkpoint model configuration differs from train.model"
        )
    return source


def _load_initial_model(
    model: torch.nn.Module, checkpoint_path: str | Path
) -> None:
    """Load a complete model state without restoring optimizer or run state."""
    source = _load_initial_checkpoint(
        getattr(model, "config", None), checkpoint_path
    )
    model.load_state_dict(source.state_dict(), strict=True)


def _apply_trainable_policy(model: torch.nn.Module, policy: str) -> None:
    """Select trainable ownership before wrapping or optimizer construction."""
    backbone = getattr(model, "backbone", model)
    processor = getattr(model, "cache_processor", None)
    if policy == "all":
        model.requires_grad_(True)
        return
    if policy == "backbone":
        backbone.requires_grad_(True)
        if isinstance(processor, torch.nn.Module):
            processor.requires_grad_(False)
        return
    if policy in {"lora", "lora_and_cache_processor"}:
        model.requires_grad_(False)
        adapters = [
            module
            for module in backbone.modules()
            if isinstance(module, LoRALinear)
        ]
        if not adapters:
            raise ValueError(
                f"trainable={policy!r} requires attached LoRA projections"
            )
        for adapter in adapters:
            adapter.lora_a.requires_grad_(True)
            adapter.lora_b.requires_grad_(True)
        if policy == "lora_and_cache_processor":
            if not isinstance(processor, torch.nn.Module):
                raise ValueError(
                    "trainable='lora_and_cache_processor' requires a cache processor"
                )
            processor.requires_grad_(True)
        return
    if policy == "cache_processor":
        if not isinstance(processor, torch.nn.Module):
            raise ValueError(
                "trainable='cache_processor' requires a cache processor"
            )
        backbone.requires_grad_(False)
        processor.requires_grad_(True)
        return
    raise ValueError(f"unsupported trainable policy: {policy}")


def _model_dtype(config: TrainConfig) -> torch.dtype:
    """Parameter dtype implied by the precision mode; mixed modes keep float32 and autocast."""
    return torch.bfloat16 if config.precision == "bf16" else torch.float32


def _build_training_model(
    config: TrainConfig, device: torch.device, *, rank: int, use_fsdp: bool
) -> torch.nn.Module:
    model_dtype = _model_dtype(config)
    if use_fsdp:
        if rank == 0:
            if config.init_checkpoint:
                model = _load_initial_checkpoint(
                    config.model, config.init_checkpoint
                )
                return model.to(dtype=model_dtype)
            model = build_model(config.model)
            model.init_weights()
            if config.backbone_checkpoint:
                load_backbone_checkpoint(model, config.backbone_checkpoint)
            return model.to(dtype=model_dtype)
        with torch.device("meta"):
            return build_model(config.model).to(dtype=model_dtype)

    model = build_model(config.model).to(device)
    model.init_weights()
    if config.init_checkpoint:
        _load_initial_model(model, config.init_checkpoint)
    elif config.backbone_checkpoint:
        load_backbone_checkpoint(model, config.backbone_checkpoint)
    return model.to(dtype=model_dtype)
