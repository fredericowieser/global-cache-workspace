"""Native causal language-model backbones supported by GCW."""

from __future__ import annotations

from gcw.config import ModelConfig
from gcw.models.backbones.base import Backbone
from gcw.models.backbones.transformer import TF
from gcw.models.lora import apply_lora


def build_backbone(config: ModelConfig) -> Backbone:
    """Construct the native backbone selected by a validated model config."""
    if config.family == "qwen3_5":
        from gcw.models.backbones.qwen3 import Qwen35TF

        return apply_lora(Qwen35TF(config), config.lora)
    if config.family == "qwen3_5_moe":
        from gcw.models.backbones.qwen3 import Qwen35MoeTF

        return apply_lora(Qwen35MoeTF(config), config.lora)
    if config.family == "gemma4":
        from gcw.models.backbones.gemma4 import Gemma4TF

        return apply_lora(Gemma4TF(config), config.lora)
    return apply_lora(TF(config), config.lora)


__all__ = ["Backbone", "TF", "build_backbone"]
