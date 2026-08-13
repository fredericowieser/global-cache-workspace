"""Native causal language-model backbones supported by GCW."""

from __future__ import annotations

from gcw.config import ModelConfig
from gcw.models.backbones.base import Backbone
from gcw.models.backbones.transformer import TF


def build_backbone(config: ModelConfig) -> Backbone:
    """Construct the native backbone selected by a validated model config."""
    if config.family == "qwen3_5":
        from gcw.models.backbones.qwen3 import Qwen35TF

        return Qwen35TF(config)
    if config.family == "qwen3_5_moe":
        from gcw.models.backbones.qwen3 import Qwen35MoeTF

        return Qwen35MoeTF(config)
    if config.family == "gemma4":
        from gcw.models.backbones.gemma4 import Gemma4TF

        return Gemma4TF(config)
    return TF(config)


__all__ = ["Backbone", "TF", "build_backbone"]
