from __future__ import annotations

from gcw.config import ModelConfig
from gcw.models.transformer import TF


def build_backbone(config: ModelConfig):
    if config.family == "qwen3_5_moe":
        from gcw.models.qwen3 import Qwen35MoeTF

        return Qwen35MoeTF(config)
    return TF(config)
