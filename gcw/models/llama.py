from __future__ import annotations

from gcw.config import ModelConfig
from gcw.models.transformer import TF


def llama_config(**kwargs) -> ModelConfig:
    return ModelConfig(family="llama", qk_norm=False, **kwargs)


LlamaTF = TF
