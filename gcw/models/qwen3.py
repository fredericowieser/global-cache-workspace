from __future__ import annotations

from gcw.config import ModelConfig
from gcw.models.transformer import TF


def qwen3_config(**kwargs) -> ModelConfig:
    if kwargs.get("layer_types") or kwargs.get("sliding_window"):
        raise NotImplementedError("Qwen3 sliding attention is not implemented in gcw yet.")
    kwargs.pop("layer_types", None)
    kwargs.pop("sliding_window", None)
    return ModelConfig(family="qwen3", qk_norm=True, **kwargs)


Qwen3TF = TF
