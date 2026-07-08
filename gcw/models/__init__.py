from gcw.config import ModelConfig
from gcw.models.bgt import BGT
from gcw.models.bt import BT
from gcw.models.transformer import TF


def build_model(config: ModelConfig):
    if config.variant == "tf":
        return TF(config)
    if config.variant == "bt":
        return BT(config)
    if config.variant == "bgt":
        return BGT(config)
    raise ValueError(f"unknown model variant: {config.variant}")


__all__ = ["BGT", "BT", "TF", "build_model"]
