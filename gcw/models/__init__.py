from gcw.config import ModelConfig
from gcw.models.bgt import BGT
from gcw.models.backbone import build_backbone
from gcw.models.bt import BT
from gcw.models.icarus import Icarus
from gcw.models.longshot import Longshot
from gcw.models.transformer import TF


def build_model(config: ModelConfig):
    if config.variant == "tf":
        return build_backbone(config)
    if config.variant == "bt":
        return BT(config)
    if config.variant == "bgt":
        return BGT(config)
    if config.variant == "icarus":
        return Icarus(config)
    if config.variant == "longshot":
        return Longshot(config)
    raise ValueError(f"unknown model variant: {config.variant}")


__all__ = [
    "BGT",
    "BT",
    "Icarus",
    "Longshot",
    "TF",
    "build_backbone",
    "build_model",
]
