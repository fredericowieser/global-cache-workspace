from importlib.metadata import PackageNotFoundError, version

from gcw.config import CacheProcessorConfig, EvalConfig, ModelConfig, TrainConfig
from gcw.models import Backbone, CacheProcessorBase, LanguageModel, TF, build_model

try:
    __version__ = version("gcw")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "Backbone",
    "CacheProcessorBase",
    "CacheProcessorConfig",
    "EvalConfig",
    "LanguageModel",
    "ModelConfig",
    "TF",
    "TrainConfig",
    "build_model",
]
