"""Backbone and cache-processor model assembly."""

from __future__ import annotations

from gcw.architecture import resolve_model
from gcw.config import ModelConfig
from gcw.models.backbones import Backbone, TF, build_backbone
from gcw.models.cache_processors.base import CacheProcessorBase
from gcw.models.cache_processors.diffusion import TaskDirectedDiffusion
from gcw.models.cache_processors.flow import LatentFlow, set_flow_solver_steps
from gcw.models.model import LanguageModel, build_cache_processor, build_language_model


def build_model(config: ModelConfig) -> Backbone | LanguageModel:
    """Validate and construct an explicit backbone/processor composition."""
    resolve_model(config).require()
    return build_language_model(config)


__all__ = [
    "Backbone",
    "CacheProcessorBase",
    "LanguageModel",
    "LatentFlow",
    "TaskDirectedDiffusion",
    "TF",
    "build_backbone",
    "build_cache_processor",
    "build_model",
    "set_flow_solver_steps",
]
