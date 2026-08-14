from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models.backbones import Backbone, build_backbone
from gcw.models.cache_processors.base import CacheProcessorBase


class LanguageModel(nn.Module):
    """A causal backbone with an optional cache processor.

    The wrapper owns exactly two state-dict domains: ``backbone`` and
    ``cache_processor``. A processor-free model returns the backbone directly,
    so baseline checkpoints retain their established unprefixed keys.
    """

    def __init__(
        self,
        config: ModelConfig,
        backbone: Backbone,
        cache_processor: CacheProcessorBase,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.cache_processor = cache_processor

    def init_weights(self) -> None:
        self.backbone.init_weights()

    def get_device(self) -> torch.device:
        return self.backbone.get_device()

    def _processor_config(self) -> CacheProcessorConfig:
        config = self.config.cache_processor
        if config is None:
            raise RuntimeError("processed language model has no cache processor configuration")
        return config

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_reduction: str = "mean",
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        processor = self._processor_config()
        # Prepared tensors are local to this context and therefore to this
        # forward's autograd graph. Generation deliberately keeps the eager path.
        with self.cache_processor.prepare_forward_execution() as execution:
            return self.backbone(
                input_ids,
                targets,
                loss_reduction,
                processor=execution,
                process_every=processor.process_every,
                attention_mask=attention_mask,
            )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        processor = self._processor_config()
        relative = self.config.processor_generation_mode == "post_prefill_relative"
        return self.backbone.generate(
            input_ids,
            max_new_tokens,
            eos_token_id=eos_token_id,
            processor=self.cache_processor,
            process_every=processor.process_every,
            process_prefill_windows=not relative,
            process_after_prefill=relative,
            relative_generation_cadence=relative,
        )

    @torch.no_grad()
    def generate_fast(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        stop_sequences: list[list[int]] | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        processor = self._processor_config()
        relative = self.config.processor_generation_mode == "post_prefill_relative"
        return self.backbone.generate_fast(
            input_ids,
            max_new_tokens,
            eos_token_id=eos_token_id,
            processor=self.cache_processor,
            process_every=processor.process_every,
            process_prefill_windows=not relative,
            process_after_prefill=relative,
            relative_generation_cadence=relative,
            stop_sequences=stop_sequences,
            pad_token_id=pad_token_id,
        )

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(parameter.numel() for parameter in self.parameters())
        if not non_embedding:
            return total
        embedding_layer = getattr(self.backbone, "embed_tokens")
        if not isinstance(embedding_layer, nn.Embedding):
            raise TypeError("backbone embed_tokens must be an embedding module")
        embedding = embedding_layer.weight.numel()
        lm_head = getattr(self.backbone, "lm_head", None)
        if lm_head is not None and not isinstance(lm_head, nn.Linear):
            raise TypeError("backbone lm_head must be a linear module or null")
        head = 0 if lm_head is None else lm_head.weight.numel()
        return total - embedding - head


def build_language_model(config: ModelConfig) -> Backbone | LanguageModel:
    """Build the explicit ``backbone × processor`` composition."""
    if config.variant == "tf":
        return build_backbone(config)

    processor = build_cache_processor(config)
    backbone_config = replace(
        config,
        flow_processor=None,
        diffusion_processor=None,
        processor_generation_mode="windowed",
    )
    if config.variant == "longshot":
        backbone_config = replace(backbone_config, workspace_flow_refinement=None)
    backbone = build_backbone(backbone_config)
    return LanguageModel(config, backbone, processor)


def build_cache_processor(config: ModelConfig) -> CacheProcessorBase:
    """Construct one cache processor without constructing a backbone."""
    if config.variant == "bt":
        from gcw.models.cache_processors.standard import PerLayerCacheProcessor

        return PerLayerCacheProcessor(config)
    if config.variant == "bgt":
        from gcw.models.cache_processors.standard import GlobalCacheProcessor as BGTProcessor

        return BGTProcessor(config)
    if config.variant == "icarus":
        from gcw.models.cache_processors.icarus import GlobalCacheProcessor as IcarusProcessor

        return IcarusProcessor(config)
    if config.variant == "longshot":
        from gcw.models.cache_processors.longshot import GlobalCacheProcessor as LongshotProcessor

        return LongshotProcessor(config)
    raise ValueError(f"variant {config.variant!r} does not define a cache processor")
