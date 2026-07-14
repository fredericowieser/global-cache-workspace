from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import torch
import torch.nn as nn

from gcw.config import CacheProcessorConfig, ModelConfig, Variant
from gcw.models.cache_processor import GlobalCacheProcessor
from gcw.models.transformer import TF


class BGT(nn.Module):
    processor_cls: ClassVar[type[nn.Module]] = GlobalCacheProcessor
    variant: ClassVar[Variant] = "bgt"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = replace(config, variant=self.variant)
        self.backbone = TF(config)
        self.cache_processor = self.processor_cls(self.config)

    def init_weights(self) -> None:
        self.backbone.init_weights()

    def get_device(self) -> torch.device:
        return self.backbone.get_device()

    def _cache_processor_config(self) -> CacheProcessorConfig:
        config = self.config.cache_processor
        if config is None:
            raise RuntimeError(f"{self.variant} model has no cache processor configuration")
        return config

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None, loss_reduction: str = "mean") -> torch.Tensor:
        cp = self._cache_processor_config()
        return self.backbone(input_ids, targets, loss_reduction, processor=self.cache_processor, process_every=cp.process_every)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None = None) -> torch.Tensor:
        cp = self._cache_processor_config()
        legacy = self.config.legacy_bottleneck_generate
        return self.backbone.generate(
            input_ids,
            max_new_tokens,
            eos_token_id=eos_token_id,
            processor=self.cache_processor,
            process_every=cp.process_every,
            process_prefill_windows=not legacy,
            process_after_prefill=legacy,
            relative_generation_cadence=legacy,
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
        cp = self._cache_processor_config()
        legacy = self.config.legacy_bottleneck_generate
        return self.backbone.generate_fast(
            input_ids,
            max_new_tokens,
            eos_token_id=eos_token_id,
            processor=self.cache_processor,
            process_every=cp.process_every,
            process_prefill_windows=not legacy,
            process_after_prefill=legacy,
            relative_generation_cadence=legacy,
            stop_sequences=stop_sequences,
            pad_token_id=pad_token_id,
        )

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not non_embedding:
            return total
        return total - self.backbone.embed_tokens.weight.numel() - (0 if self.backbone.lm_head is None else self.backbone.lm_head.weight.numel())
