from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from gcw.config import ModelConfig
from gcw.models.cache_processor import GlobalCacheProcessor
from gcw.models.transformer import TF


class BGT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = replace(config, variant="bgt")
        self.backbone = TF(config)
        self.cache_processor = GlobalCacheProcessor(self.config)

    def init_weights(self) -> None:
        self.backbone.init_weights()

    def get_device(self) -> torch.device:
        return self.backbone.get_device()

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None, loss_reduction: str = "mean") -> torch.Tensor:
        cp = self.config.cache_processor
        return self.backbone(input_ids, targets, loss_reduction, processor=self.cache_processor, process_every=cp.process_every)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None = None) -> torch.Tensor:
        cp = self.config.cache_processor
        return self.backbone.generate(
            input_ids,
            max_new_tokens,
            eos_token_id=eos_token_id,
            processor=self.cache_processor,
            process_every=cp.process_every,
        )

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not non_embedding:
            return total
        return total - self.backbone.embed_tokens.weight.numel() - (0 if self.backbone.lm_head is None else self.backbone.lm_head.weight.numel())
