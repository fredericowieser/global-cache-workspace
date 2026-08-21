from __future__ import annotations

import torch
import torch.nn as nn

from gcw.config import ModelConfig
from gcw.models.cache_processors.state import StaticCacheState
from gcw.models.backbones.transformer.attention import CausalSelfAttention
from gcw.models.backbones.transformer.layers import MLP, RMSNorm


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.norm_eps)
        self.self_attn = CausalSelfAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.norm_eps
        )
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        past: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, keys, values, scores = self.self_attn(
            self.input_layernorm(x), cos_sin, past
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, keys, values, scores

    def forward_static(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cache: StaticCacheState,
        start_pos: int,
        end_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, scores = self.self_attn.forward_static(
            self.input_layernorm(x), cos_sin, cache, start_pos, end_pos
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, scores

    def forward_tensor_past(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, keys, values, scores = self.self_attn.forward_tensor_past(
            self.input_layernorm(x),
            cos_sin,
            past_keys,
            past_values,
            attention_bias,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, keys, values, scores
