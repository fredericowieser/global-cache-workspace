from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcw.config import ModelConfig
from gcw.models.cache_processors.state import StaticCacheState
from gcw.models.backbones.transformer import Linear
from gcw.models.backbones.qwen3.attention import Qwen35Attention
from gcw.models.backbones.qwen3.linear_attention import Qwen35GatedDeltaNet
from gcw.models.backbones.qwen3.norms import Qwen35RMSNorm


class Qwen35MLP(nn.Module):
    def __init__(self, config: ModelConfig, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(
            config.hidden_size, intermediate_size, bias=False
        )
        self.up_proj = Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(
            intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35Block(nn.Module):
    def __init__(
        self, config: ModelConfig, layer_idx: int, cache_idx: int | None
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = (
            config.layer_types[layer_idx]
            if config.layer_types is not None
            else "full_attention"
        )
        self.input_layernorm = Qwen35RMSNorm(
            config.hidden_size, config.norm_eps
        )
        self.linear_attn: Qwen35GatedDeltaNet | None
        self.self_attn: Qwen35Attention | None
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(config, layer_idx)
            self.self_attn = None
        else:
            if cache_idx is None:
                raise ValueError("full_attention layers require a cache index")
            self.self_attn = Qwen35Attention(config, layer_idx, cache_idx)
            self.linear_attn = None
        self.post_attention_layernorm = Qwen35RMSNorm(
            config.hidden_size, config.norm_eps
        )
        self.mlp: nn.Module = Qwen35MLP(config, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        past: tuple[torch.Tensor, torch.Tensor] | None,
        linear_state: tuple[torch.Tensor, torch.Tensor | None] | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor | None] | None,
    ]:
        h = self.input_layernorm(x)
        if self.layer_type == "linear_attention":
            if self.linear_attn is None:
                raise RuntimeError(
                    "linear-attention block has no linear attention module"
                )
            attn_out, new_linear_state = self.linear_attn(h, linear_state)
            keys = values = None
            scores = attn_out.float().norm(dim=-1)
        else:
            if self.self_attn is None:
                raise RuntimeError(
                    "full-attention block has no attention module"
                )
            attn_out, keys, values, scores = self.self_attn(h, cos_sin, past)
            new_linear_state = None
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, keys, values, scores, new_linear_state

    def forward_static(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cache: StaticCacheState,
        start_pos: int,
        end_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_layernorm(x)
        if self.layer_type == "linear_attention":
            if self.linear_attn is None:
                raise RuntimeError(
                    "linear-attention block has no linear attention module"
                )
            states = cache.linear_states or {}
            attn_out, new_state = self.linear_attn(
                h, states.get(self.layer_idx)
            )
            states[self.layer_idx] = new_state
            cache.linear_states = states
            scores = attn_out.float().norm(dim=-1)
        else:
            if self.self_attn is None:
                raise RuntimeError(
                    "full-attention block has no attention module"
                )
            attn_out, scores = self.self_attn.forward_static(
                h, cos_sin, cache, start_pos, end_pos
            )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, scores
