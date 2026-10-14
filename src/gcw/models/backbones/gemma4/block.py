from __future__ import annotations

import torch
import torch.nn as nn

from gcw.config import ModelConfig
from gcw.models.cache_processors.state import StaticCacheState
from gcw.models.backbones.transformer import Linear
from gcw.models.backbones.gemma4.attention import Gemma4Attention
from gcw.models.backbones.gemma4.norms import Gemma4RMSNorm, _act_fn


class Gemma4MLP(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        first_shared = config.num_layers - config.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_shared > 0
        intermediate = config.intermediate_size * (
            2 if config.use_double_wide_mlp and is_kv_shared_layer else 1
        )
        self.gate_proj = Linear(config.hidden_size, intermediate, bias=False)
        self.up_proj = Linear(config.hidden_size, intermediate, bias=False)
        self.down_proj = Linear(intermediate, config.hidden_size, bias=False)
        self.act = _act_fn(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class Gemma4Block(nn.Module):
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
        self.self_attn = Gemma4Attention(config, layer_idx, cache_idx)
        self.mlp = Gemma4MLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.norm_eps
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.norm_eps
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.norm_eps
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.norm_eps
        )
        self.layer_scalar: torch.Tensor
        self.register_buffer("layer_scalar", torch.ones(1))
        self.hidden_size_per_layer_input = int(
            config.hidden_size_per_layer_input
        )
        if self.hidden_size_per_layer_input:
            self.act = _act_fn(config.hidden_act)
            self.per_layer_input_gate = Linear(
                config.hidden_size, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = Linear(
                self.hidden_size_per_layer_input, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                config.hidden_size, config.norm_eps
            )

    def forward(
        self,
        x: torch.Tensor,
        per_layer_input: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        past: tuple[torch.Tensor, torch.Tensor] | None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor, int]],
        start_pos: int,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor
    ]:
        residual = x
        attn_out, k, v, scores = self.self_attn(
            self.input_layernorm(x),
            cos_sin,
            past,
            shared_kv_states,
            start_pos,
        )
        x = residual + self.post_attention_layernorm(attn_out)
        residual = x
        x = residual + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x))
        )
        if self.hidden_size_per_layer_input:
            if per_layer_input is None:
                raise ValueError("Gemma4 PLE layer requires per_layer_input")
            residual = x
            x = self.act(self.per_layer_input_gate(x)) * per_layer_input
            x = residual + self.post_per_layer_input_norm(
                self.per_layer_projection(x)
            )
        return x * self.layer_scalar.to(x.dtype), k, v, scores

    def forward_static(
        self,
        x: torch.Tensor,
        per_layer_input: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cache: StaticCacheState,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor, int]],
        start_pos: int,
        end_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        attn_out, scores = self.self_attn.forward_static(
            self.input_layernorm(x),
            cos_sin,
            cache,
            shared_kv_states,
            start_pos,
            end_pos,
        )
        x = residual + self.post_attention_layernorm(attn_out)
        residual = x
        x = residual + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x))
        )
        if self.hidden_size_per_layer_input:
            if per_layer_input is None:
                raise ValueError("Gemma4 PLE layer requires per_layer_input")
            residual = x
            x = self.act(self.per_layer_input_gate(x)) * per_layer_input
            x = residual + self.post_per_layer_input_norm(
                self.per_layer_projection(x)
            )
        return x * self.layer_scalar.to(x.dtype), scores
