from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcw.config import ModelConfig
from gcw.models.cache_processors.state import StaticCacheState
from gcw.models.backbones.transformer import Linear
from gcw.models.backbones.qwen3.gated_delta import (
    _HF_ACT2FN,
    _hf_use_experts_implementation,
)
from gcw.models.backbones.qwen3.blocks import Qwen35Block, Qwen35MLP


class _Qwen35MoeExperts(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if _hf_use_experts_implementation is None or _HF_ACT2FN is None:
            raise ImportError(
                "Qwen3.5-MoE expert dispatch requires Transformers' ExpertsInterface. "
                "Install a Transformers version with `transformers.integrations.use_experts_implementation`."
            )
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = int(config.moe_intermediate_size or config.intermediate_size)
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = _HF_ACT2FN[config.hidden_act]

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        raise RuntimeError("Qwen35MoeExperts must be run through Transformers' ExpertsInterface backend.")


if _hf_use_experts_implementation is not None:
    Qwen35MoeExperts = _hf_use_experts_implementation(_Qwen35MoeExperts)
else:
    Qwen35MoeExperts = _Qwen35MoeExperts


class Qwen35MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.gate = Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen35MoeExperts(config)
        self.shared_expert = Qwen35MLP(config, int(config.shared_expert_intermediate_size or config.intermediate_size))
        self.shared_expert_gate = Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate(flat)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        routing_weights, selected_experts = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat.dtype)
        expert_output = self.experts(flat, selected_experts, routing_weights)
        shared = torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        return (expert_output + shared).reshape(batch, seq_len, hidden_dim), router_logits


def qwen35_moe_load_balancing_loss(
    router_logits: tuple[torch.Tensor, ...] | list[torch.Tensor] | None,
    *,
    num_experts: int,
    top_k: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not router_logits:
        raise ValueError("router_logits must contain at least one layer")
    compute_device = router_logits[0].device
    gate_logits = torch.cat([logits.to(compute_device) for logits in router_logits], dim=0)
    routing_weights = F.softmax(gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = F.one_hot(selected_experts, num_experts)
    if attention_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = gate_logits.shape[0] // (batch_size * sequence_length)
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )
        router_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )
        tokens_per_expert = (expert_mask.float() * expert_attention_mask).sum(dim=0) / expert_attention_mask.sum(
            dim=0
        ).clamp_min(1)
        router_prob_per_expert = (routing_weights * router_attention_mask).sum(dim=0) / router_attention_mask.sum(
            dim=0
        ).clamp_min(1)
    return torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) * num_experts


class Qwen35MoeBlock(Qwen35Block):
    def __init__(self, config: ModelConfig, layer_idx: int, cache_idx: int | None):
        super().__init__(config, layer_idx, cache_idx)
        self.mlp = Qwen35MoeSparseMoeBlock(config)

    def forward(  # type: ignore[override]
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
        torch.Tensor,
    ]:
        h = self.input_layernorm(x)
        if self.layer_type == "linear_attention":
            if self.linear_attn is None:
                raise RuntimeError("linear-attention block has no linear attention module")
            attn_out, new_linear_state = self.linear_attn(h, linear_state)
            keys = values = None
            scores = attn_out.float().norm(dim=-1)
        else:
            if self.self_attn is None:
                raise RuntimeError("full-attention block has no attention module")
            attn_out, keys, values, scores = self.self_attn(h, cos_sin, past)
            new_linear_state = None
        x = x + attn_out
        mlp_out, router_logits = self.mlp(self.post_attention_layernorm(x))
        x = x + mlp_out
        return x, keys, values, scores, new_linear_state, router_logits

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
                raise RuntimeError("linear-attention block has no linear attention module")
            states = cache.linear_states or {}
            attn_out, new_state = self.linear_attn(h, states.get(self.layer_idx))
            states[self.layer_idx] = new_state
            cache.linear_states = states
            scores = attn_out.float().norm(dim=-1)
        else:
            if self.self_attn is None:
                raise RuntimeError("full-attention block has no attention module")
            attn_out, scores = self.self_attn.forward_static(h, cos_sin, cache, start_pos, end_pos)
        x = x + attn_out
        mlp_out, _ = self.mlp(self.post_attention_layernorm(x))
        x = x + mlp_out
        return x, scores
