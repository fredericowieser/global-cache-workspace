from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcw.config import ModelConfig
from gcw.models.cache_state import CacheState, StaticCacheState
from gcw.models.transformer import (
    Linear,
    RotaryCache,
    TF,
    _autocast_dtype,
    _causal_chunk_bias,
    apply_rotary,
    flash_sdpa,
)
from gcw.models.windowed_cache import WindowedCache

try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule as _fla_recurrent_gated_delta_rule,
    )
except Exception:
    _fla_chunk_gated_delta_rule = None
    _fla_recurrent_gated_delta_rule = None

try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn
except Exception:
    _causal_conv1d_fn = None

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule as _hf_qwen35_torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule as _hf_qwen35_torch_recurrent_gated_delta_rule,
    )
except Exception:
    _hf_qwen35_torch_chunk_gated_delta_rule = None
    _hf_qwen35_torch_recurrent_gated_delta_rule = None

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        torch_chunk_gated_delta_rule as _hf_qwen35_moe_torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule as _hf_qwen35_moe_torch_recurrent_gated_delta_rule,
    )
except Exception:
    _hf_qwen35_moe_torch_chunk_gated_delta_rule = None
    _hf_qwen35_moe_torch_recurrent_gated_delta_rule = None

_hf_torch_chunk_gated_delta_rule = _hf_qwen35_torch_chunk_gated_delta_rule or _hf_qwen35_moe_torch_chunk_gated_delta_rule
_hf_torch_recurrent_gated_delta_rule = (
    _hf_qwen35_torch_recurrent_gated_delta_rule or _hf_qwen35_moe_torch_recurrent_gated_delta_rule
)

try:
    from transformers.activations import ACT2FN as _HF_ACT2FN
    from transformers.integrations import use_experts_implementation as _hf_use_experts_implementation
except Exception:
    _HF_ACT2FN = None
    _hf_use_experts_implementation = None


def _has_fla_gated_delta_rule() -> bool:
    return _fla_chunk_gated_delta_rule is not None and _fla_recurrent_gated_delta_rule is not None


def _has_hf_torch_gated_delta_rule() -> bool:
    return _hf_torch_chunk_gated_delta_rule is not None and _hf_torch_recurrent_gated_delta_rule is not None


def qwen3_config(**kwargs) -> ModelConfig:
    if kwargs.get("layer_types") or kwargs.get("sliding_window"):
        raise NotImplementedError("Qwen3 sliding attention is not implemented in gcw yet.")
    kwargs.pop("layer_types", None)
    kwargs.pop("sliding_window", None)
    return ModelConfig(family="qwen3", qk_norm=True, **kwargs)


class Qwen35RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).to(x.dtype)


class Qwen35RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = self.weight.float() * y
        y = y * F.silu(gate.float())
        return y.to(x.dtype)


def _apply_partial_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1] * 2
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = apply_rotary(q_rot, cos, sin, style="llama")
    k_rot = apply_rotary(k_rot, cos, sin, style="llama")
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


class Qwen35Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, cache_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.cache_idx = cache_idx
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = int(config.head_dim)
        self.rotary_style = config.rotary_style
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = Linear(config.hidden_size, q_dim * 2, bias=False)
        self.k_proj = Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = Linear(q_dim, config.hidden_size, bias=False)
        self.q_norm = Qwen35RMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        past: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, _ = x.shape
        q_raw = self.q_proj(x).view(batch, steps, self.num_heads, -1)
        q, gate = q_raw.chunk(2, dim=-1)
        gate = gate.reshape(batch, steps, -1)
        k = self.k_proj(x).view(batch, steps, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, steps, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        rotary = (cos_sin[0].to(q.dtype), cos_sin[1].to(q.dtype))
        q, k = _apply_partial_rotary(q, k, *rotary)
        q, k = q.to(v.dtype), k.to(v.dtype)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        past_len = 0
        if past is not None:
            pk, pv = past
            past_len = pk.shape[2]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        mask, is_causal = _causal_chunk_bias(steps, past_len, x.device)
        y = flash_sdpa(q, k, v, attn_mask=mask, is_causal=is_causal, enable_gqa=self.num_kv_heads != self.num_heads)
        y = y.transpose(1, 2).contiguous().view(batch, steps, self.num_heads * self.head_dim)
        y = y * torch.sigmoid(gate.to(y.dtype))
        attn_out = self.o_proj(y)
        return attn_out, k, v, attn_out.float().norm(dim=-1)

    def forward_static(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cache: StaticCacheState,
        start_pos: int,
        end_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = x.shape
        q_raw = self.q_proj(x).view(batch, steps, self.num_heads, -1)
        q, gate = q_raw.chunk(2, dim=-1)
        gate = gate.reshape(batch, steps, -1)
        k = self.k_proj(x).view(batch, steps, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, steps, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        rotary = (cos_sin[0].to(q.dtype), cos_sin[1].to(q.dtype))
        q, k = _apply_partial_rotary(q, k, *rotary)
        q, k = q.to(v.dtype), k.to(v.dtype)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cache.write_layer(self.cache_idx, start_pos, k, v)
        keys, values = cache.layer_view(self.cache_idx, end_pos)
        mask, is_causal = _causal_chunk_bias(steps, start_pos, x.device)
        y = flash_sdpa(q, keys, values, attn_mask=mask, is_causal=is_causal, enable_gqa=self.num_kv_heads != self.num_heads)
        y = y.transpose(1, 2).contiguous().view(batch, steps, self.num_heads * self.head_dim)
        y = y * torch.sigmoid(gate.to(y.dtype))
        attn_out = self.o_proj(y)
        return attn_out, attn_out.float().norm(dim=-1)


class Qwen35GatedDeltaNet(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        if not _has_fla_gated_delta_rule() and not _has_hf_torch_gated_delta_rule():
            raise ImportError(
                "Qwen3.5 linear attention requires FLA kernels or Transformers' "
                "Qwen3.5 torch fallback functions."
            )
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.hidden_size = config.hidden_size
        self.num_v_heads = int(config.linear_num_value_heads or config.num_heads)
        self.num_k_heads = int(config.linear_num_key_heads or config.num_kv_heads)
        self.head_k_dim = int(config.linear_key_head_dim or config.head_dim)
        self.head_v_dim = int(config.linear_value_head_dim or config.head_dim)
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = int(config.linear_conv_kernel_dim)
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.norm = Qwen35RMSNormGated(self.head_v_dim, config.norm_eps)
        self.out_proj = Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = Linear(self.hidden_size, self.num_v_heads, bias=False)

    def effective_backends(self, device: torch.device) -> dict[str, str]:
        use_fast_gated_delta = device.type == "cuda" and _has_fla_gated_delta_rule()
        return {
            "causal_conv": "causal-conv1d"
            if device.type == "cuda" and _causal_conv1d_fn is not None
            else "torch-conv1d",
            "gated_delta": "fla" if use_fast_gated_delta else "transformers-torch",
        }

    @torch.no_grad()
    def init_state_weights(self) -> None:
        nn.init.ones_(self.dt_bias)
        self.A_log.copy_(torch.empty_like(self.A_log).uniform_(1e-4, 16).log_())

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_state: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).view(batch, seq_len, -1, self.head_v_dim)
        beta = torch.sigmoid(self.in_proj_b(hidden_states))
        a = self.in_proj_a(hidden_states)
        conv_state = past_state[0] if past_state is not None else None
        recurrent_state = past_state[1] if past_state is not None else None

        conv_out, new_conv_state = self._causal_conv(mixed_qkv, conv_state)
        query, key, value = torch.split(conv_out.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)
        core_attn_out, new_recurrent_state = self._gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
        )
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z).reshape(batch, seq_len, -1)
        return self.out_proj(core_attn_out), (new_conv_state, new_recurrent_state)

    def _causal_conv(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_len = self.conv_kernel_size - 1
        if conv_state is not None:
            mixed = torch.cat([conv_state.to(mixed_qkv.dtype), mixed_qkv], dim=-1)
        else:
            mixed = mixed_qkv
        if mixed_qkv.device.type == "cuda" and _causal_conv1d_fn is not None:
            out = _causal_conv1d_fn(
                x=mixed,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        else:
            out = self.conv1d(mixed)[:, :, : mixed.shape[-1]]
            out = _HF_ACT2FN[self.activation](out) if _HF_ACT2FN is not None else F.silu(out)
        out = out[:, :, -mixed_qkv.shape[-1] :]
        if state_len > 0:
            new_state = mixed[:, :, -state_len:].detach() if not mixed.requires_grad else mixed[:, :, -state_len:]
        else:
            new_state = mixed[:, :, :0]
        return out.to(mixed_qkv.dtype), new_state

    def _gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_fla = query.device.type == "cuda" and _has_fla_gated_delta_rule()
        if not use_fla and not _has_hf_torch_gated_delta_rule():
            raise RuntimeError("CPU Qwen3.5 gated-delta execution requires Transformers' torch fallback.")
        needs_grad = torch.is_grad_enabled() and any(
            t is not None and t.requires_grad for t in (query, key, value, g, beta, initial_state)
        )
        if query.shape[1] == 1 and not needs_grad:
            recurrent_impl = _fla_recurrent_gated_delta_rule if use_fla else _hf_torch_recurrent_gated_delta_rule
            return recurrent_impl(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        chunk_impl = _fla_chunk_gated_delta_rule if use_fla else _hf_torch_chunk_gated_delta_rule
        return chunk_impl(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )


class Qwen35MLP(nn.Module):
    def __init__(self, config: ModelConfig, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, cache_idx: int | None):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx] if config.layer_types is not None else "full_attention"
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, config.norm_eps)
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(config, layer_idx)
            self.self_attn = None
        else:
            if cache_idx is None:
                raise ValueError("full_attention layers require a cache index")
            self.self_attn = Qwen35Attention(config, layer_idx, cache_idx)
            self.linear_attn = None
        self.post_attention_layernorm = Qwen35RMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = Qwen35MLP(config, config.intermediate_size)

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
            attn_out, new_linear_state = self.linear_attn(h, linear_state)
            keys = values = None
            scores = attn_out.float().norm(dim=-1)
        else:
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
            states = cache.linear_states or {}
            attn_out, new_state = self.linear_attn(h, states.get(self.layer_idx))
            states[self.layer_idx] = new_state
            cache.linear_states = states
            scores = attn_out.float().norm(dim=-1)
        else:
            attn_out, scores = self.self_attn.forward_static(h, cos_sin, cache, start_pos, end_pos)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, scores


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

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
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
) -> torch.Tensor | int:
    if not router_logits:
        return 0
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
        tokens_per_expert = (expert_mask.float() * expert_attention_mask).sum(dim=0) / expert_attention_mask.sum(dim=0).clamp_min(1)
        router_prob_per_expert = (routing_weights * router_attention_mask).sum(dim=0) / router_attention_mask.sum(dim=0).clamp_min(1)
    return torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) * num_experts


class Qwen35MoeBlock(Qwen35Block):
    def __init__(self, config: ModelConfig, layer_idx: int, cache_idx: int | None):
        super().__init__(config, layer_idx, cache_idx)
        self.mlp = Qwen35MoeSparseMoeBlock(config)

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
        torch.Tensor,
    ]:
        h = self.input_layernorm(x)
        if self.layer_type == "linear_attention":
            attn_out, new_linear_state = self.linear_attn(h, linear_state)
            keys = values = None
            scores = attn_out.float().norm(dim=-1)
        else:
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
            states = cache.linear_states or {}
            attn_out, new_state = self.linear_attn(h, states.get(self.layer_idx))
            states[self.layer_idx] = new_state
            cache.linear_states = states
            scores = attn_out.float().norm(dim=-1)
        else:
            attn_out, scores = self.self_attn.forward_static(h, cos_sin, cache, start_pos, end_pos)
        x = x + attn_out
        mlp_out, _ = self.mlp(self.post_attention_layernorm(x))
        x = x + mlp_out
        return x, scores


class Qwen35TF(TF):
    block_cls = Qwen35Block

    def __init__(self, config: ModelConfig, pad_vocab_size_to: int = 128):
        nn.Module.__init__(self)
        self.config = replace(config, variant="tf")
        vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.padded_vocab_size = vocab
        self.embed_tokens = nn.Embedding(vocab, config.hidden_size)
        cache_index_by_layer = {layer: idx for idx, layer in enumerate(config.cache_layer_indices)}
        self.layers = nn.ModuleList(self.block_cls(config, i, cache_index_by_layer.get(i)) for i in range(config.num_layers))
        self.norm = Qwen35RMSNorm(config.hidden_size, config.norm_eps)
        rotary_dim = int(int(config.head_dim) * config.partial_rotary_factor)
        self.rotary = RotaryCache(config.sequence_len, rotary_dim, config.rope_theta)
        self.lm_head = None if config.tie_embeddings else Linear(config.hidden_size, vocab, bias=False)

    @torch.no_grad()
    def init_weights(self, std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, 0.0, std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0.0, std)
            elif isinstance(module, Qwen35MoeExperts):
                nn.init.normal_(module.gate_up_proj, 0.0, std)
                nn.init.normal_(module.down_proj, 0.0, std)
            elif isinstance(module, Qwen35RMSNorm):
                nn.init.zeros_(module.weight)
            elif isinstance(module, Qwen35RMSNormGated):
                nn.init.ones_(module.weight)
            elif isinstance(module, Qwen35GatedDeltaNet):
                module.init_state_weights()

    def forward_full(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_reduction: str = "mean",
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        cos_sin = self.rotary.get_range(0, input_ids.shape[1], input_ids.device, _autocast_dtype(x.device) or x.dtype)
        for layer in self.layers:
            x, _, _, _, _ = layer(x, cos_sin, None, None)
        x = self.norm(x)
        if targets is not None:
            return self.loss_from_hidden(x, targets, loss_reduction)
        return self.logits_from_hidden(x)

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        cache: CacheState | WindowedCache | None = None,
        *,
        logits: bool = True,
        cache_format: str = "list",
    ) -> tuple[torch.Tensor, CacheState | WindowedCache]:
        x = self.embed_tokens(input_ids)
        cos_sin = self.rotary.get_range(
            start_pos,
            input_ids.shape[1],
            input_ids.device,
            _autocast_dtype(x.device) or x.dtype,
        )
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        layer_scores: list[torch.Tensor] = []
        old_scores = input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32) if cache is None else cache.scores
        has_cache = cache is not None and cache.scores.numel() > 0
        linear_states = {} if cache is None or cache.linear_states is None else dict(cache.linear_states)
        cache_idx = 0
        for layer in self.layers:
            if layer.layer_type == "full_attention":
                past = None if not has_cache else cache.layer(cache_idx)
                x, k, v, scores, _ = layer(x, cos_sin, past, None)
                keys.append(k)
                values.append(v)
                cache_idx += 1
            else:
                x, _, _, scores, new_state = layer(x, cos_sin, None, linear_states.get(layer.layer_idx))
                linear_states[layer.layer_idx] = new_state
            layer_scores.append(scores)
        x = self.norm(x)
        out = self.logits_from_hidden(x) if logits else x
        chunk_scores = torch.stack(layer_scores).mean(dim=0)
        new_scores = chunk_scores if old_scores.numel() == 0 else torch.cat([old_scores, chunk_scores], dim=1)
        if cache_format == "windowed" or isinstance(cache, WindowedCache):
            return out, WindowedCache.from_lists(keys, values, new_scores, linear_states)
        return out, CacheState(keys, values, new_scores, linear_states)

    def forward_chunk_static(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        cache: StaticCacheState,
        *,
        logits: bool = True,
    ) -> tuple[torch.Tensor, StaticCacheState]:
        x = self.embed_tokens(input_ids)
        end_pos = start_pos + input_ids.shape[1]
        pos = torch.arange(start_pos, end_pos, device=input_ids.device)
        cos_sin = self.rotary.get(pos, x.dtype)
        layer_scores: list[torch.Tensor] = []
        for layer in self.layers:
            x, scores = layer.forward_static(x, cos_sin, cache, start_pos, end_pos)
            layer_scores.append(scores)
        x = self.norm(x)
        out = self.logits_from_hidden(x) if logits else x
        cache.write_scores(start_pos, torch.stack(layer_scores).mean(dim=0))
        return out, cache


class Qwen35MoeTF(Qwen35TF):
    block_cls = Qwen35MoeBlock

    def forward_full(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_reduction: str = "mean",
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        cos_sin = self.rotary.get_range(0, input_ids.shape[1], input_ids.device, _autocast_dtype(x.device) or x.dtype)
        router_logits = []
        for layer in self.layers:
            x, _, _, _, _, layer_router_logits = layer(x, cos_sin, None, None)
            router_logits.append(layer_router_logits)
        x = self.norm(x)
        if targets is not None:
            loss = self.loss_from_hidden(x, targets, loss_reduction)
            if loss_reduction == "mean" and self.config.router_aux_loss_coef:
                aux_loss = qwen35_moe_load_balancing_loss(
                    tuple(router_logits),
                    num_experts=self.config.num_experts,
                    top_k=self.config.num_experts_per_tok,
                    attention_mask=attention_mask,
                )
                loss = loss + self.config.router_aux_loss_coef * aux_loss.to(loss.device)
            return loss
        return self.logits_from_hidden(x)

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        cache: CacheState | WindowedCache | None = None,
        *,
        logits: bool = True,
        cache_format: str = "list",
    ) -> tuple[torch.Tensor, CacheState | WindowedCache]:
        x = self.embed_tokens(input_ids)
        cos_sin = self.rotary.get_range(
            start_pos,
            input_ids.shape[1],
            input_ids.device,
            _autocast_dtype(x.device) or x.dtype,
        )
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        layer_scores: list[torch.Tensor] = []
        old_scores = input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32) if cache is None else cache.scores
        has_cache = cache is not None and cache.scores.numel() > 0
        linear_states = {} if cache is None or cache.linear_states is None else dict(cache.linear_states)
        cache_idx = 0
        for layer in self.layers:
            if layer.layer_type == "full_attention":
                past = None if not has_cache else cache.layer(cache_idx)
                x, k, v, scores, _, _ = layer(x, cos_sin, past, None)
                keys.append(k)
                values.append(v)
                cache_idx += 1
            else:
                x, _, _, scores, new_state, _ = layer(x, cos_sin, None, linear_states.get(layer.layer_idx))
                linear_states[layer.layer_idx] = new_state
            layer_scores.append(scores)
        x = self.norm(x)
        out = self.logits_from_hidden(x) if logits else x
        chunk_scores = torch.stack(layer_scores).mean(dim=0)
        new_scores = chunk_scores if old_scores.numel() == 0 else torch.cat([old_scores, chunk_scores], dim=1)
        if cache_format == "windowed" or isinstance(cache, WindowedCache):
            return out, WindowedCache.from_lists(keys, values, new_scores, linear_states)
        return out, CacheState(keys, values, new_scores, linear_states)


Qwen35MoeRMSNorm = Qwen35RMSNorm
Qwen35MoeGatedDeltaNet = Qwen35GatedDeltaNet
Qwen3TF = TF
