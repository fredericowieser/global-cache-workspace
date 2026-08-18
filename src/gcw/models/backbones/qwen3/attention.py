from __future__ import annotations

import torch
import torch.nn as nn

from gcw.config import ModelConfig
from gcw.models.cache_processors.state import StaticCacheState
from gcw.models.backbones.transformer import (
    Linear,
    _causal_chunk_bias,
    apply_rotary,
    flash_sdpa,
)
from gcw.models.backbones.qwen3.norms import Qwen35RMSNorm


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
        self.head_dim = config.head_dim_for_layer(layer_idx)
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
        y = flash_sdpa(
            q, keys, values, attn_mask=mask, is_causal=is_causal, enable_gqa=self.num_kv_heads != self.num_heads
        )
        y = y.transpose(1, 2).contiguous().view(batch, steps, self.num_heads * self.head_dim)
        y = y * torch.sigmoid(gate.to(y.dtype))
        attn_out = self.o_proj(y)
        return attn_out, attn_out.float().norm(dim=-1)
