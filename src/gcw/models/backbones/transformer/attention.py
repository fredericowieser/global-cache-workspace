from __future__ import annotations

import torch
import torch.nn as nn

from gcw.config import ModelConfig
from gcw.models.cache_processors.state import StaticCacheState
import gcw.models.backbones.transformer as tf_pkg
from gcw.models.backbones.transformer.layers import (
    Linear,
    RMSNorm,
    apply_rotary,
    _causal_chunk_bias,
)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        if config.head_dim is None:
            raise ValueError("attention requires a resolved model head_dim")
        self.head_dim = config.head_dim
        self.rotary_style = config.rotary_style
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = Linear(config.hidden_size, q_dim, bias=False)
        self.k_proj = Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = Linear(q_dim, config.hidden_size, bias=False)
        self.q_norm = (
            RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        )
        self.k_norm = (
            RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        )

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        past: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, _ = x.shape
        q = self.q_proj(x).view(batch, steps, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, steps, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, steps, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        rotary = (cos_sin[0].to(q.dtype), cos_sin[1].to(q.dtype))
        q, k = apply_rotary(q, *rotary, style=self.rotary_style), apply_rotary(
            k, *rotary, style=self.rotary_style
        )
        q, k = q.to(v.dtype), k.to(v.dtype)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        past_len = 0
        if past is not None:
            pk, pv = past
            past_len = pk.shape[2]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        mask, is_causal = _causal_chunk_bias(steps, past_len, x.device)
        y = tf_pkg.flash_sdpa(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=is_causal,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(batch, steps, self.num_heads * self.head_dim)
        )
        attn_out = self.o_proj(y)
        return attn_out, k, v, attn_out.float().norm(dim=-1)

    def forward_tensor_past(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attention path with tensor cache inputs and an externally built causal bias."""
        batch, steps, _ = x.shape
        q = self.q_proj(x).view(batch, steps, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, steps, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, steps, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        rotary = (cos_sin[0].to(q.dtype), cos_sin[1].to(q.dtype))
        q = apply_rotary(q, *rotary, style=self.rotary_style)
        k = apply_rotary(k, *rotary, style=self.rotary_style)
        q, k = q.to(v.dtype), k.to(v.dtype)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        k = torch.cat([past_keys, k], dim=2)
        v = torch.cat([past_values, v], dim=2)
        y = tf_pkg.flash_sdpa(
            q,
            k,
            v,
            attn_mask=attention_bias,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(batch, steps, self.num_heads * self.head_dim)
        )
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
        q = self.q_proj(x).view(batch, steps, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, steps, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, steps, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        rotary = (cos_sin[0].to(q.dtype), cos_sin[1].to(q.dtype))
        q, k = apply_rotary(q, *rotary, style=self.rotary_style), apply_rotary(
            k, *rotary, style=self.rotary_style
        )
        q, k = q.to(v.dtype), k.to(v.dtype)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        cache.write_layer(self.layer_idx, start_pos, k, v)
        keys, values = cache.layer_view(self.layer_idx, end_pos)
        mask, is_causal = _causal_chunk_bias(steps, start_pos, x.device)
        y = tf_pkg.flash_sdpa(
            q,
            keys,
            values,
            attn_mask=mask,
            is_causal=is_causal,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(batch, steps, self.num_heads * self.head_dim)
        )
        attn_out = self.o_proj(y)
        return attn_out, attn_out.float().norm(dim=-1)
