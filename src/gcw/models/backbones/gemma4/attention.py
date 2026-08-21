from __future__ import annotations

import torch
import torch.nn as nn

from gcw.config import ModelConfig
from gcw.models.cache_processors.state import StaticCacheState
from gcw.models.backbones.transformer import (
    Linear,
    _causal_chunk_bias,
)
import gcw.models.backbones.gemma4 as gemma4_pkg
from gcw.models.backbones.gemma4.norms import Gemma4RMSNorm
from gcw.models.backbones.gemma4.rotary import _apply_rotary_pos_emb


def _sliding_attention_mask(
    *,
    steps: int,
    total: int,
    start_pos: int,
    sliding_window: int | None,
    device: torch.device,
) -> torch.Tensor | None:
    if sliding_window is None:
        raise ValueError("sliding_attention requires sliding_window")
    q_pos = torch.arange(start_pos, start_pos + steps, device=device)[:, None]
    kv_pos = torch.arange(total, device=device)[None, :]
    keep = (kv_pos <= q_pos) & ((q_pos - kv_pos) < sliding_window)
    if bool(keep.all()):
        return None
    mask = torch.zeros(steps, total, dtype=torch.float32, device=device)
    return mask.masked_fill(~keep, torch.finfo(torch.float32).min)[
        None, None, :, :
    ]


def _gemma4_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    is_causal: bool,
    groups: int,
    scale: float,
    try_flash: bool,
) -> torch.Tensor:
    y = gemma4_pkg.flash_sdpa(
        q,
        k,
        v,
        attn_mask=mask,
        is_causal=is_causal,
        enable_gqa=groups != 1,
        scale=scale,
        try_flash=try_flash,
    )
    return y.transpose(1, 2).contiguous()


class Gemma4Attention(nn.Module):
    def __init__(
        self, config: ModelConfig, layer_idx: int, cache_idx: int | None
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.cache_idx = cache_idx
        self.layer_type = (
            config.layer_types[layer_idx]
            if config.layer_types is not None
            else "full_attention"
        )
        self.is_sliding = self.layer_type == "sliding_attention"
        first_shared = config.num_layers - config.num_kv_shared_layers
        self.is_kv_shared_layer = layer_idx >= first_shared
        prev_layers = (config.layer_types or [])[:first_shared]
        self.store_full_length_kv = False
        if not self.is_kv_shared_layer and self.layer_type in prev_layers:
            self.store_full_length_kv = layer_idx == len(
                prev_layers
            ) - 1 - prev_layers[::-1].index(self.layer_type)
        self.num_heads = config.num_heads
        self.num_kv_heads = config.kv_heads_for_layer(layer_idx)
        self.head_dim = config.head_dim_for_layer(layer_idx)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = 1.0
        self.use_alternative_attention = (
            config.attention_k_eq_v and not self.is_sliding
        )
        self.q_proj = Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.q_norm = Gemma4RMSNorm(self.head_dim, config.norm_eps)
        if not self.is_kv_shared_layer:
            self.k_proj = Linear(
                config.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=False,
            )
            self.v_proj = (
                None
                if self.use_alternative_attention
                else Linear(
                    config.hidden_size,
                    self.num_kv_heads * self.head_dim,
                    bias=False,
                )
            )
            self.k_norm = Gemma4RMSNorm(self.head_dim, config.norm_eps)
            self.v_norm = Gemma4RMSNorm(
                self.head_dim, config.norm_eps, with_scale=False
            )
        self.o_proj = Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        past: tuple[torch.Tensor, torch.Tensor] | None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        start_pos: int,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor
    ]:
        batch, steps, _ = x.shape
        q = self.q_proj(x).view(batch, steps, self.num_heads, self.head_dim)
        q = _apply_rotary_pos_emb(self.q_norm(q), *cos_sin).transpose(1, 2)
        if self.is_kv_shared_layer:
            k, v = shared_kv_states[self.layer_type]
        else:
            k = self.k_proj(x).view(
                batch, steps, self.num_kv_heads, self.head_dim
            )
            v = (
                k
                if self.v_proj is None
                else self.v_proj(x).view(
                    batch, steps, self.num_kv_heads, self.head_dim
                )
            )
            k = _apply_rotary_pos_emb(self.k_norm(k), *cos_sin).transpose(1, 2)
            v = self.v_norm(v).transpose(1, 2)
            if past is not None:
                pk, pv = past
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            if self.store_full_length_kv:
                shared_kv_states[self.layer_type] = (k, v)
        mask, is_causal, try_flash = self._attention_bias(
            steps, k.shape[2], start_pos, x.device
        )
        y = _gemma4_attention(
            q,
            k,
            v,
            mask,
            is_causal,
            self.num_key_value_groups,
            self.scaling,
            try_flash,
        )
        y = y.reshape(batch, steps, self.num_heads * self.head_dim)
        out = self.o_proj(y)
        return (
            out,
            None if self.is_kv_shared_layer else k,
            None if self.is_kv_shared_layer else v,
            out.float().norm(dim=-1),
        )

    def forward_static(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cache: StaticCacheState,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        start_pos: int,
        end_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = x.shape
        q = self.q_proj(x).view(batch, steps, self.num_heads, self.head_dim)
        q = _apply_rotary_pos_emb(self.q_norm(q), *cos_sin).transpose(1, 2)
        if self.is_kv_shared_layer:
            k, v = shared_kv_states[self.layer_type]
        else:
            if self.cache_idx is None:
                raise ValueError(
                    "non-shared Gemma4 attention layers require a cache index"
                )
            current_k = self.k_proj(x).view(
                batch, steps, self.num_kv_heads, self.head_dim
            )
            current_v = (
                current_k
                if self.v_proj is None
                else self.v_proj(x).view(
                    batch, steps, self.num_kv_heads, self.head_dim
                )
            )
            current_k = _apply_rotary_pos_emb(
                self.k_norm(current_k), *cos_sin
            ).transpose(1, 2)
            current_v = self.v_norm(current_v).transpose(1, 2)
            cache.write_layer(self.cache_idx, start_pos, current_k, current_v)
            k, v = cache.layer_view(self.cache_idx, end_pos)
            if self.store_full_length_kv:
                shared_kv_states[self.layer_type] = (k, v)
        mask, is_causal, try_flash = self._attention_bias(
            steps, k.shape[2], start_pos, x.device
        )
        y = _gemma4_attention(
            q,
            k,
            v,
            mask,
            is_causal,
            self.num_key_value_groups,
            self.scaling,
            try_flash,
        )
        y = y.reshape(batch, steps, self.num_heads * self.head_dim)
        out = self.o_proj(y)
        return out, out.float().norm(dim=-1)

    def _attention_bias(
        self,
        steps: int,
        total: int,
        start_pos: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, bool, bool]:
        if not self.is_sliding:
            mask, is_causal = _causal_chunk_bias(steps, total - steps, device)
            return mask, is_causal, True
        mask = _sliding_attention_mask(
            steps=steps,
            total=total,
            start_pos=start_pos,
            sliding_window=self.config.sliding_window,
            device=device,
        )
        return mask, False, mask is None
