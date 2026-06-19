from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models.cache_state import CacheState
from gcw.models.transformer import RMSNorm, apply_rotary, flash_sdpa


def rotary_for_indices(indices: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=indices.device, dtype=torch.float32) / head_dim))
    freqs = indices.float().unsqueeze(-1) * inv
    return freqs.cos().unsqueeze(2).to(dtype), freqs.sin().unsqueeze(2).to(dtype)


class ProcessorAttention(nn.Module):
    def __init__(self, config: CacheProcessorConfig):
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("processor hidden_size must be divisible by num_heads")
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError("processor num_heads must be divisible by num_kv_heads")
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.attention_dropout
        # TODO: P2 this is currently single head per kv cache head, but we could have multiple heads per kv head if we wanted to
        # TODO: P1 currently a single shared projection for all kv cache heads, but we could have separate projections for each kv head if we wanted to - not sure how to with nn.Linear
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, theta: float, selected_heads: torch.Tensor) -> torch.Tensor:
        batch, cache_heads, seq, _ = x.shape
        if selected_heads.shape[-1] == 0:
            return torch.zeros_like(x)
        q = self.q_proj(x).view(batch, cache_heads, seq, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, cache_heads, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, cache_heads, seq, self.num_kv_heads, self.head_dim)
        cos, sin = rotary_for_indices(positions, self.head_dim, theta, q.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        k, v = (_gather_cache_heads(t, selected_heads) for t in (k, v))
        q = q.permute(0, 3, 1, 2, 4).reshape(batch, self.num_heads, cache_heads * seq, self.head_dim)
        k, v = (
            t.permute(0, 3, 1, 2, 4).reshape(batch, self.num_kv_heads, selected_heads.shape[-1] * seq, self.head_dim)
            for t in (k, v)
        )
        q, k = q.to(v.dtype), k.to(v.dtype)
        y = flash_sdpa(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        y = y.view(batch, self.num_heads, cache_heads, seq, self.head_dim)
        y = y.permute(0, 2, 3, 1, 4).contiguous().view(batch, cache_heads, seq, -1)
        return self.o_proj(y)


class ProcessorLayer(nn.Module):
    def __init__(self, config: CacheProcessorConfig):
        super().__init__()
        self.attn = ProcessorAttention(config)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.input_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.theta = config.rope_theta

    def forward(self, x: torch.Tensor, positions: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), positions, self.theta, selected_heads)
        h = self.post_norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class CacheProcessor(nn.Module):
    def __init__(self, input_dim: int, config: CacheProcessorConfig):
        super().__init__()
        self.config = config
        self.in_norm = RMSNorm(input_dim, config.norm_eps)
        self.in_proj = nn.Linear(input_dim, config.hidden_size)
        self.layers = nn.ModuleList(ProcessorLayer(config) for _ in range(config.num_layers))
        self.out_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.out_proj = nn.Linear(config.hidden_size, input_dim)
        self.gate = nn.Parameter(torch.full((1,), config.gate_init))
        self.init_weights()

    def init_weights(self, std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        nn.init.constant_(self.gate, self.config.gate_init)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.in_proj(self.in_norm(x))
        for layer in self.layers:
            h = layer(h, positions, selected_heads)
        return residual + self.out_proj(self.out_norm(h)) * torch.sigmoid(self.gate)


class GlobalCacheProcessor(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        if model_config.cache_processor is None:
            raise ValueError("BGT requires cache_processor config")
        self.config = model_config.cache_processor
        self.num_layers = model_config.num_layers
        self.num_kv_heads = model_config.num_kv_heads
        self.head_dim = int(model_config.head_dim)
        self.block = CacheProcessor(2 * self.head_dim, self.config)

    def forward(self, cache: CacheState, *, select_all: bool = False) -> CacheState:
        indices = cache.all_indices() if select_all else cache.select_indices(self.config.process_every, self.config.topk)
        parts = []
        shapes = []
        # TODO: P3 vectorise this across layers to avoid python for loop
        for layer_idx in range(self.num_layers):
            keys, values = cache.gather_layer(layer_idx, indices)
            # TODO: P2 look to unrope keys
            shapes.append((keys.shape[1], keys.shape[3]))
            parts.append(_merge_kv(keys, values))
        merged = torch.cat(parts, dim=1)
        selected_heads = cache.all_heads() if select_all else _select_heads_from_merged(merged, self.config.topk_heads)
        updated = self.block(merged, indices, selected_heads)  # concatenate along head dimension to process all layers at once
        for layer_idx, part in enumerate(torch.split(updated, [p.shape[1] for p in parts], dim=1)):  # TODO: P1 check this split is correct, should be along head dimension
            num_heads, head_dim = shapes[layer_idx]
            keys, values = _split_kv(part, num_heads, head_dim)
            cache.scatter_layer(layer_idx, indices, keys, values)
        return cache


def _merge_kv(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return torch.cat([keys, values], dim=-1)  # [batch, heads, seq, 2 * dim]


def _gather_cache_heads(x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
    if selected_heads.ndim != 2:
        raise ValueError("selected_heads must have shape (batch, selected_heads)")
    index = selected_heads[:, :, None, None, None].expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
    return torch.gather(x, dim=1, index=index)


def _select_heads_from_merged(x: torch.Tensor, topk: int) -> torch.Tensor:
    batch = x.shape[0]
    if topk <= 0:
        return torch.empty(batch, 0, dtype=torch.long, device=x.device)
    scores = x.float().flatten(start_dim=2).norm(dim=-1)
    k = min(topk, scores.shape[-1])
    _, indices = torch.topk(scores, k=k, dim=-1)
    return indices.sort(dim=-1).values


def _split_kv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    half = x.shape[-1] // 2
    keys, values = x[..., :half], x[..., half:]
    return keys, values  # [batch, heads, seq, dim], [batch, heads, seq, dim]
