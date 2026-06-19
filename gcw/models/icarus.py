from __future__ import annotations
from linecache import cache

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
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.attention_dropout
        # TODO: P2 this is currently single head per kv cache head, but we could have multiple heads per kv head if we wanted to
        # TODO: P1 currently a single shared projection for all kv cache heads, but we could have separate projections for each kv head if we wanted to - not sure how to with nn.Linear
        self.q_proj = nn.Linear(config.hidden_size, self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, theta: float, selected_heads: torch.Tensor) -> torch.Tensor:
        batch, seq, heads, head_dim = x.shape
        num_selected_heads = selected_heads.shape[-1]  # [batch, num_selected_heads]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x).transpose(1, 2)
        cos_sin = rotary_for_indices(positions, head_dim, theta, q.dtype)
        q = apply_rotary(q, *cos_sin).transpose(1, 2)
        k = apply_rotary(k, *cos_sin).transpose(1, 2)
        k, v = (t[..., selected_heads, :] for t in (k, v))  # select only the heads we want to take values from - TODO: P1 check slicing is correct
        q = q.view(batch, seq * heads, head_dim)  # flatten heads into seq dimension for attention across all heads
        k, v = (t.view(batch, seq * num_selected_heads, head_dim) for t in (k, v))
        y = flash_sdpa(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        y = y.view(batch, seq, heads, head_dim)  # unflatten heads back into separate dimension
        return self.o_proj(y.transpose(1, 2).contiguous().view(batch, seq, -1))


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
        per_layer = 2 * model_config.num_kv_heads * int(model_config.head_dim)
        self.num_layers = model_config.num_layers
        self.num_kv_heads = model_config.num_kv_heads
        self.head_dim = int(model_config.head_dim)
        self.block = CacheProcessor(self.head_dim, self.config)

    def forward(self, cache: CacheState, *, select_all: bool = False) -> CacheState:
        if select_all:
            indices = cache.all_indices()
            selected_heads = cache.all_heads()  # TODO: P1 this needs implementing
        else:
            indices = cache.select_indices(self.config.process_every, self.config.topk)
            selected_heads = cache.select_heads(self.config.process_every, self.config.topk_heads)  # TODO: P1 this needs implementing
        parts = []
        shapes = []
        # TODO: P3 vectorise this across layers to avoid python for loop
        for layer_idx in range(self.num_layers):
            keys, values = cache.gather_layer(layer_idx, indices)
            # TODO: P2 look to unrope keys
            shapes.append((keys.shape[1], keys.shape[3]))
            parts.append(_merge_kv(keys, values))
        updated = self.block(torch.cat(parts, dim=1), indices, selected_heads)  # concatenate along head dimension to process all layers at once
        for layer_idx, part in enumerate(torch.split(updated, [p.shape[1] for p in parts], dim=1)):  # TODO: P1 check this split is correct, should be along head dimension
            num_heads, head_dim = shapes[layer_idx]
            keys, values = _split_kv(part, num_heads, head_dim)
            cache.scatter_layer(layer_idx, indices, keys, values)
        return cache


def _merge_kv(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return torch.cat([keys, values], dim=-1)  # [batch, heads, seq, 2 * dim]


def _split_kv(x: torch.Tensor, num_heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    half = x.shape[-1] // 2
    keys, values = x[..., :half], x[..., half:]
    return keys, values  # [batch, heads, seq, dim], [batch, heads, seq, dim]