from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models.cache_state import CacheState
from gcw.models.cache_processor_ops import (
    gather_cache_layers,
    merge_global_kv,
    merge_layer_kv,
    rotary_for_indices,
    run_batched_cache_processors,
    scatter_cache_layers,
    split_global_kv,
    split_layer_kv,
)
from gcw.models.transformer import RMSNorm, apply_rotary, flash_sdpa


class ProcessorAttention(nn.Module):
    def __init__(self, config: CacheProcessorConfig):
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("processor hidden_size must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos_sin = rotary_for_indices(positions, self.head_dim, theta, q.dtype)
        q = apply_rotary(q, *cos_sin).transpose(1, 2)
        k = apply_rotary(k, *cos_sin).transpose(1, 2)
        q, k = q.to(v.dtype), k.to(v.dtype)
        y = flash_sdpa(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
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

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), positions, self.theta)
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

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.in_proj(self.in_norm(x))
        for layer in self.layers:
            h = layer(h, positions)
        return residual + self.out_proj(self.out_norm(h)) * torch.sigmoid(self.gate)


class PerLayerCacheProcessor(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        if model_config.cache_processor is None:
            raise ValueError("BT requires cache_processor config")
        self.config = model_config.cache_processor
        layer_dim = 2 * model_config.num_kv_heads * int(model_config.head_dim)
        self.blocks = nn.ModuleList(CacheProcessor(layer_dim, self.config) for _ in range(model_config.num_layers))

    def forward(self, cache: CacheState, *, select_all: bool = False) -> CacheState:
        indices = cache.all_indices() if select_all else cache.select_indices(self.config.process_every, self.config.topk)
        keys, values = gather_cache_layers(cache, indices)
        updated = run_batched_cache_processors(self.blocks, merge_layer_kv(keys, values), indices)
        new_keys, new_values = split_layer_kv(updated, keys.shape[2], keys.shape[4])
        scatter_cache_layers(cache, indices, new_keys, new_values)
        return cache


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
        self.block = CacheProcessor(self.num_layers * per_layer, self.config)

    def forward(self, cache: CacheState, *, select_all: bool = False) -> CacheState:
        indices = cache.all_indices() if select_all else cache.select_indices(self.config.process_every, self.config.topk)
        keys, values = gather_cache_layers(cache, indices)
        updated = self.block(merge_global_kv(keys, values), indices)
        new_keys, new_values = split_global_kv(updated, self.num_layers, self.num_kv_heads, self.head_dim)
        scatter_cache_layers(cache, indices, new_keys, new_values)
        return cache


def _merge_kv(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    batch, heads, seq, dim = keys.shape
    k = keys.permute(0, 2, 1, 3).reshape(batch, seq, heads * dim)
    v = values.permute(0, 2, 1, 3).reshape(batch, seq, heads * dim)
    return torch.cat([k, v], dim=-1)


def _split_kv(x: torch.Tensor, num_heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq, dim = x.shape
    half = dim // 2
    keys, values = x[..., :half], x[..., half:]
    keys = keys.view(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    values = values.view(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    return keys, values
