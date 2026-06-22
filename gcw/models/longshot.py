from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models.bgt import BGT
from gcw.models.cache_state import CacheState
from gcw.models.transformer import RMSNorm, flash_sdpa


EncoderMethod = Literal["linear", "query_attention"]


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
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        batch, cache_heads, _ = x.shape
        if selected_heads.shape[-1] == 0:
            return torch.zeros_like(x)
        q = self.q_proj(x).view(batch, cache_heads, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, cache_heads, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, cache_heads, self.num_kv_heads, self.head_dim)
        k, v = (_gather_cache_heads(t, selected_heads) for t in (k, v))
        q = q.permute(0, 2, 1, 3)
        k, v = (t.permute(0, 2, 1, 3) for t in (k, v))
        q, k = q.to(v.dtype), k.to(v.dtype)
        y = flash_sdpa(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        y = y.permute(0, 2, 1, 3).contiguous().view(batch, cache_heads, -1)
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

    def forward(self, x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), selected_heads)
        h = self.post_norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class PerCacheHeadLinear(nn.Module):
    def __init__(self, num_cache_heads: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.num_cache_heads = num_cache_heads
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_cache_heads, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_cache_heads, out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.einsum("bci,cio->bco", x, self.weight)
        if self.bias is not None:
            y = y + self.bias[None, :, :]
        return y


class CacheProcessor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_cache_heads: int,
        config: CacheProcessorConfig,
        *,
        encoder_method: EncoderMethod = "linear",
        encoder_seq_len: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.encoder_method = encoder_method
        self.encoder_seq_len = encoder_seq_len or config.process_every + max(config.topk, 0)
        self.per_cache_head_projections = config.per_cache_head_projections
        self.in_norm = RMSNorm(input_dim, config.norm_eps)
        if encoder_method == "linear":
            in_features = self.encoder_seq_len * input_dim
        elif encoder_method == "query_attention":
            in_features = input_dim
            self.query = nn.Parameter(torch.empty(input_dim))
        else:
            raise ValueError(f"unknown encoder_method: {encoder_method}")
        if self.per_cache_head_projections:
            self.in_proj = PerCacheHeadLinear(num_cache_heads, in_features, config.hidden_size)
        else:
            self.in_proj = nn.Linear(in_features, config.hidden_size)
        self.layers = nn.ModuleList(ProcessorLayer(config) for _ in range(config.num_layers))
        self.out_norm = RMSNorm(config.hidden_size, config.norm_eps)
        if self.per_cache_head_projections:
            self.out_proj = PerCacheHeadLinear(num_cache_heads, config.hidden_size, input_dim)
        else:
            self.out_proj = nn.Linear(config.hidden_size, input_dim)
        self.gate = nn.Parameter(torch.full((1,), config.gate_init))
        self.init_weights()

    def init_weights(self, std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, PerCacheHeadLinear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        if hasattr(self, "query"):
            nn.init.normal_(self.query, mean=0.0, std=std)
        nn.init.constant_(self.gate, self.config.gate_init)

    def forward(self, x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        if x.shape[2] == 0:
            raise ValueError("Longshot cache processor requires at least one selected time position")
        residual = x[:, :, -1, :]
        h = self._encode_time(self.in_norm(x))
        for layer in self.layers:
            h = layer(h, selected_heads)
        return residual + self.out_proj(self.out_norm(h)) * torch.sigmoid(self.gate)

    def _encode_time(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_method == "linear":
            return self.in_proj(x.flatten(start_dim=2))
        scale = self.input_dim**-0.5
        scores = torch.einsum("bcsi,i->bcs", x, self.query.to(x.dtype)) * scale
        weights = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        context = torch.einsum("bcs,bcsi->bci", weights, x)
        return self.in_proj(context)


class GlobalCacheProcessor(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        *,
        encoder_method: EncoderMethod = "linear",
        encoder_seq_len: int | None = None,
    ):
        super().__init__()
        if model_config.cache_processor is None:
            raise ValueError("Longshot requires cache_processor config")
        self.config = model_config.cache_processor
        self.num_layers = model_config.num_layers
        self.num_kv_heads = model_config.num_kv_heads
        self.head_dim = int(model_config.head_dim)
        self.block = CacheProcessor(
            2 * self.head_dim,
            self.num_layers * self.num_kv_heads,
            self.config,
            encoder_method=encoder_method,
            encoder_seq_len=encoder_seq_len,
        )

    def forward(self, cache: CacheState, *, select_all: bool = False) -> CacheState:
        indices = cache.all_indices() if select_all else cache.select_indices(self.config.process_every, self.config.topk)
        if indices.shape[-1] == 0:
            return cache
        parts = []
        for layer_idx in range(self.num_layers):
            keys, values = cache.gather_layer(layer_idx, indices)
            parts.append(_merge_kv(keys, values))
        merged = torch.cat(parts, dim=1)
        selected_heads = cache.all_heads() if select_all else _select_heads_from_merged(merged, self.config.topk_heads)
        updated = self.block(merged, selected_heads)
        latest = indices[:, -1:]
        for layer_idx, part in enumerate(torch.split(updated, [p.shape[1] for p in parts], dim=1)):
            keys, values = _split_kv(part)
            cache.scatter_layer(layer_idx, latest, keys.unsqueeze(2), values.unsqueeze(2))
        return cache


class Longshot(BGT):
    processor_cls = GlobalCacheProcessor
    variant = "longshot"


def _merge_kv(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return torch.cat([keys, values], dim=-1)  # [batch, heads, seq, 2 * dim]


def _split_kv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    half = x.shape[-1] // 2
    keys, values = x[..., :half], x[..., half:]
    return keys, values  # [batch, heads, dim], [batch, heads, dim]


def _gather_cache_heads(x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
    if selected_heads.ndim != 2:
        raise ValueError("selected_heads must have shape (batch, selected_heads)")
    index = selected_heads[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
    return torch.gather(x, dim=1, index=index)


def _select_heads_from_merged(x: torch.Tensor, topk: int) -> torch.Tensor:
    batch = x.shape[0]
    if topk <= 0:
        return torch.empty(batch, 0, dtype=torch.long, device=x.device)
    scores = x.float().flatten(start_dim=2).norm(dim=-1)
    k = min(topk, scores.shape[-1])
    _, indices = torch.topk(scores, k=k, dim=-1)
    return indices.sort(dim=-1).values
