from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from gcw.config import CacheEncoderMethod, CacheEncoderQuerySource, CacheProcessorConfig, ModelConfig
from gcw.models.bgt import BGT
from gcw.models.cache_processor_ops import rotate_cache_keys
from gcw.models.cache_state import CacheState, StaticCacheState
from gcw.models.transformer import RMSNorm, flash_sdpa
from gcw.models.windowed_cache import WindowedCache

CacheLike = CacheState | StaticCacheState | WindowedCache


EncoderMethod = CacheEncoderMethod
EncoderQuerySource = CacheEncoderQuerySource


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
        selected_x = _gather_cache_heads(x, selected_heads)
        k = self.k_proj(selected_x).view(batch, selected_heads.shape[-1], self.num_kv_heads, self.head_dim)
        v = self.v_proj(selected_x).view(batch, selected_heads.shape[-1], self.num_kv_heads, self.head_dim)
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
        self.weight = nn.Parameter(torch.empty(num_cache_heads, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_cache_heads, out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.einsum("bci,cio->bco", x, self.weight)
        if self.bias is not None:
            y = y + self.bias[None, :, :]
        return y


class QueryAttentionTimeEncoder(nn.Module):
    def __init__(self, input_dim: int, num_cache_heads: int, config: CacheProcessorConfig):
        super().__init__()
        self.input_dim = input_dim
        self.encoder_query_source = config.encoder_query_source
        if self.encoder_query_source == "learned":
            self.query = nn.Parameter(torch.empty(num_cache_heads, input_dim))
        elif self.encoder_query_source == "final_token":
            self.query_proj: nn.Module
            if config.per_cache_head_projections:
                self.query_proj = PerCacheHeadLinear(num_cache_heads, input_dim, input_dim)
            else:
                self.query_proj = nn.Linear(input_dim, input_dim)
        else:
            raise ValueError(f"unknown encoder_query_source: {self.encoder_query_source}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self._query(x)
        scale = self.input_dim**-0.5
        scores = torch.einsum("bcsi,bci->bcs", x, query.to(x.dtype)) * scale
        weights = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        return torch.einsum("bcs,bcsi->bci", weights, x)

    def _query(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_query_source == "learned":
            return self.query[None, :, :].expand(x.shape[0], -1, -1)
        if self.encoder_query_source == "final_token":
            return self.query_proj(x[:, :, -1, :])
        raise ValueError(f"unsupported encoder_query_source: {self.encoder_query_source}")


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
        self.encoder_query_source = config.encoder_query_source
        self.encoder_seq_len = encoder_seq_len or config.encoder_seq_len or config.process_every + max(config.topk, 0)
        self.in_norm = RMSNorm(input_dim, config.norm_eps)
        if encoder_method == "linear":
            in_features = self.encoder_seq_len * input_dim
        elif encoder_method == "query_attention":
            in_features = input_dim
            self.time_encoder = QueryAttentionTimeEncoder(input_dim, num_cache_heads, config)
        else:
            raise ValueError(f"unknown encoder_method: {encoder_method}")
        self.in_proj: nn.Module
        if config.per_cache_head_projections:
            self.in_proj = PerCacheHeadLinear(num_cache_heads, in_features, config.hidden_size)
        else:
            self.in_proj = nn.Linear(in_features, config.hidden_size)
        self.layers = nn.ModuleList(ProcessorLayer(config) for _ in range(config.num_layers))
        self.out_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.out_proj: nn.Module
        if config.per_cache_head_projections:
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
            elif isinstance(module, QueryAttentionTimeEncoder) and hasattr(module, "query"):
                nn.init.normal_(module.query, mean=0.0, std=std)
        nn.init.constant_(self.gate, self.config.gate_init)
        if self.config.out_proj_init == "zero":
            _zero_projection(self.out_proj)

    def compute_update(self, x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        h = self._encode_time(self.in_norm(x))
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                h = checkpoint(layer, h, selected_heads, use_reentrant=False)
            else:
                h = layer(h, selected_heads)
        return self.out_proj(self.out_norm(h))

    def scaled_update(self, x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        delta = self.compute_update(x, selected_heads)
        return delta * self._gate_scale(delta.dtype)

    def forward(self, x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
        if x.shape[2] == 0:
            raise ValueError("Longshot cache processor requires at least one selected time position")
        return x[:, :, -1, :] + self.scaled_update(x, selected_heads)

    def _gate_scale(self, dtype: torch.dtype) -> torch.Tensor:
        if self.config.gate_mode == "sigmoid":
            return torch.sigmoid(self.gate).to(dtype)
        if self.config.gate_mode == "direct":
            return self.gate.to(dtype)
        raise ValueError(f"unsupported cache processor gate_mode: {self.config.gate_mode}")

    def _encode_time(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_method == "linear":
            x = _fit_time_length(x, self.encoder_seq_len)
            return self.in_proj(x.flatten(start_dim=2))
        if self.encoder_method == "query_attention":
            return self.in_proj(self.time_encoder(x))
        raise ValueError(f"unknown encoder_method: {self.encoder_method}")


class GlobalCacheProcessor(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        *,
        encoder_method: EncoderMethod | None = None,
        encoder_seq_len: int | None = None,
    ):
        super().__init__()
        if model_config.cache_processor is None:
            raise ValueError("Longshot requires cache_processor config")
        self.config = model_config.cache_processor
        if self.config.processor_mode != "standard":
            raise ValueError("Longshot supports only cache_processor.processor_mode='standard'")
        if self.config.update_stream != "kv":
            raise ValueError("Longshot supports only cache_processor.update_stream='kv'")
        if self.config.score != "attn_l2":
            raise ValueError("Longshot supports only cache_processor.score='attn_l2'")
        encoder_method = encoder_method or self.config.encoder_method
        self.num_layers = model_config.num_layers
        self.num_kv_heads = model_config.num_kv_heads
        if model_config.head_dim is None:
            raise ValueError("Longshot requires a resolved model head_dim")
        self.head_dim = model_config.head_dim
        self.rope_theta = model_config.rope_theta
        self.rotary_style = model_config.rotary_style
        self.num_cache_heads = self.num_layers * self.num_kv_heads
        self.block = CacheProcessor(
            2 * self.head_dim,
            self.num_cache_heads,
            self.config,
            encoder_method=encoder_method,
            encoder_seq_len=encoder_seq_len or self.config.encoder_seq_len,
        )

    def forward(self, cache: CacheLike, *, select_all: bool = False) -> CacheLike:
        indices = cache.all_indices() if select_all else cache.select_indices(self.config.process_every, self.config.topk)
        if indices.shape[-1] == 0:
            return cache
        merged = _gather_merged(
            cache,
            indices,
            unrope_keys=self.config.unrope_keys,
            rope_theta=self.rope_theta,
            rotary_style=self.rotary_style,
        )
        selected_heads = cache.all_heads() if select_all else _select_heads_from_merged(merged, self.config.topk_heads)
        updated = self.block(merged, selected_heads)
        latest = indices[:, -1:]
        keys, values = _split_merged(updated[:, :, None, :], self.num_layers, self.num_kv_heads, self.head_dim)
        if self.config.unrope_keys:
            keys = rotate_cache_keys(keys, latest, theta=self.rope_theta, rotary_style=self.rotary_style)
        return _scatter_latest(cache, latest, keys, values)


class Longshot(BGT):
    processor_cls = GlobalCacheProcessor
    variant = "longshot"


def _zero_projection(module: nn.Module) -> None:
    if not isinstance(module, (nn.Linear, PerCacheHeadLinear)):
        raise TypeError(f"unsupported projection type: {type(module).__name__}")
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _cache_tensors(cache: CacheLike) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(cache, WindowedCache):
        return cache.keys, cache.values
    if isinstance(cache, StaticCacheState):
        keys = [cache.layer_view(i, cache.length)[0] for i in range(cache.num_layers)]
        values = [cache.layer_view(i, cache.length)[1] for i in range(cache.num_layers)]
        return torch.stack(keys, dim=0), torch.stack(values, dim=0)
    return torch.stack(cache.keys, dim=0), torch.stack(cache.values, dim=0)


def _gather_merged(
    cache: CacheLike,
    indices: torch.Tensor,
    *,
    unrope_keys: bool = False,
    rope_theta: float = 10000.0,
    rotary_style: str = "llama",
) -> torch.Tensor:
    keys, values = _cache_tensors(cache)
    index = indices[None, :, None, :, None].expand(keys.shape[0], -1, keys.shape[2], -1, keys.shape[4])
    keys = torch.gather(keys, 3, index)
    values = torch.gather(values, 3, index)
    if unrope_keys:
        keys = rotate_cache_keys(keys, indices, theta=rope_theta, rotary_style=rotary_style, inverse=True)
    return torch.cat([keys, values], dim=-1).permute(1, 0, 2, 3, 4).reshape(
        keys.shape[1], keys.shape[0] * keys.shape[2], keys.shape[3], 2 * keys.shape[4]
    )


def _scatter_latest(cache: CacheLike, indices: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> CacheLike:
    if isinstance(cache, WindowedCache):
        index = indices[None, :, None, :, None].expand_as(keys)
        cache.keys = cache.keys.scatter(3, index, keys.to(cache.keys.dtype))
        cache.values = cache.values.scatter(3, index, values.to(cache.values.dtype))
        return cache
    if isinstance(cache, StaticCacheState):
        for layer_idx in range(cache.num_layers):
            cache.scatter_layer(layer_idx, indices, keys[layer_idx], values[layer_idx])
        return cache
    for layer_idx in range(cache.num_layers):
        cache.scatter_layer(layer_idx, indices, keys[layer_idx], values[layer_idx])
    return cache


def _split_merged(
    x: torch.Tensor,
    num_layers: int,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, cache_heads, seq, _ = x.shape
    split = x.view(batch, num_layers, num_heads, seq, 2, head_dim)
    keys = split[:, :, :, :, 0].permute(1, 0, 2, 3, 4).contiguous()
    values = split[:, :, :, :, 1].permute(1, 0, 2, 3, 4).contiguous()
    return keys, values


def _fit_time_length(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    current = x.shape[2]
    if current == seq_len:
        return x
    warnings.warn(
        f"Longshot linear encoder expected {seq_len} selected positions but got {current}; "
        "padding/truncating the selected cache history.",
        stacklevel=2,
    )
    if current > seq_len:
        return x[:, :, -seq_len:, :]
    out = x.new_zeros(x.shape[0], x.shape[1], seq_len, x.shape[3])
    out[:, :, :current, :] = x
    return out


def _gather_cache_heads(x: torch.Tensor, selected_heads: torch.Tensor) -> torch.Tensor:
    if selected_heads.ndim != 2:
        raise ValueError("selected_heads must have shape (batch, selected_heads)")
    if x.ndim != 3:
        raise ValueError("x must have shape (batch, cache_heads, hidden)")
    index = selected_heads[:, :, None].expand(-1, -1, x.shape[2])
    return torch.gather(x, dim=1, index=index)


def _select_heads_from_merged(x: torch.Tensor, topk: int) -> torch.Tensor:
    batch = x.shape[0]
    if topk <= 0:
        return torch.empty(batch, 0, dtype=torch.long, device=x.device)
    scores = x.float().flatten(start_dim=2).norm(dim=-1)
    k = min(topk, scores.shape[-1])
    _, indices = torch.topk(scores, k=k, dim=-1)
    return indices.sort(dim=-1).values
