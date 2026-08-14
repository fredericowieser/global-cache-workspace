from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from gcw.models.cache_processors.state import CacheState, StaticCacheState, select_l2_indices
from gcw.models.backbones.transformer import apply_rotary, flash_sdpa
from gcw.models.cache_processors.windowed import WindowedCache

CacheLike = CacheState | StaticCacheState | WindowedCache


def rotary_for_indices(
    indices: torch.Tensor,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=indices.device, dtype=torch.float32) / head_dim))
    freqs = indices.float().unsqueeze(-1) * inv
    return freqs.cos().unsqueeze(-2).to(dtype), freqs.sin().unsqueeze(-2).to(dtype)


def rotate_cache_keys(
    keys: torch.Tensor,
    indices: torch.Tensor,
    *,
    theta: float,
    rotary_style: str,
    inverse: bool = False,
    rotary_dim: int | None = None,
) -> torch.Tensor:
    """Apply or undo RoPE on layer-major or flattened cache-key tensors."""
    rotary_dim = keys.shape[-1] if rotary_dim is None else rotary_dim
    if rotary_dim <= 0 or rotary_dim > keys.shape[-1] or rotary_dim % 2:
        raise ValueError("rotary_dim must be a positive even value no larger than the key dimension")
    cos, sin = rotary_for_indices(indices, rotary_dim, theta, keys.dtype)
    if keys.shape[:-1] == indices.shape:
        cos = cos.squeeze(-2)
        sin = sin.squeeze(-2)
    elif keys.ndim == 5 and indices.ndim == 2 and keys.shape[1] == indices.shape[0] and keys.shape[3] == indices.shape[1]:
        cos = cos.permute(0, 2, 1, 3).unsqueeze(0)
        sin = sin.permute(0, 2, 1, 3).unsqueeze(0)
    else:
        raise ValueError("keys and indices shapes are not compatible for cache-key RoPE")
    if inverse:
        sin = -sin
    rotated = apply_rotary(keys[..., :rotary_dim], cos, sin, style=rotary_style)
    if rotary_dim == keys.shape[-1]:
        return rotated
    return torch.cat([rotated, keys[..., rotary_dim:]], dim=-1)


def expand_cache_head_indices(indices: torch.Tensor, num_cache_heads: int) -> torch.Tensor:
    if indices.ndim == 3:
        return indices
    if indices.ndim != 2:
        raise ValueError("indices must have shape (batch, selected) or (batch, cache_heads, selected)")
    return indices[:, None, :].expand(-1, num_cache_heads, -1)


def select_per_cache_head_kv_norm_indices_from_scores(
    scores: torch.Tensor,
    process_every: int,
    topk: int,
) -> torch.Tensor:
    if scores.ndim != 3:
        raise ValueError("scores must have shape (batch, cache_heads, seq)")
    batch, cache_heads, seq = scores.shape
    if seq == 0:
        return torch.empty(batch, cache_heads, 0, dtype=torch.long, device=scores.device)
    recent = min(process_every, seq)
    hist_len = seq - recent
    recent_idx = torch.arange(hist_len, seq, device=scores.device).view(1, 1, recent).expand(batch, cache_heads, -1)
    if hist_len > 0 and topk > 0:
        k = min(topk, hist_len)
        _, hist_idx = torch.topk(scores[:, :, :hist_len], k=k, dim=-1)
        selected = torch.cat([hist_idx, recent_idx], dim=-1)
    else:
        selected = recent_idx
    return selected.sort(dim=-1).values


def gather_per_cache_head_scores(scores: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 3:
        raise ValueError("scores must have shape (batch, cache_heads, seq)")
    if indices.ndim != 3:
        raise ValueError("indices must have shape (batch, cache_heads, selected)")
    return torch.gather(scores, dim=2, index=indices)


def select_heads_from_per_cache_head_scores(scores: torch.Tensor, topk: int) -> torch.Tensor:
    if scores.ndim != 3:
        raise ValueError("scores must have shape (batch, cache_heads, selected)")
    assert topk > 0, "topk must select at least one cache head"
    head_scores = scores.float().sum(dim=-1)
    k = min(topk, head_scores.shape[-1])
    _, indices = torch.topk(head_scores, k=k, dim=-1)
    return indices.sort(dim=-1).values


def gather_per_cache_head_kv(
    cache: CacheLike,
    indices: torch.Tensor,
    *,
    unrope_keys: bool = False,
    rope_theta: float = 10000.0,
    rotary_style: str = "llama",
    rotary_dim: int | None = None,
) -> torch.Tensor:
    keys, values = _cache_tensors(cache)
    keys = _flatten_cache_heads(keys)
    values = _flatten_cache_heads(values)
    if indices.ndim != 3:
        raise ValueError("indices must have shape (batch, cache_heads, selected)")
    index = indices[..., None].expand(-1, -1, -1, keys.shape[-1])
    keys = torch.gather(keys, dim=2, index=index)
    values = torch.gather(values, dim=2, index=index)
    if unrope_keys:
        keys = rotate_cache_keys(
            keys,
            indices,
            theta=rope_theta,
            rotary_style=rotary_style,
            inverse=True,
            rotary_dim=rotary_dim,
        )
    return torch.cat([keys, values], dim=-1)


def latest_shared_indices(indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim == 2:
        return indices[:, -1:]
    if indices.ndim != 3:
        raise ValueError("indices must have shape (batch, selected) or (batch, cache_heads, selected)")
    return indices[:, :1, -1]


def gather_cache_layers(cache: CacheState | WindowedCache, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(cache, WindowedCache):
        keys, values = cache.keys, cache.values
    else:
        keys = torch.stack(cache.keys, dim=0)
        values = torch.stack(cache.values, dim=0)
    index = indices[None, :, None, :, None].expand(keys.shape[0], -1, keys.shape[2], -1, keys.shape[4])
    return torch.gather(keys, 3, index), torch.gather(values, 3, index)


def gather_cache_values(cache: CacheState | WindowedCache, indices: torch.Tensor) -> torch.Tensor:
    values = cache.values if isinstance(cache, WindowedCache) else torch.stack(cache.values, dim=0)
    index = indices[None, :, None, :, None].expand(values.shape[0], -1, values.shape[2], -1, values.shape[4])
    return torch.gather(values, 3, index)


def select_kv_norm_indices(
    cache: CacheState | WindowedCache,
    score_logits: torch.Tensor,
    process_every: int,
    topk: int,
) -> torch.Tensor:
    scores = kv_norm_scores(cache, score_logits)
    token_scores = scores.amax(dim=(0, 2))
    return select_l2_indices(token_scores, process_every=process_every, topk=topk)


def kv_norm_scores(cache: CacheState | WindowedCache, score_logits: torch.Tensor) -> torch.Tensor:
    keys, values = _cache_tensors(cache)
    weights = torch.softmax(score_logits.float(), dim=0)
    key_norm = keys.float().pow(2).mean(dim=-1).sqrt()
    value_norm = values.float().pow(2).mean(dim=-1).sqrt()
    return weights[0] * key_norm + weights[1] * value_norm


def workspace_writer_values(
    cache: CacheState | WindowedCache,
    indices: torch.Tensor,
    score_logits: torch.Tensor,
    writer_head_fraction: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = gather_cache_values(cache, indices)
    selected_scores = _selected_kv_norm_scores(cache, indices, score_logits)
    gate = _workspace_gate(selected_scores, writer_head_fraction, temperature)
    return values * gate.to(values.dtype), values


def workspace_writer_kv(
    cache: CacheState | WindowedCache,
    indices: torch.Tensor,
    score_logits: torch.Tensor,
    writer_head_fraction: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keys, values = gather_cache_layers(cache, indices)
    selected_scores = _selected_kv_norm_scores(cache, indices, score_logits)
    gate = _workspace_gate(selected_scores, writer_head_fraction, temperature)
    return keys * gate.to(keys.dtype), values * gate.to(values.dtype), keys, values


def scatter_cache_layers(
    cache: CacheState | WindowedCache,
    indices: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> CacheState | WindowedCache:
    if isinstance(cache, WindowedCache):
        old_keys, old_values = cache.keys, cache.values
    else:
        old_keys = torch.stack(cache.keys, dim=0)
        old_values = torch.stack(cache.values, dim=0)
    index = indices[None, :, None, :, None].expand_as(keys)
    new_keys = old_keys.scatter(3, index, keys.to(old_keys.dtype))
    new_values = old_values.scatter(3, index, values.to(old_values.dtype))
    if isinstance(cache, WindowedCache):
        cache.keys = new_keys
        cache.values = new_values
    else:
        cache.keys = list(new_keys.unbind(0))
        cache.values = list(new_values.unbind(0))
    return cache


def scatter_cache_values(
    cache: CacheState | WindowedCache,
    indices: torch.Tensor,
    values: torch.Tensor,
) -> CacheState | WindowedCache:
    old_values = cache.values if isinstance(cache, WindowedCache) else torch.stack(cache.values, dim=0)
    index = indices[None, :, None, :, None].expand_as(values)
    new_values = old_values.scatter(3, index, values.to(old_values.dtype))
    if isinstance(cache, WindowedCache):
        cache.values = new_values
    else:
        cache.values = list(new_values.unbind(0))
    return cache


def _cache_tensors(cache: CacheLike) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(cache, WindowedCache):
        return cache.keys, cache.values
    if isinstance(cache, StaticCacheState):
        keys = [cache.layer_view(i, cache.length)[0] for i in range(cache.num_layers)]
        values = [cache.layer_view(i, cache.length)[1] for i in range(cache.num_layers)]
        return torch.stack(keys, dim=0), torch.stack(values, dim=0)
    return torch.stack(cache.keys, dim=0), torch.stack(cache.values, dim=0)


def _flatten_cache_heads(x: torch.Tensor) -> torch.Tensor:
    layers, batch, heads, seq, dim = x.shape
    return x.permute(1, 0, 2, 3, 4).reshape(batch, layers * heads, seq, dim)


def _selected_kv_norm_scores(
    cache: CacheState | WindowedCache,
    indices: torch.Tensor,
    score_logits: torch.Tensor,
) -> torch.Tensor:
    scores = kv_norm_scores(cache, score_logits)
    score_index = indices[None, :, None, :].expand(scores.shape[0], -1, scores.shape[2], -1)
    return torch.gather(scores, 3, score_index)


def _workspace_gate(
    selected_scores: torch.Tensor,
    writer_head_fraction: float,
    temperature: float,
) -> torch.Tensor:
    return _writer_head_gate(selected_scores, writer_head_fraction, temperature) * _token_gate(selected_scores, temperature)


def _writer_head_gate(
    selected_scores: torch.Tensor,
    writer_head_fraction: float,
    temperature: float,
) -> torch.Tensor:
    layers, _, heads, _ = selected_scores.shape
    head_scores = selected_scores.mean(dim=(1, 3))
    writer_count = max(1, math.ceil(layers * heads * writer_head_fraction))
    top = torch.topk(head_scores.flatten(), k=writer_count).indices
    mask = torch.zeros_like(head_scores.flatten(), dtype=torch.bool)
    mask.scatter_(0, top, True)
    logits = (head_scores.flatten() / temperature).masked_fill(~mask, torch.finfo(head_scores.dtype).min)
    gate = torch.softmax(logits, dim=0).view(layers, heads) * writer_count
    return gate[:, None, :, None, None]


def _token_gate(selected_scores: torch.Tensor, temperature: float) -> torch.Tensor:
    token_scores = selected_scores.amax(dim=(0, 2))
    gate = torch.softmax(token_scores / temperature, dim=-1) * selected_scores.shape[-1]
    return gate[None, :, None, :, None]


def merge_layer_kv(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    layers, batch, heads, seq, dim = keys.shape
    k = keys.permute(0, 1, 3, 2, 4).reshape(layers, batch, seq, heads * dim)
    v = values.permute(0, 1, 3, 2, 4).reshape(layers, batch, seq, heads * dim)
    return torch.cat([k, v], dim=-1)


def merge_layer_values(values: torch.Tensor) -> torch.Tensor:
    layers, batch, heads, seq, dim = values.shape
    return values.permute(0, 1, 3, 2, 4).reshape(layers, batch, seq, heads * dim)


def split_layer_kv(x: torch.Tensor, num_heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    layers, batch, seq, _ = x.shape
    feature_dim = num_heads * head_dim
    keys, values = torch.split(x, [feature_dim, feature_dim], dim=-1)
    keys = keys.view(layers, batch, seq, num_heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()
    values = values.view(layers, batch, seq, num_heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()
    return keys, values


def split_layer_values(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    layers, batch, seq, _ = x.shape
    return x.view(layers, batch, seq, num_heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()


def merge_global_kv(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    layers, batch, heads, seq, dim = keys.shape
    by_layer = torch.cat(
        [
            keys.permute(1, 3, 0, 2, 4).reshape(batch, seq, layers, heads * dim),
            values.permute(1, 3, 0, 2, 4).reshape(batch, seq, layers, heads * dim),
        ],
        dim=-1,
    )
    return by_layer.reshape(batch, seq, layers * 2 * heads * dim)


def merge_global_values(values: torch.Tensor) -> torch.Tensor:
    layers, batch, heads, seq, dim = values.shape
    return values.permute(1, 3, 0, 2, 4).reshape(batch, seq, layers * heads * dim)


def split_global_kv(
    x: torch.Tensor,
    num_layers: int,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq, _ = x.shape
    by_layer = x.view(batch, seq, num_layers, 2, num_heads, head_dim)
    keys = by_layer[:, :, :, 0].permute(2, 0, 3, 1, 4).contiguous()
    values = by_layer[:, :, :, 1].permute(2, 0, 3, 1, 4).contiguous()
    return keys, values


def split_global_values(
    x: torch.Tensor,
    num_layers: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    batch, seq, _ = x.shape
    return x.view(batch, seq, num_layers, num_heads, head_dim).permute(2, 0, 3, 1, 4).contiguous()


@dataclass(frozen=True)
class PreparedLinearParameters:
    """One differentiable stack of equivalent per-layer linear parameters."""

    weight: torch.Tensor
    bias: torch.Tensor | None


@dataclass(frozen=True)
class PreparedRMSNormParameters:
    weight: torch.Tensor
    eps: float


@dataclass(frozen=True)
class PreparedAttentionParameters:
    q_proj: PreparedLinearParameters
    k_proj: PreparedLinearParameters
    v_proj: PreparedLinearParameters
    o_proj: PreparedLinearParameters
    num_heads: int
    num_kv_heads: int
    head_dim: int
    theta: float
    dropout: float
    training: bool


@dataclass(frozen=True)
class PreparedProcessorLayerParameters:
    input_norm: PreparedRMSNormParameters
    post_norm: PreparedRMSNormParameters
    attention: PreparedAttentionParameters
    gate_proj: PreparedLinearParameters
    up_proj: PreparedLinearParameters
    down_proj: PreparedLinearParameters


@dataclass(frozen=True)
class PreparedBatchedCacheProcessorParameters:
    """Graph-local stacked views of a homogeneous cache-processor collection.

    These tensors are derived from, not replacements for, the parameters owned by
    the source ``ModuleList``. The caller must bound this object's lifetime to one
    language-model forward. Reusing it in several processor events within that
    forward is safe; retaining it for a later forward would retain or reuse the
    old autograd graph.
    """

    in_norm: PreparedRMSNormParameters
    in_proj: PreparedLinearParameters
    layers: tuple[PreparedProcessorLayerParameters, ...]
    out_norm: PreparedRMSNormParameters
    out_proj: PreparedLinearParameters
    gate: torch.Tensor
    gate_mode: str
    activation_checkpointing: bool


def prepare_batched_cache_processor_parameters(blocks) -> PreparedBatchedCacheProcessorParameters:
    """Stack homogeneous processor parameters once for graph-local reuse."""
    block_list = list(blocks)
    if not block_list:
        raise ValueError("cannot prepare an empty cache-processor collection")
    layer_count = len(block_list[0].layers)
    if any(len(block.layers) != layer_count for block in block_list):
        raise ValueError("batched cache processors must have the same number of layers")
    checkpointing = {bool(block.activation_checkpointing) for block in block_list}
    if len(checkpointing) != 1:
        raise ValueError("batched cache processors must use one activation-checkpointing policy")

    prepared_layers = []
    for layer_idx in range(layer_count):
        layers = [block.layers[layer_idx] for block in block_list]
        attns = [layer.attn for layer in layers]
        prepared_layers.append(
            PreparedProcessorLayerParameters(
                input_norm=_prepare_rms_norm([layer.input_norm for layer in layers]),
                post_norm=_prepare_rms_norm([layer.post_norm for layer in layers]),
                attention=PreparedAttentionParameters(
                    q_proj=_prepare_linear(attns, "q_proj"),
                    k_proj=_prepare_linear(attns, "k_proj"),
                    v_proj=_prepare_linear(attns, "v_proj"),
                    o_proj=_prepare_linear(attns, "o_proj"),
                    num_heads=attns[0].num_heads,
                    num_kv_heads=attns[0].num_kv_heads,
                    head_dim=attns[0].head_dim,
                    theta=layers[0].theta,
                    dropout=attns[0].dropout,
                    training=attns[0].training,
                ),
                gate_proj=_prepare_linear(layers, "gate_proj"),
                up_proj=_prepare_linear(layers, "up_proj"),
                down_proj=_prepare_linear(layers, "down_proj"),
            )
        )

    return PreparedBatchedCacheProcessorParameters(
        in_norm=_prepare_rms_norm([block.in_norm for block in block_list]),
        in_proj=_prepare_linear(block_list, "in_proj"),
        layers=tuple(prepared_layers),
        out_norm=_prepare_rms_norm([block.out_norm for block in block_list]),
        out_proj=_prepare_linear(block_list, "out_proj"),
        gate=torch.stack([block.gate for block in block_list], dim=0).view(len(block_list), 1, 1, 1),
        gate_mode=block_list[0].config.gate_mode,
        activation_checkpointing=checkpointing.pop(),
    )


def run_prepared_batched_cache_processors(
    prepared: PreparedBatchedCacheProcessorParameters,
    x: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Run BT using stacks prepared for this language-model forward."""
    h = _prepared_linear(_prepared_rms_norm(x, prepared.in_norm), prepared.in_proj)
    for layer in prepared.layers:
        if prepared.activation_checkpointing and layer.attention.training and torch.is_grad_enabled():
            # The complete transformer layer is the smallest boundary that drops
            # attention and MLP intermediates while retaining BT's stacked kernels.
            h = checkpoint(
                _run_prepared_processor_layer,
                h,
                positions,
                layer,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            h = _run_prepared_processor_layer(h, positions, layer)
    delta = _prepared_linear(_prepared_rms_norm(h, prepared.out_norm), prepared.out_proj)
    return x + delta * _prepared_gate_scale(prepared, delta.dtype)


def _run_prepared_processor_layer(
    h: torch.Tensor,
    positions: torch.Tensor,
    layer: PreparedProcessorLayerParameters,
) -> torch.Tensor:
    h = h + _prepared_attention(_prepared_rms_norm(h, layer.input_norm), layer.attention, positions)
    m = _prepared_rms_norm(h, layer.post_norm)
    return h + _prepared_linear(
        F.silu(_prepared_linear(m, layer.gate_proj)) * _prepared_linear(m, layer.up_proj),
        layer.down_proj,
    )


def _prepare_linear(modules, name: str) -> PreparedLinearParameters:
    linears = [getattr(module, name) for module in modules]
    weight = torch.stack([linear.weight for linear in linears], dim=0)
    bias = None
    if linears[0].bias is not None:
        bias = torch.stack([linear.bias for linear in linears], dim=0)
    return PreparedLinearParameters(weight=weight, bias=bias)


def _prepare_rms_norm(norms) -> PreparedRMSNormParameters:
    return PreparedRMSNormParameters(
        weight=torch.stack([norm.weight for norm in norms], dim=0),
        eps=norms[0].eps,
    )


def _prepared_linear(x: torch.Tensor, linear: PreparedLinearParameters) -> torch.Tensor:
    layers, batch, seq, in_features = x.shape
    out_features = linear.weight.shape[1]
    out = torch.bmm(
        x.reshape(layers, batch * seq, in_features),
        linear.weight.transpose(1, 2),
    ).view(layers, batch, seq, out_features)
    if linear.bias is not None:
        out = out + linear.bias[:, None, None, :]
    return out


def _prepared_rms_norm(x: torch.Tensor, norm: PreparedRMSNormParameters) -> torch.Tensor:
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + norm.eps)
    return y * norm.weight.to(x.dtype)[:, None, None, :]


def _prepared_attention(
    x: torch.Tensor,
    attention: PreparedAttentionParameters,
    positions: torch.Tensor,
) -> torch.Tensor:
    num_layers, batch, seq, _ = x.shape
    q = _prepared_linear(x, attention.q_proj).view(
        num_layers, batch, seq, attention.num_heads, attention.head_dim
    )
    k = _prepared_linear(x, attention.k_proj).view(
        num_layers, batch, seq, attention.num_kv_heads, attention.head_dim
    )
    v = _prepared_linear(x, attention.v_proj).view(
        num_layers, batch, seq, attention.num_kv_heads, attention.head_dim
    )

    cos_sin = rotary_for_indices(positions, attention.head_dim, attention.theta, q.dtype)
    q = apply_rotary(q, *cos_sin).permute(0, 1, 3, 2, 4).contiguous()
    k = apply_rotary(k, *cos_sin).permute(0, 1, 3, 2, 4).contiguous()
    v = v.permute(0, 1, 3, 2, 4).contiguous()
    q, k = q.to(v.dtype), k.to(v.dtype)

    y = flash_sdpa(
        q.view(num_layers * batch, attention.num_heads, seq, attention.head_dim),
        k.view(num_layers * batch, attention.num_kv_heads, seq, attention.head_dim),
        v.view(num_layers * batch, attention.num_kv_heads, seq, attention.head_dim),
        dropout_p=attention.dropout if attention.training else 0.0,
        enable_gqa=attention.num_kv_heads != attention.num_heads,
    )
    y = y.view(num_layers, batch, attention.num_heads, seq, attention.head_dim)
    y = y.permute(0, 1, 3, 2, 4).contiguous()
    return _prepared_linear(
        y.view(num_layers, batch, seq, attention.num_heads * attention.head_dim),
        attention.o_proj,
    )


def _prepared_gate_scale(
    prepared: PreparedBatchedCacheProcessorParameters,
    dtype: torch.dtype,
) -> torch.Tensor:
    if prepared.gate_mode == "sigmoid":
        return torch.sigmoid(prepared.gate).to(dtype)
    if prepared.gate_mode == "direct":
        return prepared.gate.to(dtype)
    raise ValueError(f"unsupported cache processor gate_mode: {prepared.gate_mode}")


def run_batched_cache_processors(blocks, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    block_list = list(blocks)
    h = _linear(_rms_norm(x, [b.in_norm for b in block_list]), block_list, "in_proj")
    for layer_idx in range(len(block_list[0].layers)):
        layers = [b.layers[layer_idx] for b in block_list]
        h = h + _attention(_rms_norm(h, [layer.input_norm for layer in layers]), layers, positions)
        m = _rms_norm(h, [layer.post_norm for layer in layers])
        h = h + _linear(F.silu(_linear(m, layers, "gate_proj")) * _linear(m, layers, "up_proj"), layers, "down_proj")
    delta = _linear(_rms_norm(h, [b.out_norm for b in block_list]), block_list, "out_proj")
    return x + delta * _gate_scale(block_list, delta.dtype)


def _gate_scale(blocks, dtype: torch.dtype) -> torch.Tensor:
    gate = torch.stack([b.gate for b in blocks], dim=0).view(len(blocks), 1, 1, 1)
    mode = blocks[0].config.gate_mode
    if mode == "sigmoid":
        return torch.sigmoid(gate).to(dtype)
    if mode == "direct":
        return gate.to(dtype)
    raise ValueError(f"unsupported cache processor gate_mode: {mode}")


def _linear(x: torch.Tensor, modules, name: str) -> torch.Tensor:
    linears = [getattr(module, name) for module in modules]
    weight = torch.stack([linear.weight for linear in linears], dim=0)
    layers, batch, seq, in_features = x.shape
    out_features = weight.shape[1]
    out = torch.bmm(
        x.reshape(layers, batch * seq, in_features),
        weight.transpose(1, 2),
    ).view(layers, batch, seq, out_features)
    if linears[0].bias is not None:
        bias = torch.stack([linear.bias for linear in linears], dim=0)
        out = out + bias[:, None, None, :]
    return out


def _rms_norm(x: torch.Tensor, norms) -> torch.Tensor:
    eps = norms[0].eps
    weight = torch.stack([norm.weight for norm in norms], dim=0).to(x.dtype)
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return y * weight[:, None, None, :]


def _attention(x: torch.Tensor, layers, positions: torch.Tensor) -> torch.Tensor:
    attns = [layer.attn for layer in layers]
    num_layers, batch, seq, _ = x.shape
    num_heads = attns[0].num_heads
    num_kv_heads = attns[0].num_kv_heads
    head_dim = attns[0].head_dim

    q = _linear(x, attns, "q_proj").view(num_layers, batch, seq, num_heads, head_dim)
    k = _linear(x, attns, "k_proj").view(num_layers, batch, seq, num_kv_heads, head_dim)
    v = _linear(x, attns, "v_proj").view(num_layers, batch, seq, num_kv_heads, head_dim)

    cos_sin = rotary_for_indices(positions, head_dim, layers[0].theta, q.dtype)
    q = apply_rotary(q, *cos_sin).permute(0, 1, 3, 2, 4).contiguous()
    k = apply_rotary(k, *cos_sin).permute(0, 1, 3, 2, 4).contiguous()
    v = v.permute(0, 1, 3, 2, 4).contiguous()
    q, k = q.to(v.dtype), k.to(v.dtype)

    y = flash_sdpa(
        q.view(num_layers * batch, num_heads, seq, head_dim),
        k.view(num_layers * batch, num_kv_heads, seq, head_dim),
        v.view(num_layers * batch, num_kv_heads, seq, head_dim),
        dropout_p=attns[0].dropout if attns[0].training else 0.0,
        enable_gqa=num_kv_heads != num_heads,
    )
    y = y.view(num_layers, batch, num_heads, seq, head_dim).permute(0, 1, 3, 2, 4).contiguous()
    return _linear(y.view(num_layers, batch, seq, num_heads * head_dim), attns, "o_proj")

def zero_projection(module: nn.Module, *, per_head_type: type[nn.Module]) -> None:
    if not isinstance(module, (nn.Linear, per_head_type)):
        raise TypeError(f"unsupported projection type: {type(module).__name__}")
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"projection {type(module).__name__} has no tensor weight")
    nn.init.zeros_(weight)
    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError(f"projection {type(module).__name__} has a non-tensor bias")
        nn.init.zeros_(bias)


def scatter_latest(cache, indices: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
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


def scatter_latest_values(cache, indices: torch.Tensor, values: torch.Tensor):
    if isinstance(cache, WindowedCache):
        index = indices[None, :, None, :, None].expand_as(values)
        cache.values = cache.values.scatter(3, index, values.to(cache.values.dtype))
        return cache
    if isinstance(cache, StaticCacheState):
        for layer_idx in range(cache.num_layers):
            target = cache.values[layer_idx][:, :, : cache.length, :]
            index = indices[:, None, :, None].expand_as(values[layer_idx])
            target.scatter_(2, index, values[layer_idx].to(target.dtype))
        return cache
    for layer_idx in range(cache.num_layers):
        index = indices[:, None, :, None].expand_as(values[layer_idx])
        cache.values[layer_idx] = cache.values[layer_idx].scatter(
            2,
            index,
            values[layer_idx].to(cache.values[layer_idx].dtype),
        )
    return cache


def split_merged_cache(
    x: torch.Tensor,
    num_layers: int,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _cache_heads, seq, _ = x.shape
    split = x.view(batch, num_layers, num_heads, seq, 2, head_dim)
    keys = split[:, :, :, :, 0].permute(1, 0, 2, 3, 4).contiguous()
    values = split[:, :, :, :, 1].permute(1, 0, 2, 3, 4).contiguous()
    return keys, values


def split_merged_cache_values(
    x: torch.Tensor,
    num_layers: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    batch, _cache_heads, seq, _ = x.shape
    return x.view(batch, num_layers, num_heads, seq, head_dim).permute(1, 0, 2, 3, 4).contiguous()
