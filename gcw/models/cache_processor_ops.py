from __future__ import annotations

import torch
import torch.nn.functional as F

from gcw.models.cache_state import CacheState
from gcw.models.transformer import apply_rotary, flash_sdpa


def rotary_for_indices(
    indices: torch.Tensor,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=indices.device, dtype=torch.float32) / head_dim))
    freqs = indices.float().unsqueeze(-1) * inv
    return freqs.cos().unsqueeze(2).to(dtype), freqs.sin().unsqueeze(2).to(dtype)


def gather_cache_layers(cache: CacheState, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.stack(cache.keys, dim=0)
    values = torch.stack(cache.values, dim=0)
    index = indices[None, :, None, :, None].expand(keys.shape[0], -1, keys.shape[2], -1, keys.shape[4])
    return torch.gather(keys, 3, index), torch.gather(values, 3, index)


def scatter_cache_layers(
    cache: CacheState,
    indices: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> None:
    old_keys = torch.stack(cache.keys, dim=0)
    old_values = torch.stack(cache.values, dim=0)
    index = indices[None, :, None, :, None].expand_as(keys)
    new_keys = old_keys.scatter(3, index, keys.to(old_keys.dtype))
    new_values = old_values.scatter(3, index, values.to(old_values.dtype))
    cache.keys = list(new_keys.unbind(0))
    cache.values = list(new_values.unbind(0))


def merge_layer_kv(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    layers, batch, heads, seq, dim = keys.shape
    k = keys.permute(0, 1, 3, 2, 4).reshape(layers, batch, seq, heads * dim)
    v = values.permute(0, 1, 3, 2, 4).reshape(layers, batch, seq, heads * dim)
    return torch.cat([k, v], dim=-1)


def split_layer_kv(x: torch.Tensor, num_heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    layers, batch, seq, _ = x.shape
    feature_dim = num_heads * head_dim
    keys, values = torch.split(x, [feature_dim, feature_dim], dim=-1)
    keys = keys.view(layers, batch, seq, num_heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()
    values = values.view(layers, batch, seq, num_heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()
    return keys, values


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


def run_batched_cache_processors(blocks, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    block_list = list(blocks)
    h = _linear(_rms_norm(x, [b.in_norm for b in block_list]), block_list, "in_proj")
    for layer_idx in range(len(block_list[0].layers)):
        layers = [b.layers[layer_idx] for b in block_list]
        h = h + _attention(_rms_norm(h, [layer.input_norm for layer in layers]), layers, positions)
        m = _rms_norm(h, [layer.post_norm for layer in layers])
        h = h + _linear(F.silu(_linear(m, layers, "gate_proj")) * _linear(m, layers, "up_proj"), layers, "down_proj")
    delta = _linear(_rms_norm(h, [b.out_norm for b in block_list]), block_list, "out_proj")
    gate = torch.stack([b.gate for b in block_list], dim=0).view(len(block_list), 1, 1, 1)
    return x + delta * torch.sigmoid(gate).to(delta.dtype)


def _linear(x: torch.Tensor, modules, name: str) -> torch.Tensor:
    linears = [getattr(module, name) for module in modules]
    weight = torch.stack([linear.weight for linear in linears], dim=0)
    out = torch.einsum("lbsi,loi->lbso", x, weight)
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
