"""Vectorized BT processor execution with forward-local prepared parameters.

The cache processor continues to own every parameter and state-dict entry. This
module creates differentiable stacked views whose lifetime is bounded by the
model's existing prepared-forward execution context.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from gcw.models.backbones.transformer import apply_rotary, flash_sdpa
from gcw.models.cache_processors.ops import rotary_for_indices


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


def prepare_batched_cache_processor_parameters(
    blocks,
) -> PreparedBatchedCacheProcessorParameters:
    """Stack homogeneous processor parameters once for graph-local reuse."""
    block_list = list(blocks)
    if not block_list:
        raise ValueError("cannot prepare an empty cache-processor collection")
    layer_count = len(block_list[0].layers)
    if any(len(block.layers) != layer_count for block in block_list):
        raise ValueError(
            "batched cache processors must have the same number of layers"
        )
    checkpointing = {
        bool(block.activation_checkpointing) for block in block_list
    }
    if len(checkpointing) != 1:
        raise ValueError(
            "batched cache processors must use one activation-checkpointing policy"
        )

    prepared_layers = []
    for layer_idx in range(layer_count):
        layers = [block.layers[layer_idx] for block in block_list]
        attns = [layer.attn for layer in layers]
        prepared_layers.append(
            PreparedProcessorLayerParameters(
                input_norm=_prepare_rms_norm(
                    [layer.input_norm for layer in layers]
                ),
                post_norm=_prepare_rms_norm(
                    [layer.post_norm for layer in layers]
                ),
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
        gate=torch.stack([block.gate for block in block_list], dim=0).view(
            len(block_list), 1, 1, 1
        ),
        gate_mode=block_list[0].config.gate_mode,
        activation_checkpointing=checkpointing.pop(),
    )


def run_prepared_batched_cache_processors(
    prepared: PreparedBatchedCacheProcessorParameters,
    x: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Run BT using stacks prepared for this language-model forward."""
    h = _prepared_linear(
        _prepared_rms_norm(x, prepared.in_norm), prepared.in_proj
    )
    for layer in prepared.layers:
        if (
            prepared.activation_checkpointing
            and layer.attention.training
            and torch.is_grad_enabled()
        ):
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
    delta = _prepared_linear(
        _prepared_rms_norm(h, prepared.out_norm), prepared.out_proj
    )
    return x + delta * _prepared_gate_scale(prepared, delta.dtype)


def _run_prepared_processor_layer(
    h: torch.Tensor,
    positions: torch.Tensor,
    layer: PreparedProcessorLayerParameters,
) -> torch.Tensor:
    h = h + _prepared_attention(
        _prepared_rms_norm(h, layer.input_norm), layer.attention, positions
    )
    m = _prepared_rms_norm(h, layer.post_norm)
    return h + _prepared_linear(
        F.silu(_prepared_linear(m, layer.gate_proj))
        * _prepared_linear(m, layer.up_proj),
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


def _prepared_linear(
    x: torch.Tensor, linear: PreparedLinearParameters
) -> torch.Tensor:
    layers, batch, seq, in_features = x.shape
    out_features = linear.weight.shape[1]
    out = torch.bmm(
        x.reshape(layers, batch * seq, in_features),
        linear.weight.transpose(1, 2),
    ).view(layers, batch, seq, out_features)
    if linear.bias is not None:
        out = out + linear.bias[:, None, None, :]
    return out


def _prepared_rms_norm(
    x: torch.Tensor, norm: PreparedRMSNormParameters
) -> torch.Tensor:
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

    cos_sin = rotary_for_indices(
        positions, attention.head_dim, attention.theta, q.dtype
    )
    q = apply_rotary(q, *cos_sin).permute(0, 1, 3, 2, 4).contiguous()
    k = apply_rotary(k, *cos_sin).permute(0, 1, 3, 2, 4).contiguous()
    v = v.permute(0, 1, 3, 2, 4).contiguous()
    q, k = q.to(v.dtype), k.to(v.dtype)

    y = flash_sdpa(
        q.view(
            num_layers * batch, attention.num_heads, seq, attention.head_dim
        ),
        k.view(
            num_layers * batch, attention.num_kv_heads, seq, attention.head_dim
        ),
        v.view(
            num_layers * batch, attention.num_kv_heads, seq, attention.head_dim
        ),
        dropout_p=attention.dropout if attention.training else 0.0,
        enable_gqa=attention.num_kv_heads != attention.num_heads,
    )
    y = y.view(num_layers, batch, attention.num_heads, seq, attention.head_dim)
    y = y.permute(0, 1, 3, 2, 4).contiguous()
    return _prepared_linear(
        y.view(
            num_layers, batch, seq, attention.num_heads * attention.head_dim
        ),
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
    raise ValueError(
        f"unsupported cache processor gate_mode: {prepared.gate_mode}"
    )


def run_batched_cache_processors(
    blocks, x: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Run the vectorized BT reference without retaining prepared stacks."""
    block_list = list(blocks)
    h = _linear(
        _rms_norm(x, [b.in_norm for b in block_list]), block_list, "in_proj"
    )
    for layer_idx in range(len(block_list[0].layers)):
        layers = [b.layers[layer_idx] for b in block_list]
        h = h + _attention(
            _rms_norm(h, [layer.input_norm for layer in layers]),
            layers,
            positions,
        )
        m = _rms_norm(h, [layer.post_norm for layer in layers])
        h = h + _linear(
            F.silu(_linear(m, layers, "gate_proj"))
            * _linear(m, layers, "up_proj"),
            layers,
            "down_proj",
        )
    delta = _linear(
        _rms_norm(h, [b.out_norm for b in block_list]), block_list, "out_proj"
    )
    return x + delta * _gate_scale(block_list, delta.dtype)


def _gate_scale(blocks, dtype: torch.dtype) -> torch.Tensor:
    gate = torch.stack([b.gate for b in blocks], dim=0).view(
        len(blocks), 1, 1, 1
    )
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


def _attention(
    x: torch.Tensor, layers, positions: torch.Tensor
) -> torch.Tensor:
    attns = [layer.attn for layer in layers]
    num_layers, batch, seq, _ = x.shape
    num_heads = attns[0].num_heads
    num_kv_heads = attns[0].num_kv_heads
    head_dim = attns[0].head_dim

    q = _linear(x, attns, "q_proj").view(
        num_layers, batch, seq, num_heads, head_dim
    )
    k = _linear(x, attns, "k_proj").view(
        num_layers, batch, seq, num_kv_heads, head_dim
    )
    v = _linear(x, attns, "v_proj").view(
        num_layers, batch, seq, num_kv_heads, head_dim
    )

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
    y = (
        y.view(num_layers, batch, num_heads, seq, head_dim)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    return _linear(
        y.view(num_layers, batch, seq, num_heads * head_dim), attns, "o_proj"
    )
