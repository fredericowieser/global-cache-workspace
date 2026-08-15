"""Private Triton JIT kernels for exact cache movement.

This module is imported lazily by :mod:`gcw.performance.triton.cache_movement`.
Payload tensors are presented as signed 16-bit integer views so BF16 bit
patterns are copied without floating-point conversion.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_per_cache_head_bits_kernel(
    keys,
    values,
    indices,
    output,
    history,
    index_stride_batch,
    index_stride_head,
    index_stride_selected,
    LAYERS: tl.constexpr,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    SELECTED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    feature_block = tl.program_id(0)
    row = tl.program_id(1)
    selected_index = row % SELECTED
    cache_head = (row // SELECTED) % (LAYERS * HEADS)
    batch = row // (SELECTED * LAYERS * HEADS)
    layer = cache_head // HEADS
    head = cache_head % HEADS

    source_position = tl.load(
        indices + batch * index_stride_batch + cache_head * index_stride_head + selected_index * index_stride_selected
    )
    features = feature_block * BLOCK + tl.arange(0, BLOCK)
    valid = features < 2 * HEAD_DIM
    dimension = features % HEAD_DIM
    source_offset = (((layer * BATCH + batch) * HEADS + head) * history + source_position) * HEAD_DIM + dimension
    key_bits = tl.load(keys + source_offset, mask=valid & (features < HEAD_DIM), other=0)
    value_bits = tl.load(values + source_offset, mask=valid & (features >= HEAD_DIM), other=0)
    bits = tl.where(features < HEAD_DIM, key_bits, value_bits)
    output_offset = row * (2 * HEAD_DIM) + features
    tl.store(output + output_offset, bits, mask=valid)


@triton.jit
def _gather_shared_kv_bits_kernel(
    keys,
    values,
    indices,
    output,
    history,
    LAYERS: tl.constexpr,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    SELECTED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GLOBAL_LAYOUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    feature_block = tl.program_id(0)
    row = tl.program_id(1)
    if GLOBAL_LAYOUT:
        selected_index = row % SELECTED
        batch = row // SELECTED
        features = feature_block * BLOCK + tl.arange(0, BLOCK)
        valid = features < LAYERS * 2 * HEADS * HEAD_DIM
        layer_width = 2 * HEADS * HEAD_DIM
        layer = features // layer_width
        within_layer = features % layer_width
    else:
        selected_index = row % SELECTED
        batch = (row // SELECTED) % BATCH
        layer = row // (SELECTED * BATCH)
        features = feature_block * BLOCK + tl.arange(0, BLOCK)
        valid = features < 2 * HEADS * HEAD_DIM
        within_layer = features
    stream = within_layer // (HEADS * HEAD_DIM)
    within_stream = within_layer % (HEADS * HEAD_DIM)
    head = within_stream // HEAD_DIM
    dimension = within_stream % HEAD_DIM
    source_position = tl.load(indices + batch * SELECTED + selected_index)
    source_offset = (((layer * BATCH + batch) * HEADS + head) * history + source_position) * HEAD_DIM + dimension
    key_bits = tl.load(keys + source_offset, mask=valid & (stream == 0), other=0)
    value_bits = tl.load(values + source_offset, mask=valid & (stream == 1), other=0)
    bits = tl.where(stream == 0, key_bits, value_bits)
    if GLOBAL_LAYOUT:
        output_offset = row * (LAYERS * 2 * HEADS * HEAD_DIM) + features
    else:
        output_offset = row * (2 * HEADS * HEAD_DIM) + features
    tl.store(output + output_offset, bits, mask=valid)


@triton.jit
def _copy_kv_bits_kernel(source_keys, source_values, target_keys, target_values, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < elements
    tl.store(target_keys + offsets, tl.load(source_keys + offsets, mask=valid, other=0), mask=valid)
    tl.store(target_values + offsets, tl.load(source_values + offsets, mask=valid, other=0), mask=valid)


@triton.jit
def _scatter_shared_update_bits_kernel(
    update,
    indices,
    target_keys,
    target_values,
    history,
    LAYERS: tl.constexpr,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    SELECTED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GLOBAL_LAYOUT: tl.constexpr,
    ZERO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    feature_block = tl.program_id(0)
    row = tl.program_id(1)
    if GLOBAL_LAYOUT:
        selected_index = row % SELECTED
        batch = row // SELECTED
        features = feature_block * BLOCK + tl.arange(0, BLOCK)
        valid = features < LAYERS * 2 * HEADS * HEAD_DIM
        layer_width = 2 * HEADS * HEAD_DIM
        layer = features // layer_width
        within_layer = features % layer_width
        update_offset = row * (LAYERS * 2 * HEADS * HEAD_DIM) + features
    else:
        selected_index = row % SELECTED
        batch = (row // SELECTED) % BATCH
        layer = row // (SELECTED * BATCH)
        features = feature_block * BLOCK + tl.arange(0, BLOCK)
        valid = features < 2 * HEADS * HEAD_DIM
        within_layer = features
        update_offset = row * (2 * HEADS * HEAD_DIM) + features
    stream = within_layer // (HEADS * HEAD_DIM)
    within_stream = within_layer % (HEADS * HEAD_DIM)
    head = within_stream // HEAD_DIM
    dimension = within_stream % HEAD_DIM
    target_position = tl.load(indices + batch * SELECTED + selected_index)
    cache_offset = (((layer * BATCH + batch) * HEADS + head) * history + target_position) * HEAD_DIM + dimension
    bits = 0 if ZERO else tl.load(update + update_offset, mask=valid, other=0)
    tl.store(target_keys + cache_offset, bits, mask=valid & (stream == 0))
    tl.store(target_values + cache_offset, bits, mask=valid & (stream == 1))


@triton.jit
def _scatter_shared_kv_grad_bits_kernel(
    grad_output,
    indices,
    grad_keys,
    grad_values,
    history,
    LAYERS: tl.constexpr,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    SELECTED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GLOBAL_LAYOUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    feature_block = tl.program_id(0)
    row = tl.program_id(1)
    if GLOBAL_LAYOUT:
        selected_index = row % SELECTED
        batch = row // SELECTED
        features = feature_block * BLOCK + tl.arange(0, BLOCK)
        valid = features < LAYERS * 2 * HEADS * HEAD_DIM
        layer_width = 2 * HEADS * HEAD_DIM
        layer = features // layer_width
        within_layer = features % layer_width
        output_offset = row * (LAYERS * 2 * HEADS * HEAD_DIM) + features
    else:
        selected_index = row % SELECTED
        batch = (row // SELECTED) % BATCH
        layer = row // (SELECTED * BATCH)
        features = feature_block * BLOCK + tl.arange(0, BLOCK)
        valid = features < 2 * HEADS * HEAD_DIM
        within_layer = features
        output_offset = row * (2 * HEADS * HEAD_DIM) + features
    stream = within_layer // (HEADS * HEAD_DIM)
    within_stream = within_layer % (HEADS * HEAD_DIM)
    head = within_stream // HEAD_DIM
    dimension = within_stream % HEAD_DIM
    source_position = tl.load(indices + batch * SELECTED + selected_index)
    cache_offset = (((layer * BATCH + batch) * HEADS + head) * history + source_position) * HEAD_DIM + dimension
    bits = tl.load(grad_output + output_offset, mask=valid, other=0)
    tl.store(grad_keys + cache_offset, bits, mask=valid & (stream == 0))
    tl.store(grad_values + cache_offset, bits, mask=valid & (stream == 1))


@triton.jit
def _scatter_per_cache_head_grad_bits_kernel(
    grad_output,
    indices,
    grad_keys,
    grad_values,
    history,
    index_stride_batch,
    index_stride_head,
    index_stride_selected,
    LAYERS: tl.constexpr,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    SELECTED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    feature_block = tl.program_id(0)
    row = tl.program_id(1)
    selected_index = row % SELECTED
    cache_head = (row // SELECTED) % (LAYERS * HEADS)
    batch = row // (SELECTED * LAYERS * HEADS)
    layer = cache_head // HEADS
    head = cache_head % HEADS

    source_position = tl.load(
        indices + batch * index_stride_batch + cache_head * index_stride_head + selected_index * index_stride_selected
    )
    dimensions = feature_block * BLOCK + tl.arange(0, BLOCK)
    valid = dimensions < HEAD_DIM
    cache_offset = (((layer * BATCH + batch) * HEADS + head) * history + source_position) * HEAD_DIM + dimensions
    output_row_offset = row * (2 * HEAD_DIM)
    key_bits = tl.load(grad_output + output_row_offset + dimensions, mask=valid, other=0)
    value_bits = tl.load(
        grad_output + output_row_offset + HEAD_DIM + dimensions,
        mask=valid,
        other=0,
    )
    tl.store(grad_keys + cache_offset, key_bits, mask=valid)
    tl.store(grad_values + cache_offset, value_bits, mask=valid)


def gather_per_cache_head_bits(
    keys: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Launch the raw-bit per-cache-head gather."""
    layers, batch, heads, _history, head_dim = keys.shape
    selected = indices.shape[2]
    output = torch.empty(
        batch,
        layers * heads,
        selected,
        2 * head_dim,
        dtype=torch.int16,
        device=keys.device,
    )
    block = 128
    grid = (triton.cdiv(2 * head_dim, block), batch * layers * heads * selected)
    _gather_per_cache_head_bits_kernel[grid](
        keys,
        values,
        indices,
        output,
        keys.shape[3],
        indices.stride(0),
        indices.stride(1),
        indices.stride(2),
        LAYERS=layers,
        BATCH=batch,
        HEADS=heads,
        SELECTED=selected,
        HEAD_DIM=head_dim,
        BLOCK=block,
        num_warps=4,
    )
    return output


def copy_kv_bits(
    source_keys: torch.Tensor,
    source_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy complete raw K/V payloads into functional outputs."""
    target_keys = torch.empty_like(source_keys)
    target_values = torch.empty_like(source_values)
    block = 256
    grid = (triton.cdiv(source_keys.numel(), block),)
    _copy_kv_bits_kernel[grid](
        source_keys,
        source_values,
        target_keys,
        target_values,
        source_keys.numel(),
        BLOCK=block,
        num_warps=4,
    )
    return target_keys, target_values


def scatter_shared_update_bits(
    update: torch.Tensor,
    indices: torch.Tensor,
    target_keys: torch.Tensor,
    target_values: torch.Tensor,
    *,
    global_layout: bool,
    zero: bool = False,
) -> None:
    """Scatter merged updates, or zeros, into selected cache positions."""
    layers, batch, heads, history, head_dim = target_keys.shape
    selected = indices.shape[1]
    block = 128
    features = layers * 2 * heads * head_dim if global_layout else 2 * heads * head_dim
    rows = batch * selected if global_layout else layers * batch * selected
    grid = (triton.cdiv(features, block), rows)
    _scatter_shared_update_bits_kernel[grid](
        update,
        indices,
        target_keys,
        target_values,
        history,
        LAYERS=layers,
        BATCH=batch,
        HEADS=heads,
        SELECTED=selected,
        HEAD_DIM=head_dim,
        GLOBAL_LAYOUT=global_layout,
        ZERO=zero,
        BLOCK=block,
        num_warps=4,
    )


def gather_shared_kv_bits(
    keys: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    *,
    global_layout: bool,
) -> torch.Tensor:
    """Launch raw-bit shared-index gather into BT or BGT layout."""
    layers, batch, heads, _history, head_dim = keys.shape
    selected = indices.shape[1]
    shape = (
        (batch, selected, layers * 2 * heads * head_dim)
        if global_layout
        else (layers, batch, selected, 2 * heads * head_dim)
    )
    output = torch.empty(shape, dtype=torch.int16, device=keys.device)
    block = 128
    features = layers * 2 * heads * head_dim if global_layout else 2 * heads * head_dim
    rows = batch * selected if global_layout else layers * batch * selected
    grid = (triton.cdiv(features, block), rows)
    _gather_shared_kv_bits_kernel[grid](
        keys,
        values,
        indices,
        output,
        keys.shape[3],
        LAYERS=layers,
        BATCH=batch,
        HEADS=heads,
        SELECTED=selected,
        HEAD_DIM=head_dim,
        GLOBAL_LAYOUT=global_layout,
        BLOCK=block,
        num_warps=4,
    )
    return output


def scatter_shared_kv_grad_bits(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    cache_shape: tuple[int, int, int, int, int],
    *,
    global_layout: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch inverse raw-bit movement for BT/BGT gather backward."""
    layers, batch, heads, history, head_dim = cache_shape
    selected = indices.shape[1]
    grad_keys = torch.zeros(cache_shape, dtype=torch.int16, device=grad_output.device)
    grad_values = torch.zeros_like(grad_keys)
    block = 128
    features = layers * 2 * heads * head_dim if global_layout else 2 * heads * head_dim
    rows = batch * selected if global_layout else layers * batch * selected
    grid = (triton.cdiv(features, block), rows)
    _scatter_shared_kv_grad_bits_kernel[grid](
        grad_output,
        indices,
        grad_keys,
        grad_values,
        history,
        LAYERS=layers,
        BATCH=batch,
        HEADS=heads,
        SELECTED=selected,
        HEAD_DIM=head_dim,
        GLOBAL_LAYOUT=global_layout,
        BLOCK=block,
        num_warps=4,
    )
    return grad_keys, grad_values


def scatter_per_cache_head_grad_bits(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    cache_shape: tuple[int, int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the inverse raw-bit movement used by gather backward."""
    layers, batch, heads, history, head_dim = cache_shape
    selected = indices.shape[2]
    grad_keys = torch.zeros(cache_shape, dtype=torch.int16, device=grad_output.device)
    grad_values = torch.zeros_like(grad_keys)
    block = 128
    grid = (triton.cdiv(head_dim, block), batch * layers * heads * selected)
    _scatter_per_cache_head_grad_bits_kernel[grid](
        grad_output,
        indices,
        grad_keys,
        grad_values,
        history,
        indices.stride(0),
        indices.stride(1),
        indices.stride(2),
        LAYERS=layers,
        BATCH=batch,
        HEADS=heads,
        SELECTED=selected,
        HEAD_DIM=head_dim,
        BLOCK=block,
        num_warps=4,
    )
    return grad_keys, grad_values
