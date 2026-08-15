"""State-neutral Longshot cache access implementations."""

from __future__ import annotations

import torch


def gather_windowed_per_cache_head_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather selected layer/head histories without flattening the full cache.

    Args:
        keys: Layer-major keys with shape ``[layers, batch, heads, seq, dim]``.
        values: Values with the same shape as ``keys``.
        indices: Per-cache-head positions with shape
            ``[batch, layers * heads, selected]``.

    Returns:
        Selected keys and values with shape
        ``[batch, layers * heads, selected, dim]``.
    """
    _validate_inputs(keys, values, indices)
    layers, batch, heads, _, head_dim = keys.shape
    selected = indices.shape[-1]
    by_layer = indices.reshape(batch, layers, heads, selected).permute(1, 0, 2, 3)
    gather_index = by_layer[..., None].expand(layers, batch, heads, selected, head_dim)
    selected_keys = torch.gather(keys, dim=3, index=gather_index)
    selected_values = torch.gather(values, dim=3, index=gather_index)
    output_shape = (batch, layers * heads, selected, head_dim)
    return (
        selected_keys.permute(1, 0, 2, 3, 4).reshape(output_shape),
        selected_values.permute(1, 0, 2, 3, 4).reshape(output_shape),
    )


def _validate_inputs(keys: torch.Tensor, values: torch.Tensor, indices: torch.Tensor) -> None:
    if keys.ndim != 5:
        raise ValueError("Longshot windowed cache tensors must have shape (layers, batch, heads, seq, dim)")
    if keys.shape != values.shape:
        raise ValueError("Longshot cache keys and values must have identical shapes")
    if keys.dtype != values.dtype or keys.device != values.device:
        raise ValueError("Longshot cache keys and values must share dtype and device")
    if indices.ndim != 3:
        raise ValueError("Longshot cache indices must have shape (batch, cache_heads, selected)")
    layers, batch, heads, _, _ = keys.shape
    if indices.shape[:2] != (batch, layers * heads):
        raise ValueError("Longshot cache indices do not match the layer-major cache shape")
    if indices.dtype != torch.long or indices.device != keys.device:
        raise ValueError("Longshot cache indices must be long tensors on the cache device")
