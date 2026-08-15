"""Autograd-connected wrappers for exact Triton cache movement."""

from __future__ import annotations

from typing import Any

import torch

from gcw.performance.triton.runtime import load_kernel_module

_KERNEL_MODULE = "gcw.performance.triton._cache_kernels"


class _GatherSharedKV(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        keys: torch.Tensor,
        values: torch.Tensor,
        indices: torch.Tensor,
        global_layout: bool,
    ) -> torch.Tensor:
        ctx.cache_shape = tuple(keys.shape)
        ctx.global_layout = global_layout
        ctx.save_for_backward(indices)
        layers, batch, heads, _history, head_dim = keys.shape
        if indices.shape[1] == 0:
            shape = (
                (batch, 0, layers * 2 * heads * head_dim) if global_layout else (layers, batch, 0, 2 * heads * head_dim)
            )
            return keys.new_empty(shape)
        kernels = load_kernel_module(_KERNEL_MODULE, keys.device)
        output_bits = kernels.gather_shared_kv_bits(
            keys.view(torch.int16),
            values.view(torch.int16),
            indices,
            global_layout=global_layout,
        )
        return output_bits.view(keys.dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        (indices,) = ctx.saved_tensors
        cache_shape = ctx.cache_shape
        if indices.shape[1] == 0:
            return (
                grad_output.new_zeros(cache_shape),
                grad_output.new_zeros(cache_shape),
                None,
                None,
            )
        kernels = load_kernel_module(_KERNEL_MODULE, grad_output.device)
        grad_key_bits, grad_value_bits = kernels.scatter_shared_kv_grad_bits(
            grad_output.contiguous().view(torch.int16),
            indices,
            cache_shape,
            global_layout=ctx.global_layout,
        )
        return (
            grad_key_bits.view(grad_output.dtype),
            grad_value_bits.view(grad_output.dtype),
            None,
            None,
        )


class _GatherPerCacheHeadKV(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        keys: torch.Tensor,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        ctx.cache_shape = tuple(keys.shape)
        ctx.save_for_backward(indices)
        if indices.shape[2] == 0:
            layers, batch, heads, _history, head_dim = keys.shape
            return keys.new_empty(batch, layers * heads, 0, 2 * head_dim)
        kernels = load_kernel_module(_KERNEL_MODULE, keys.device)
        output_bits = kernels.gather_per_cache_head_bits(
            keys.view(torch.int16),
            values.view(torch.int16),
            indices,
        )
        return output_bits.view(keys.dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        (indices,) = ctx.saved_tensors
        cache_shape = ctx.cache_shape
        if indices.shape[2] == 0:
            return (
                grad_output.new_zeros(cache_shape),
                grad_output.new_zeros(cache_shape),
                None,
            )
        contiguous_grad = grad_output.contiguous()
        kernels = load_kernel_module(_KERNEL_MODULE, grad_output.device)
        grad_key_bits, grad_value_bits = kernels.scatter_per_cache_head_grad_bits(
            contiguous_grad.view(torch.int16),
            indices,
            cache_shape,
        )
        return grad_key_bits.view(grad_output.dtype), grad_value_bits.view(grad_output.dtype), None


def gather_shared_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    *,
    layout: str,
    validate_indices: bool = True,
) -> torch.Tensor:
    """Gather shared selected positions directly into BT or BGT K/V layout.

    ``layout='layer'`` returns ``[layers, 2, selected, 2 * heads * dim]``.
    ``layout='global'`` returns ``[2, selected, layers * 2 * heads * dim]``.
    """
    _validate_cache_inputs(keys, values)
    if layout not in {"layer", "global"}:
        raise ValueError("Triton shared cache layout must be 'layer' or 'global'")
    if indices.device != keys.device or indices.dtype != torch.long:
        raise ValueError("Triton shared cache indices must be torch.long on the K/V device")
    if indices.ndim != 2 or indices.shape[0] != keys.shape[1] or not indices.is_contiguous():
        raise ValueError("Triton shared cache indices must be contiguous (2, selected)")
    if validate_indices:
        _validate_index_values(indices, keys.shape[3])
    return _GatherSharedKV.apply(keys, values, indices, layout == "global")


def gather_per_cache_head_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    *,
    validate_indices: bool = True,
) -> torch.Tensor:
    """Gather live layer-major K/V directly into per-cache-head layout.

    ``keys`` and ``values`` must be contiguous ``[layers, 2, heads, history,
    head_dim]`` BF16 tensors. ``indices`` may be shared ``[2, selected]`` or
    per-cache-head ``[2, layers * heads, selected]``. The result has shape
    ``[2, layers * heads, selected, 2 * head_dim]`` with key features first.

    Validation synchronizes when ``validate_indices`` is true. Production
    callers may disable repeated validation only after their internal selector
    contract has been qualified; shape, dtype, layout and runtime checks always
    remain active.
    """
    _validate_cache_inputs(keys, values)
    normalized = _normalize_indices(indices, keys)
    if validate_indices:
        _validate_index_values(normalized, keys.shape[3])
    return _GatherPerCacheHeadKV.apply(keys, values, normalized)


def _validate_cache_inputs(keys: torch.Tensor, values: torch.Tensor) -> None:
    if keys.ndim != 5 or keys.shape != values.shape:
        raise ValueError("Triton cache gather requires matching five-dimensional K/V")
    if keys.shape[1] != 2:
        raise ValueError("Triton cache gather requires fixed physical batch size 2")
    if keys.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
        raise TypeError("Triton cache gather currently supports BF16 K/V only")
    if keys.device != values.device or keys.device.type != "cuda":
        raise ValueError("Triton cache gather requires K/V on one CUDA device")
    if not keys.is_contiguous() or not values.is_contiguous():
        raise ValueError("Triton cache gather requires contiguous layer-major K/V")
    if keys.shape[0] <= 0 or keys.shape[2] <= 0 or keys.shape[3] <= 0 or keys.shape[4] <= 0:
        raise ValueError("Triton cache gather requires positive cache dimensions")


def _normalize_indices(indices: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    cache_heads = keys.shape[0] * keys.shape[2]
    if indices.device != keys.device:
        raise ValueError("Triton cache indices must be on the K/V device")
    if indices.dtype != torch.long:
        raise TypeError("Triton cache indices must use torch.long")
    if indices.ndim == 2:
        if indices.shape[0] != keys.shape[1]:
            raise ValueError("shared Triton cache indices must have shape (2, selected)")
        normalized = indices[:, None, :].expand(-1, cache_heads, -1)
    elif indices.ndim == 3:
        if indices.shape[:2] != (keys.shape[1], cache_heads):
            raise ValueError("per-head Triton cache indices must have shape (2, layers * heads, selected)")
        normalized = indices
    else:
        raise ValueError("Triton cache indices must be shared or per-cache-head")
    if normalized.stride(-1) != 1:
        raise ValueError("Triton cache indices must be contiguous in the selected dimension")
    return normalized


def _validate_index_values(indices: torch.Tensor, history: int) -> None:
    if indices.numel() == 0:
        return
    if not bool(torch.all((indices >= 0) & (indices < history)).item()):
        raise IndexError("Triton cache indices are outside the live history")
    if indices.shape[-1] > 1:
        differences = indices[..., 1:] - indices[..., :-1]
        if not bool(torch.all(differences > 0).item()):
            raise ValueError("Triton cache indices must be strictly increasing and unique")
