from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch


# Bounds each FP32 cast and elementwise square. A BF16 score chunk can therefore
# hold at most two 32 MiB FP32 temporaries before its reduction releases them.
LONGSHOT_SCORE_FP32_CHUNK_BYTES = 32 * 1024 * 1024
_FP32_BYTES = 4


def eager_fp32_squared_l2(x: torch.Tensor) -> torch.Tensor:
    """Reference Longshot per-position score; keep this expression unchanged."""
    return x.float().pow(2).sum(dim=-1)


def eager_fp32_l2_norm(x: torch.Tensor) -> torch.Tensor:
    """Reference Longshot cache-head score; keep this expression unchanged."""
    return x.float().flatten(start_dim=2).norm(dim=-1)


def _eager_fp32_l2_norm_rows(x: torch.Tensor) -> torch.Tensor:
    return x.float().flatten(start_dim=1).norm(dim=-1)


def chunked_fp32_squared_l2(
    x: torch.Tensor,
    *,
    max_temp_bytes: int = LONGSHOT_SCORE_FP32_CHUNK_BYTES,
) -> torch.Tensor:
    """Evaluate the reference FP32 squared-L2 score in bounded outer-row chunks."""
    return _ChunkedSquaredL2.apply(x, max_temp_bytes)


def chunked_fp32_l2_norm(
    x: torch.Tensor,
    *,
    max_temp_bytes: int = LONGSHOT_SCORE_FP32_CHUNK_BYTES,
) -> torch.Tensor:
    """Evaluate the reference FP32 flattened L2 norm in bounded cache-head chunks."""
    return _ChunkedL2Norm.apply(x, max_temp_bytes)


class _ChunkedSquaredL2(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, max_temp_bytes: int) -> torch.Tensor:
        _validate_input(x, reduction_start=x.ndim - 1, max_temp_bytes=max_temp_bytes)
        ctx.max_temp_bytes = max_temp_bytes
        ctx.save_for_backward(x)
        return _run_chunked(x, x.ndim - 1, max_temp_bytes, eager_fp32_squared_l2)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None]:
        (x,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None
        return (
            _recomputed_chunked_backward(
                x,
                grad_output,
                reduction_start=x.ndim - 1,
                max_temp_bytes=ctx.max_temp_bytes,
                oracle=eager_fp32_squared_l2,
            ),
            None,
        )


class _ChunkedL2Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, max_temp_bytes: int) -> torch.Tensor:
        _validate_input(x, reduction_start=2, max_temp_bytes=max_temp_bytes)
        ctx.max_temp_bytes = max_temp_bytes
        ctx.save_for_backward(x)
        return _run_chunked(x, 2, max_temp_bytes, _eager_fp32_l2_norm_rows)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None]:
        (x,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None
        return (
            _recomputed_chunked_backward(
                x,
                grad_output,
                reduction_start=2,
                max_temp_bytes=ctx.max_temp_bytes,
                oracle=_eager_fp32_l2_norm_rows,
            ),
            None,
        )


def _run_chunked(
    x: torch.Tensor,
    reduction_start: int,
    max_temp_bytes: int,
    oracle: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    outer_shape = x.shape[:reduction_start]
    reduction_shape = x.shape[reduction_start:]
    rows = x.view(math.prod(outer_shape), *reduction_shape)
    rows_per_chunk = _rows_per_chunk(reduction_shape, max_temp_bytes)
    result = torch.empty(rows.shape[0], dtype=torch.float32, device=x.device)
    for start in range(0, rows.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, rows.shape[0])
        result[start:end] = oracle(rows[start:end])
    return result.view(outer_shape)


def _recomputed_chunked_backward(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    reduction_start: int,
    max_temp_bytes: int,
    oracle: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    outer_shape = x.shape[:reduction_start]
    reduction_shape = x.shape[reduction_start:]
    rows = x.view(math.prod(outer_shape), *reduction_shape)
    grad_rows = grad_output.contiguous().view(-1)
    result = torch.empty_like(rows)
    rows_per_chunk = _rows_per_chunk(reduction_shape, max_temp_bytes)
    create_graph = torch.is_grad_enabled()
    for start in range(0, rows.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, rows.shape[0])
        with torch.enable_grad():
            chunk = rows[start:end]
            scores = oracle(chunk)
            (chunk_grad,) = torch.autograd.grad(
                scores,
                chunk,
                grad_rows[start:end],
                create_graph=create_graph,
            )
        result[start:end] = chunk_grad
    return result.view_as(x)


def _validate_input(x: torch.Tensor, *, reduction_start: int, max_temp_bytes: int) -> None:
    if not x.is_floating_point():
        raise TypeError("Longshot score reduction requires a floating-point tensor")
    if reduction_start < 0 or reduction_start >= x.ndim:
        raise ValueError("Longshot score reduction received an invalid tensor rank")
    if not x.is_contiguous():
        raise ValueError("Longshot score reduction requires contiguous gathered cache storage")
    if max_temp_bytes <= 0:
        raise ValueError("Longshot score max_temp_bytes must be positive")


def _rows_per_chunk(reduction_shape: torch.Size, max_temp_bytes: int) -> int:
    fp32_bytes_per_row = math.prod(reduction_shape) * _FP32_BYTES
    return max(1, max_temp_bytes // max(fp32_bytes_per_row, 1))


def chunked_fp32_concatenated_kv_squared_l2(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    max_temp_bytes: int = LONGSHOT_SCORE_FP32_CHUNK_BYTES,
) -> torch.Tensor:
    """Score layer-major K/V without materializing the full merged cache.

    Each feature reduction remains exactly
    ``cat([key, value]).float().pow(2).sum(-1)``. Only independent outer
    rows are chunked.
    """
    return _ChunkedConcatenatedKVSquaredL2.apply(keys, values, max_temp_bytes)


class _ChunkedConcatenatedKVSquaredL2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        keys: torch.Tensor,
        values: torch.Tensor,
        max_temp_bytes: int,
    ) -> torch.Tensor:
        _validate_layer_major_kv(keys, values, max_temp_bytes=max_temp_bytes)
        ctx.max_temp_bytes = max_temp_bytes
        ctx.save_for_backward(keys, values)
        return _run_layer_major_kv_scores(keys, values, max_temp_bytes=max_temp_bytes)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        keys, values = ctx.saved_tensors
        key_grad, value_grad = _recomputed_layer_major_kv_backward(
            keys,
            values,
            grad_output,
            max_temp_bytes=ctx.max_temp_bytes,
            need_key_grad=ctx.needs_input_grad[0],
            need_value_grad=ctx.needs_input_grad[1],
        )
        return key_grad, value_grad, None


def _run_layer_major_kv_scores(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    max_temp_bytes: int,
) -> torch.Tensor:
    layers, batch, heads, seq, head_dim = keys.shape
    rows_per_chunk = _rows_per_chunk(torch.Size((2 * head_dim,)), max_temp_bytes)
    key_rows = keys.view(-1, head_dim)
    value_rows = values.view(-1, head_dim)
    score_rows = torch.empty(key_rows.shape[0], dtype=torch.float32, device=keys.device)
    for start in range(0, key_rows.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, key_rows.shape[0])
        merged = torch.cat([key_rows[start:end], value_rows[start:end]], dim=-1)
        score_rows[start:end] = eager_fp32_squared_l2(merged)
    by_layer = score_rows.view(layers, batch, heads, seq)
    return by_layer.permute(1, 0, 2, 3).reshape(batch, layers * heads, seq)


def _recomputed_layer_major_kv_backward(
    keys: torch.Tensor,
    values: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    max_temp_bytes: int,
    need_key_grad: bool,
    need_value_grad: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    layers, batch, heads, seq, head_dim = keys.shape
    rows_per_chunk = _rows_per_chunk(torch.Size((2 * head_dim,)), max_temp_bytes)
    grad_rows = (
        grad_output.reshape(batch, layers, heads, seq)
        .permute(1, 0, 2, 3)
        .contiguous()
        .view(-1)
    )
    key_rows = keys.view(-1, head_dim)
    value_rows = values.view(-1, head_dim)
    key_result = torch.empty_like(key_rows) if need_key_grad else None
    value_result = torch.empty_like(value_rows) if need_value_grad else None
    create_graph = torch.is_grad_enabled()
    for start in range(0, key_rows.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, key_rows.shape[0])
        with torch.enable_grad():
            key_chunk = key_rows[start:end]
            value_chunk = value_rows[start:end]
            if not key_chunk.requires_grad:
                key_chunk = key_chunk.detach().requires_grad_(True)
            if not value_chunk.requires_grad:
                value_chunk = value_chunk.detach().requires_grad_(True)
            scores = eager_fp32_squared_l2(torch.cat([key_chunk, value_chunk], dim=-1))
            key_chunk_grad, value_chunk_grad = torch.autograd.grad(
                scores,
                (key_chunk, value_chunk),
                grad_rows[start:end],
                create_graph=create_graph,
            )
        if key_result is not None:
            key_result[start:end] = key_chunk_grad
        if value_result is not None:
            value_result[start:end] = value_chunk_grad
    return (
        None if key_result is None else key_result.view_as(keys),
        None if value_result is None else value_result.view_as(values),
    )


def _validate_layer_major_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    max_temp_bytes: int,
) -> None:
    if keys.ndim != 5:
        raise ValueError(
            "Longshot layer-major cache tensors must have shape "
            "(layers, batch, heads, seq, dim)"
        )
    if keys.shape != values.shape:
        raise ValueError("Longshot cache keys and values must have identical shapes")
    if not keys.is_floating_point() or not values.is_floating_point():
        raise TypeError("Longshot cache keys and values must be floating-point tensors")
    if keys.dtype != values.dtype or keys.device != values.device:
        raise ValueError("Longshot cache keys and values must share dtype and device")
    if not keys.is_contiguous() or not values.is_contiguous():
        raise ValueError("Longshot layer-major cache tensors must be contiguous")
    if max_temp_bytes <= 0:
        raise ValueError("Longshot score max_temp_bytes must be positive")
