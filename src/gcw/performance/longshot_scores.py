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
