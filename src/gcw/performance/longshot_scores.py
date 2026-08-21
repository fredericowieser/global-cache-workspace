from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from gcw.models.cache_processors.ops import rotate_cache_keys

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
        _validate_input(
            x, reduction_start=x.ndim - 1, max_temp_bytes=max_temp_bytes
        )
        ctx.max_temp_bytes = max_temp_bytes
        ctx.save_for_backward(x)
        return _run_chunked(
            x, x.ndim - 1, max_temp_bytes, eager_fp32_squared_l2
        )

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, None]:
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
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, None]:
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


def _validate_input(
    x: torch.Tensor, *, reduction_start: int, max_temp_bytes: int
) -> None:
    if not x.is_floating_point():
        raise TypeError(
            "Longshot score reduction requires a floating-point tensor"
        )
    if reduction_start < 0 or reduction_start >= x.ndim:
        raise ValueError(
            "Longshot score reduction received an invalid tensor rank"
        )
    if not x.is_contiguous():
        raise ValueError(
            "Longshot score reduction requires contiguous gathered cache storage"
        )
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


def chunked_fp32_unroped_concatenated_kv_squared_l2(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    theta: float,
    rotary_style: str,
    rotary_dim: int,
    max_temp_bytes: int = LONGSHOT_SCORE_FP32_CHUNK_BYTES,
) -> torch.Tensor:
    """Score chunk-local inverse-RoPE K/V with the eager per-row oracle.

    The function rotates independent layer-major cache rows only inside the
    bounded score chunk. It never materializes an unrotated full history and
    never splits or reorders the feature reduction of one cache row.
    """
    return _ChunkedUnropedConcatenatedKVSquaredL2.apply(
        keys,
        values,
        theta,
        rotary_style,
        rotary_dim,
        max_temp_bytes,
    )


def unroped_score_estimated_temp_bytes_per_row(
    *,
    dtype: torch.dtype,
    head_dim: int,
    rotary_dim: int,
) -> int:
    """Return a conservative analytical bound for one score-row working set.

    This includes chunk-local inverse-RoPE output, trigonometric tensors and
    pointwise intermediates, merged K/V storage, and both FP32 score
    temporaries. Allocator peak telemetry remains authoritative.
    """
    if (
        head_dim <= 0
        or rotary_dim <= 0
        or rotary_dim > head_dim
        or rotary_dim % 2
    ):
        raise ValueError(
            "Longshot un-RoPE dimensions must be positive, even, and within head_dim"
        )
    element_bytes = torch.empty((), dtype=dtype).element_size()
    rotation_bytes = unrope_rotation_estimated_temp_bytes_per_row(
        dtype=dtype, head_dim=head_dim, rotary_dim=rotary_dim
    )
    merged_bytes = 2 * head_dim * element_bytes
    fp32_score_bytes = 4 * head_dim * _FP32_BYTES
    return rotation_bytes + merged_bytes + fp32_score_bytes


def unrope_rotation_estimated_temp_bytes_per_row(
    *,
    dtype: torch.dtype,
    head_dim: int,
    rotary_dim: int,
) -> int:
    """Conservatively bound inverse-RoPE output and pointwise row temporaries."""
    if (
        head_dim <= 0
        or rotary_dim <= 0
        or rotary_dim > head_dim
        or rotary_dim % 2
    ):
        raise ValueError(
            "Longshot un-RoPE dimensions must be positive, even, and within head_dim"
        )
    element_bytes = torch.empty((), dtype=dtype).element_size()
    dtype_elements = head_dim + 6 * rotary_dim
    fp32_elements = 2 * rotary_dim
    return dtype_elements * element_bytes + fp32_elements * _FP32_BYTES + 8


def unrope_rotation_estimated_fixed_temp_bytes(*, rotary_dim: int) -> int:
    """Bound fixed-size allocations made inside one ``rotate_cache_keys`` call.

    The inverse-frequency construction allocates FP32 arange, division, power,
    reciprocal, and scaling results. Scalar conversion overhead is included so
    tiny chunks are not reported as though rotation had only row-sized costs.
    """
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("Longshot rotary_dim must be positive and even")
    return 5 * (rotary_dim // 2) * _FP32_BYTES + 5 * _FP32_BYTES


def unrope_rotation_estimated_inner_allocation_bytes(
    rows: int,
    *,
    dtype: torch.dtype,
    head_dim: int,
    rotary_dim: int,
    calls: int = 1,
) -> int:
    """Bound allocations inside ``rotate_cache_keys``, including call overhead."""
    if rows < 0 or calls < 0 or (rows == 0) != (calls == 0):
        raise ValueError(
            "Longshot rotation rows and calls must both be zero or both be positive"
        )
    return rows * unrope_rotation_estimated_temp_bytes_per_row(
        dtype=dtype, head_dim=head_dim, rotary_dim=rotary_dim
    ) + calls * unrope_rotation_estimated_fixed_temp_bytes(
        rotary_dim=rotary_dim
    )


class _ChunkedUnropedConcatenatedKVSquaredL2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        keys: torch.Tensor,
        values: torch.Tensor,
        theta: float,
        rotary_style: str,
        rotary_dim: int,
        max_temp_bytes: int,
    ) -> torch.Tensor:
        _validate_layer_major_kv(keys, values, max_temp_bytes=max_temp_bytes)
        _validate_unrope(
            keys,
            rotary_dim=rotary_dim,
            max_temp_bytes=max_temp_bytes,
        )
        ctx.theta = theta
        ctx.rotary_style = rotary_style
        ctx.rotary_dim = rotary_dim
        ctx.max_temp_bytes = max_temp_bytes
        ctx.save_for_backward(keys, values)
        return _run_layer_major_unroped_kv_scores(
            keys,
            values,
            theta=theta,
            rotary_style=rotary_style,
            rotary_dim=rotary_dim,
            max_temp_bytes=max_temp_bytes,
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None, torch.Tensor | None, None, None, None, None
    ]:
        keys, values = ctx.saved_tensors
        key_grad, value_grad = _recomputed_layer_major_unroped_kv_backward(
            keys,
            values,
            grad_output,
            theta=ctx.theta,
            rotary_style=ctx.rotary_style,
            rotary_dim=ctx.rotary_dim,
            max_temp_bytes=ctx.max_temp_bytes,
            need_key_grad=ctx.needs_input_grad[0],
            need_value_grad=ctx.needs_input_grad[1],
        )
        return key_grad, value_grad, None, None, None, None


def _run_layer_major_unroped_kv_scores(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    theta: float,
    rotary_style: str,
    rotary_dim: int,
    max_temp_bytes: int,
) -> torch.Tensor:
    layers, batch, heads, seq, head_dim = keys.shape
    bytes_per_row = unroped_score_estimated_temp_bytes_per_row(
        dtype=keys.dtype,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
    )
    rows_per_chunk = max(1, max_temp_bytes // bytes_per_row)
    key_rows = keys.view(-1, head_dim)
    value_rows = values.view(-1, head_dim)
    score_rows = torch.empty(
        key_rows.shape[0], dtype=torch.float32, device=keys.device
    )
    for start in range(0, key_rows.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, key_rows.shape[0])
        positions = torch.arange(
            start, end, device=keys.device, dtype=torch.long
        ).remainder(seq)
        unroped_keys = rotate_cache_keys(
            key_rows[start:end],
            positions,
            theta=theta,
            rotary_style=rotary_style,
            inverse=True,
            rotary_dim=rotary_dim,
        )
        merged = torch.cat([unroped_keys, value_rows[start:end]], dim=-1)
        score_rows[start:end] = eager_fp32_squared_l2(merged)
    by_layer = score_rows.view(layers, batch, heads, seq)
    return by_layer.permute(1, 0, 2, 3).reshape(batch, layers * heads, seq)


def _recomputed_layer_major_unroped_kv_backward(
    keys: torch.Tensor,
    values: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    theta: float,
    rotary_style: str,
    rotary_dim: int,
    max_temp_bytes: int,
    need_key_grad: bool,
    need_value_grad: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    layers, batch, heads, seq, head_dim = keys.shape
    bytes_per_row = unroped_score_estimated_temp_bytes_per_row(
        dtype=keys.dtype,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
    )
    rows_per_chunk = max(1, max_temp_bytes // bytes_per_row)
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
            positions = torch.arange(
                start, end, device=keys.device, dtype=torch.long
            ).remainder(seq)
            unroped_keys = rotate_cache_keys(
                key_chunk,
                positions,
                theta=theta,
                rotary_style=rotary_style,
                inverse=True,
                rotary_dim=rotary_dim,
            )
            scores = eager_fp32_squared_l2(
                torch.cat([unroped_keys, value_chunk], dim=-1)
            )
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


def _validate_unrope(
    keys: torch.Tensor,
    *,
    rotary_dim: int,
    max_temp_bytes: int,
) -> None:
    bytes_per_row = unroped_score_estimated_temp_bytes_per_row(
        dtype=keys.dtype,
        head_dim=keys.shape[-1],
        rotary_dim=rotary_dim,
    )
    if max_temp_bytes < bytes_per_row:
        raise ValueError(
            "Longshot un-RoPE score max_temp_bytes must fit one complete rotation and score row "
            f"({bytes_per_row})"
        )


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
        return _run_layer_major_kv_scores(
            keys, values, max_temp_bytes=max_temp_bytes
        )

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
    rows_per_chunk = _rows_per_chunk(
        torch.Size((2 * head_dim,)), max_temp_bytes
    )
    key_rows = keys.view(-1, head_dim)
    value_rows = values.view(-1, head_dim)
    score_rows = torch.empty(
        key_rows.shape[0], dtype=torch.float32, device=keys.device
    )
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
    rows_per_chunk = _rows_per_chunk(
        torch.Size((2 * head_dim,)), max_temp_bytes
    )
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
            scores = eager_fp32_squared_l2(
                torch.cat([key_chunk, value_chunk], dim=-1)
            )
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
        raise ValueError(
            "Longshot cache keys and values must have identical shapes"
        )
    if not keys.is_floating_point() or not values.is_floating_point():
        raise TypeError(
            "Longshot cache keys and values must be floating-point tensors"
        )
    if keys.dtype != values.dtype or keys.device != values.device:
        raise ValueError(
            "Longshot cache keys and values must share dtype and device"
        )
    if not keys.is_contiguous() or not values.is_contiguous():
        raise ValueError(
            "Longshot layer-major cache tensors must be contiguous"
        )
    if max_temp_bytes <= 0:
        raise ValueError("Longshot score max_temp_bytes must be positive")
