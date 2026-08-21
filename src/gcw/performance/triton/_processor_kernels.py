"""Private fixed-launch processor pointwise Triton kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from gcw.performance.kernel_manifest import CACHE_COPY_BLOCK, CACHE_NUM_WARPS


@triton.jit
def _silu_mul_forward_kernel(gate, up, output, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < elements
    gate_values = tl.load(gate + offsets, mask=valid, other=0.0).to(tl.float32)
    up_values = tl.load(up + offsets, mask=valid, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate_values))
    tl.store(output + offsets, gate_values * sigmoid * up_values, mask=valid)


@triton.jit
def _silu_mul_backward_kernel(
    gate, up, grad_output, grad_gate, grad_up, elements, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < elements
    gate_values = tl.load(gate + offsets, mask=valid, other=0.0).to(tl.float32)
    up_values = tl.load(up + offsets, mask=valid, other=0.0).to(tl.float32)
    output_gradient = tl.load(grad_output + offsets, mask=valid, other=0.0).to(
        tl.float32
    )
    sigmoid = 1.0 / (1.0 + tl.exp(-gate_values))
    silu_derivative = sigmoid * (1.0 + gate_values * (1.0 - sigmoid))
    tl.store(
        grad_gate + offsets,
        output_gradient * up_values * silu_derivative,
        mask=valid,
    )
    tl.store(
        grad_up + offsets, output_gradient * gate_values * sigmoid, mask=valid
    )


def processor_silu_mul_forward(
    gate: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    output = torch.empty_like(gate)
    grid = (triton.cdiv(gate.numel(), CACHE_COPY_BLOCK),)
    _silu_mul_forward_kernel[grid](
        gate,
        up,
        output,
        gate.numel(),
        BLOCK=CACHE_COPY_BLOCK,
        num_warps=CACHE_NUM_WARPS,
    )
    return output


def processor_silu_mul_backward(
    gate: torch.Tensor,
    up: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_gate = torch.empty_like(gate)
    grad_up = torch.empty_like(up)
    grid = (triton.cdiv(gate.numel(), CACHE_COPY_BLOCK),)
    _silu_mul_backward_kernel[grid](
        gate,
        up,
        grad_output,
        grad_gate,
        grad_up,
        gate.numel(),
        BLOCK=CACHE_COPY_BLOCK,
        num_warps=CACHE_NUM_WARPS,
    )
    return grad_gate, grad_up
