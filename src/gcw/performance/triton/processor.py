"""Benchmark-only fused cache-processor pointwise operations."""

from __future__ import annotations

from typing import Any

import torch

from gcw.performance.triton.runtime import load_kernel_module

_KERNEL_MODULE = "gcw.performance.triton._processor_kernels"


class _FusedSiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(gate, up)
        kernels = load_kernel_module(_KERNEL_MODULE, gate.device)
        return kernels.processor_silu_mul_forward(gate, up)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        gate, up = ctx.saved_tensors
        kernels = load_kernel_module(_KERNEL_MODULE, grad_output.device)
        return kernels.processor_silu_mul_backward(
            gate, up, grad_output.contiguous()
        )


def fused_processor_silu_mul(
    gate: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    """Fuse ``silu(gate) * up`` under one fixed Hopper launch policy."""
    if (
        gate.shape != up.shape
        or gate.dtype != up.dtype
        or gate.device != up.device
    ):
        raise ValueError(
            "fused processor SiLU-multiply inputs must have matching metadata"
        )
    if gate.dtype != torch.bfloat16 or gate.device.type != "cuda":
        raise ValueError(
            "fused processor SiLU-multiply requires CUDA BF16 inputs"
        )
    if not gate.is_contiguous() or not up.is_contiguous() or gate.numel() == 0:
        raise ValueError(
            "fused processor SiLU-multiply requires nonempty contiguous inputs"
        )
    return _FusedSiluMul.apply(gate, up)
