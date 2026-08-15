from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from gcw.performance.triton import fused_processor_silu_mul

pytestmark = [pytest.mark.gpu, pytest.mark.performance]


def _require_gate() -> None:
    if os.environ.get("GCW_RUN_TRITON_PROCESSOR_TESTS") != "1":
        pytest.skip("set GCW_RUN_TRITON_PROCESSOR_TESTS=1")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("processor Triton gate requires H100/H200")


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((actual - expected).float())
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(torch.finfo(torch.float32).tiny)
    return float((numerator / denominator).cpu())


def test_fused_processor_silu_mul_forward_backward_matches_bf16_oracle() -> None:
    _require_gate()
    generator = torch.Generator(device="cuda").manual_seed(913)
    shape = (28, 2, 96, 384)
    reference_gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator).requires_grad_(True)
    reference_up = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator).requires_grad_(True)
    candidate_gate = reference_gate.detach().clone().requires_grad_(True)
    candidate_up = reference_up.detach().clone().requires_grad_(True)
    upstream = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    reference = F.silu(reference_gate) * reference_up
    candidate = fused_processor_silu_mul(candidate_gate, candidate_up)
    reference.backward(upstream)
    candidate.backward(upstream)
    assert reference_gate.grad is not None and candidate_gate.grad is not None
    assert reference_up.grad is not None and candidate_up.grad is not None
    torch.testing.assert_close(candidate, reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(candidate_gate.grad, reference_gate.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(candidate_up.grad, reference_up.grad, atol=2e-2, rtol=2e-2)
    assert _relative_l2(candidate, reference) <= 5e-4
    assert _relative_l2(candidate_gate.grad, reference_gate.grad) <= 5e-4
    assert _relative_l2(candidate_up.grad, reference_up.grad) <= 5e-4
