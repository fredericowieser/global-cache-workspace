from __future__ import annotations

import pytest
import torch

from gcw.performance.triton import fused_processor_silu_mul


def test_fused_processor_silu_mul_rejects_non_cuda_or_mismatched_inputs() -> (
    None
):
    with pytest.raises(ValueError, match="CUDA BF16"):
        fused_processor_silu_mul(torch.ones(2, 3), torch.ones(2, 3))
    with pytest.raises(ValueError, match="matching metadata"):
        fused_processor_silu_mul(
            torch.ones(2, 3, dtype=torch.bfloat16),
            torch.ones(3, 2, dtype=torch.bfloat16),
        )
