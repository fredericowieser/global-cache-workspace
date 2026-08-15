from __future__ import annotations

import os

import pytest
import torch

from gcw.models.cache_processors.ops import gather_per_cache_head_kv as eager_gather
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.performance.triton import gather_per_cache_head_kv, require_triton_runtime

pytestmark = [pytest.mark.gpu, pytest.mark.performance]


def _require_cuda_gate() -> None:
    if os.environ.get("GCW_RUN_TRITON_CACHE_CUDA_TESTS") != "1":
        pytest.skip("set GCW_RUN_TRITON_CACHE_CUDA_TESTS=1 for Triton cache tests")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")


def _indices(batch: int, cache_heads: int, selected: int, history: int) -> torch.Tensor:
    base = torch.arange(selected, device="cuda", dtype=torch.long) * max(1, history // selected)
    base = base.clamp_max(history - 1)
    rows = []
    for row in range(batch):
        heads = []
        for head in range(cache_heads):
            offset = (row + head) % max(1, history - int(base[-1]) if selected else 1)
            heads.append((base + offset).clamp_max(history - 1))
        rows.append(torch.stack(heads))
    return torch.stack(rows)


@pytest.mark.parametrize(
    ("layers", "heads", "history", "head_dim", "selected"),
    ((2, 1, 17, 13, 5), (4, 2, 64, 32, 9), (28, 2, 192, 128, 96)),
)
def test_triton_per_cache_head_gather_forward_backward_matches_eager(
    layers: int,
    heads: int,
    history: int,
    head_dim: int,
    selected: int,
) -> None:
    _require_cuda_gate()
    torch.manual_seed(1234)
    shape = (layers, 2, heads, history, head_dim)
    eager_keys = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    eager_values = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    candidate_keys = eager_keys.detach().clone().requires_grad_(True)
    candidate_values = eager_values.detach().clone().requires_grad_(True)
    indices = _indices(2, layers * heads, selected, history)

    eager = eager_gather(WindowedCache(eager_keys, eager_values, torch.empty(2, 0, device="cuda")), indices)
    candidate = gather_per_cache_head_kv(candidate_keys, candidate_values, indices)
    assert torch.equal(eager.view(torch.int16), candidate.view(torch.int16))

    upstream = torch.randn_like(eager)
    eager.backward(upstream)
    candidate.backward(upstream)
    assert eager_keys.grad is not None and candidate_keys.grad is not None
    assert eager_values.grad is not None and candidate_values.grad is not None
    assert torch.equal(eager_keys.grad.view(torch.int16), candidate_keys.grad.view(torch.int16))
    assert torch.equal(eager_values.grad.view(torch.int16), candidate_values.grad.view(torch.int16))


def test_triton_per_cache_head_gather_supports_shared_stride_zero_indices() -> None:
    _require_cuda_gate()
    keys = torch.randn(3, 2, 2, 19, 16, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    shared = torch.tensor([[0, 3, 11, 18], [1, 4, 12, 17]], device="cuda")
    eager_indices = shared[:, None, :].expand(-1, keys.shape[0] * keys.shape[2], -1)
    eager = eager_gather(
        WindowedCache(keys, values, torch.empty(2, 0, device="cuda")), eager_indices
    )
    candidate = gather_per_cache_head_kv(keys, values, shared)
    assert torch.equal(eager.view(torch.int16), candidate.view(torch.int16))


def test_triton_cache_runtime_records_hopper_identity() -> None:
    _require_cuda_gate()
    identity = require_triton_runtime(torch.device("cuda"))
    assert identity.compute_capability == (9, 0)
    assert identity.triton_version == "3.6.0"
    assert identity.device_name in {"NVIDIA H100 80GB HBM3", "NVIDIA H200"}
