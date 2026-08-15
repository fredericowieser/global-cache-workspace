from __future__ import annotations

import os

import pytest
import torch

from gcw.models.cache_processors.ops import (
    gather_cache_layers,
    gather_per_cache_head_kv as eager_gather,
    merge_global_kv,
    merge_layer_kv,
    scatter_cache_layers,
    split_global_kv,
    split_layer_kv,
)
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.performance.triton import (
    gather_per_cache_head_kv,
    gather_shared_kv,
    require_triton_runtime,
    scatter_shared_kv,
)

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
    eager = eager_gather(WindowedCache(keys, values, torch.empty(2, 0, device="cuda")), eager_indices)
    candidate = gather_per_cache_head_kv(keys, values, shared)
    assert torch.equal(eager.view(torch.int16), candidate.view(torch.int16))


@pytest.mark.parametrize("layout", ("layer", "global"))
@pytest.mark.parametrize(
    ("layers", "heads", "history", "head_dim", "selected"),
    ((3, 2, 19, 16, 4), (28, 2, 192, 128, 96)),
)
def test_triton_shared_gather_layout_and_backward_match_eager(
    layout: str,
    layers: int,
    heads: int,
    history: int,
    head_dim: int,
    selected: int,
) -> None:
    _require_cuda_gate()
    shape = (layers, 2, heads, history, head_dim)
    eager_keys = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    eager_values = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    candidate_keys = eager_keys.detach().clone().requires_grad_(True)
    candidate_values = eager_values.detach().clone().requires_grad_(True)
    indices = _indices(2, 1, selected, history)[:, 0].contiguous()
    cache = WindowedCache(eager_keys, eager_values, torch.empty(2, 0, device="cuda"))
    selected_keys, selected_values = gather_cache_layers(cache, indices)
    eager = (
        merge_layer_kv(selected_keys, selected_values)
        if layout == "layer"
        else merge_global_kv(selected_keys, selected_values)
    )
    candidate = gather_shared_kv(candidate_keys, candidate_values, indices, layout=layout)
    assert torch.equal(eager.view(torch.int16), candidate.view(torch.int16))

    upstream = torch.randn_like(eager)
    eager.backward(upstream)
    candidate.backward(upstream)
    assert eager_keys.grad is not None and candidate_keys.grad is not None
    assert eager_values.grad is not None and candidate_values.grad is not None
    assert torch.equal(eager_keys.grad.view(torch.int16), candidate_keys.grad.view(torch.int16))
    assert torch.equal(eager_values.grad.view(torch.int16), candidate_values.grad.view(torch.int16))


@pytest.mark.parametrize("layout", ("layer", "global"))
def test_triton_functional_split_scatter_and_backward_match_eager(layout: str) -> None:
    _require_cuda_gate()
    layers, heads, history, head_dim, selected = 4, 2, 23, 16, 7
    shape = (layers, 2, heads, history, head_dim)
    eager_keys = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    eager_values = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    candidate_keys = eager_keys.detach().clone().requires_grad_(True)
    candidate_values = eager_values.detach().clone().requires_grad_(True)
    indices = _indices(2, 1, selected, history)[:, 0].contiguous()
    update_shape = (
        (layers, 2, selected, 2 * heads * head_dim)
        if layout == "layer"
        else (2, selected, layers * 2 * heads * head_dim)
    )
    eager_update = torch.randn(update_shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    candidate_update = eager_update.detach().clone().requires_grad_(True)
    if layout == "layer":
        update_keys, update_values = split_layer_kv(eager_update, heads, head_dim)
    else:
        update_keys, update_values = split_global_kv(eager_update, layers, heads, head_dim)
    eager_cache = scatter_cache_layers(
        WindowedCache(eager_keys, eager_values, torch.empty(2, 0, device="cuda")),
        indices,
        update_keys,
        update_values,
    )
    assert isinstance(eager_cache, WindowedCache)
    candidate_output = scatter_shared_kv(
        candidate_keys,
        candidate_values,
        indices,
        candidate_update,
        layout=layout,
    )
    assert torch.equal(eager_cache.keys.view(torch.int16), candidate_output[0].view(torch.int16))
    assert torch.equal(eager_cache.values.view(torch.int16), candidate_output[1].view(torch.int16))

    key_upstream = torch.randn_like(eager_cache.keys)
    value_upstream = torch.randn_like(eager_cache.values)
    torch.autograd.backward((eager_cache.keys, eager_cache.values), (key_upstream, value_upstream))
    torch.autograd.backward(candidate_output, (key_upstream, value_upstream))
    assert eager_keys.grad is not None and candidate_keys.grad is not None
    assert eager_values.grad is not None and candidate_values.grad is not None
    assert eager_update.grad is not None and candidate_update.grad is not None
    assert torch.equal(eager_keys.grad.view(torch.int16), candidate_keys.grad.view(torch.int16))
    assert torch.equal(eager_values.grad.view(torch.int16), candidate_values.grad.view(torch.int16))
    assert torch.equal(eager_update.grad.view(torch.int16), candidate_update.grad.view(torch.int16))


def test_triton_cache_runtime_records_hopper_identity() -> None:
    _require_cuda_gate()
    identity = require_triton_runtime(torch.device("cuda"))
    assert identity.compute_capability == (9, 0)
    assert identity.triton_version == "3.6.0"
    assert identity.device_name in {"NVIDIA H100 80GB HBM3", "NVIDIA H200"}
