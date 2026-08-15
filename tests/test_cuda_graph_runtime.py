from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from gcw.performance.backends.qwen3 import validate_qwen3_backbone_chunk_backend
from gcw.performance.cuda_graphs import (
    clear_cuda_graph_runtime,
    cuda_graph_capture_stats,
    run_graphed_callable,
    tensor_graph_signature,
)


def test_tensor_graph_signature_binds_shape_stride_dtype_device_and_grad() -> None:
    contiguous = torch.zeros(2, 3, requires_grad=True)
    transposed = torch.zeros(3, 2).t()
    first = tensor_graph_signature("region", (contiguous,))
    second = tensor_graph_signature("region", (transposed,))
    assert first != second
    assert first.tensors[0][0] == (2, 3)
    assert first.tensors[0][-1] is True


def test_cuda_graph_backend_is_quarantined_without_benchmark_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("GCW_ALLOW_UNQUALIFIED_CUDA_GRAPH", raising=False)
    with pytest.raises(RuntimeError, match="benchmark-only"):
        validate_qwen3_backbone_chunk_backend("cuda_graph")
    monkeypatch.setenv("GCW_ALLOW_UNQUALIFIED_CUDA_GRAPH", "1")
    validate_qwen3_backbone_chunk_backend("cuda_graph")


def test_cuda_graph_request_fails_closed_on_cpu() -> None:
    owner = nn.Linear(3, 2)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        run_graphed_callable(
            owner,
            region="linear",
            operation=owner,
            args=(torch.zeros(1, 3),),
        )
    assert cuda_graph_capture_stats(owner) == ()


def test_cuda_graph_runtime_is_external_to_copy_and_state_dict() -> None:
    owner = nn.Linear(3, 2)
    copied = copy.deepcopy(owner)
    assert owner.state_dict().keys() == copied.state_dict().keys()
    assert cuda_graph_capture_stats(owner) == cuda_graph_capture_stats(copied) == ()
    clear_cuda_graph_runtime(owner)
