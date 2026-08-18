from __future__ import annotations

import copy
import os

import pytest
import torch
from torch import nn

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models import build_model
from gcw.models.model import LanguageModel
from gcw.performance.backends.qwen3 import configure_qwen3_backbone_chunk_backend
from gcw.performance.cuda_graphs import (
    clear_cuda_graph_runtime,
    cuda_graph_capture_stats,
    cuda_graph_replay_counts,
    run_graphed_callable,
)
from gcw.training.model_setup import _apply_trainable_policy

pytestmark = [pytest.mark.gpu, pytest.mark.performance]


def _require_cuda_gate() -> None:
    if os.environ.get("GCW_RUN_CUDA_GRAPH_RUNTIME_TESTS") != "1":
        pytest.skip("set GCW_RUN_CUDA_GRAPH_RUNTIME_TESTS=1 for CUDA Graph tests")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("CUDA Graph tests require H100/H200")


def test_graphed_callable_replays_distinct_inputs_and_matching_backward() -> None:
    _require_cuda_gate()
    torch.manual_seed(9)
    eager = nn.Linear(16, 16, bias=False, device="cuda").requires_grad_(False)
    candidate = copy.deepcopy(eager)
    for seed in (1, 2):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        eager_input = torch.randn(8, 16, device="cuda", generator=generator, requires_grad=True)
        candidate_input = eager_input.detach().clone().requires_grad_(True)
        expected = eager(eager_input).sin()
        actual = run_graphed_callable(
            candidate,
            region="linear_sin",
            operation=lambda values: candidate(values).sin(),
            args=(candidate_input,),
        )
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        expected.sum().backward()
        actual.sum().backward()
        assert eager_input.grad is not None and candidate_input.grad is not None
        torch.testing.assert_close(candidate_input.grad, eager_input.grad, atol=0.0, rtol=0.0)
    assert len(cuda_graph_capture_stats(candidate)) == 1
    assert tuple(cuda_graph_replay_counts(candidate).values()) == (2,)
    clear_cuda_graph_runtime(candidate)


def _model_config() -> ModelConfig:
    return ModelConfig(
        variant="bgt",
        family="qwen3",
        qk_norm=True,
        rotary_style="llama",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=128,
        cache_processor=CacheProcessorConfig(
            process_every=64,
            topk=8,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
            gate_mode="direct",
            gate_init=0.5,
            attention_dropout=0.0,
        ),
    )


def test_qwen_cuda_graph_windows_match_eager_processor_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cuda_gate()
    monkeypatch.setenv("GCW_ALLOW_UNQUALIFIED_CUDA_GRAPH", "1")
    torch.manual_seed(17)
    eager = build_model(_model_config()).cuda()
    assert isinstance(eager, LanguageModel)
    eager.init_weights()
    candidate = copy.deepcopy(eager)
    _apply_trainable_policy(eager, "cache_processor")
    _apply_trainable_policy(candidate, "cache_processor")
    configure_qwen3_backbone_chunk_backend(candidate, "cuda_graph")
    inputs = torch.randint(0, 64, (2, 128), device="cuda")
    targets = torch.randint(0, 64, (2, 128), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager_loss = eager(inputs, targets, loss_reduction="row_sum").sum()
        candidate_loss = candidate(inputs, targets, loss_reduction="row_sum").sum()
    torch.testing.assert_close(candidate_loss, eager_loss, atol=0.0, rtol=0.0)
    eager_loss.backward()
    candidate_loss.backward()
    eager_parameters = tuple(parameter for parameter in eager.parameters() if parameter.requires_grad)
    candidate_parameters = tuple(parameter for parameter in candidate.parameters() if parameter.requires_grad)
    for reference, actual in zip(eager_parameters, candidate_parameters, strict=True):
        if reference.grad is None or actual.grad is None:
            assert reference.grad is actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, reference.grad, atol=0.0, rtol=0.0)
    assert len(cuda_graph_capture_stats(candidate.backbone)) == 2
    clear_cuda_graph_runtime(candidate.backbone)
