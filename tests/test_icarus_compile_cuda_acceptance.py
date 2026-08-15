"""Opt-in, real CUDA/Inductor acceptance for the Icarus training backend."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import torch

from benchmarks.parity import (
    assert_cache_close,
    assert_parameter_gradients_close,
    assert_tensor_gradients_close,
    cache_objective,
    clone_cache,
)
from gcw.models import build_model
from gcw.models.cache_processors import icarus as processor_module
from gcw.models.cache_processors.ops import latest_shared_indices
from gcw.models.model import LanguageModel
from gcw.performance.backends.icarus import (
    IcarusProcessorBackend,
    configure_icarus_training_execution,
    icarus_training_execution,
    warmup_icarus_training_execution,
)
from tests.test_icarus_compile_backend import _cache, _config, _processor


_CUDA_ACCEPTANCE = os.environ.get("GCW_RUN_CUDA_COMPILE_ACCEPTANCE") == "1"
_FIXED_BATCH_SIZE = 2
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not _CUDA_ACCEPTANCE,
        reason="set GCW_RUN_CUDA_COMPILE_ACCEPTANCE=1 on a CUDA runner",
    ),
]


@pytest.fixture(autouse=True)
def _allow_rejected_icarus_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCW_ALLOW_REJECTED_ICARUS_COMPILE", "1")


def _paired_models(config: Any, *, device: torch.device) -> tuple[LanguageModel, LanguageModel]:
    torch.manual_seed(811)
    eager = build_model(config).to(device)
    assert isinstance(eager, LanguageModel)
    eager.init_weights()
    compiled = copy.deepcopy(eager)
    compiled.cache_processor.execution_backend = "compile"
    return eager, compiled


def _install_exact_trace(
    monkeypatch: pytest.MonkeyPatch,
    eager: LanguageModel,
    compiled: LanguageModel,
) -> dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Record selected positions/heads and the exact latest write destinations."""
    eager_block = eager.cache_processor.block
    compiled_block = compiled.cache_processor.block
    original = processor_module.run_icarus_post_selection_region
    traces: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {
        "eager": [],
        "compile": [],
    }

    def traced(
        backend: IcarusProcessorBackend,
        block: torch.nn.Module,
        x: torch.Tensor,
        positions: torch.Tensor,
        selected_heads: torch.Tensor,
    ) -> torch.Tensor:
        label = "eager" if block is eager_block else "compile" if block is compiled_block else None
        if label is not None:
            traces[label].append(
                (
                    positions.detach().cpu().clone(),
                    selected_heads.detach().cpu().clone(),
                    latest_shared_indices(positions).detach().cpu().clone(),
                )
            )
        return original(backend, block, x, positions, selected_heads)

    monkeypatch.setattr(processor_module, "run_icarus_post_selection_region", traced)
    return traces


def _flat_gradients(model: LanguageModel) -> torch.Tensor:
    gradients = [
        parameter.grad.detach().float().flatten() for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    return torch.cat(gradients)


def _assert_tight_gradient_metrics(eager: LanguageModel, compiled: LanguageModel) -> None:
    expected = _flat_gradients(eager)
    actual = _flat_gradients(compiled)
    relative_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=0)
    assert relative_l2.item() <= 5e-4
    assert cosine.item() >= 0.99999




def test_real_inductor_checkpoint_cache_and_parameter_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise two processor events and non-reentrant checkpoint replay at length 192."""
    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "compile")
    monkeypatch.setenv("GCW_STRICT_FLASH", "1")
    device = torch.device("cuda")
    config = _config(gradient_checkpointing=True)
    config.sequence_len = 192
    eager = _processor(config).to(device=device, dtype=torch.bfloat16)
    compiled = copy.deepcopy(eager)
    compiled.execution_backend = "compile"
    configure_icarus_training_execution(compiled, (2,))
    source = _cache(config, batch=2, seq=192, device=device, dtype=torch.bfloat16)
    eager_cache = clone_cache(source, requires_grad=True)
    compiled_cache = clone_cache(source, requires_grad=True)
    eager_inputs = [*eager_cache.keys, *eager_cache.values]
    compiled_inputs = [*compiled_cache.keys, *compiled_cache.values]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = eager(eager_cache)
        expected = eager(expected)
        with icarus_training_execution(compiled, 2):
            actual = compiled(compiled_cache)
            actual = compiled(actual)
            cache_objective(actual).backward()
        cache_objective(expected).backward()

    assert_cache_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert_tensor_gradients_close(compiled_inputs, eager_inputs, atol=3e-2, rtol=3e-2)
    assert_parameter_gradients_close(compiled, eager, atol=3e-2, rtol=3e-2)


def test_real_inductor_fixed_batch_two_is_rejected_by_adam_parameter_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce the H200 rejection after proving exact cache events and bounded gradients."""
    monkeypatch.delenv("GCW_ICARUS_PROCESSOR_BACKEND", raising=False)
    monkeypatch.setenv("GCW_STRICT_FLASH", "1")
    device = torch.device("cuda")
    config = _config(gradient_checkpointing=True)
    config.sequence_len = 192
    eager, compiled = _paired_models(config, device=device)
    configure_icarus_training_execution(compiled, (_FIXED_BATCH_SIZE,))
    warmup_icarus_training_execution(
        compiled,
        (_FIXED_BATCH_SIZE,),
        sequence_length=config.sequence_len,
        device=device,
        bf16=True,
    )
    eager_optimizer = torch.optim.AdamW(eager.parameters(), lr=3e-4)
    compiled_optimizer = torch.optim.AdamW(compiled.parameters(), lr=3e-4)
    traces = _install_exact_trace(monkeypatch, eager, compiled)
    generator = torch.Generator().manual_seed(3317)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()

    # Exact-layout warmup must cover the fixed oracle without first-step specialization.
    with torch._dynamo.config.patch(error_on_recompile=True):
        tokens = torch.randint(
            0,
            config.vocab_size,
            (_FIXED_BATCH_SIZE, 192),
            generator=generator,
        ).to(device)
        targets = tokens.roll(-1, dims=1)
        eager_optimizer.zero_grad(set_to_none=True)
        compiled_optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            eager_loss = eager(tokens, targets)
            with icarus_training_execution(compiled, _FIXED_BATCH_SIZE):
                compiled_loss = compiled(tokens, targets)
        eager_loss.backward()
        compiled_loss.backward()

        assert torch.allclose(compiled_loss, eager_loss, atol=2e-5, rtol=2e-4)
        assert torch.equal(torch.random.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
        eager_events = traces["eager"]
        compiled_events = traces["compile"]
        assert len(eager_events) == len(compiled_events) == 2
        for expected_event, actual_event in zip(eager_events, compiled_events, strict=True):
            expected_positions, expected_heads, expected_writes = expected_event
            actual_positions, actual_heads, actual_writes = actual_event
            assert torch.equal(actual_positions, expected_positions)
            assert torch.equal(actual_heads, expected_heads)
            assert torch.equal(actual_writes, expected_writes)
        assert compiled_events[1][0].shape[-1] > compiled_events[0][0].shape[-1]
        _assert_tight_gradient_metrics(eager, compiled)

        eager_optimizer.step()
        compiled_optimizer.step()
        formal_parameter_parity = all(
            torch.allclose(actual, expected, atol=2e-5, rtol=2e-4)
            for expected, actual in zip(eager.parameters(), compiled.parameters(), strict=True)
        )
        assert not formal_parameter_parity, (
            "the recorded H200 Adam parameter rejection no longer reproduces; "
            "rerun the complete admission review before enabling this backend"
        )


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    else:
        assert actual == expected


def test_real_inductor_production_checkpoint_survives_process_restart(tmp_path: Path) -> None:
    """Compare uninterrupted training with production save/open/restore APIs in a new process."""
    checkpoint = tmp_path / "checkpoint"
    continuous = tmp_path / "continuous.pt"
    resumed = tmp_path / "resumed.pt"
    script = Path(__file__).with_name("icarus_compile_restart_worker.py")
    env = {
        **os.environ,
        "GCW_ICARUS_PROCESSOR_BACKEND": "compile",
        "GCW_STRICT_FLASH": "1",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    subprocess.run([sys.executable, str(script), "continuous", str(continuous)], check=True, env=env)
    subprocess.run([sys.executable, str(script), "save", str(checkpoint)], check=True, env=env)
    subprocess.run([sys.executable, str(script), "resume", str(checkpoint), str(resumed)], check=True, env=env)
    expected = torch.load(continuous, map_location="cpu", weights_only=True)
    actual = torch.load(resumed, map_location="cpu", weights_only=True)
    _assert_nested_equal(actual, expected)
