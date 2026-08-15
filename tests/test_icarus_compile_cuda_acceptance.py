"""Opt-in, real CUDA/Inductor acceptance for the Icarus training backend."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys

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
from gcw.models.model import LanguageModel
from gcw.performance.backends.icarus import (
    configure_icarus_training_execution,
    icarus_training_execution,
)
from tests.test_icarus_compile_backend import _cache, _config, _processor


_CUDA_ACCEPTANCE = os.environ.get("GCW_RUN_CUDA_COMPILE_ACCEPTANCE") == "1"
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not _CUDA_ACCEPTANCE,
        reason="set GCW_RUN_CUDA_COMPILE_ACCEPTANCE=1 on a CUDA runner",
    ),
]




def test_real_inductor_checkpoint_cache_and_parameter_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "compile")
    monkeypatch.setenv("GCW_STRICT_FLASH", "1")
    device = torch.device("cuda")
    config = _config(gradient_checkpointing=True)
    eager = _processor(config).to(device=device, dtype=torch.bfloat16)
    compiled = copy.deepcopy(eager)
    compiled.execution_backend = "compile"
    configure_icarus_training_execution(compiled, (2,))
    source = _cache(config, batch=2, seq=128, device=device, dtype=torch.bfloat16)
    eager_cache = clone_cache(source, requires_grad=True)
    compiled_cache = clone_cache(source, requires_grad=True)
    eager_inputs = [*eager_cache.keys, *eager_cache.values]
    compiled_inputs = [*compiled_cache.keys, *compiled_cache.values]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = eager(eager_cache)
        with icarus_training_execution(compiled, 2):
            actual = compiled(compiled_cache)
            cache_objective(actual).backward()
        cache_objective(expected).backward()

    assert_cache_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert_tensor_gradients_close(compiled_inputs, eager_inputs, atol=3e-2, rtol=3e-2)
    assert_parameter_gradients_close(compiled, eager, atol=3e-2, rtol=3e-2)


def test_real_inductor_dynamic_adaptive_shapes_do_not_recompile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "compile")
    monkeypatch.setenv("GCW_STRICT_FLASH", "1")
    processor = _processor(_config()).to(device="cuda", dtype=torch.bfloat16)
    configure_icarus_training_execution(processor, (16, 8, 4, 2))

    # dynamic=True must retain one graph across every CUDA-qualified physical band.
    with torch._dynamo.config.patch(error_on_recompile=True):
        for rows in (16, 8, 4, 2, 16):
            cache = clone_cache(
                _cache(_config(), batch=rows, seq=64, device=torch.device("cuda"), dtype=torch.bfloat16),
                requires_grad=True,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16), icarus_training_execution(processor, rows):
                output = processor(cache)
                cache_objective(output).backward()
            processor.zero_grad(set_to_none=True)


def test_real_inductor_multi_window_optimizer_trajectory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GCW_ICARUS_PROCESSOR_BACKEND", raising=False)
    torch.manual_seed(811)
    device = torch.device("cuda")
    config = _config()
    eager = build_model(config).to(device)
    assert isinstance(eager, LanguageModel)
    eager.init_weights()
    compiled = copy.deepcopy(eager)
    compiled.cache_processor.execution_backend = "compile"
    configure_icarus_training_execution(compiled, (2,))
    eager_opt = torch.optim.AdamW(eager.parameters(), lr=3e-4)
    compiled_opt = torch.optim.AdamW(compiled.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(991)
    batches = [
        torch.randint(0, config.vocab_size, (2, 128), generator=generator).to(device)
        for _ in range(3)
    ]

    for tokens in batches:
        targets = tokens.roll(-1, dims=1)
        eager_opt.zero_grad(set_to_none=True)
        compiled_opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected_loss = eager(tokens, targets)
            with icarus_training_execution(compiled, 2):
                actual_loss = compiled(tokens, targets)
                actual_loss.backward()
        expected_loss.backward()
        assert torch.allclose(actual_loss, expected_loss, atol=3e-2, rtol=3e-2)
        eager_opt.step()
        compiled_opt.step()

    for actual, expected in zip(compiled.parameters(), eager.parameters(), strict=True):
        assert torch.allclose(actual, expected, atol=4e-2, rtol=4e-2)


def test_real_inductor_resume_survives_process_restart(tmp_path: Path) -> None:
    """Compare uninterrupted two-step training with save/restart/load training."""
    checkpoint = tmp_path / "resume.pt"
    continuous = tmp_path / "continuous.pt"
    resumed = tmp_path / "resumed.pt"
    script = Path(__file__).with_name("icarus_compile_restart_worker.py")
    env = {
        **os.environ,
        "GCW_ICARUS_PROCESSOR_BACKEND": "compile",
        "GCW_STRICT_FLASH": "1",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    subprocess.run(
        [sys.executable, str(script), "continuous", str(continuous)],
        check=True,
        env=env,
    )
    subprocess.run(
        [sys.executable, str(script), "save", str(checkpoint)],
        check=True,
        env=env,
    )
    subprocess.run(
        [sys.executable, str(script), "resume", str(checkpoint), str(resumed)],
        check=True,
        env=env,
    )
    expected = torch.load(continuous, map_location="cpu", weights_only=True)
    actual = torch.load(resumed, map_location="cpu", weights_only=True)
    assert expected.keys() == actual.keys()
    assert all(torch.equal(actual[name], expected[name]) for name in expected)
