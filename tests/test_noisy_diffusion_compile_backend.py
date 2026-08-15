from __future__ import annotations

import copy
import os
from collections.abc import Callable
from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from gcw.config import DataConfig, TrainConfig
from gcw.models.cache_processors.diffusion import TaskDirectedDiffusion
from gcw.performance import NoisyDiffusionBackendConfig, PerformanceConfig
from gcw.performance.backends import diffusion as backend_module
from gcw.performance.backends.diffusion import (
    configure_noisy_diffusion_recurrence_backend,
    noisy_diffusion_recurrence_backend_metadata,
)
from gcw.training.core import _training_fingerprint
from tests.test_diffusion import tiny_diffusion_config


class _Field(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor, _time: float) -> torch.Tensor:
        return torch.tanh(self.projection(value))


class _DiffusionHolder(nn.Module):
    def __init__(self, diffusion: TaskDirectedDiffusion) -> None:
        super().__init__()
        self.diffusion = diffusion


def _install_fake_compiler(
    monkeypatch: pytest.MonkeyPatch,
    *, fail: bool = False,
) -> tuple[list[dict[str, object]], list[torch.Tensor], list[torch.Tensor]]:
    compile_calls: list[dict[str, object]] = []
    dispatch_rng_states: list[torch.Tensor] = []
    dispatched_states: list[torch.Tensor] = []

    def fake_compile(function: Callable[..., torch.Tensor], **kwargs: object):
        compile_calls.append(dict(kwargs))

        def compiled(*args: object, **inner_kwargs: object) -> torch.Tensor:
            dispatch_rng_states.append(torch.get_rng_state().clone())
            state = args[1]
            assert isinstance(state, torch.Tensor)
            dispatched_states.append(state.detach().clone())
            if fail:
                raise RuntimeError("inductor failed")
            return function(*args, **inner_kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    backend_module._compiled_recurrence_runner = None
    return compile_calls, dispatch_rng_states, dispatched_states


def _compiled_policy() -> NoisyDiffusionBackendConfig:
    return NoisyDiffusionBackendConfig(recurrence_backend="compile")


def test_compiled_dispatch_samples_noise_eagerly_and_preserves_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(13)
    eager = TaskDirectedDiffusion(4, tiny_diffusion_config().diffusion_processor)  # type: ignore[arg-type]
    compiled = copy.deepcopy(eager)
    eager_field = _Field(4)
    compiled_field = copy.deepcopy(eager_field)
    eager_input = torch.randn(2, 3, 4, requires_grad=True)
    compiled_input = eager_input.detach().clone().requires_grad_(True)
    compile_calls, dispatch_rng_states, dispatched_states = _install_fake_compiler(monkeypatch)
    configure_noisy_diffusion_recurrence_backend(
        _DiffusionHolder(compiled), tiny_diffusion_config(), _compiled_policy()
    )

    seed = 991
    torch.manual_seed(seed)
    expected_sample = torch.randn_like(eager_input)
    expected_noisy_state = eager_input.detach() + eager.config.noise_scale * expected_sample
    torch.manual_seed(seed)
    eager_output = eager(eager_input, eager_field)
    eager_rng = torch.get_rng_state().clone()
    eager_output.square().mean().backward()

    torch.manual_seed(seed)
    compiled_output = compiled(compiled_input, compiled_field)
    compiled_rng = torch.get_rng_state().clone()
    compiled_output.square().mean().backward()

    torch.testing.assert_close(compiled_output, eager_output, atol=0, rtol=0)
    assert torch.equal(compiled_rng, eager_rng)
    assert len(dispatch_rng_states) == 1
    assert torch.equal(dispatch_rng_states[0], eager_rng)
    torch.testing.assert_close(dispatched_states[0], expected_noisy_state, atol=0, rtol=0)
    torch.testing.assert_close(compiled_input.grad, eager_input.grad, atol=0, rtol=0)
    torch.testing.assert_close(
        compiled.refinement_gate.grad,
        eager.refinement_gate.grad,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        compiled_field.projection.weight.grad,
        eager_field.projection.weight.grad,
        atol=0,
        rtol=0,
    )
    assert compile_calls == [
        {"backend": "inductor", "dynamic": False, "fullgraph": True}
    ]


def test_explicit_noise_crosses_dispatch_without_rng_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_diffusion_config()
    assert config.diffusion_processor is not None
    diffusion = TaskDirectedDiffusion(4, config.diffusion_processor)
    holder = _DiffusionHolder(diffusion)
    field = _Field(4)
    compile_calls, dispatch_rng_states, dispatched_states = _install_fake_compiler(monkeypatch)
    configure_noisy_diffusion_recurrence_backend(holder, config, _compiled_policy())
    bottleneck = torch.randn(1, 2, 4)
    noise = torch.randn_like(bottleneck)
    rng_before = torch.get_rng_state().clone()

    output = diffusion(bottleneck, field, noise=noise)

    assert output.shape == bottleneck.shape
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert len(dispatch_rng_states) == 1
    assert torch.equal(dispatch_rng_states[0], rng_before)
    torch.testing.assert_close(
        dispatched_states[0],
        bottleneck + diffusion.config.noise_scale * noise,
        atol=0,
        rtol=0,
    )
    assert len(compile_calls) == 1


def test_eager_default_never_calls_torch_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    config = tiny_diffusion_config()
    assert config.diffusion_processor is not None
    diffusion = TaskDirectedDiffusion(4, config.diffusion_processor)

    def fail_compile(*_args: object, **_kwargs: object) -> None:
        pytest.fail("eager noisy-Diffusion recurrence must not call torch.compile")

    monkeypatch.setattr(torch, "compile", fail_compile)
    output = diffusion(torch.zeros(1, 2, 4), _Field(4), noise=torch.zeros(1, 2, 4))
    assert output.shape == (1, 2, 4)


def test_unavailable_compiler_fails_before_rng_consumption(
    monkeypatch: pytest.MonkeyPatch,
 ) -> None:
    config = tiny_diffusion_config()
    assert config.diffusion_processor is not None
    holder = _DiffusionHolder(TaskDirectedDiffusion(4, config.diffusion_processor))
    monkeypatch.setattr(torch, "compile", None)
    torch.manual_seed(81)
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="requires torch.compile"):
        configure_noisy_diffusion_recurrence_backend(holder, config, _compiled_policy())
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_backend_is_fail_closed_state_neutral_and_admitted_narrowly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_diffusion_config()
    assert config.diffusion_processor is not None
    diffusion = TaskDirectedDiffusion(4, config.diffusion_processor)
    holder = _DiffusionHolder(diffusion)
    before = copy.deepcopy(holder.state_dict())
    _install_fake_compiler(monkeypatch, fail=True)
    configure_noisy_diffusion_recurrence_backend(holder, config, _compiled_policy())

    with pytest.raises(RuntimeError, match="eager fallback is disabled") as error:
        diffusion(torch.zeros(1, 2, 4), _Field(4))
    assert isinstance(error.value.__cause__, RuntimeError)
    assert tuple(holder.state_dict()) == tuple(before)
    assert all(torch.equal(value, before[name]) for name, value in holder.state_dict().items())

    no_noise = copy.deepcopy(config)
    assert no_noise.diffusion_processor is not None
    no_noise.diffusion_processor.noise_scale = 0.0
    with pytest.raises(RuntimeError, match="noise_scale>0"):
        configure_noisy_diffusion_recurrence_backend(holder, no_noise, _compiled_policy())

    wrong_variant = copy.deepcopy(config)
    wrong_variant.variant = "longshot"
    with pytest.raises(RuntimeError, match="requires BGT"):
        configure_noisy_diffusion_recurrence_backend(holder, wrong_variant, _compiled_policy())


def test_default_fingerprint_is_compatible_and_compile_is_authenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = tiny_diffusion_config()
    default = TrainConfig(model=model_config, data=DataConfig(kind="random"))
    explicit_eager = replace(
        default,
        performance=PerformanceConfig(
            noisy_diffusion=NoisyDiffusionBackendConfig(recurrence_backend="eager")
        ),
    )
    default_fingerprint = _training_fingerprint(default, 1, 8, 8, 1)
    eager_fingerprint = _training_fingerprint(explicit_eager, 1, 8, 8, 1)
    assert default_fingerprint == eager_fingerprint
    assert "noisy_diffusion_recurrence_backend_fingerprint" not in default_fingerprint

    _install_fake_compiler(monkeypatch)
    compile_config = replace(
        default,
        performance=PerformanceConfig(noisy_diffusion=_compiled_policy()),
    )
    compile_fingerprint = _training_fingerprint(compile_config, 1, 8, 8, 1)
    metadata = noisy_diffusion_recurrence_backend_metadata(
        model_config, compile_config.performance.noisy_diffusion
    )
    assert (
        compile_fingerprint["noisy_diffusion_recurrence_backend_fingerprint"]
        == metadata["fingerprint_sha256"]
    )
    assert metadata["noise_boundary"] == "eager_torch_randn_like_before_recurrence_dispatch"
    assert metadata["effective_backend"] == "compile"


def test_configuration_rejects_invalid_or_incompatible_compile_requests() -> None:
    with pytest.raises(ValueError, match="must be 'eager' or 'compile'"):
        NoisyDiffusionBackendConfig(recurrence_backend="invalid")  # type: ignore[arg-type]

    model_config = tiny_diffusion_config()
    with pytest.raises(ValueError, match="distributed='none'"):
        TrainConfig(
            model=model_config,
            data=DataConfig(kind="random"),
            distributed="ddp",
            performance=PerformanceConfig(noisy_diffusion=_compiled_policy()),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("GCW_RUN_CUDA_DIFFUSION_COMPILE_TESTS") != "1",
    reason="set GCW_RUN_CUDA_DIFFUSION_COMPILE_TESTS=1 on CUDA to exercise fullgraph Inductor",
)
def test_cuda_fullgraph_compiles_field_callable_and_preserves_backward() -> None:
    config = tiny_diffusion_config()
    assert config.diffusion_processor is not None
    eager = TaskDirectedDiffusion(8, replace(config.diffusion_processor, num_steps=4)).cuda()
    compiled = copy.deepcopy(eager)
    eager_field = _Field(8).cuda()
    compiled_field = copy.deepcopy(eager_field)
    holder = _DiffusionHolder(compiled)
    backend_module._compiled_recurrence_runner = None
    configure_noisy_diffusion_recurrence_backend(holder, config, _compiled_policy())
    eager_input = torch.randn(1, 4, 8, device="cuda", requires_grad=True)
    compiled_input = eager_input.detach().clone().requires_grad_(True)

    torch.manual_seed(2026)
    eager_output = eager(
        eager_input,
        lambda value, time: eager_field(value, time),
    )
    eager_output.sum().backward()
    eager_rng = torch.cuda.get_rng_state().clone()
    torch.manual_seed(2026)
    compiled_output = compiled(
        compiled_input,
        lambda value, time: compiled_field(value, time),
    )
    compiled_output.sum().backward()

    torch.testing.assert_close(compiled_output, eager_output, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(compiled_input.grad, eager_input.grad, atol=2e-5, rtol=2e-4)
    assert torch.equal(torch.cuda.get_rng_state(), eager_rng)
