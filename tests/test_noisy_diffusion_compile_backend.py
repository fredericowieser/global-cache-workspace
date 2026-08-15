from __future__ import annotations

import copy
import os
from collections.abc import Callable
from dataclasses import replace

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from gcw.architecture import resolve_experiment, resolve_model
from gcw.config import (
    DataConfig,
    DiffusionProcessorConfig,
    ModelConfig,
    TrainConfig,
)
from gcw.models.cache_processors.diffusion import TaskDirectedDiffusion
from gcw.models.cache_processors.standard import CacheProcessor
from gcw.performance import (
    AdaptiveBatchingConfig,
    NoisyDiffusionBackendConfig,
    PerformanceConfig,
)
from gcw.performance.backends.diffusion import (
    configure_noisy_diffusion_recurrence_backend,
    noisy_diffusion_compile_telemetry,
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


class _RecordingField(_Field):
    def __init__(self, width: int) -> None:
        super().__init__(width)
        self.times: list[float] = []

    def forward(self, value: torch.Tensor, time: float) -> torch.Tensor:
        self.times.append(time)
        return super().forward(value, time)


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
            state = args[0]
            assert isinstance(state, torch.Tensor)
            dispatched_states.append(state.detach().clone())
            if fail:
                raise RuntimeError("inductor failed")
            return function(*args, **inner_kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
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

    with pytest.raises(ValueError, match="noisy Diffusion|adaptive batching"):
        TrainConfig(
            model=copy.deepcopy(model_config),
            data=DataConfig(kind="sft", dataset="rows.jsonl"),
            mode="sft",
            batch_size=2,
            global_batch_rows=64,
            logical_pair_grouping=True,
            performance=PerformanceConfig(
                adaptive_batching=AdaptiveBatchingConfig(enabled=True),
                noisy_diffusion=_compiled_policy(),
            ),
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
    torch.testing.assert_close(
        compiled.refinement_gate.grad, eager.refinement_gate.grad, atol=2e-5, rtol=2e-4
    )
    torch.testing.assert_close(
        compiled_field.projection.weight.grad,
        eager_field.projection.weight.grad,
        atol=2e-5,
        rtol=2e-4,
    )
    before = noisy_diffusion_compile_telemetry(compiled)
    repeated = torch.randn(1, 6, 8, device="cuda", requires_grad=True)
    compiled(repeated, lambda value, time: compiled_field(value, time)).square().mean().backward()
    after = noisy_diffusion_compile_telemetry(compiled)
    assert after["runner_compile_count"] == 1
    assert after["runner_dispatch_count"] == 2
    assert int(after["dynamo_unique_graphs"]) >= int(before["dynamo_unique_graphs"])


def test_default_noisy_diffusion_policy_preserves_pre_feature_receipts() -> None:
    model = ModelConfig(
        variant="bgt",
        family="smollm",
        sequence_len=128,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        diffusion_processor=DiffusionProcessorConfig(),
    )
    train = TrainConfig(model=model, data=DataConfig(kind="random"))
    _, model_receipt = resolve_model(model).require()
    _, composition_receipt = resolve_experiment(train, model_receipt).require()

    assert model_receipt.model_config_sha256 == (
        "1db30db79c348e5f29ec2699ea9744e4c2fb37cbbbceaaae8fbf7e5a45ab2fc9"
    )
    assert model_receipt.receipt_sha256 == (
        "5a2ef93ac2e6ca753b3034116acf4846beda371dca38566d0eba255c644161eb"
    )
    assert composition_receipt.train_config_sha256 == (
        "b4bd02d2185d2017e07fa57d4154a0d087c2e20acf0fd5e3c8c983f45fd807ef"
    )
    assert composition_receipt.receipt_sha256 == (
        "75721acc440dc737d4b809370e0efbe00ac2bfd254a9aceb1bed209336450398"
    )


def test_compiled_backend_rejects_attention_dropout_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
 ) -> None:
    model_config = tiny_diffusion_config()
    assert model_config.cache_processor is not None
    model_config.cache_processor.attention_dropout = 0.1
    diffusion = TaskDirectedDiffusion(4, model_config.diffusion_processor)  # type: ignore[arg-type]
    holder = _DiffusionHolder(diffusion)
    _install_fake_compiler(monkeypatch)

    with pytest.raises(RuntimeError, match="attention_dropout=0"):
        configure_noisy_diffusion_recurrence_backend(holder, model_config, _compiled_policy())
    with pytest.raises(ValueError, match="attention_dropout=0"):
        TrainConfig(
            model=model_config,
            data=DataConfig(kind="random"),
            performance=PerformanceConfig(noisy_diffusion=_compiled_policy()),
        )


def test_compiled_runner_is_owned_per_module_without_stale_bindings(
    monkeypatch: pytest.MonkeyPatch,
 ) -> None:
    compile_calls, _, _ = _install_fake_compiler(monkeypatch)
    first_config = tiny_diffusion_config()
    second_config = copy.deepcopy(first_config)
    assert first_config.diffusion_processor is not None
    assert second_config.diffusion_processor is not None
    second_config.diffusion_processor.num_steps = 2
    first = TaskDirectedDiffusion(4, first_config.diffusion_processor)
    second = TaskDirectedDiffusion(4, second_config.diffusion_processor)
    first_holder = _DiffusionHolder(first)
    second_holder = _DiffusionHolder(second)
    configure_noisy_diffusion_recurrence_backend(first_holder, first_config, _compiled_policy())
    configure_noisy_diffusion_recurrence_backend(second_holder, second_config, _compiled_policy())
    first_keys = tuple(first_holder.state_dict())
    second_keys = tuple(second_holder.state_dict())

    first_field = _RecordingField(4)
    second_field = _RecordingField(4)
    for width in (3, 5, 3):
        first(torch.zeros(1, width, 4), first_field, noise=torch.ones(1, width, 4))
        second(torch.zeros(1, width + 1, 4), second_field, noise=torch.ones(1, width + 1, 4))

    assert len(compile_calls) == 2
    assert first_field.times == [0.0, 0.25, 0.5, 0.75] * 3
    assert second_field.times == [0.0, 0.5] * 3
    assert noisy_diffusion_compile_telemetry(first)["runner_dispatch_count"] == 3
    assert noisy_diffusion_compile_telemetry(second)["runner_dispatch_count"] == 3
    assert first.__dict__["_gcw_noisy_diffusion_compiled_recurrence_runner"] is not (
        second.__dict__["_gcw_noisy_diffusion_compiled_recurrence_runner"]
    )
    assert tuple(first_holder.state_dict()) == first_keys
    assert tuple(second_holder.state_dict()) == second_keys


def test_compiled_boundary_preserves_checkpoint_early_stop_exactly(
    monkeypatch: pytest.MonkeyPatch,
 ) -> None:
    config = tiny_diffusion_config()
    assert config.diffusion_processor is not None
    diffusion = TaskDirectedDiffusion(4, config.diffusion_processor)
    configure_noisy_diffusion_recurrence_backend(
        _DiffusionHolder(diffusion), config, _compiled_policy()
    )
    signal_type = type(
        "_StopRecomputationError",
        (Exception,),
        {"__module__": "torch.utils.checkpoint"},
    )
    signal = signal_type("checkpoint replay complete")

    def early_stop(
        _state: torch.Tensor, _bottleneck: torch.Tensor, _field: Callable[..., torch.Tensor]
    ) -> torch.Tensor:
        raise signal

    object.__setattr__(
        diffusion, "_gcw_noisy_diffusion_compiled_recurrence_runner", early_stop
    )
    with pytest.raises(signal_type) as caught:
        diffusion(torch.zeros(1, 2, 4), _Field(4))
    assert caught.value is signal


def test_compiled_checkpoint_replay_preserves_production_gradients(
    monkeypatch: pytest.MonkeyPatch,
 ) -> None:
    _install_fake_compiler(monkeypatch)
    config = tiny_diffusion_config()
    assert config.diffusion_processor is not None
    diffusion = TaskDirectedDiffusion(4, config.diffusion_processor)
    configure_noisy_diffusion_recurrence_backend(
        _DiffusionHolder(diffusion), config, _compiled_policy()
    )
    field = _Field(4)
    value = torch.randn(1, 3, 4, requires_grad=True)
    noise = torch.randn_like(value)

    output = checkpoint(
        lambda tensor: diffusion(tensor, field, noise=noise),
        value,
        use_reentrant=False,
        preserve_rng_state=True,
    )
    output.square().mean().backward()

    assert value.grad is not None
    assert diffusion.refinement_gate.grad is not None
    assert field.projection.weight.grad is not None
    assert torch.count_nonzero(diffusion.refinement_gate.grad) > 0
    assert torch.count_nonzero(field.projection.weight.grad) > 0
    assert int(noisy_diffusion_compile_telemetry(diffusion)["runner_dispatch_count"]) >= 2


def test_production_cache_processor_closure_and_parameters_are_state_neutral(
    monkeypatch: pytest.MonkeyPatch,
 ) -> None:
    _install_fake_compiler(monkeypatch)
    model_config = tiny_diffusion_config()
    assert model_config.cache_processor is not None
    assert model_config.diffusion_processor is not None
    processor = CacheProcessor(
        8, model_config.cache_processor, diffusion_config=model_config.diffusion_processor
    )
    before_keys = tuple(processor.state_dict())
    configure_noisy_diffusion_recurrence_backend(processor, model_config, _compiled_policy())
    calls: list[int] = []
    hook = processor.layers[0].register_forward_hook(lambda *_args: calls.append(1))
    value = torch.randn(1, 4, 8, requires_grad=True)
    positions = torch.arange(4).view(1, 4)

    processor.compute_update(value, positions).square().mean().backward()
    hook.remove()

    assert calls == [1] * model_config.diffusion_processor.num_steps
    assert processor.diffusion is not None
    assert processor.diffusion.refinement_gate.grad is not None
    assert torch.count_nonzero(processor.diffusion.refinement_gate.grad) > 0
    assert processor.layers[0].down_proj.weight.grad is not None
    assert torch.count_nonzero(processor.layers[0].down_proj.weight.grad) > 0
    assert tuple(processor.state_dict()) == before_keys


def _run_optimizer_steps(
    diffusion: TaskDirectedDiffusion,
    field: _Field,
    optimizer: torch.optim.Optimizer,
    value: torch.Tensor,
    steps: int,
 ) -> list[float]:
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion(value, field).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def test_compiled_optimizer_trajectory_and_rng_resume(
    monkeypatch: pytest.MonkeyPatch,
 ) -> None:
    _install_fake_compiler(monkeypatch)
    torch.manual_seed(47)
    config = tiny_diffusion_config()
    assert config.diffusion_processor is not None
    template_diffusion = TaskDirectedDiffusion(4, config.diffusion_processor)
    template_field = _Field(4)
    value = torch.randn(2, 3, 4)

    uninterrupted_diffusion = copy.deepcopy(template_diffusion)
    uninterrupted_field = copy.deepcopy(template_field)
    configure_noisy_diffusion_recurrence_backend(
        _DiffusionHolder(uninterrupted_diffusion), config, _compiled_policy()
    )
    uninterrupted_optimizer = torch.optim.AdamW(
        list(uninterrupted_diffusion.parameters()) + list(uninterrupted_field.parameters()),
        lr=1e-3,
    )
    initial_rng = torch.get_rng_state().clone()
    uninterrupted_losses = _run_optimizer_steps(
        uninterrupted_diffusion, uninterrupted_field, uninterrupted_optimizer, value, 4
    )
    uninterrupted_rng = torch.get_rng_state().clone()

    first_diffusion = copy.deepcopy(template_diffusion)
    first_field = copy.deepcopy(template_field)
    configure_noisy_diffusion_recurrence_backend(
        _DiffusionHolder(first_diffusion), config, _compiled_policy()
    )
    first_optimizer = torch.optim.AdamW(
        list(first_diffusion.parameters()) + list(first_field.parameters()), lr=1e-3
    )
    torch.set_rng_state(initial_rng)
    resumed_losses = _run_optimizer_steps(
        first_diffusion, first_field, first_optimizer, value, 2
    )
    saved_diffusion = copy.deepcopy(first_diffusion.state_dict())
    saved_field = copy.deepcopy(first_field.state_dict())
    saved_optimizer = copy.deepcopy(first_optimizer.state_dict())
    saved_rng = torch.get_rng_state().clone()

    resumed_diffusion = copy.deepcopy(template_diffusion)
    resumed_field = copy.deepcopy(template_field)
    resumed_diffusion.load_state_dict(saved_diffusion)
    resumed_field.load_state_dict(saved_field)
    configure_noisy_diffusion_recurrence_backend(
        _DiffusionHolder(resumed_diffusion), config, _compiled_policy()
    )
    resumed_optimizer = torch.optim.AdamW(
        list(resumed_diffusion.parameters()) + list(resumed_field.parameters()), lr=1e-3
    )
    resumed_optimizer.load_state_dict(saved_optimizer)
    torch.set_rng_state(saved_rng)
    resumed_losses.extend(
        _run_optimizer_steps(
            resumed_diffusion, resumed_field, resumed_optimizer, value, 2
        )
    )

    assert resumed_losses == uninterrupted_losses
    assert torch.equal(torch.get_rng_state(), uninterrupted_rng)
    for name, expected in uninterrupted_diffusion.state_dict().items():
        assert torch.equal(resumed_diffusion.state_dict()[name], expected)
    for name, expected in uninterrupted_field.state_dict().items():
        assert torch.equal(resumed_field.state_dict()[name], expected)


def _cuda_cache_processor_step(
    processor: CacheProcessor,
    optimizer: torch.optim.Optimizer,
    value: torch.Tensor,
    positions: torch.Tensor,
 ) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = processor.compute_update(value, positions).square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("GCW_RUN_CUDA_DIFFUSION_COMPILE_TESTS") != "1",
    reason="set GCW_RUN_CUDA_DIFFUSION_COMPILE_TESTS=1 for production closure/trajectory smoke",
)
def test_cuda_production_closure_optimizer_resume_and_graph_telemetry() -> None:
    config = tiny_diffusion_config()
    assert config.cache_processor is not None
    assert config.diffusion_processor is not None
    torch.manual_seed(731)
    template = CacheProcessor(
        8, config.cache_processor, diffusion_config=config.diffusion_processor
    ).cuda()
    value = torch.randn(1, 4, 8, device="cuda")
    positions = torch.arange(4, device="cuda").view(1, 4)

    eager = copy.deepcopy(template)
    compiled = copy.deepcopy(template)
    configure_noisy_diffusion_recurrence_backend(compiled, config, _compiled_policy())
    eager_optimizer = torch.optim.AdamW(eager.parameters(), lr=1e-3)
    compiled_optimizer = torch.optim.AdamW(compiled.parameters(), lr=1e-3)
    initial_cpu_rng = torch.get_rng_state().clone()
    initial_cuda_rng = torch.cuda.get_rng_state().clone()

    eager_losses = [
        _cuda_cache_processor_step(eager, eager_optimizer, value, positions)
        for _ in range(2)
    ]
    eager_cpu_rng = torch.get_rng_state().clone()
    eager_cuda_rng = torch.cuda.get_rng_state().clone()
    torch.set_rng_state(initial_cpu_rng)
    torch.cuda.set_rng_state(initial_cuda_rng)
    telemetry_before = noisy_diffusion_compile_telemetry(compiled.diffusion)  # type: ignore[arg-type]
    compiled_losses = [
        _cuda_cache_processor_step(compiled, compiled_optimizer, value, positions)
        for _ in range(2)
    ]
    telemetry_after = noisy_diffusion_compile_telemetry(compiled.diffusion)  # type: ignore[arg-type]

    torch.testing.assert_close(
        torch.tensor(compiled_losses), torch.tensor(eager_losses), atol=2e-5, rtol=2e-4
    )
    assert torch.equal(torch.get_rng_state(), eager_cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), eager_cuda_rng)
    assert compiled.diffusion is not None
    assert compiled.diffusion.refinement_gate.grad is not None
    assert compiled.layers[0].down_proj.weight.grad is not None
    assert torch.count_nonzero(compiled.diffusion.refinement_gate.grad) > 0
    assert torch.count_nonzero(compiled.layers[0].down_proj.weight.grad) > 0
    assert telemetry_after["runner_compile_count"] == 1
    assert telemetry_after["runner_dispatch_count"] == 2
    assert int(telemetry_after["dynamo_unique_graphs"]) >= int(
        telemetry_before["dynamo_unique_graphs"]
    )

    first = copy.deepcopy(template)
    configure_noisy_diffusion_recurrence_backend(first, config, _compiled_policy())
    first_optimizer = torch.optim.AdamW(first.parameters(), lr=1e-3)
    torch.set_rng_state(initial_cpu_rng)
    torch.cuda.set_rng_state(initial_cuda_rng)
    _cuda_cache_processor_step(first, first_optimizer, value, positions)
    saved_model = copy.deepcopy(first.state_dict())
    saved_optimizer = copy.deepcopy(first_optimizer.state_dict())
    saved_cpu_rng = torch.get_rng_state().clone()
    saved_cuda_rng = torch.cuda.get_rng_state().clone()

    resumed = copy.deepcopy(template)
    resumed.load_state_dict(saved_model)
    configure_noisy_diffusion_recurrence_backend(resumed, config, _compiled_policy())
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(saved_optimizer)
    torch.set_rng_state(saved_cpu_rng)
    torch.cuda.set_rng_state(saved_cuda_rng)
    resumed_loss = _cuda_cache_processor_step(
        resumed, resumed_optimizer, value, positions
    )

    assert resumed_loss == pytest.approx(compiled_losses[1], abs=2e-5, rel=2e-4)
    for name, expected in compiled.state_dict().items():
        torch.testing.assert_close(
            resumed.state_dict()[name], expected, atol=2e-5, rtol=2e-4
        )
