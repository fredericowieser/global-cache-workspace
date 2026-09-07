from __future__ import annotations

import copy
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
import pickle
from threading import Barrier, Lock
import time

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
from gcw.training.audit import _training_fingerprint
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
    *,
    fail: bool = False,
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
    compile_calls, dispatch_rng_states, dispatched_states = (
        _install_fake_compiler(monkeypatch)
    )
    configure_noisy_diffusion_recurrence_backend(
        _DiffusionHolder(compiled), tiny_diffusion_config(), _compiled_policy()
    )

    seed = 991
    torch.manual_seed(seed)
    expected_sample = torch.randn_like(eager_input)
    expected_noisy_state = (
        eager_input.detach() + eager.config.noise_scale * expected_sample
    )
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
    torch.testing.assert_close(
        dispatched_states[0], expected_noisy_state, atol=0, rtol=0
    )
    torch.testing.assert_close(
        compiled_input.grad, eager_input.grad, atol=0, rtol=0
    )
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
    compile_calls, dispatch_rng_states, dispatched_states = (
        _install_fake_compiler(monkeypatch)
    )
    configure_noisy_diffusion_recurrence_backend(
        holder, config, _compiled_policy()
    )
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


def test_eager_default_never_calls_torch_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_diffusion_config()
    assert config.diffusion_processor is not None
    diffusion = TaskDirectedDiffusion(4, config.diffusion_processor)

    def fail_compile(*_args: object, **_kwargs: object) -> None:
        pytest.fail(
            "eager noisy-Diffusion recurrence must not call torch.compile"
        )

    monkeypatch.setattr(torch, "compile", fail_compile)
    output = diffusion(
        torch.zeros(1, 2, 4), _Field(4), noise=torch.zeros(1, 2, 4)
    )
    assert output.shape == (1, 2, 4)


def test_unavailable_compiler_fails_before_rng_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_diffusion_config()
    assert config.diffusion_processor is not None
    holder = _DiffusionHolder(
        TaskDirectedDiffusion(4, config.diffusion_processor)
    )
    monkeypatch.setattr(torch, "compile", None)
    torch.manual_seed(81)
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="requires torch.compile"):
        configure_noisy_diffusion_recurrence_backend(
            holder, config, _compiled_policy()
        )
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
    configure_noisy_diffusion_recurrence_backend(
        holder, config, _compiled_policy()
    )

    with pytest.raises(
        RuntimeError, match="eager fallback is disabled"
    ) as error:
        diffusion(torch.zeros(1, 2, 4), _Field(4))
    assert isinstance(error.value.__cause__, RuntimeError)
    assert tuple(holder.state_dict()) == tuple(before)
    assert all(
        torch.equal(value, before[name])
        for name, value in holder.state_dict().items()
    )

    no_noise = copy.deepcopy(config)
    assert no_noise.diffusion_processor is not None
    no_noise.diffusion_processor.noise_scale = 0.0
    with pytest.raises(RuntimeError, match="noise_scale>0"):
        configure_noisy_diffusion_recurrence_backend(
            holder, no_noise, _compiled_policy()
        )

    wrong_variant = copy.deepcopy(config)
    wrong_variant.variant = "longshot"
    with pytest.raises(RuntimeError, match="requires BGT"):
        configure_noisy_diffusion_recurrence_backend(
            holder, wrong_variant, _compiled_policy()
        )


def test_default_fingerprint_is_compatible_and_compile_is_authenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = tiny_diffusion_config()
    default = TrainConfig(model=model_config, data=DataConfig(kind="random"))
    explicit_eager = replace(
        default,
        performance=PerformanceConfig(
            noisy_diffusion=NoisyDiffusionBackendConfig(
                recurrence_backend="eager"
            )
        ),
    )
    default_fingerprint = _training_fingerprint(default, 1, 8, 8, 1)
    eager_fingerprint = _training_fingerprint(explicit_eager, 1, 8, 8, 1)
    assert default_fingerprint == eager_fingerprint
    assert (
        "noisy_diffusion_recurrence_backend_fingerprint"
        not in default_fingerprint
    )

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
    assert (
        metadata["noise_boundary"]
        == "eager_torch_randn_like_before_recurrence_dispatch"
    )
    assert metadata["effective_backend"] == "compile"


def test_configuration_rejects_invalid_or_incompatible_compile_requests() -> (
    None
):
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
    not torch.cuda.is_available()
    or os.environ.get("GCW_RUN_CUDA_DIFFUSION_COMPILE_TESTS") != "1",
    reason="set GCW_RUN_CUDA_DIFFUSION_COMPILE_TESTS=1 on CUDA to exercise fullgraph Inductor",
)
def test_cuda_fullgraph_compiles_field_callable_and_preserves_backward() -> (
    None
):
    config = tiny_diffusion_config()
    assert config.diffusion_processor is not None
    eager = TaskDirectedDiffusion(
        8, replace(config.diffusion_processor, num_steps=4)
    ).cuda()
    compiled = copy.deepcopy(eager)
    eager_field = _Field(8).cuda()
    compiled_field = copy.deepcopy(eager_field)
    holder = _DiffusionHolder(compiled)
    configure_noisy_diffusion_recurrence_backend(
        holder, config, _compiled_policy()
    )
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

    torch.testing.assert_close(
        compiled_output, eager_output, atol=2e-5, rtol=2e-4
    )
    torch.testing.assert_close(
        compiled_input.grad, eager_input.grad, atol=2e-5, rtol=2e-4
    )
    assert torch.equal(torch.cuda.get_rng_state(), eager_rng)
    torch.testing.assert_close(
        compiled.refinement_gate.grad,
        eager.refinement_gate.grad,
        atol=2e-5,
        rtol=2e-4,
    )
    torch.testing.assert_close(
        compiled_field.projection.weight.grad,
        eager_field.projection.weight.grad,
        atol=2e-5,
        rtol=2e-4,
    )
    before = noisy_diffusion_compile_telemetry(compiled)
    repeated = torch.randn(1, 6, 8, device="cuda", requires_grad=True)
    compiled(
        repeated, lambda value, time: compiled_field(value, time)
    ).square().mean().backward()
    after = noisy_diffusion_compile_telemetry(compiled)
    assert after["runner_creation_count"] == 1
    assert after["runner_dispatch_count"] == 2
    # Process-wide counters are diagnostics only, but this isolated CUDA test must
    # observe a new static-shape graph without introducing a graph break.
    assert after["dynamo_process_diagnostics_acceptance_eligible"] is False
    assert int(after["dynamo_process_delta_unique_graphs"]) > int(
        before["dynamo_process_delta_unique_graphs"]
    )
    assert (
        after["dynamo_process_delta_graph_breaks"]
        == before["dynamo_process_delta_graph_breaks"]
    )
    assert before["runner_creation_count"] == 1


def test_workspace_sunset_updates_default_receipts() -> None:
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
        "af5e1820986481ef2f0b99e79e10a173e2c91146e0372b9042bc667e0d0faef7"
    )
    assert model_receipt.receipt_sha256 == (
        "4067252544760def3a922df3aa431739cc01a846e17cfdb97e9e085d7920f9c7"
    )
    assert composition_receipt.train_config_sha256 == (
        "49da481ae4e8ac03f5ec87182109184613e00b61d9cd5eeeda3a9f28419e3a1f"
    )
    assert composition_receipt.receipt_sha256 == (
        "20124b45911238bdd442d2f3256cda2599de5740f04a79a0849916e812f82bc8"
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
        configure_noisy_diffusion_recurrence_backend(
            holder, model_config, _compiled_policy()
        )
    with pytest.raises(ValueError, match="attention_dropout=0"):
        TrainConfig(
            model=model_config,
            data=DataConfig(kind="random"),
            performance=PerformanceConfig(noisy_diffusion=_compiled_policy()),
        )


def test_materialized_runner_does_not_bind_deepcopy_to_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls, _, _ = _install_fake_compiler(monkeypatch)
    config = tiny_diffusion_config()
    assert config.cache_processor is not None
    assert config.diffusion_processor is not None
    source = CacheProcessor(
        8, config.cache_processor, diffusion_config=config.diffusion_processor
    )
    configure_noisy_diffusion_recurrence_backend(
        source, config, _compiled_policy()
    )
    value = torch.randn(1, 4, 8)
    positions = torch.arange(4).view(1, 4)
    source_output = source.compute_update(value, positions)
    assert source.diffusion is not None
    assert (
        noisy_diffusion_compile_telemetry(source.diffusion)[
            "runner_creation_count"
        ]
        == 1
    )

    copied = copy.deepcopy(source)
    assert copied.diffusion is not None
    assert (
        noisy_diffusion_compile_telemetry(copied.diffusion)["runner_owned"]
        is False
    )
    with torch.no_grad():
        copied.diffusion.refinement_gate.add_(1.0)
        copied.layers[0].down_proj.weight.add_(0.25)
    copied_output = copied.compute_update(value, positions)
    assert len(compile_calls) == 2
    assert not torch.allclose(copied_output, source_output)

    source.zero_grad(set_to_none=True)
    copied.zero_grad(set_to_none=True)
    copied.compute_update(value, positions).square().mean().backward()
    assert source.diffusion.refinement_gate.grad is None
    assert source.layers[0].down_proj.weight.grad is None
    assert copied.diffusion.refinement_gate.grad is not None
    assert copied.layers[0].down_proj.weight.grad is not None
    assert torch.count_nonzero(copied.diffusion.refinement_gate.grad) > 0
    assert torch.count_nonzero(copied.layers[0].down_proj.weight.grad) > 0


def test_materialized_runner_is_not_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls, _, _ = _install_fake_compiler(monkeypatch)
    config = tiny_diffusion_config()
    assert config.diffusion_processor is not None
    diffusion = TaskDirectedDiffusion(4, config.diffusion_processor)
    configure_noisy_diffusion_recurrence_backend(
        _DiffusionHolder(diffusion), config, _compiled_policy()
    )
    field = _Field(4)
    value = torch.randn(1, 3, 4)
    expected = diffusion(value, field, noise=torch.ones_like(value))
    restored = pickle.loads(pickle.dumps(diffusion))
    assert noisy_diffusion_compile_telemetry(restored)["runner_owned"] is False
    actual = restored(value, field, noise=torch.ones_like(value))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(compile_calls) == 2


def test_dynamo_telemetry_labels_process_wide_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from torch._dynamo.testing import CompileCounterWithBackend
        from torch._dynamo.utils import counters
    except ImportError:  # pragma: no cover - version-dependent support
        pytest.skip("Torch Dynamo compile counters are unavailable")
    torch._dynamo.reset()
    counters.clear()
    real_compile = torch.compile
    compile_counter = CompileCounterWithBackend("eager")

    def counting_compile(
        function: Callable[..., torch.Tensor], **kwargs: object
    ):
        assert kwargs == {
            "backend": "inductor",
            "dynamic": False,
            "fullgraph": True,
        }
        return real_compile(
            function,
            backend=compile_counter,
            dynamic=False,
            fullgraph=True,
        )

    monkeypatch.setattr(torch, "compile", counting_compile)
    config = tiny_diffusion_config()
    assert config.diffusion_processor is not None
    diffusion = TaskDirectedDiffusion(4, config.diffusion_processor)
    configure_noisy_diffusion_recurrence_backend(
        _DiffusionHolder(diffusion), config, _compiled_policy()
    )
    before = noisy_diffusion_compile_telemetry(diffusion)
    assert before["runner_creation_count"] == 0
    assert before["dynamo_process_delta_unique_graphs"] == 0

    field = _Field(4)
    for width in (3, 3, 5, 3, 5):
        value = torch.zeros(1, width, 4)
        diffusion(value, field, noise=torch.ones_like(value))
    after = noisy_diffusion_compile_telemetry(diffusion)

    assert after["runner_creation_count"] == 1
    assert after["runner_dispatch_count"] == 5
    assert compile_counter.frame_count == 2
    assert (
        after["telemetry_schema"] == "gcw.noisy_diffusion_compile_telemetry.v2"
    )
    assert after["dynamo_process_diagnostics_acceptance_eligible"] is False
    # CompileCounterWithBackend is scoped to this runner; process counters are not.
    assert int(after["dynamo_process_delta_unique_graphs"]) >= 0
    assert int(after["dynamo_process_delta_graph_breaks"]) >= 0
    assert int(after["dynamo_process_delta_recompiles"]) >= 0


def test_concurrent_first_dispatch_serializes_lazy_shape_specialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls: list[dict[str, object]] = []
    call_guard = Lock()
    active_calls = 0
    maximum_active_calls = 0

    def fake_compile(function: Callable[..., torch.Tensor], **kwargs: object):
        nonlocal active_calls, maximum_active_calls
        compile_calls.append(dict(kwargs))

        def compiled(*args: object, **inner_kwargs: object) -> torch.Tensor:
            nonlocal active_calls, maximum_active_calls
            with call_guard:
                active_calls += 1
                maximum_active_calls = max(maximum_active_calls, active_calls)
            try:
                # An unsafe backend would overlap here. Avoid invoking real Dynamo
                # concurrently because that failure can terminate the test process.
                time.sleep(0.02)
                return function(*args, **inner_kwargs)
            finally:
                with call_guard:
                    active_calls -= 1

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    torch.manual_seed(227)
    config = tiny_diffusion_config()
    assert config.diffusion_processor is not None
    diffusion = TaskDirectedDiffusion(4, config.diffusion_processor)
    eager = copy.deepcopy(diffusion)
    field = _Field(4)
    eager_field = copy.deepcopy(field)
    configure_noisy_diffusion_recurrence_backend(
        _DiffusionHolder(diffusion), config, _compiled_policy()
    )
    values = [torch.randn(1, width, 4, requires_grad=True) for width in (3, 5)]
    eager_values = [
        value.detach().clone().requires_grad_(True) for value in values
    ]
    start = Barrier(3)

    def dispatch(index: int) -> torch.Tensor:
        start.wait()
        value = values[index]
        return diffusion(value, field, noise=torch.zeros_like(value))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(dispatch, index) for index in range(2)]
        start.wait()
        outputs = [future.result(timeout=5) for future in futures]

    expected_outputs = [
        eager(value, eager_field, noise=torch.zeros_like(value))
        for value in eager_values
    ]
    for output, expected in zip(outputs, expected_outputs, strict=True):
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
    sum(output.square().mean() for output in outputs).backward()

    assert compile_calls == [
        {"backend": "inductor", "dynamic": False, "fullgraph": True}
    ]
    assert maximum_active_calls == 1
    assert active_calls == 0
    assert all(value.grad is not None for value in values)
    assert diffusion.refinement_gate.grad is not None
    assert torch.count_nonzero(diffusion.refinement_gate.grad) > 0
    assert field.projection.weight.grad is not None
    assert torch.count_nonzero(field.projection.weight.grad) > 0
    assert eager.refinement_gate.grad is None
    assert eager_field.projection.weight.grad is None
    telemetry = noisy_diffusion_compile_telemetry(diffusion)
    assert telemetry["runner_creation_count"] == 1
    assert telemetry["runner_dispatch_count"] == 2


def test_compiled_boundary_preserves_checkpoint_early_stop_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_compiler(monkeypatch)
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
        _diffusion: TaskDirectedDiffusion,
        _state: torch.Tensor,
        _bottleneck: torch.Tensor,
        _field: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        raise signal

    # Replace only the external execution runner; no model state is modified.
    from gcw.performance.backends import diffusion as diffusion_backend

    diffusion(torch.zeros(1, 2, 4), _Field(4), noise=torch.zeros(1, 2, 4))
    runner_state = diffusion_backend._compiled_runner_states[diffusion]
    runner_state.runner = early_stop
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
    assert (
        int(
            noisy_diffusion_compile_telemetry(diffusion)[
                "runner_dispatch_count"
            ]
        )
        >= 2
    )


def test_production_cache_processor_closure_and_parameters_are_state_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_compiler(monkeypatch)
    model_config = tiny_diffusion_config()
    assert model_config.cache_processor is not None
    assert model_config.diffusion_processor is not None
    processor = CacheProcessor(
        8,
        model_config.cache_processor,
        diffusion_config=model_config.diffusion_processor,
    )
    before_keys = tuple(processor.state_dict())
    configure_noisy_diffusion_recurrence_backend(
        processor, model_config, _compiled_policy()
    )
    calls: list[int] = []
    hook = processor.layers[0].register_forward_hook(
        lambda *_args: calls.append(1)
    )
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
        list(uninterrupted_diffusion.parameters())
        + list(uninterrupted_field.parameters()),
        lr=1e-3,
    )
    initial_rng = torch.get_rng_state().clone()
    uninterrupted_losses = _run_optimizer_steps(
        uninterrupted_diffusion,
        uninterrupted_field,
        uninterrupted_optimizer,
        value,
        4,
    )
    uninterrupted_rng = torch.get_rng_state().clone()

    first_diffusion = copy.deepcopy(template_diffusion)
    first_field = copy.deepcopy(template_field)
    configure_noisy_diffusion_recurrence_backend(
        _DiffusionHolder(first_diffusion), config, _compiled_policy()
    )
    first_optimizer = torch.optim.AdamW(
        list(first_diffusion.parameters()) + list(first_field.parameters()),
        lr=1e-3,
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
        list(resumed_diffusion.parameters()) + list(resumed_field.parameters()),
        lr=1e-3,
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


def _assert_module_state_close(
    actual: nn.Module,
    expected: nn.Module,
    *,
    atol: float = 2e-5,
    rtol: float = 2e-4,
) -> None:
    actual_state = actual.state_dict()
    expected_state = expected.state_dict()
    assert tuple(actual_state) == tuple(expected_state)
    for name, expected_value in expected_state.items():
        torch.testing.assert_close(
            actual_state[name], expected_value, atol=atol, rtol=rtol
        )


def _assert_optimizer_state_close(
    actual: torch.optim.Optimizer,
    expected: torch.optim.Optimizer,
    *,
    atol: float = 2e-5,
    rtol: float = 2e-4,
) -> None:
    actual_state = actual.state_dict()
    expected_state = expected.state_dict()
    assert actual_state["param_groups"] == expected_state["param_groups"]
    assert actual_state["state"].keys() == expected_state["state"].keys()
    for parameter_id, expected_slots in expected_state["state"].items():
        actual_slots = actual_state["state"][parameter_id]
        assert actual_slots.keys() == expected_slots.keys()
        for slot_name, expected_value in expected_slots.items():
            actual_value = actual_slots[slot_name]
            if isinstance(expected_value, torch.Tensor):
                assert isinstance(actual_value, torch.Tensor)
                torch.testing.assert_close(
                    actual_value, expected_value, atol=atol, rtol=rtol
                )
            else:
                assert actual_value == expected_value


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or os.environ.get("GCW_RUN_CUDA_DIFFUSION_COMPILE_TESTS") != "1",
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
    configure_noisy_diffusion_recurrence_backend(
        compiled, config, _compiled_policy()
    )
    eager_optimizer = torch.optim.AdamW(eager.parameters(), lr=1e-3)
    compiled_optimizer = torch.optim.AdamW(compiled.parameters(), lr=1e-3)
    initial_cpu_rng = torch.get_rng_state().clone()
    initial_cuda_rng = torch.cuda.get_rng_state().clone()

    eager_losses = [
        _cuda_cache_processor_step(eager, eager_optimizer, value, positions)
        for _ in range(4)
    ]
    eager_cpu_rng = torch.get_rng_state().clone()
    eager_cuda_rng = torch.cuda.get_rng_state().clone()
    torch.set_rng_state(initial_cpu_rng)
    torch.cuda.set_rng_state(initial_cuda_rng)
    compiled_losses = [
        _cuda_cache_processor_step(
            compiled, compiled_optimizer, value, positions
        )
        for _ in range(4)
    ]
    telemetry_after = noisy_diffusion_compile_telemetry(compiled.diffusion)  # type: ignore[arg-type]

    torch.testing.assert_close(
        torch.tensor(compiled_losses),
        torch.tensor(eager_losses),
        atol=2e-5,
        rtol=2e-4,
    )
    assert torch.equal(torch.get_rng_state(), eager_cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), eager_cuda_rng)
    _assert_module_state_close(compiled, eager)
    _assert_optimizer_state_close(compiled_optimizer, eager_optimizer)
    assert compiled.diffusion is not None
    assert compiled.diffusion.refinement_gate.grad is not None
    assert compiled.layers[0].down_proj.weight.grad is not None
    assert torch.count_nonzero(compiled.diffusion.refinement_gate.grad) > 0
    assert torch.count_nonzero(compiled.layers[0].down_proj.weight.grad) > 0
    assert telemetry_after["runner_creation_count"] == 1
    assert telemetry_after["runner_dispatch_count"] == 4
    assert (
        telemetry_after["telemetry_schema"]
        == "gcw.noisy_diffusion_compile_telemetry.v2"
    )
    assert (
        telemetry_after["dynamo_process_diagnostics_acceptance_eligible"]
        is False
    )
    # Process-global Dynamo diagnostics are informational, never acceptance gates.
    assert int(telemetry_after["dynamo_process_delta_unique_graphs"]) >= 0
    assert int(telemetry_after["dynamo_process_delta_graph_breaks"]) >= 0
    assert int(telemetry_after["dynamo_process_delta_recompiles"]) >= 0

    first = copy.deepcopy(template)
    configure_noisy_diffusion_recurrence_backend(
        first, config, _compiled_policy()
    )
    first_optimizer = torch.optim.AdamW(first.parameters(), lr=1e-3)
    torch.set_rng_state(initial_cpu_rng)
    torch.cuda.set_rng_state(initial_cuda_rng)
    resumed_losses = [
        _cuda_cache_processor_step(first, first_optimizer, value, positions)
        for _ in range(2)
    ]
    saved_model = copy.deepcopy(first.state_dict())
    saved_optimizer = copy.deepcopy(first_optimizer.state_dict())
    saved_cpu_rng = torch.get_rng_state().clone()
    saved_cuda_rng = torch.cuda.get_rng_state().clone()

    resumed = copy.deepcopy(template)
    resumed.load_state_dict(saved_model)
    configure_noisy_diffusion_recurrence_backend(
        resumed, config, _compiled_policy()
    )
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(saved_optimizer)
    torch.set_rng_state(saved_cpu_rng)
    torch.cuda.set_rng_state(saved_cuda_rng)
    resumed_losses.extend(
        _cuda_cache_processor_step(resumed, resumed_optimizer, value, positions)
        for _ in range(2)
    )

    torch.testing.assert_close(
        torch.tensor(resumed_losses),
        torch.tensor(compiled_losses),
        atol=2e-5,
        rtol=2e-4,
    )
    _assert_module_state_close(resumed, compiled)
    _assert_optimizer_state_close(resumed_optimizer, compiled_optimizer)
    assert torch.equal(torch.get_rng_state(), eager_cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), eager_cuda_rng)
