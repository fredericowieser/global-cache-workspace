from __future__ import annotations

import copy
from collections.abc import Callable

import pytest
import torch

from scripts.benchmarks.parity import (
    assert_cache_close,
    assert_parameter_gradients_close,
    assert_tensor_gradients_close,
    cache_objective,
    clone_cache,
)
from gcw.config import (
    CacheProcessorConfig,
    DataConfig,
    DiffusionProcessorConfig,
    FlowProcessorConfig,
    ModelConfig,
    TrainConfig,
)
from gcw.models import build_model
from gcw.performance.backends import cache_processors as backend_module
from gcw.performance.backends.cache_processors import (
    cache_processor_backend_metadata,
)
from gcw.models.cache_processors.standard import GlobalCacheProcessor
from gcw.models.cache_processors.state import CacheState
from gcw.models.model import LanguageModel
from gcw.performance.checkpointing import (
    configure_native_activation_checkpointing as _configure_native_activation_checkpointing,
)
from gcw.training.audit import _training_fingerprint


def _config(kind: str = "standard", *, process_every: int = 64) -> ModelConfig:
    flow = FlowProcessorConfig(num_steps=2) if kind == "flow" else None
    diffusion = (
        DiffusionProcessorConfig(num_steps=2, noise_scale=0.0)
        if kind == "diffusion"
        else None
    )
    return ModelConfig(
        variant="bgt",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=192,
        cache_processor=CacheProcessorConfig(
            process_every=process_every,
            topk=32,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
            attention_dropout=0.0,
            gate_mode="direct",
            gate_init=0.5,
        ),
        flow_processor=flow,
        diffusion_processor=diffusion,
    )


def _model(config: ModelConfig) -> LanguageModel:
    model = build_model(config)
    assert isinstance(model, LanguageModel)
    model.init_weights()
    return model


def _processor(model: LanguageModel) -> GlobalCacheProcessor:
    processor = model.cache_processor
    assert isinstance(processor, GlobalCacheProcessor)
    return processor


def _cache(config: ModelConfig, batch: int, seq: int) -> CacheState:
    generator = torch.Generator().manual_seed(1000 + batch + seq)
    num_heads, head_dim = config.stored_cache_shape
    shape = (batch, num_heads, seq, head_dim)
    keys = [
        torch.randn(shape, generator=generator)
        for _ in range(config.cache_num_layers)
    ]
    values = [
        torch.randn(shape, generator=generator)
        for _ in range(config.cache_num_layers)
    ]
    scores = torch.randn((batch, seq), generator=generator)
    return CacheState(keys, values, scores)


def _install_fake_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[str, dict[str, object]]], list[tuple[str, int, int]]]:
    compile_calls: list[tuple[str, dict[str, object]]] = []
    region_calls: list[tuple[str, int, int]] = []

    def fake_compile(function: Callable[..., torch.Tensor], **kwargs: object):
        compile_calls.append((function.__name__, dict(kwargs)))

        def compiled(*args: object, **inner_kwargs: object) -> torch.Tensor:
            x = args[1]
            assert isinstance(x, torch.Tensor)
            region_calls.append((function.__name__, x.shape[0], x.shape[1]))
            return function(*args, **inner_kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    backend_module._compiled_bgt_transform_runner = None
    backend_module._compiled_bgt_scaled_update_runner = None
    return compile_calls, region_calls


@pytest.mark.parametrize("kind", ("standard", "flow", "diffusion"))
def test_compiled_bgt_regions_match_forward_backward_and_selected_indices(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    torch.manual_seed(2026)
    config = _config(kind)
    eager = _processor(_model(config))
    compiled = copy.deepcopy(eager)
    monkeypatch.setenv("GCW_CACHE_PROCESSOR_BACKEND", "compile")
    compile_calls, region_calls = _install_fake_compiler(monkeypatch)
    compiled.execution_backend = "compile"

    for batch, seq, selected in ((1, 64, 64), (2, 128, 96)):
        source = _cache(config, batch, seq)
        eager_cache = clone_cache(source, requires_grad=True)
        compiled_cache = clone_cache(source, requires_grad=True)
        eager_keys, eager_values = list(eager_cache.keys), list(
            eager_cache.values
        )
        compiled_keys, compiled_values = list(compiled_cache.keys), list(
            compiled_cache.values
        )
        expected_indices = eager._select_indices(eager_cache, False)
        actual_indices = compiled._select_indices(compiled_cache, False)

        eager.zero_grad(set_to_none=True)
        compiled.zero_grad(set_to_none=True)
        expected = eager(eager_cache)
        actual = compiled(compiled_cache)
        cache_objective(expected).backward()
        cache_objective(actual).backward()

        assert torch.equal(actual_indices, expected_indices)
        assert actual_indices.shape == (batch, selected)
        assert_cache_close(actual, expected)
        assert_tensor_gradients_close(compiled_keys, eager_keys)
        assert_tensor_gradients_close(compiled_values, eager_values)
        assert_parameter_gradients_close(compiled, eager)

    assert compile_calls == [
        (
            "_run_bgt_transform",
            {"backend": "inductor", "dynamic": False, "fullgraph": True},
        )
    ]
    assert region_calls == [
        ("_run_bgt_transform", 1, 64),
        ("_run_bgt_transform", 2, 96),
    ]


def test_compiled_bgt_callable_is_reused_across_repeated_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    processor = _processor(_model(config))
    monkeypatch.setenv("GCW_CACHE_PROCESSOR_BACKEND", "compile")
    compile_calls, region_calls = _install_fake_compiler(monkeypatch)
    processor.execution_backend = "compile"

    for _ in range(2):
        processor.zero_grad(set_to_none=True)
        output = processor(
            clone_cache(_cache(config, 2, 128), requires_grad=True)
        )
        cache_objective(output).backward()

    assert len(compile_calls) == 1
    assert region_calls == [
        ("_run_bgt_transform", 2, 96),
        ("_run_bgt_transform", 2, 96),
    ]


def test_compiled_bgt_composes_with_segment_and_processor_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(2041)
    config = _config(process_every=4)
    config.sequence_len = 12
    eager = _model(config)
    compiled = copy.deepcopy(eager)
    monkeypatch.setenv("GCW_CACHE_PROCESSOR_BACKEND", "compile")
    compile_calls, region_calls = _install_fake_compiler(monkeypatch)
    compiled_processor = _processor(compiled)
    compiled_processor.execution_backend = "compile"
    _configure_native_activation_checkpointing(
        compiled, "standard_processor_layers"
    )
    inputs = torch.randint(0, config.vocab_size, (2, 12))
    targets = torch.roll(inputs, shifts=-1, dims=1)

    eager_loss = eager(inputs, targets)
    compiled_loss = compiled(inputs, targets, activation_checkpoint_windows=2)
    eager_loss.backward()
    compiled_loss.backward()

    torch.testing.assert_close(compiled_loss, eager_loss)
    for expected, actual in zip(
        eager.parameters(), compiled.parameters(), strict=True
    ):
        assert expected.grad is not None and actual.grad is not None
        torch.testing.assert_close(
            actual.grad, expected.grad, atol=2e-5, rtol=2e-4
        )
    assert len(compile_calls) == 1
    assert len(region_calls) == 4
    assert compiled_processor.pop_call_count() == 2


def test_eager_default_never_compiles_and_unavailable_opt_in_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_compile(*_args: object, **_kwargs: object) -> None:
        pytest.fail("eager cache processing must not call torch.compile")

    monkeypatch.setenv("GCW_CACHE_PROCESSOR_BACKEND", "eager")
    monkeypatch.setattr(torch, "compile", fail_compile)
    model = _model(_config())
    model.cache_processor(
        clone_cache(_cache(model.config, 1, 64), requires_grad=False)
    )

    monkeypatch.setenv("GCW_CACHE_PROCESSOR_BACKEND", "compile")
    monkeypatch.setattr(torch, "compile", None)
    with pytest.raises(RuntimeError, match="requires torch.compile"):
        build_model(_config())


def test_backend_rejects_bt_longshot_and_noisy_diffusion_without_consuming_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCW_CACHE_PROCESSOR_BACKEND", "compile")
    _install_fake_compiler(monkeypatch)
    bt = _config()
    bt.variant = "bt"
    with pytest.raises(RuntimeError, match="BT's prepared backend"):
        build_model(bt)

    longshot = _config()
    longshot.variant = "longshot"
    with pytest.raises(RuntimeError, match="explicitly unsupported"):
        build_model(longshot)

    noisy = _config("diffusion")
    assert noisy.diffusion_processor is not None
    noisy.diffusion_processor.noise_scale = 0.1
    torch.manual_seed(55)
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="noise_scale>0"):
        build_model(noisy)
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_backend_is_fail_closed_state_dict_neutral_and_exact_resume_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    monkeypatch.setenv("GCW_CACHE_PROCESSOR_BACKEND", "compile")
    _install_fake_compiler(monkeypatch)
    model = _model(config)
    processor = _processor(model)
    before = {
        name: tensor.clone() for name, tensor in model.state_dict().items()
    }
    processor(clone_cache(_cache(config, 1, 64), requires_grad=False))

    assert tuple(model.state_dict()) == tuple(before)
    assert all(
        torch.equal(tensor, before[name])
        for name, tensor in model.state_dict().items()
    )
    restored = _model(config)
    incompatible = restored.load_state_dict(before, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    metadata = processor.execution_backend_metadata()
    assert metadata == cache_processor_backend_metadata(config, "compile")
    assert metadata["architecture"] == "bgt"
    assert metadata["effective_backend"] == "compile"
    assert isinstance(metadata["fingerprint_sha256"], str)
    assert len(metadata["fingerprint_sha256"]) == 64

    train_config = TrainConfig(model=config, data=DataConfig(kind="random"))
    fingerprint = _training_fingerprint(train_config, 1, 192, 192, 1)
    assert (
        fingerprint["cache_processor_backend_fingerprint"]
        == metadata["fingerprint_sha256"]
    )
    monkeypatch.setenv("GCW_CACHE_PROCESSOR_BACKEND", "eager")
    eager_fingerprint = _training_fingerprint(train_config, 1, 192, 192, 1)
    assert (
        eager_fingerprint["cache_processor_backend_fingerprint"]
        != fingerprint["cache_processor_backend_fingerprint"]
    )
    monkeypatch.setenv("GCW_CACHE_PROCESSOR_BACKEND", "compile")

    def fail_compile(_function: Callable[..., torch.Tensor], **_kwargs: object):
        def fail(*_args: object, **_inner_kwargs: object) -> torch.Tensor:
            raise RuntimeError("inductor failed")

        return fail

    monkeypatch.setattr(torch, "compile", fail_compile)
    backend_module._compiled_bgt_transform_runner = None
    backend_module._compiled_bgt_scaled_update_runner = None
    failing = _model(config)
    with pytest.raises(
        RuntimeError, match="eager fallback is disabled"
    ) as error:
        failing.cache_processor(
            clone_cache(_cache(config, 1, 64), requires_grad=False)
        )
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "inductor failed"
