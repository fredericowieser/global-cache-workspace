from __future__ import annotations

import copy

import pytest
import torch

from gcw.config import CacheProcessorConfig, DataConfig, ModelConfig, TrainConfig
from gcw.models import build_model
from gcw.performance.backends import cache_processors as backend_module
from gcw.performance.backends.cache_processors import bt_processor_backend_metadata
from gcw.models.model import LanguageModel
from gcw.performance.checkpointing import (
    configure_native_activation_checkpointing as _configure_native_activation_checkpointing,
)
from gcw.training.audit import _training_fingerprint


def _config() -> ModelConfig:
    return ModelConfig(
        variant="bt",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=32,
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=2,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
            attention_dropout=0.0,
            gate_mode="direct",
            gate_init=0.5,
        ),
    )


def _model() -> LanguageModel:
    model = build_model(_config())
    assert isinstance(model, LanguageModel)
    model.init_weights()
    return model


def _install_fake_compiler(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict[str, object]], list[int]]:
    compile_calls: list[dict[str, object]] = []
    region_calls: list[int] = []

    def fake_compile(function, **kwargs):
        compile_calls.append(kwargs)

        def compiled(*args, **inner_kwargs):
            region_calls.append(args[1].shape[2])
            return function(*args, **inner_kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(backend_module, "_compiled_runner", None)
    return compile_calls, region_calls


def test_bt_backend_dispatches_only_prepared_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCW_BT_PROCESSOR_BACKEND", "compile")
    compile_calls, region_calls = _install_fake_compiler(monkeypatch)
    model = _model()

    inputs = torch.randint(0, model.config.vocab_size, (1, 9))
    model.generate(inputs, max_new_tokens=1)
    assert compile_calls == []
    assert model.cache_processor.pop_call_count() > 0

    model(inputs)

    assert compile_calls == [{"backend": "inductor", "dynamic": False, "fullgraph": True}]
    assert region_calls == [4, 6]
    assert model.cache_processor.pop_call_count() == 2
    assert model.cache_processor.execution_backend == "compile"


def test_explicit_eager_is_cpu_safe_and_never_calls_torch_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCW_BT_PROCESSOR_BACKEND", "eager")

    def fail_compile(*_args, **_kwargs):
        pytest.fail("explicit eager backend must not call torch.compile")

    monkeypatch.setattr(torch, "compile", fail_compile)
    model = _model()
    output = model(torch.randint(0, model.config.vocab_size, (1, 9)))

    assert output.shape == (1, 9, model.config.vocab_size)
    assert output.device.type == "cpu"


def test_compiled_dispatch_preserves_outputs_gradients_train_eval_and_repeated_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(41)
    eager = _model()
    compiled = copy.deepcopy(eager)
    monkeypatch.setenv("GCW_BT_PROCESSOR_BACKEND", "compile")
    _install_fake_compiler(monkeypatch)
    compiled.cache_processor.execution_backend = "compile"
    inputs = torch.randint(0, eager.config.vocab_size, (2, 9))
    targets = torch.randint(0, eager.config.vocab_size, (2, 9))

    for _ in range(2):
        eager.zero_grad(set_to_none=True)
        compiled.zero_grad(set_to_none=True)
        eager_loss = eager(inputs, targets)
        compiled_loss = compiled(inputs, targets)
        eager_loss.backward()
        compiled_loss.backward()
        torch.testing.assert_close(compiled_loss, eager_loss)
        for eager_parameter, compiled_parameter in zip(eager.parameters(), compiled.parameters(), strict=True):
            assert eager_parameter.grad is not None
            assert compiled_parameter.grad is not None
            torch.testing.assert_close(compiled_parameter.grad, eager_parameter.grad, atol=2e-5, rtol=2e-4)

    eager.eval()
    compiled.eval()
    with torch.no_grad():
        torch.testing.assert_close(compiled(inputs), eager(inputs))


def test_compiled_backend_composes_with_segment_and_processor_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(43)
    eager = _model()
    checkpointed = copy.deepcopy(eager)
    monkeypatch.setenv("GCW_BT_PROCESSOR_BACKEND", "compile")
    compile_calls, region_calls = _install_fake_compiler(monkeypatch)
    checkpointed.cache_processor.execution_backend = "compile"
    _configure_native_activation_checkpointing(checkpointed, "standard_processor_layers")
    inputs = torch.randint(0, eager.config.vocab_size, (2, 12))
    targets = torch.randint(0, eager.config.vocab_size, (2, 12))

    eager_loss = eager(inputs, targets)
    checkpointed_loss = checkpointed(
        inputs,
        targets,
        activation_checkpoint_windows=2,
        activation_checkpoint_min_padded_sequence_length=12,
    )
    eager_loss.backward()
    checkpointed_loss.backward()

    torch.testing.assert_close(checkpointed_loss, eager_loss)
    for expected, actual in zip(eager.parameters(), checkpointed.parameters(), strict=True):
        assert expected.grad is not None and actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-5, rtol=2e-4)
    assert compile_calls == [{"backend": "inductor", "dynamic": False, "fullgraph": True}]
    assert len(region_calls) == 4
    assert checkpointed.cache_processor.pop_call_count() == 2


def test_compiled_backend_is_fail_closed_and_unavailable_compile_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCW_BT_PROCESSOR_BACKEND", "compile")
    monkeypatch.setattr(torch, "compile", None)
    with pytest.raises(RuntimeError, match="requires torch.compile"):
        _model()

    def fake_compile(_function, **_kwargs):
        def fail(*_args, **_inner_kwargs):
            raise RuntimeError("inductor failed")

        return fail

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(backend_module, "_compiled_runner", None)
    model = _model()
    with pytest.raises(RuntimeError, match="eager fallback is disabled") as error:
        model(torch.randint(0, model.config.vocab_size, (1, 9)))
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "inductor failed"


def test_backend_is_state_dict_neutral_and_visible_in_metadata_and_training_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCW_BT_PROCESSOR_BACKEND", "compile")
    _install_fake_compiler(monkeypatch)
    model = _model()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    model(torch.randint(0, model.config.vocab_size, (1, 9)))

    assert tuple(model.state_dict()) == tuple(before)
    assert all("compiled" not in name and "prepared" not in name for name in before)
    assert all(torch.equal(tensor, before[name]) for name, tensor in model.state_dict().items())
    restored = _model()
    incompatible = restored.load_state_dict(before, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    metadata = model.cache_processor.execution_backend_metadata()
    assert metadata == bt_processor_backend_metadata("compile")
    assert metadata["effective_backend"] == "compile"
    assert metadata["region"] == "gcw.models.cache_processors.ops.run_prepared_batched_cache_processors"
    assert isinstance(metadata["fingerprint_sha256"], str)
    assert len(metadata["fingerprint_sha256"]) == 64

    train_config = TrainConfig(model=model.config, data=DataConfig(kind="random"))
    fingerprint = _training_fingerprint(train_config, 1, 32, 32, 1)
    assert fingerprint["bt_processor_backend_fingerprint"] == metadata["fingerprint_sha256"]
