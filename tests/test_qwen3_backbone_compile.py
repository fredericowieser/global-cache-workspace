from __future__ import annotations

import copy
from dataclasses import replace
import json

import pytest
import torch

from gcw.config import CacheProcessorConfig, DataConfig, ModelConfig, ResumeConfig, TrainConfig
from gcw.models import build_model
from gcw.models.backbones.backend import (
    configure_qwen3_backbone_chunk_backend,
    qwen3_backbone_chunk_backend_metadata,
)
from gcw.models.cache_processors import backend as processor_backend_module
from gcw.models.cache_processors.state import CacheState
from gcw.models.model import LanguageModel
from gcw.training.checkpoint import CheckpointCompatibilityError
from gcw.training.core import (
    _apply_trainable_policy,
    _configure_native_activation_checkpointing,
    _training_fingerprint,
    train,
)


def _model_config(*, variant: str = "bt") -> ModelConfig:
    return ModelConfig(
        variant=variant,  # type: ignore[arg-type]
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
        sequence_len=256,
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


def _model() -> LanguageModel:
    model = build_model(_model_config())
    assert isinstance(model, LanguageModel)
    model.init_weights()
    return model


def _install_fake_compiler(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict[str, object]], list[int]]:
    compile_calls: list[dict[str, object]] = []
    qwen_past_lengths: list[int] = []

    def fake_compile(function, **kwargs):
        compile_calls.append(kwargs)

        def compiled(*args, **inner_kwargs):
            if len(args) == 6:
                qwen_past_lengths.append(args[3].shape[3])
            return function(*args, **inner_kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(processor_backend_module, "_compiled_runner", None)
    return compile_calls, qwen_past_lengths


def _freeze_and_compile(model: LanguageModel) -> None:
    _apply_trainable_policy(model, "cache_processor")
    configure_qwen3_backbone_chunk_backend(model, "compile")


def test_fake_compiler_dispatches_one_dynamic_region_not_the_window_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_calls, past_lengths = _install_fake_compiler(monkeypatch)
    model = _model()
    _freeze_and_compile(model)

    model(torch.randint(0, model.config.vocab_size, (1, 141)))

    assert compile_calls == [{"backend": "inductor", "dynamic": True, "fullgraph": True}]
    assert past_lengths == [64]
    assert model.cache_processor.pop_call_count() == 2


def test_compiled_region_preserves_forward_backward_and_processor_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(17)
    eager = _model()
    compiled = copy.deepcopy(eager)
    _apply_trainable_policy(eager, "cache_processor")
    _install_fake_compiler(monkeypatch)
    _freeze_and_compile(compiled)
    inputs = torch.randint(0, eager.config.vocab_size, (2, 128))
    targets = torch.randint(0, eager.config.vocab_size, (2, 128))

    eager_loss = eager(inputs, targets)
    compiled_loss = compiled(inputs, targets)
    eager_loss.backward()
    compiled_loss.backward()

    torch.testing.assert_close(compiled_loss, eager_loss, atol=2e-6, rtol=2e-5)
    eager_gradients = dict(eager.cache_processor.named_parameters())
    compiled_gradients = dict(compiled.cache_processor.named_parameters())
    assert eager_gradients.keys() == compiled_gradients.keys()
    for name, eager_parameter in eager_gradients.items():
        compiled_parameter = compiled_gradients[name]
        assert eager_parameter.grad is not None
        assert compiled_parameter.grad is not None
        torch.testing.assert_close(compiled_parameter.grad, eager_parameter.grad, atol=2e-5, rtol=2e-4)


def test_compiled_chunk_cache_and_scores_match_varied_past_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(29)
    eager_model = _model()
    compiled_model = copy.deepcopy(eager_model)
    compile_calls, past_lengths = _install_fake_compiler(monkeypatch)
    _freeze_and_compile(compiled_model)
    eager = eager_model.backbone
    compiled = compiled_model.backbone

    for past_len in (0, 7, 79):
        scores = torch.randn(1, past_len)
        keys = [torch.randn(1, 1, past_len, 8) for _ in range(2)]
        values = [torch.randn(1, 1, past_len, 8) for _ in range(2)]
        eager_cache = CacheState(keys, values, scores)
        compiled_cache = CacheState([key.clone() for key in keys], [value.clone() for value in values], scores.clone())
        input_ids = torch.randint(0, eager_model.config.vocab_size, (1, 64))

        eager_out, eager_next = eager.forward_chunk(input_ids, past_len, eager_cache, cache_format="windowed")
        compiled_out, compiled_next = compiled.forward_chunk(
            input_ids, past_len, compiled_cache, cache_format="windowed"
        )

        torch.testing.assert_close(compiled_out, eager_out, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(compiled_next.keys, eager_next.keys, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(compiled_next.values, eager_next.values, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(compiled_next.scores, eager_next.scores, atol=2e-6, rtol=2e-5)

    assert compile_calls == [{"backend": "inductor", "dynamic": True, "fullgraph": True}]
    assert past_lengths == [7, 79]


def test_dynamic_past_lengths_reuse_one_dynamo_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        from torch._dynamo.testing import CompileCounterWithBackend
    except ImportError:  # pragma: no cover - version-dependent test support
        pytest.skip("Torch Dynamo compile counter is unavailable")
    real_compile = torch.compile
    counter = CompileCounterWithBackend("eager")

    def counting_compile(function, **_kwargs):
        return real_compile(function, backend=counter, dynamic=True, fullgraph=True)

    monkeypatch.setattr(torch, "compile", counting_compile)
    model = _model()
    _freeze_and_compile(model)

    for past_len in (7, 79, 143):
        cache = CacheState(
            [torch.randn(1, 1, past_len, 8) for _ in range(2)],
            [torch.randn(1, 1, past_len, 8) for _ in range(2)],
            torch.randn(1, past_len),
        )
        model.backbone.forward_chunk(
            torch.randint(0, model.config.vocab_size, (1, 64)),
            past_len,
            cache,
        )

    assert counter.frame_count == 1


def test_compile_is_fail_closed_and_requires_a_frozen_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    with pytest.raises(ValueError, match="requires a frozen backbone"):
        configure_qwen3_backbone_chunk_backend(model, "compile")

    _apply_trainable_policy(model, "cache_processor")
    monkeypatch.setattr(torch, "compile", None)
    with pytest.raises(RuntimeError, match="requires torch.compile"):
        configure_qwen3_backbone_chunk_backend(model, "compile")

    def fake_compile(_function, **_kwargs):
        def fail(*_args, **_inner_kwargs):
            raise RuntimeError("inductor failed")

        return fail

    monkeypatch.setattr(torch, "compile", fake_compile)
    configure_qwen3_backbone_chunk_backend(model, "compile")
    with pytest.raises(RuntimeError, match="eager fallback is disabled") as error:
        model(torch.randint(0, model.config.vocab_size, (1, 128)))
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "inductor failed"


def test_backend_is_state_dict_neutral_and_has_stable_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_compiler(monkeypatch)
    model = _model()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    _freeze_and_compile(model)
    model(torch.randint(0, model.config.vocab_size, (1, 128)))

    assert tuple(model.state_dict()) == tuple(before)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert not any("compiled" in name or "backend" in name for name in before)
    restored = _model()
    incompatible = restored.load_state_dict(before, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    metadata = qwen3_backbone_chunk_backend_metadata("compile")
    assert metadata["effective_backend"] == "compile"
    assert metadata["chunk_steps"] == 64
    assert metadata["dynamic_past_length"] is True
    assert metadata["attention_bias_boundary"] == "causal_lower_right"
    assert metadata["initial_chunk"] == "eager"
    assert metadata["partial_chunks"] == "eager"
    assert isinstance(metadata["fingerprint_sha256"], str)
    assert len(metadata["fingerprint_sha256"]) == 64


def test_qwen_compile_composes_with_bt_compile_and_activation_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCW_BT_PROCESSOR_BACKEND", "compile")
    compile_calls, past_lengths = _install_fake_compiler(monkeypatch)
    model = _model()
    _apply_trainable_policy(model, "cache_processor")
    runtime = _configure_native_activation_checkpointing(model, "standard_processor_layers")
    configure_qwen3_backbone_chunk_backend(model, "compile")

    loss = model(torch.randint(0, model.config.vocab_size, (1, 128)), torch.randint(0, 64, (1, 128)))
    loss.backward()

    assert {tuple(sorted(call.items())) for call in compile_calls} == {
        tuple(sorted({"backend": "inductor", "dynamic": True, "fullgraph": True}.items())),
        tuple(sorted({"backend": "inductor", "dynamic": False, "fullgraph": True}.items())),
    }
    assert past_lengths == [64]
    assert runtime["policy"] == "standard_processor_layers"
    assert any(parameter.grad is not None for parameter in model.cache_processor.parameters())


def test_qwen_compile_preserves_segmented_checkpoint_early_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(47)
    eager = _model()
    segmented = copy.deepcopy(eager)
    _apply_trainable_policy(eager, "cache_processor")
    _install_fake_compiler(monkeypatch)
    _freeze_and_compile(segmented)
    inputs = torch.randint(0, eager.config.vocab_size, (2, 192))
    targets = torch.randint(0, eager.config.vocab_size, (2, 192))

    eager_loss = eager(inputs, targets)
    segmented_loss = segmented(inputs, targets, activation_checkpoint_windows=2)
    eager_loss.backward()
    segmented_loss.backward()

    torch.testing.assert_close(segmented_loss, eager_loss, atol=2e-6, rtol=2e-5)
    eager_gradients = dict(eager.cache_processor.named_parameters())
    segmented_gradients = dict(segmented.cache_processor.named_parameters())
    assert eager_gradients.keys() == segmented_gradients.keys()
    for name, eager_parameter in eager_gradients.items():
        segmented_parameter = segmented_gradients[name]
        assert eager_parameter.grad is not None
        assert segmented_parameter.grad is not None
        torch.testing.assert_close(
            segmented_parameter.grad,
            eager_parameter.grad,
            atol=2e-5,
            rtol=2e-4,
        )


def test_config_is_explicit_and_rejects_unsupported_combinations() -> None:
    assert TrainConfig().backbone_chunk_backend == "eager"
    accepted = TrainConfig(
        model=_model_config(),
        data=DataConfig(kind="random"),
        trainable="cache_processor",
        activation_checkpointing="standard_processor_layers",
        backbone_chunk_backend="compile",
        sequence_len=128,
    )
    assert accepted.backbone_chunk_backend == "compile"

    with pytest.raises(ValueError, match="must be 'eager' or 'compile'"):
        replace(accepted, backbone_chunk_backend="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="family='qwen3'"):
        replace(accepted, model=replace(accepted.model, family="smollm"))
    with pytest.raises(ValueError, match="trainable='cache_processor'"):
        replace(accepted, trainable="all")
    with pytest.raises(ValueError, match="process_every=64"):
        replace(
            accepted,
            model=replace(accepted.model, cache_processor=replace(accepted.model.cache_processor, process_every=32)),
        )
    with pytest.raises(ValueError, match="not supported with FSDP"):
        replace(accepted, distributed="fsdp")


def test_backend_fingerprint_blocks_resume(tmp_path) -> None:
    eager = TrainConfig(
        model=_model_config(),
        data=DataConfig(kind="random"),
        trainable="cache_processor",
        output_dir=str(tmp_path / "run"),
        batch_size=1,
        sequence_len=128,
        max_steps=1,
        warmup_steps=1,
        save_final=True,
        eval_every=0,
    )
    train(eager)
    checkpoint = tmp_path / "run" / "checkpoint"
    state = json.loads((checkpoint / "train_state.json").read_text(encoding="utf-8"))
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    eager_fingerprint = _training_fingerprint(eager, 1, 128, 128, 1)
    compiled = replace(
        eager,
        backbone_chunk_backend="compile",
        resume=ResumeConfig(checkpoint=str(checkpoint)),
    )

    assert state["qwen3_backbone_chunk_backend"]["effective_backend"] == "eager"
    assert (
        manifest["training_fingerprint"]["qwen3_backbone_chunk_backend_fingerprint"]
        == eager_fingerprint["qwen3_backbone_chunk_backend_fingerprint"]
    )
    with pytest.raises(CheckpointCompatibilityError, match="qwen3_backbone_chunk_backend_fingerprint"):
        train(compiled)
