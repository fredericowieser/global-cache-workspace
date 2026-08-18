from __future__ import annotations

import copy
from dataclasses import replace
import json
import os

import pytest
import torch

from gcw.config import CacheProcessorConfig, DataConfig, ModelConfig, ResumeConfig, TrainConfig
from gcw.models import build_model
import gcw.performance.backends.qwen3 as qwen3_backend_module
from gcw.performance.backends.qwen3 import (
    Qwen3BackboneChunkBackend,
    configure_qwen3_backbone_chunk_backend,
    qwen3_backbone_chunk_backend_metadata,
)
from gcw.performance.backends import cache_processors as processor_backend_module
from gcw.performance.checkpointing import (
    configure_native_activation_checkpointing as _configure_native_activation_checkpointing,
)
from gcw.performance.compile_telemetry import install_compile_acceptance_recorder
from gcw.models.cache_processors.state import CacheState
from gcw.models.model import LanguageModel
from gcw.provenance import canonical_json_sha256
from gcw.training.checkpoint import CheckpointCompatibilityError
from gcw.training.audit import _training_fingerprint
from gcw.training.model_setup import _apply_trainable_policy
from gcw.training.core import train


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


def _freeze_and_compile(model: LanguageModel, backend: Qwen3BackboneChunkBackend = "compile") -> None:
    _apply_trainable_policy(model, "cache_processor")
    configure_qwen3_backbone_chunk_backend(model, backend)


def test_fake_compiler_dispatches_one_dynamic_region_not_the_window_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_calls, past_lengths = _install_fake_compiler(monkeypatch)
    model = _model()
    _freeze_and_compile(model)

    model(torch.randint(0, model.config.vocab_size, (1, 141)))

    assert compile_calls == [{"backend": "inductor", "dynamic": True, "fullgraph": True}]
    assert past_lengths == [64]
    assert model.cache_processor.pop_call_count() == 2


@pytest.mark.parametrize("backend", ["compile", "compile_full_windows"])
def test_compiled_region_preserves_forward_backward_and_processor_gradients(
    monkeypatch: pytest.MonkeyPatch,
    backend: Qwen3BackboneChunkBackend,
) -> None:
    torch.manual_seed(17)
    eager = _model()
    compiled = copy.deepcopy(eager)
    _apply_trainable_policy(eager, "cache_processor")
    _install_fake_compiler(monkeypatch)
    _freeze_and_compile(compiled, backend)
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
    torch._dynamo.reset()
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


def test_initial_full_window_is_one_dynamo_fullgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        from torch._dynamo.testing import CompileCounterWithBackend
    except ImportError:  # pragma: no cover - version-dependent test support
        pytest.skip("Torch Dynamo compile counter is unavailable")
    torch._dynamo.reset()
    real_compile = torch.compile
    counter = CompileCounterWithBackend("eager")

    def counting_compile(function, **kwargs):
        backend = counter if kwargs.get("dynamic") is False else "eager"
        return real_compile(
            function,
            backend=backend,
            dynamic=kwargs.get("dynamic"),
            fullgraph=kwargs.get("fullgraph", False),
        )

    monkeypatch.setattr(torch, "compile", counting_compile)
    model = _model()
    _apply_trainable_policy(model, "cache_processor")
    configure_qwen3_backbone_chunk_backend(model, "compile_full_windows")
    model.backbone.forward_chunk(torch.randint(0, model.config.vocab_size, (1, 64)), 0)

    assert counter.frame_count == 1


def test_variable_batches_record_expected_graphs_and_reuse_model_local_runners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from torch._dynamo.testing import CompileCounterWithBackend
    except ImportError:  # pragma: no cover - version-dependent test support
        pytest.skip("Torch Dynamo compile counter is unavailable")
    torch._dynamo.reset()
    real_compile = torch.compile
    initial_counter = CompileCounterWithBackend("eager")
    past_counter = CompileCounterWithBackend("eager")

    def counting_compile(function, **kwargs):
        counter = initial_counter if kwargs.get("dynamic") is False else past_counter
        return real_compile(
            function,
            backend=counter,
            dynamic=kwargs.get("dynamic"),
            fullgraph=kwargs.get("fullgraph", False),
        )

    monkeypatch.setattr(torch, "compile", counting_compile)
    torch.manual_seed(211)
    compiled_model = _model()
    eager_model = copy.deepcopy(compiled_model)
    _apply_trainable_policy(eager_model, "cache_processor")
    _apply_trainable_policy(compiled_model, "cache_processor")
    configure_qwen3_backbone_chunk_backend(compiled_model, "compile_full_windows")
    recorder = install_compile_acceptance_recorder(compiled_model)

    for batch_rows in (16, 8, 4, 2, 8):
        first_ids = torch.randint(0, compiled_model.config.vocab_size, (batch_rows, 64))
        second_ids = torch.randint(0, compiled_model.config.vocab_size, (batch_rows, 64))
        eager_first, eager_cache = eager_model.backbone.forward_chunk(first_ids, 0)
        compiled_first, compiled_cache = compiled_model.backbone.forward_chunk(first_ids, 0)
        eager_second, eager_cache = eager_model.backbone.forward_chunk(second_ids, 64, eager_cache)
        compiled_second, compiled_cache = compiled_model.backbone.forward_chunk(
            second_ids,
            64,
            compiled_cache,
        )

        torch.testing.assert_close(compiled_first, eager_first, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(compiled_second, eager_second, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(compiled_cache.keys, eager_cache.keys, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(compiled_cache.values, eager_cache.values, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(compiled_cache.scores, eager_cache.scores, atol=2e-6, rtol=2e-5)

    recorder.record_graph_counts(
        "initial_full_window",
        graph_count=initial_counter.frame_count,
        recompile_count=max(0, initial_counter.frame_count - 1),
    )
    recorder.record_graph_counts(
        "full_window_with_past",
        graph_count=past_counter.frame_count,
        recompile_count=max(0, past_counter.frame_count - 1),
    )
    evidence = recorder.snapshot()
    regions = evidence["regions"]
    assert isinstance(regions, dict)
    initial = regions["initial_full_window"]
    past = regions["full_window_with_past"]
    assert isinstance(initial, dict)
    assert isinstance(past, dict)

    assert initial_counter.frame_count == 4
    assert past_counter.frame_count == 4
    assert initial["calls"] == 5
    assert initial["graph_count"] == 4
    assert initial["recompile_count"] == 3
    assert initial["observed_signatures"] == [[2, 64], [4, 64], [8, 64], [16, 64]]
    assert past["calls"] == 5
    assert past["graph_count"] == 4
    assert past["recompile_count"] == 3
    assert past["observed_signatures"] == [
        [2, 64, 64],
        [4, 64, 64],
        [8, 64, 64],
        [16, 64, 64],
    ]
    expected_cold_signatures = {
        "initial_full_window": {"2,64", "4,64", "8,64", "16,64"},
        "full_window_with_past": {"2,64,64", "4,64,64", "8,64,64", "16,64,64"},
    }
    for name, region in (("initial_full_window", initial), ("full_window_with_past", past)):
        cold_costs = region["cold_call_seconds"]
        assert isinstance(cold_costs, dict)
        assert set(cold_costs) == expected_cold_signatures[name]
        assert all(isinstance(seconds, float) and seconds > 0 for seconds in cold_costs.values())
    assert evidence["expected_initial_static_graphs"] == 4
    assert evidence["expected_initial_shape_recompiles"] == 3
    assert evidence["expected_past_batch_specializations"] == 4
    assert evidence["expected_past_shape_recompiles"] == 3
    assert evidence["dynamic_past_length_within_batch_specialization"] is True


def test_compiled_runners_remain_bound_to_their_own_backbone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_compiler(monkeypatch)
    torch.manual_seed(223)
    first = _model()
    second = copy.deepcopy(first)
    with torch.no_grad():
        second.backbone.embed_tokens.weight.add_(0.25)
    first_eager = copy.deepcopy(first)
    second_eager = copy.deepcopy(second)
    for model in (first, second):
        _apply_trainable_policy(model, "cache_processor")
        configure_qwen3_backbone_chunk_backend(model, "compile_full_windows")
    input_ids = torch.randint(0, first.config.vocab_size, (4, 64))

    first_output, _ = first.backbone.forward_chunk(input_ids, 0)
    second_output, _ = second.backbone.forward_chunk(input_ids, 0)
    first_expected, _ = first_eager.backbone.forward_chunk(input_ids, 0)
    second_expected, _ = second_eager.backbone.forward_chunk(input_ids, 0)

    torch.testing.assert_close(first_output, first_expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(second_output, second_expected, atol=2e-6, rtol=2e-5)
    assert not torch.equal(first_output, second_output)
    assert (
        first.backbone.__dict__["_gcw_qwen3_compiled_initial_chunk_runner"]
        is not second.backbone.__dict__["_gcw_qwen3_compiled_initial_chunk_runner"]
    )


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


@pytest.mark.parametrize("backend", ["compile", "compile_full_windows"])
def test_qwen_compile_preserves_segmented_checkpoint_early_stop_and_processor_gradients(
    monkeypatch: pytest.MonkeyPatch,
    backend: Qwen3BackboneChunkBackend,
) -> None:
    torch.manual_seed(47)
    eager = _model()
    segmented = copy.deepcopy(eager)
    _apply_trainable_policy(eager, "cache_processor")
    _install_fake_compiler(monkeypatch)
    early_stop_types: list[str] = []
    original_early_stop_check = qwen3_backend_module._is_non_reentrant_checkpoint_early_stop

    def record_early_stop(error: Exception) -> bool:
        is_early_stop = original_early_stop_check(error)
        if is_early_stop:
            early_stop_types.append(type(error).__name__)
        return is_early_stop

    monkeypatch.setattr(
        qwen3_backend_module,
        "_is_non_reentrant_checkpoint_early_stop",
        record_early_stop,
    )
    _freeze_and_compile(segmented, backend)
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
    assert early_stop_types
    assert set(early_stop_types) == {"_StopRecomputationError"}


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
    assert replace(accepted, backbone_chunk_backend="cuda_graph").backbone_chunk_backend == "cuda_graph"

    with pytest.raises(ValueError, match="must be 'eager', 'compile', 'compile_full_windows', or 'cuda_graph'"):
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


def test_compile_full_windows_dispatches_initial_and_past_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_calls, past_lengths = _install_fake_compiler(monkeypatch)
    model = _model()
    _apply_trainable_policy(model, "cache_processor")
    configure_qwen3_backbone_chunk_backend(model, "compile_full_windows")

    model(torch.randint(0, model.config.vocab_size, (1, 141)))

    assert compile_calls == [
        {"backend": "inductor", "dynamic": False, "fullgraph": True},
        {"backend": "inductor", "dynamic": True, "fullgraph": True},
    ]
    assert past_lengths == [64]
    assert model.cache_processor.pop_call_count() == 2


def test_compiled_initial_region_preserves_outputs_cache_scores_and_input_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(53)
    eager_model = _model()
    compiled_model = copy.deepcopy(eager_model)
    _apply_trainable_policy(eager_model, "cache_processor")
    _install_fake_compiler(monkeypatch)
    _apply_trainable_policy(compiled_model, "cache_processor")
    configure_qwen3_backbone_chunk_backend(compiled_model, "compile_full_windows")

    eager_inputs = (
        eager_model.backbone.embed_tokens(torch.randint(0, eager_model.config.vocab_size, (2, 64)))
        .detach()
        .requires_grad_(True)
    )
    compiled_inputs = eager_inputs.detach().clone().requires_grad_(True)
    eager_out, eager_cache = eager_model.backbone.forward_embedding_chunk(
        eager_inputs, 0, logits=False, cache_format="windowed"
    )
    compiled_out, compiled_cache = compiled_model.backbone.forward_embedding_chunk(
        compiled_inputs, 0, logits=False, cache_format="windowed"
    )

    torch.testing.assert_close(compiled_out, eager_out, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(compiled_cache.keys, eager_cache.keys, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(compiled_cache.values, eager_cache.values, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(compiled_cache.scores, eager_cache.scores, atol=2e-6, rtol=2e-5)
    eager_objective = eager_out.square().mean() + eager_cache.keys.square().mean()
    compiled_objective = compiled_out.square().mean() + compiled_cache.keys.square().mean()
    eager_objective.backward()
    compiled_objective.backward()
    assert eager_inputs.grad is not None
    assert compiled_inputs.grad is not None
    torch.testing.assert_close(compiled_inputs.grad, eager_inputs.grad, atol=2e-5, rtol=2e-4)


def test_compiled_initial_region_preserves_flash_causal_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gcw.models.backbones import transformer as transformer_module

    calls: list[tuple[torch.Tensor | None, bool]] = []
    original = transformer_module.flash_sdpa

    def recording_flash_sdpa(q, k, v, **kwargs):
        calls.append((kwargs.get("attn_mask"), bool(kwargs.get("is_causal", False))))
        return original(q, k, v, **kwargs)

    _install_fake_compiler(monkeypatch)
    monkeypatch.setattr(transformer_module, "flash_sdpa", recording_flash_sdpa)
    model = _model()
    _apply_trainable_policy(model, "cache_processor")
    configure_qwen3_backbone_chunk_backend(model, "compile_full_windows")
    model.backbone.forward_chunk(torch.randint(0, model.config.vocab_size, (1, 64)), 0)

    assert calls
    assert all(mask is None and is_causal for mask, is_causal in calls)


def test_compile_full_windows_is_state_neutral_versioned_and_preserves_v1_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_compiler(monkeypatch)
    model = _model()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    _apply_trainable_policy(model, "cache_processor")
    configure_qwen3_backbone_chunk_backend(model, "compile_full_windows")
    model(torch.randint(0, model.config.vocab_size, (1, 128)))

    assert tuple(model.state_dict()) == tuple(before)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    for backend in ("eager", "compile"):
        expected_v1 = {
            "schema": "gcw.qwen3_backbone_chunk_backend.v1",
            "effective_backend": backend,
            "region": "gcw.models.backbones.backend._qwen3_backbone_chunk_tensor_region",
            "chunk_steps": 64,
            "dynamic_past_length": backend == "compile",
            "compile_options": (
                {"backend": "inductor", "dynamic": True, "fullgraph": True} if backend == "compile" else None
            ),
            "attention_bias_boundary": "causal_lower_right",
            "initial_chunk": "eager",
            "partial_chunks": "eager",
            "torch_version": str(torch.__version__),
        }
        legacy_metadata = qwen3_backbone_chunk_backend_metadata(backend)
        assert {key: legacy_metadata[key] for key in expected_v1} == expected_v1
        assert legacy_metadata["fingerprint_sha256"] == canonical_json_sha256(expected_v1)
    metadata = qwen3_backbone_chunk_backend_metadata("compile_full_windows")
    assert metadata["schema"] == "gcw.qwen3_backbone_chunk_backend.v2"
    assert metadata["initial_chunk"] == "compile_fullgraph"
    assert metadata["partial_chunks"] == "eager"
    assert metadata["effective_backend"] == "compile_full_windows"
    assert metadata["fingerprint_sha256"] not in {
        qwen3_backbone_chunk_backend_metadata("eager")["fingerprint_sha256"],
        qwen3_backbone_chunk_backend_metadata("compile")["fingerprint_sha256"],
    }


def test_compile_full_windows_config_is_explicit_and_rejects_fsdp() -> None:
    accepted = TrainConfig(
        model=_model_config(),
        data=DataConfig(kind="random"),
        trainable="cache_processor",
        backbone_chunk_backend="compile_full_windows",
        sequence_len=128,
    )
    assert accepted.backbone_chunk_backend == "compile_full_windows"
    with pytest.raises(ValueError, match="not supported with FSDP"):
        replace(accepted, distributed="fsdp")


def test_compile_full_windows_fails_closed_on_initial_runner_error(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    _apply_trainable_policy(model, "cache_processor")

    def fake_compile(_function, **_kwargs):
        def fail(*_args, **_inner_kwargs):
            raise RuntimeError("initial inductor failed")

        return fail

    monkeypatch.setattr(torch, "compile", fake_compile)
    configure_qwen3_backbone_chunk_backend(model, "compile_full_windows")
    with pytest.raises(RuntimeError, match="compiled initial Qwen3") as error:
        model(torch.randint(0, model.config.vocab_size, (1, 128)))
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "initial inductor failed"


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("GCW_RUN_QWEN3_INITIAL_COMPILE_CUDA_SMOKE") != "1"
    or not torch.cuda.is_available()
    or not torch.cuda.is_bf16_supported(),
    reason="set GCW_RUN_QWEN3_INITIAL_COMPILE_CUDA_SMOKE=1 on a BF16 CUDA host",
)
def test_compiled_initial_full_window_cuda_forward_backward_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCW_STRICT_FLASH", "1")
    torch.manual_seed(61)
    eager = _model().cuda()
    compiled = copy.deepcopy(eager)
    _apply_trainable_policy(eager, "cache_processor")
    _apply_trainable_policy(compiled, "cache_processor")
    configure_qwen3_backbone_chunk_backend(compiled, "compile_full_windows")
    inputs = torch.randint(0, eager.config.vocab_size, (2, 128), device="cuda")
    targets = torch.randint(0, eager.config.vocab_size, (2, 128), device="cuda")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profile:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eager_loss = eager(inputs, targets)
            compiled_loss = compiled(inputs, targets)
        eager_loss.backward()
        compiled_loss.backward()
        torch.cuda.synchronize()

    torch.testing.assert_close(compiled_loss, eager_loss, atol=3e-3, rtol=3e-3)
    for eager_parameter, compiled_parameter in zip(
        eager.cache_processor.parameters(),
        compiled.cache_processor.parameters(),
        strict=True,
    ):
        assert eager_parameter.grad is not None
        assert compiled_parameter.grad is not None
        torch.testing.assert_close(
            compiled_parameter.grad,
            eager_parameter.grad,
            atol=3e-3,
            rtol=3e-3,
        )

    sdpa_events = sorted(
        event.key
        for event in profile.key_averages()
        if "scaled_dot_product" in event.key or "flash_attention" in event.key
    )
    assert os.environ["GCW_STRICT_FLASH"] == "1"
    assert any("flash_attention" in event for event in sdpa_events), sdpa_events
    print({"strict_flash": True, "sdpa_profiler_events": sdpa_events})
