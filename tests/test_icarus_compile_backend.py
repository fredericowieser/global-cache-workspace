from __future__ import annotations

import copy
import pickle
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
from gcw.config import CacheProcessorConfig, DataConfig, FlowProcessorConfig, ModelConfig, TrainConfig
from gcw.models import build_model
from gcw.models.cache_processors.icarus import GlobalCacheProcessor, _select_heads_from_merged
from gcw.models.cache_processors.ops import expand_cache_head_indices, gather_per_cache_head_kv
from gcw.models.cache_processors.state import CacheState
from gcw.models.model import LanguageModel
from gcw.performance.backends import icarus as backend_module
from gcw.performance.backends.icarus import (
    configure_icarus_training_execution,
    icarus_processor_backend_metadata,
    icarus_training_execution,
    validate_icarus_training_admission,
)
from gcw.training.core import _training_fingerprint


@pytest.fixture(autouse=True)
def _allow_rejected_icarus_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCW_ALLOW_REJECTED_ICARUS_COMPILE", "1")


def _config(
    *,
    score: str = "kv_norm",
    update_stream: str = "kv",
    attention_dropout: float = 0.0,
    flow: bool = False,
    gradient_checkpointing: bool = False,
) -> ModelConfig:
    return ModelConfig(
        variant="icarus",
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
            topk=32,
            topk_heads=1,
            score=score,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            attention_dropout=attention_dropout,
            gradient_checkpointing=gradient_checkpointing,
            update_stream=update_stream,
            gate_mode="direct",
            gate_init=0.5,
        ),
        flow_processor=FlowProcessorConfig(num_steps=2) if flow else None,
    )


def _processor(config: ModelConfig) -> GlobalCacheProcessor:
    model = build_model(config)
    assert isinstance(model, LanguageModel)
    model.init_weights()
    processor = model.cache_processor
    assert isinstance(processor, GlobalCacheProcessor)
    return processor


def _cache(
    config: ModelConfig,
    *,
    batch: int = 2,
    seq: int = 128,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> CacheState:
    generator = torch.Generator(device=device).manual_seed(9000 + batch + seq)
    num_heads, head_dim = config.stored_cache_shape
    shape = (batch, num_heads, seq, head_dim)
    keys = [torch.randn(shape, generator=generator, device=device, dtype=dtype) for _ in range(config.cache_num_layers)]
    values = [
        torch.randn(shape, generator=generator, device=device, dtype=dtype) for _ in range(config.cache_num_layers)
    ]
    scores = torch.randn((batch, seq), generator=generator, device=device, dtype=torch.float32)
    return CacheState(keys, values, scores)


def _expected_selection(
    processor: GlobalCacheProcessor,
    cache: CacheState,
    *,
    select_all: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if processor.config.score == "kv_norm" and not select_all:
        all_indices = expand_cache_head_indices(cache.all_indices(), processor.num_cache_heads)
        merged = gather_per_cache_head_kv(
            cache,
            all_indices,
            unrope_keys=processor.config.unrope_keys,
            rope_theta=processor.rope_theta,
            rotary_style=processor.rotary_style,
            rotary_dim=processor.rotary_dim,
        )
        indices, selected_heads = processor._select_indices_and_heads(cache, merged, select_all)
        merged = torch.gather(
            merged,
            dim=2,
            index=indices[..., None].expand(-1, -1, -1, merged.shape[-1]),
        )
    else:
        indices, selected_heads = processor._select_indices_and_heads(cache, None, select_all)
        merged = gather_per_cache_head_kv(
            cache,
            indices,
            unrope_keys=processor.config.unrope_keys,
            rope_theta=processor.rope_theta,
            rotary_style=processor.rotary_style,
            rotary_dim=processor.rotary_dim,
        )
    if selected_heads is None:
        selected_heads = _select_heads_from_merged(merged, processor.config.topk_heads)
    return indices, selected_heads


def _install_fake_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[str, dict[str, object]]], list[tuple[torch.Tensor, torch.Tensor]]]:
    compile_calls: list[tuple[str, dict[str, object]]] = []
    region_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_compile(function: Callable[..., torch.Tensor], **kwargs: object):
        compile_calls.append((function.__name__, dict(kwargs)))

        def compiled(*args: object, **inner_kwargs: object) -> torch.Tensor:
            positions, selected_heads = args[2], args[3]
            assert isinstance(positions, torch.Tensor)
            assert isinstance(selected_heads, torch.Tensor)
            region_inputs.append((positions.detach().clone(), selected_heads.detach().clone()))
            return function(*args, **inner_kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    return compile_calls, region_inputs


@pytest.mark.parametrize(("score", "update_stream"), (("kv_norm", "kv"), ("attn_l2", "v")))
def test_compiled_icarus_post_selection_matches_outputs_indices_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
    score: str,
    update_stream: str,
) -> None:
    torch.manual_seed(2026)
    config = _config(score=score, update_stream=update_stream)
    eager = _processor(config)
    compiled = copy.deepcopy(eager)
    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "compile")
    compile_calls, region_inputs = _install_fake_compiler(monkeypatch)
    compiled.execution_backend = "compile"

    source = _cache(config)
    eager_cache = clone_cache(source, requires_grad=True)
    compiled_cache = clone_cache(source, requires_grad=True)
    eager_inputs = [*eager_cache.keys, *eager_cache.values]
    compiled_inputs = [*compiled_cache.keys, *compiled_cache.values]
    expected_positions, expected_heads = _expected_selection(eager, eager_cache)

    eager.zero_grad(set_to_none=True)
    compiled.zero_grad(set_to_none=True)
    expected = eager(eager_cache)
    # Requested compilation is inert for direct calls and generation.
    direct = compiled(clone_cache(source, requires_grad=False))
    assert compile_calls == []
    assert_cache_close(direct, eager(clone_cache(source, requires_grad=False)))
    configure_icarus_training_execution(compiled, (2,))
    with icarus_training_execution(compiled, 2):
        actual = compiled(compiled_cache)
        cache_objective(actual).backward()
    cache_objective(expected).backward()

    assert_cache_close(actual, expected)
    assert_tensor_gradients_close(compiled_inputs, eager_inputs)
    assert_parameter_gradients_close(compiled, eager)
    assert compile_calls == [
        (
            "_run_icarus_post_selection",
            {"backend": "inductor", "dynamic": True, "fullgraph": True},
        )
    ]
    assert len(region_inputs) == 1
    positions, selected_heads = region_inputs[0]
    assert positions.dtype == torch.long
    assert selected_heads.dtype == torch.long
    assert torch.equal(positions, expected_positions)
    assert torch.equal(selected_heads, expected_heads)

    expected_destinations = [
        ~torch.isclose(before, after) for before, after in zip(source.values, expected.values, strict=True)
    ]
    actual_destinations = [
        ~torch.isclose(before, after) for before, after in zip(source.values, actual.values, strict=True)
    ]
    assert sum(mask.any(dim=-1).sum().item() for mask in actual_destinations) > 0
    assert all(
        torch.equal(actual_mask, expected_mask)
        for actual_mask, expected_mask in zip(actual_destinations, expected_destinations, strict=True)
    )


def test_compiled_icarus_preserves_processor_checkpointing(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(2027)
    config = _config(gradient_checkpointing=True)
    eager = _processor(config)
    compiled = copy.deepcopy(eager)
    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "compile")
    compile_calls, _ = _install_fake_compiler(monkeypatch)
    compiled.execution_backend = "compile"
    source = _cache(config, batch=1, seq=64)
    eager_cache = clone_cache(source, requires_grad=True)
    compiled_cache = clone_cache(source, requires_grad=True)

    expected = eager(eager_cache)
    expected = eager(expected)
    configure_icarus_training_execution(compiled, (1,))
    with icarus_training_execution(compiled, 1):
        actual = compiled(compiled_cache)
        actual = compiled(actual)
        cache_objective(actual).backward()
    cache_objective(expected).backward()

    assert_cache_close(actual, expected)
    assert_parameter_gradients_close(compiled, eager)
    assert len(compile_calls) == 1


def test_icarus_backend_is_eager_by_default_and_preserves_training_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()

    def fail_compile(*_args: object, **_kwargs: object) -> None:
        pytest.fail("eager Icarus must not call torch.compile")

    monkeypatch.delenv("GCW_ICARUS_PROCESSOR_BACKEND", raising=False)
    monkeypatch.setattr(torch, "compile", fail_compile)
    processor = _processor(config)
    before = tuple(processor.state_dict())
    processor(_cache(config))
    assert tuple(processor.state_dict()) == before

    train_config = TrainConfig(model=config, data=DataConfig(kind="random"))
    fingerprint = _training_fingerprint(train_config, 1, 128, 128, 1)
    assert "icarus_processor_backend_fingerprint" not in fingerprint
    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "eager")
    explicit_eager = _training_fingerprint(train_config, 1, 128, 128, 1)
    assert explicit_eager == fingerprint
    assert processor.execution_backend_metadata()["effective_backend"] == "eager"


def test_compiled_icarus_is_not_campaign_admitted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "compile")
    monkeypatch.delenv("GCW_ALLOW_REJECTED_ICARUS_COMPILE", raising=False)
    with pytest.raises(RuntimeError, match="not campaign-admitted"):
        _processor(_config())


def test_compiled_icarus_is_state_neutral_fingerprinted_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "compile")
    _install_fake_compiler(monkeypatch)
    processor = _processor(config)
    state = {name: tensor.clone() for name, tensor in processor.state_dict().items()}
    processor(_cache(config))

    assert tuple(processor.state_dict()) == tuple(state)
    assert all(torch.equal(tensor, state[name]) for name, tensor in processor.state_dict().items())
    restored = _processor(config)
    incompatible = restored.load_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    metadata = processor.execution_backend_metadata()
    assert metadata == icarus_processor_backend_metadata(config, "compile")
    assert metadata["selection_policy"] == "eager_fp32_exact_topk"
    assert metadata["writeback_policy"] == "eager_exact"
    assert metadata["campaign_admitted"] is False
    assert metadata["rejection"] == "fixed_batch_2_adam_parameter_tolerance"
    assert len(str(metadata["fingerprint_sha256"])) == 64
    fingerprint = _training_fingerprint(TrainConfig(model=config, data=DataConfig(kind="random")), 1, 128, 128, 1)
    assert fingerprint["icarus_processor_backend_fingerprint"] == metadata["fingerprint_sha256"]

    def failing_compile(_function: Callable[..., torch.Tensor], **_kwargs: object):
        def fail(*_args: object, **_inner_kwargs: object) -> torch.Tensor:
            raise RuntimeError("inductor failed")

        return fail

    monkeypatch.setattr(torch, "compile", failing_compile)
    failing = _processor(config)
    # Direct invocation is deliberately eager even when compilation was requested.
    failing(_cache(config))
    configure_icarus_training_execution(failing, (2,))
    with pytest.raises(RuntimeError, match="shape=.*eager fallback is disabled") as error:
        with icarus_training_execution(failing, 2):
            failing(_cache(config))
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "inductor failed"


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (_config(attention_dropout=0.1), "attention_dropout=0"),
        (_config(flow=True), "ordinary Icarus transform only"),
    ),
)
def test_compiled_icarus_rejects_unproven_stochastic_or_refined_regions(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelConfig,
    message: str,
) -> None:
    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "compile")
    _install_fake_compiler(monkeypatch)
    with pytest.raises(RuntimeError, match=message):
        build_model(config)


def test_compiled_icarus_rejects_wrong_architecture_and_missing_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "compile")
    _install_fake_compiler(monkeypatch)
    wrong = _config()
    wrong.variant = "longshot"
    with pytest.raises(RuntimeError, match="supports deterministic Icarus only"):
        build_model(wrong)

    monkeypatch.setattr(torch, "compile", None)
    with pytest.raises(RuntimeError, match="requires torch.compile"):
        build_model(_config())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke requires a GPU")
def test_compiled_icarus_cuda_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real Inductor/Flash forward and backward boundary on CUDA."""
    torch.manual_seed(2048)
    device = torch.device("cuda")
    config = _config()
    eager = _processor(config).to(device=device, dtype=torch.bfloat16)
    compiled = copy.deepcopy(eager)
    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "compile")
    monkeypatch.setenv("GCW_STRICT_FLASH", "1")
    compiled.execution_backend = "compile"
    configure_icarus_training_execution(compiled, (1,))
    source = _cache(config, batch=1, seq=64, device=device, dtype=torch.bfloat16)
    eager_cache = clone_cache(source, requires_grad=True)
    compiled_cache = clone_cache(source, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = eager(eager_cache)
        with icarus_training_execution(compiled, 1):
            actual = compiled(compiled_cache)
            actual_loss = cache_objective(actual)
            actual_loss.backward()
        expected_loss = cache_objective(expected)
    expected_loss.backward()

    assert_cache_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert_parameter_gradients_close(compiled, eager, atol=2e-2, rtol=2e-2)
    assert torch.isfinite(actual_loss)


def test_compile_warmup_is_rng_gradient_and_serialized_state_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "compile")
    compile_calls, region_inputs = _install_fake_compiler(monkeypatch)
    first = _processor(_config())
    second = _processor(_config())
    configure_icarus_training_execution(first, (2, 4))
    configure_icarus_training_execution(second, (2,))
    rng_before = torch.random.get_rng_state().clone()
    backend_module.warmup_icarus_training_execution(
        first, (2, 4), sequence_length=128, device=torch.device("cpu"), bf16=False
    )
    backend_module.warmup_icarus_training_execution(
        second, (2,), sequence_length=128, device=torch.device("cpu"), bf16=False
    )

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert all(parameter.grad is None for parameter in first.parameters())
    assert all(parameter.grad is None for parameter in second.parameters())
    assert len(compile_calls) == 2
    assert len(region_inputs) == 3
    assert {positions.shape[-1] for positions, _heads in region_inputs} == {64}
    assert all(positions.is_contiguous() and heads.is_contiguous() for positions, heads in region_inputs)
    assert first.block in backend_module._COMPILED_RUNNERS
    assert second.block in backend_module._COMPILED_RUNNERS
    assert backend_module._COMPILED_RUNNERS[first.block] is not backend_module._COMPILED_RUNNERS[second.block]
    assert not any("compiled_runner" in name for name in first.block.__dict__)
    calls_after_warmup = len(region_inputs)
    first(_cache(_config(), batch=2))
    assert len(region_inputs) == calls_after_warmup


def test_materialized_runner_is_deepcopy_neutral_and_copy_compiles_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(4101)
    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "compile")
    compile_calls, _ = _install_fake_compiler(monkeypatch)
    original = _processor(_config())
    original.execution_backend = "compile"
    configure_icarus_training_execution(original, (2,))
    with icarus_training_execution(original, 2):
        original(_cache(_config(), batch=2))
    assert len(compile_calls) == 1

    copied = copy.deepcopy(original)
    eager = copy.deepcopy(copied)
    eager.execution_backend = "eager"
    assert copied.block not in backend_module._COMPILED_RUNNERS
    with torch.no_grad():
        next(original.parameters()).add_(7.0)
    assert not torch.equal(next(original.parameters()), next(copied.parameters()))

    source = _cache(_config(), batch=2)
    eager_cache = clone_cache(source, requires_grad=True)
    copied_cache = clone_cache(source, requires_grad=True)
    configure_icarus_training_execution(copied, (2,))
    expected = eager(eager_cache)
    with icarus_training_execution(copied, 2):
        actual = copied(copied_cache)
        cache_objective(actual).backward()
    cache_objective(expected).backward()

    assert len(compile_calls) == 2
    assert copied.block in backend_module._COMPILED_RUNNERS
    assert_cache_close(actual, expected)
    assert_parameter_gradients_close(copied, eager)


def test_materialized_runner_is_whole_module_pickle_neutral_and_restored_copy_is_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(4102)
    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "compile")
    compile_calls, _ = _install_fake_compiler(monkeypatch)
    model = build_model(_config())
    assert isinstance(model, LanguageModel)
    model.init_weights()
    model.cache_processor.execution_backend = "compile"
    configure_icarus_training_execution(model, (2,))
    with icarus_training_execution(model, 2):
        model.cache_processor(_cache(_config(), batch=2))
    assert len(compile_calls) == 1

    payload = pickle.dumps(model)
    restored = pickle.loads(payload)
    assert isinstance(restored, LanguageModel)
    assert restored.cache_processor.block not in backend_module._COMPILED_RUNNERS
    assert tuple(restored.state_dict()) == tuple(model.state_dict())
    assert all(torch.equal(restored.state_dict()[name], tensor) for name, tensor in model.state_dict().items())

    eager = copy.deepcopy(restored)
    eager.cache_processor.execution_backend = "eager"
    source = _cache(_config(), batch=2)
    configure_icarus_training_execution(restored, (2,))
    expected = eager.cache_processor(clone_cache(source, requires_grad=True))
    with icarus_training_execution(restored, 2):
        actual = restored.cache_processor(clone_cache(source, requires_grad=True))
        cache_objective(actual).backward()
    cache_objective(expected).backward()

    assert len(compile_calls) == 2
    assert_cache_close(actual, expected)
    assert_parameter_gradients_close(restored.cache_processor, eager.cache_processor)


def test_compile_scope_rejects_unadmitted_shapes_and_preserves_checkpoint_control_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "compile")
    processor = _processor(_config())
    configure_icarus_training_execution(processor, (2,))
    with pytest.raises(RuntimeError, match="batch 4 was not admitted"):
        with icarus_training_execution(processor, 4):
            pass

    stop_type = type(
        "_StopRecomputationError",
        (Exception,),
        {"__module__": "torch.utils.checkpoint"},
    )

    def control_flow_compile(_function: Callable[..., torch.Tensor], **_kwargs: object):
        def fail(*_args: object, **_inner_kwargs: object) -> torch.Tensor:
            raise stop_type()

        return fail

    monkeypatch.setattr(torch, "compile", control_flow_compile)
    with pytest.raises(stop_type):
        with icarus_training_execution(processor, 2):
            processor(_cache(_config(), batch=2))


def test_requested_compile_keeps_language_model_generation_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls: list[object] = []

    def observe_compile(function: object, **_kwargs: object) -> object:
        compile_calls.append(function)
        return function

    monkeypatch.setenv("GCW_ICARUS_PROCESSOR_BACKEND", "compile")
    monkeypatch.setattr(torch, "compile", observe_compile)
    model = build_model(_config())
    assert isinstance(model, LanguageModel)
    model.init_weights()
    prompt = torch.arange(128).view(1, -1) % model.config.vocab_size
    generated = model.generate(prompt, max_new_tokens=1)

    assert generated.shape == (1, 129)
    assert compile_calls == []


@pytest.mark.parametrize(("distributed", "world_size"), (("ddp", 1), ("fsdp", 1), ("none", 2)))
def test_compile_training_admission_rejects_unqualified_distributed_topologies(
    distributed: str,
    world_size: int,
) -> None:
    with pytest.raises(RuntimeError, match="DDP or FSDP"):
        validate_icarus_training_admission(
            "compile",
            distributed=distributed,
            world_size=world_size,
            distributed_initialized=False,
            batch_size=2,
            adaptive_batch_sizes=None,
            device=torch.device("cuda"),
        )


def test_compile_training_admission_rejects_nonoracle_batches() -> None:
    with pytest.raises(RuntimeError, match="fixed batch_size=2"):
        validate_icarus_training_admission(
            "compile",
            distributed="none",
            world_size=1,
            distributed_initialized=False,
            batch_size=4,
            adaptive_batch_sizes=None,
            device=torch.device("cuda"),
        )
    with pytest.raises(RuntimeError, match="does not admit adaptive batching"):
        validate_icarus_training_admission(
            "compile",
            distributed="none",
            world_size=1,
            distributed_initialized=False,
            batch_size=2,
            adaptive_batch_sizes=(16, 8, 4, 2),
            device=torch.device("cuda"),
        )
    validate_icarus_training_admission(
        "compile",
        distributed="none",
        world_size=1,
        distributed_initialized=False,
        batch_size=2,
        adaptive_batch_sizes=None,
        device=torch.device("cuda"),
    )
