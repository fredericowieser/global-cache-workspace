from __future__ import annotations

import copy
import os
import pickle
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gcw.architecture import resolve_experiment, resolve_model
from gcw.config import CacheProcessorConfig, DataConfig, ModelConfig, TrainConfig, to_dict
from gcw.config import FlowProcessorConfig
from gcw.data.training import RandomCursor
from gcw.models import build_model
from gcw.models.backbones.transformer import _flatten_windowed_cache, _unflatten_windowed_cache
from gcw.models.cache_processors import longshot as longshot_module
from gcw.models.cache_processors.longshot import GlobalCacheProcessor
from gcw.models.cache_processors.ops import expand_cache_head_indices, gather_per_cache_head_kv
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.performance.config import (
    AdaptiveBatchingConfig,
    KVScoreSidecarConfig,
    LongshotFullHistoryConfig,
    PerformanceConfig,
)
from gcw.performance.kv_score_sidecar import (
    KVScoreSidecarExecutionContext,
    KVScoreSidecarState,
    configure_kv_score_sidecar_execution,
    kv_score_sidecar_execution,
    kv_score_sidecar_execution_metadata,
    kv_score_sidecar_spec,
    refresh_kv_score_sidecar,
    synchronize_kv_score_sidecar,
)
from gcw.performance.longshot_cache import configure_longshot_full_history_execution
from gcw.training.audit import _training_fingerprint
from gcw.training.checkpoint import ResumeState, save_training_checkpoint


def _model_config(
    *,
    unrope_keys: bool = False,
    update_stream: str = "kv",
    iteration_mode: str = "internal",
    iterations: int = 1,
    gradient_checkpointing: bool = False,
) -> ModelConfig:
    return ModelConfig(
        variant="longshot",
        family="qwen3",
        sequence_len=16,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=2,
            topk_heads=1,
            score="kv_norm",
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
            attention_dropout=0.0,
            update_stream=update_stream,  # type: ignore[arg-type]
            unrope_keys=unrope_keys,
            iterations=iterations,
            iteration_mode=iteration_mode,  # type: ignore[arg-type]
            internal_iteration_score="kv_norm",
            gradient_checkpointing=gradient_checkpointing,
            encoder_method="query_attention",
            encoder_query_source="final_token",
        ),
    )


def _processor(**kwargs: object) -> GlobalCacheProcessor:
    return GlobalCacheProcessor(_model_config(**kwargs))


def _sidecar_context(**changes: object) -> KVScoreSidecarExecutionContext:
    context = KVScoreSidecarExecutionContext(
        model_variant="longshot",
        model_family="qwen3",
        physical_batch_size=2,
        distributed="none",
        backbone_chunk_backend="eager",
        adaptive_batching_enabled=False,
        longshot_full_history_enabled=False,
        whole_model_compile_requested=False,
        activation_checkpointing="none",
        incompatible_refinement_enabled=False,
    )
    return replace(context, **changes)


def _cache(
    *,
    dtype: torch.dtype = torch.float32,
    sequence: int = 9,
    requires_grad: bool = False,
) -> WindowedCache:
    generator = torch.Generator().manual_seed(41 + sequence)
    shape = (2, 2, 1, sequence, 8)
    keys = torch.randn(shape, generator=generator, dtype=dtype)
    values = torch.randn(shape, generator=generator, dtype=dtype)
    keys.requires_grad_(requires_grad)
    values.requires_grad_(requires_grad)
    return WindowedCache(keys.contiguous(), values.contiguous(), torch.randn(2, sequence, generator=generator))


def _clone_cache(cache: WindowedCache, *, requires_grad: bool = False) -> WindowedCache:
    keys = cache.keys.detach().clone().requires_grad_(requires_grad)
    values = cache.values.detach().clone().requires_grad_(requires_grad)
    return WindowedCache(keys, values, cache.scores.clone(), kv_score_sidecar=copy.deepcopy(cache.kv_score_sidecar))


def _oracle_scores(processor: GlobalCacheProcessor, cache: WindowedCache) -> torch.Tensor:
    indices = expand_cache_head_indices(cache.all_indices(), processor.num_kv_cache_heads)
    merged = gather_per_cache_head_kv(
        cache,
        indices,
        unrope_keys=processor.config.unrope_keys,
        rope_theta=processor.rope_theta,
        rotary_style=processor.rotary_style,
        rotary_dim=processor.rotary_dim,
    )
    return merged.float().pow(2).sum(dim=-1)


def _assert_fp32_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.equal(actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("unrope_keys", [False, True])
def test_score_table_bitwise_oracle_parity_bootstrap_append_ties_and_refresh(
    dtype: torch.dtype,
    unrope_keys: bool,
) -> None:
    processor = _processor(unrope_keys=unrope_keys)
    spec = kv_score_sidecar_spec(processor)
    cache = _cache(dtype=dtype, sequence=6)
    # Force exact ties without changing the complete-row expression.
    cache.keys[..., 1, :] = cache.keys[..., 0, :]
    cache.values[..., 1, :] = cache.values[..., 0, :]

    state = synchronize_kv_score_sidecar(None, cache.keys, cache.values, spec)
    _assert_fp32_bits_equal(state.scores, _oracle_scores(processor, cache))
    assert not state.scores.requires_grad and state.scores.grad_fn is None

    tail = _cache(dtype=dtype, sequence=2)
    cache.keys = torch.cat([cache.keys, tail.keys], dim=3).contiguous()
    cache.values = torch.cat([cache.values, tail.values], dim=3).contiguous()
    cache.scores = torch.cat([cache.scores, tail.scores], dim=1)
    state = synchronize_kv_score_sidecar(state, cache.keys, cache.values, spec)
    _assert_fp32_bits_equal(state.scores, _oracle_scores(processor, cache))

    positions = torch.tensor([[2], [4]])
    cache.values = cache.values.scatter(
        3,
        positions[None, :, None, :, None].expand(2, 2, 1, 1, 8),
        torch.full((2, 2, 1, 1, 8), 0.75, dtype=dtype),
    )
    state = refresh_kv_score_sidecar(state, cache.keys, cache.values, positions, spec)
    _assert_fp32_bits_equal(state.scores, _oracle_scores(processor, cache))


def _install_selection_trace(
    processor: GlobalCacheProcessor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    token_trace: list[torch.Tensor] = []
    head_trace: list[torch.Tensor] = []
    select_tokens = processor._select_kv_norm_history
    select_heads = processor._select_combined_heads

    def traced_tokens(cache: WindowedCache):
        result = select_tokens(cache)
        token_trace.append(result[2].detach().clone())
        return result

    def traced_heads(**kwargs: object):
        result = select_heads(**kwargs)  # type: ignore[arg-type]
        head_trace.append(result.detach().clone())
        return result

    object.__setattr__(processor, "_select_kv_norm_history", traced_tokens)
    object.__setattr__(processor, "_select_combined_heads", traced_heads)
    return token_trace, head_trace


def test_policy_selection_keeps_full_live_gather_and_refreshes_value_and_kv_writes() -> None:
    torch.manual_seed(53)
    for update_stream in ("v", "kv"):
        for iteration_mode in ("internal", "external"):
            iterations = 2 if iteration_mode == "external" else 1
            reference = _processor(update_stream=update_stream, iteration_mode=iteration_mode, iterations=iterations)
            sidecar = copy.deepcopy(reference)
            configure_kv_score_sidecar_execution(
                sidecar, KVScoreSidecarConfig(enabled=True), context=_sidecar_context()
            )
            source = _cache(sequence=10, requires_grad=True)
            expected_cache = _clone_cache(source, requires_grad=True)
            actual_cache = _clone_cache(source, requires_grad=True)

            expected_merged, expected_scores, expected_indices = reference._select_kv_norm_history(expected_cache)
            actual_merged, actual_scores, actual_indices = sidecar._select_kv_norm_history(actual_cache)
            assert torch.equal(actual_indices, expected_indices)
            assert torch.equal(actual_merged, expected_merged)
            _assert_fp32_bits_equal(actual_scores, expected_scores)

            expected_token_trace, expected_head_trace = _install_selection_trace(reference)
            actual_token_trace, actual_head_trace = _install_selection_trace(sidecar)
            expected = reference(expected_cache)
            actual = sidecar(actual_cache)
            assert len(actual_token_trace) == len(expected_token_trace) == iterations
            assert len(actual_head_trace) == len(expected_head_trace) == iterations
            for actual_selection, expected_selection in zip(actual_token_trace, expected_token_trace, strict=True):
                assert torch.equal(actual_selection, expected_selection)
                assert torch.equal(actual_selection[:, :1, -1], expected_selection[:, :1, -1])
            for actual_selection, expected_selection in zip(actual_head_trace, expected_head_trace, strict=True):
                assert torch.equal(actual_selection, expected_selection)
            assert torch.equal(actual.keys, expected.keys)
            assert torch.equal(actual.values, expected.values)
            assert actual.kv_score_sidecar is not None
            _assert_fp32_bits_equal(actual.kv_score_sidecar.scores, _oracle_scores(sidecar, actual))


def test_selected_kv_loss_cache_and_parameter_gradients_and_adam_state_match_multiple_steps() -> None:
    torch.manual_seed(59)
    reference = _processor(update_stream="kv", iteration_mode="external", iterations=2)
    sidecar = copy.deepcopy(reference)
    before_keys = tuple(sidecar.state_dict())
    configure_kv_score_sidecar_execution(
        sidecar, KVScoreSidecarConfig(enabled=True), context=_sidecar_context()
    )
    assert tuple(sidecar.state_dict()) == before_keys
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4)
    sidecar_optimizer = torch.optim.AdamW(sidecar.parameters(), lr=3e-4)

    for sequence in (9, 12, 11):
        source = _cache(sequence=sequence)
        reference_cache = _clone_cache(source, requires_grad=True)
        sidecar_cache = _clone_cache(source, requires_grad=True)
        reference_input_keys, reference_input_values = reference_cache.keys, reference_cache.values
        sidecar_input_keys, sidecar_input_values = sidecar_cache.keys, sidecar_cache.values
        reference_optimizer.zero_grad(set_to_none=True)
        sidecar_optimizer.zero_grad(set_to_none=True)
        reference_output = reference(reference_cache)
        sidecar_output = sidecar(sidecar_cache)
        reference_loss = reference_output.keys.float().square().mean() + reference_output.values.float().square().mean()
        sidecar_loss = sidecar_output.keys.float().square().mean() + sidecar_output.values.float().square().mean()
        assert torch.equal(sidecar_loss, reference_loss)
        reference_loss.backward()
        sidecar_loss.backward()
        assert torch.equal(sidecar_input_keys.grad, reference_input_keys.grad)
        assert torch.equal(sidecar_input_values.grad, reference_input_values.grad)
        for actual_parameter, expected_parameter in zip(sidecar.parameters(), reference.parameters(), strict=True):
            assert torch.equal(actual_parameter.grad, expected_parameter.grad)
        reference_optimizer.step()
        sidecar_optimizer.step()
        for actual_parameter, expected_parameter in zip(sidecar.parameters(), reference.parameters(), strict=True):
            assert torch.equal(actual_parameter, expected_parameter)
            actual_state = sidecar_optimizer.state[actual_parameter]
            expected_state = reference_optimizer.state[expected_parameter]
            assert actual_state.keys() == expected_state.keys()
            for key in actual_state:
                if isinstance(actual_state[key], torch.Tensor):
                    assert torch.equal(actual_state[key], expected_state[key])
                else:
                    assert actual_state[key] == expected_state[key]
        assert sidecar_output.kv_score_sidecar is not None
        assert sidecar_output.kv_score_sidecar.scores.grad_fn is None


@pytest.mark.parametrize("activation_checkpoint_windows", [0, 2])
def test_native_and_segmented_checkpoint_replay_match_eager_gradients(
    activation_checkpoint_windows: int,
) -> None:
    torch.manual_seed(61)
    config = _model_config(gradient_checkpointing=True)
    reference = build_model(config).train()
    reference.init_weights()
    sidecar = copy.deepcopy(reference)
    configure_kv_score_sidecar_execution(
        sidecar, KVScoreSidecarConfig(enabled=True), context=_sidecar_context()
    )
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    targets = torch.randint(0, config.vocab_size, (2, 12))
    expected = reference(input_ids, targets, activation_checkpoint_windows=activation_checkpoint_windows)
    actual = sidecar(input_ids, targets, activation_checkpoint_windows=activation_checkpoint_windows)
    assert torch.equal(actual, expected)
    expected.backward()
    actual.backward()
    for actual_parameter, expected_parameter in zip(sidecar.parameters(), reference.parameters(), strict=True):
        assert torch.equal(actual_parameter.grad, expected_parameter.grad)


def test_sidecar_logits_and_reference_generation_preserve_public_windowed_paths() -> None:
    torch.manual_seed(63)
    config = _model_config()
    reference = build_model(config).eval()
    reference.init_weights()
    sidecar = copy.deepcopy(reference)
    configure_kv_score_sidecar_execution(
        sidecar, KVScoreSidecarConfig(enabled=True), context=_sidecar_context()
    )
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    with torch.no_grad():
        expected_logits = reference(input_ids)
        actual_logits = sidecar(input_ids)
        expected_tokens = reference.generate(input_ids[:, :8], max_new_tokens=3)
        actual_tokens = sidecar.generate(input_ids[:, :8], max_new_tokens=3)
    assert torch.equal(actual_logits, expected_logits)
    assert torch.equal(actual_tokens, expected_tokens)
    with pytest.raises(TypeError, match="WindowedCache"):
        sidecar.generate_fast(input_ids[:, :8], max_new_tokens=1)


def test_select_all_refreshes_sidecar_and_disabled_path_does_not_touch_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(67)
    reference = _processor(update_stream="v")
    disabled_cache = _cache(sequence=8)
    monkeypatch.setattr(
        longshot_module,
        "synchronize_kv_score_sidecar",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("disabled lifecycle call")),
    )
    reference(_clone_cache(disabled_cache))
    monkeypatch.undo()

    sidecar = copy.deepcopy(reference)
    configure_kv_score_sidecar_execution(
        sidecar, KVScoreSidecarConfig(enabled=True), context=_sidecar_context()
    )
    actual = sidecar(_clone_cache(disabled_cache), select_all=True)
    assert isinstance(actual, WindowedCache)
    assert actual.kv_score_sidecar is not None
    _assert_fp32_bits_equal(actual.kv_score_sidecar.scores, _oracle_scores(sidecar, actual))


def test_cache_copy_deepcopy_pickle_and_fail_closed_state_validation() -> None:
    processor = _processor()
    spec = kv_score_sidecar_spec(processor)
    cache = _cache(sequence=7)
    cache.kv_score_sidecar = synchronize_kv_score_sidecar(None, cache.keys, cache.values, spec)
    for restored in (copy.copy(cache), copy.deepcopy(cache), pickle.loads(pickle.dumps(cache))):
        assert isinstance(restored.kv_score_sidecar, KVScoreSidecarState)
        _assert_fp32_bits_equal(restored.kv_score_sidecar.scores, cache.kv_score_sidecar.scores)
        tensors, layout = _flatten_windowed_cache(restored)
        flattened = _unflatten_windowed_cache(tensors, layout)
        assert flattened.kv_score_sidecar is not None
        _assert_fp32_bits_equal(flattened.kv_score_sidecar.scores, cache.kv_score_sidecar.scores)
    stale = replace(cache.kv_score_sidecar, scores=cache.kv_score_sidecar.scores[:, :, :-1].requires_grad_())
    with pytest.raises(RuntimeError, match="detached selector-only"):
        synchronize_kv_score_sidecar(stale, cache.keys, cache.values, spec)
    with pytest.raises(RuntimeError, match="does not match"):
        synchronize_kv_score_sidecar(
            replace(cache.kv_score_sidecar, spec=replace(spec, head_dim=4)), cache.keys, cache.values, spec
        )


def test_fresh_python_process_training_checkpoint_resume_reconstructs_sidecar(tmp_path: Path) -> None:
    torch.manual_seed(71)
    config = _model_config()
    continuous = build_model(config).train()
    continuous.init_weights()
    resumed = copy.deepcopy(continuous)
    policy = KVScoreSidecarConfig(enabled=True)
    configure_kv_score_sidecar_execution(continuous, policy, context=_sidecar_context())
    configure_kv_score_sidecar_execution(resumed, policy, context=_sidecar_context())
    continuous_optimizer = torch.optim.AdamW(continuous.parameters(), lr=3e-4)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=3e-4)
    first = (torch.randint(0, 32, (2, 12)), torch.randint(0, 32, (2, 12)))
    second = (torch.randint(0, 32, (2, 12)), torch.randint(0, 32, (2, 12)))

    def step(
        model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
        model(*batch).backward()
        optimizer.step()

    step(continuous, continuous_optimizer, first)
    step(continuous, continuous_optimizer, second)
    step(resumed, resumed_optimizer, first)
    fingerprint = {
        "distributed": "none",
        "kv_score_sidecar_execution": kv_score_sidecar_execution_metadata(policy, unrope_keys=False),
    }
    checkpoint_path = tmp_path / "checkpoint"
    save_training_checkpoint(
        checkpoint_path,
        resumed,
        config,
        resumed_optimizer,
        ResumeState(
            step=1,
            smooth_loss=0.0,
            loader_cursor=RandomCursor(rng_state=(1, 2, 3)),
            saved_world_size=1,
            exact_topology=True,
        ),
        fingerprint,
    )
    batch_path = tmp_path / "batch.pt"
    output_path = tmp_path / "output.pt"
    torch.save(second, batch_path)
    worker = Path(__file__).with_name("kv_score_sidecar_resume_worker.py")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(Path(__file__).parents[1] / "src"), environment.get("PYTHONPATH", "")])
    )
    subprocess.run(
        [sys.executable, str(worker), str(checkpoint_path), str(batch_path), str(output_path)],
        check=True,
        cwd=Path(__file__).parents[1],
        env=environment,
    )
    fresh = torch.load(output_path, weights_only=True)
    for name, expected in continuous.state_dict().items():
        assert torch.equal(fresh["model"][name], expected), name
    _assert_nested_equal(fresh["optimizer"], continuous_optimizer.state_dict())


def _assert_nested_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _train_config(*, enabled: bool = False, **overrides: object) -> TrainConfig:
    values: dict[str, object] = {
        "model": _model_config(),
        "data": DataConfig(kind="random"),
        "performance": PerformanceConfig(kv_score_sidecar=KVScoreSidecarConfig(enabled=enabled)),
        "batch_size": 2,
        "sequence_len": 16,
        "total_batch_size": 32,
        "trainable": "cache_processor",
    }
    values.update(overrides)
    return TrainConfig(**values)  # type: ignore[arg-type]


def test_default_off_config_receipts_and_enabled_fingerprint_metadata() -> None:
    implicit = _train_config()
    explicit = replace(implicit, performance=PerformanceConfig())
    assert "kv_score_sidecar" not in to_dict(implicit.performance)
    assert to_dict(implicit) == to_dict(explicit)
    model_receipt = resolve_model(implicit.model).require()[1]
    explicit_model_receipt = resolve_model(explicit.model).require()[1]
    assert model_receipt == explicit_model_receipt
    assert (
        resolve_experiment(implicit, model_receipt).require()[1]
        == resolve_experiment(explicit, explicit_model_receipt).require()[1]
    )
    implicit_fingerprint = _training_fingerprint(implicit, 1, 32, 32, 1)
    assert "kv_score_sidecar_execution" not in implicit_fingerprint

    enabled = _train_config(enabled=True)
    metadata = kv_score_sidecar_execution_metadata(enabled.performance.kv_score_sidecar, unrope_keys=False)
    fingerprint = _training_fingerprint(enabled, 1, 32, 32, 1)
    assert fingerprint["kv_score_sidecar_execution"] == metadata
    assert metadata["campaign_admitted"] is False
    assert metadata["schema"] == "gcw.kv_score_sidecar_execution.v1"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"batch_size": 4}, "batch_size=2"),
        ({"distributed": "ddp"}, "single-process"),
        ({"backbone_chunk_backend": "compile"}, "process_every=64"),
        (
            {
                "performance": PerformanceConfig(
                    kv_score_sidecar=KVScoreSidecarConfig(True), adaptive_batching=AdaptiveBatchingConfig(True)
                )
            },
            "adaptive batching",
        ),
        (
            {
                "performance": PerformanceConfig(
                    kv_score_sidecar=KVScoreSidecarConfig(True), longshot_full_history=LongshotFullHistoryConfig(True)
                )
            },
            "longshot_full_history",
        ),
    ],
)
def test_invalid_policy_compositions_fail_closed(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _train_config(enabled=True, **overrides)


@pytest.mark.parametrize(
    ("context", "message"),
    [
        (None, "explicit execution context"),
        (_sidecar_context(physical_batch_size=4), "physical batch size 2"),
        (_sidecar_context(distributed="ddp"), "single-process"),
        (_sidecar_context(backbone_chunk_backend="compile"), "backbone_chunk_backend='eager'"),
        (_sidecar_context(adaptive_batching_enabled=True), "adaptive batching"),
        (_sidecar_context(longshot_full_history_enabled=True), "full-history"),
        (_sidecar_context(whole_model_compile_requested=True), "whole-model compilation"),
        (_sidecar_context(activation_checkpointing="standard_processor_layers"), "native and segmented"),
        (_sidecar_context(incompatible_refinement_enabled=True), "Flow, Diffusion"),
    ],
)
def test_direct_enable_requires_validated_execution_context(
    context: KVScoreSidecarExecutionContext | None, message: str
) -> None:
    model = build_model(_model_config())
    with pytest.raises(ValueError, match=message):
        configure_kv_score_sidecar_execution(
            model, KVScoreSidecarConfig(enabled=True), context=context
        )
    assert not kv_score_sidecar_execution(model.cache_processor).enabled


@pytest.mark.parametrize(
    "model_config",
    [
        replace(_model_config(), family="gemma4"),
        replace(_model_config(), flow_processor=FlowProcessorConfig()),
    ],
)
def test_direct_enable_rejects_spoofed_actual_model_composition(
    model_config: ModelConfig,
 ) -> None:
    processor = GlobalCacheProcessor(model_config)
    with pytest.raises(ValueError, match="does not match the actual model composition"):
        configure_kv_score_sidecar_execution(
            processor, KVScoreSidecarConfig(enabled=True), context=_sidecar_context()
        )
    assert not kv_score_sidecar_execution(processor).enabled


def test_direct_enable_rejects_an_already_configured_full_history_policy() -> None:
    processor = _processor()
    configure_longshot_full_history_execution(processor, LongshotFullHistoryConfig(enabled=True))
    with pytest.raises(ValueError, match="configured Longshot full-history"):
        configure_kv_score_sidecar_execution(
            processor, KVScoreSidecarConfig(enabled=True), context=_sidecar_context()
        )


def test_explicit_disable_removes_forward_local_execution_policy() -> None:
    processor = _processor()
    configure_kv_score_sidecar_execution(
        processor, KVScoreSidecarConfig(enabled=True), context=_sidecar_context()
    )
    assert kv_score_sidecar_execution(processor).enabled
    configure_kv_score_sidecar_execution(processor, KVScoreSidecarConfig())
    assert not kv_score_sidecar_execution(processor).enabled
