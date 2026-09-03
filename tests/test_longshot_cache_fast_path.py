from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models import build_model
from gcw.models.cache_processors.base import cache_processor_recomputation
from gcw.models.cache_processors import longshot as longshot_module
from gcw.models.cache_processors import longshot_selection
from gcw.models.cache_processors.longshot import GlobalCacheProcessor
from gcw.models.cache_processors.ops import (
    expand_cache_head_indices,
    gather_per_cache_head_kv,
    rotate_cache_keys,
    select_per_cache_head_kv_norm_indices_from_scores,
)
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.performance.config import LongshotFullHistoryConfig
from gcw.performance.longshot_cache import (
    configure_longshot_full_history_execution,
    gather_windowed_per_cache_head_kv,
    pop_longshot_full_history_metrics,
    record_longshot_full_history_event,
)
from gcw.performance.longshot_scores import (
    chunked_fp32_concatenated_kv_squared_l2,
    chunked_fp32_unroped_concatenated_kv_squared_l2,
    chunked_fp32_squared_l2,
    unroped_score_estimated_temp_bytes_per_row,
    unrope_rotation_estimated_inner_allocation_bytes,
)
from gcw.training.objectives import row_loss_sums_and_counts


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_layer_major_kv_scores_match_identity_gather_forward_backward(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(71)
    keys = torch.randn(3, 2, 2, 7, 4, dtype=dtype, requires_grad=True)
    values = torch.randn(3, 2, 2, 7, 4, dtype=dtype, requires_grad=True)
    cache = WindowedCache(keys=keys, values=values, scores=torch.randn(2, 7))
    all_indices = expand_cache_head_indices(cache.all_indices(), 6)
    identity_merged = gather_per_cache_head_kv(cache, all_indices)

    expected = chunked_fp32_squared_l2(identity_merged, max_temp_bytes=8 * 4)
    actual = chunked_fp32_concatenated_kv_squared_l2(
        keys, values, max_temp_bytes=8 * 4
    )

    assert torch.equal(actual, expected)
    upstream = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(
        expected, (keys, values), upstream, retain_graph=True
    )
    actual_grads = torch.autograd.grad(actual, (keys, values), upstream)
    assert torch.equal(actual_grads[0], expected_grads[0])
    assert torch.equal(actual_grads[1], expected_grads[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("rotary_style", "rotary_dim"), [("llama", 4), ("llama", 8), ("ppt", 8)]
)
def test_chunk_local_unrope_scores_and_gradients_match_full_history_oracle(
    dtype: torch.dtype,
    rotary_style: str,
    rotary_dim: int,
) -> None:
    torch.manual_seed(72)
    keys = torch.randn(3, 2, 2, 7, 8, dtype=dtype, requires_grad=True)
    values = torch.randn(3, 2, 2, 7, 8, dtype=dtype, requires_grad=True)
    cache = WindowedCache(keys=keys, values=values, scores=torch.randn(2, 7))
    all_indices = expand_cache_head_indices(cache.all_indices(), 6)
    merged = gather_per_cache_head_kv(
        cache,
        all_indices,
        unrope_keys=True,
        rotary_style=rotary_style,
        rotary_dim=rotary_dim,
    )
    expected = chunked_fp32_squared_l2(merged, max_temp_bytes=1024)
    actual = chunked_fp32_unroped_concatenated_kv_squared_l2(
        keys,
        values,
        theta=10000.0,
        rotary_style=rotary_style,
        rotary_dim=rotary_dim,
        max_temp_bytes=1024,
    )

    assert torch.equal(actual, expected)
    upstream = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(
        expected, (keys, values), upstream, retain_graph=True
    )
    actual_grads = torch.autograd.grad(actual, (keys, values), upstream)
    assert torch.equal(actual_grads[0], expected_grads[0])
    assert torch.equal(actual_grads[1], expected_grads[1])


def test_chunk_local_unrope_never_rotates_more_rows_than_the_budget() -> None:
    keys = torch.randn(2, 2, 2, 7, 8, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    bytes_per_row = unroped_score_estimated_temp_bytes_per_row(
        dtype=keys.dtype, head_dim=8, rotary_dim=8
    )
    with patch(
        "gcw.performance.longshot_scores.rotate_cache_keys",
        wraps=rotate_cache_keys,
    ) as rotate:
        chunked_fp32_unroped_concatenated_kv_squared_l2(
            keys,
            values,
            theta=10000.0,
            rotary_style="llama",
            rotary_dim=8,
            max_temp_bytes=3 * bytes_per_row,
        )

    assert len(rotate.call_args_list) > 1
    assert max(call.args[0].shape[0] for call in rotate.call_args_list) <= 3


def test_rotate_cache_keys_inner_allocator_events_fit_the_reported_bound() -> (
    None
):
    rows = 5
    head_dim = rotary_dim = 8
    keys = torch.randn(rows, head_dim, dtype=torch.bfloat16)
    positions = torch.arange(rows)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
    ) as profile:
        rotate_cache_keys(
            keys,
            positions,
            theta=10000.0,
            rotary_style="llama",
            inverse=True,
            rotary_dim=rotary_dim,
        )

    inner_events = [
        event
        for event in profile.events()
        if event.cpu_memory_usage > 0
        and not any(child.cpu_memory_usage > 0 for child in event.cpu_children)
    ]
    names = {event.name for event in inner_events}
    assert {"aten::cos", "aten::sin", "aten::cat"} <= names
    profiled_inner_allocation_bytes = sum(
        event.cpu_memory_usage for event in inner_events
    )
    output_bytes = rows * head_dim * keys.element_size()
    reported_bound = unrope_rotation_estimated_inner_allocation_bytes(
        rows,
        dtype=keys.dtype,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
    )
    assert output_bytes < profiled_inner_allocation_bytes <= reported_bound


def test_layer_major_kv_score_saves_sources_not_full_merged_history() -> None:
    keys = torch.randn(3, 2, 2, 7, 4, dtype=torch.bfloat16, requires_grad=True)
    values = torch.randn(
        3, 2, 2, 7, 4, dtype=torch.bfloat16, requires_grad=True
    )
    saved: list[torch.Tensor] = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: saved.append(tensor) or tensor,
        lambda tensor: tensor,
    ):
        chunked_fp32_concatenated_kv_squared_l2(
            keys, values, max_temp_bytes=8 * 4
        )

    assert len(saved) == 2
    assert saved[0].data_ptr() == keys.data_ptr()
    assert saved[1].data_ptr() == values.data_ptr()
    assert all(tensor.shape != (2, 6, 7, 8) for tensor in saved)


@pytest.mark.parametrize("unrope_keys", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_direct_selected_kv_gather_matches_flatten_then_gather(
    dtype: torch.dtype, unrope_keys: bool
) -> None:
    torch.manual_seed(73)
    keys = torch.randn(3, 2, 2, 7, 4, dtype=dtype, requires_grad=True)
    values = torch.randn(3, 2, 2, 7, 4, dtype=dtype, requires_grad=True)
    indices = torch.randint(0, 7, (2, 6, 3))
    cache = WindowedCache(keys=keys, values=values, scores=torch.randn(2, 7))

    expected = gather_per_cache_head_kv(
        cache, indices, unrope_keys=unrope_keys, rotary_dim=4
    )
    selected_keys, selected_values = gather_windowed_per_cache_head_kv(
        keys, values, indices, unrope_keys=unrope_keys, rotary_dim=4
    )
    actual = torch.cat([selected_keys, selected_values], dim=-1)

    assert torch.equal(actual, expected)
    upstream = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(
        expected, (keys, values), upstream, retain_graph=True
    )
    actual_grads = torch.autograd.grad(actual, (keys, values), upstream)
    assert torch.equal(actual_grads[0], expected_grads[0])
    assert torch.equal(actual_grads[1], expected_grads[1])


def test_qwen3_backbone_cache_positions_drive_exact_llama_unrope_history_scoring() -> (
    None
):
    torch.manual_seed(74)
    config = ModelConfig(
        variant="longshot",
        family="qwen3",
        sequence_len=16,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        qk_norm=True,
        rotary_style="llama",
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=2,
            topk_heads=2,
            score="kv_norm",
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            encoder_method="query_attention",
            encoder_query_source="final_token",
            update_stream="v",
            unrope_keys=True,
            gradient_checkpointing=False,
        ),
    )
    model = build_model(config).eval()
    model.init_weights()
    pre_rotary_keys: list[torch.Tensor] = []
    hooks = []
    for layer in model.backbone.layers:
        assert layer.self_attn is not None
        hooks.append(
            layer.self_attn.k_norm.register_forward_hook(
                lambda _module, _inputs, output: pre_rotary_keys.append(
                    output.detach()
                )
            )
        )
    try:
        _, cache = model.backbone.forward_chunk(
            torch.randint(0, config.vocab_size, (2, 11)),
            0,
            cache_format="windowed",
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert isinstance(cache, WindowedCache)
    assert cache.keys.shape == (2, 2, 1, 11, 8)
    expected_unrotated = torch.stack(pre_rotary_keys).permute(0, 1, 3, 2, 4)
    all_indices = expand_cache_head_indices(
        cache.all_indices(), config.num_layers
    )
    gathered = gather_per_cache_head_kv(
        cache,
        all_indices,
        unrope_keys=True,
        rope_theta=config.rope_theta,
        rotary_style=config.rotary_style,
        rotary_dim=config.head_dim,
    )
    actual_unrotated = (
        gathered[..., : config.head_dim]
        .reshape(2, config.num_layers, config.num_kv_heads, 11, config.head_dim)
        .permute(1, 0, 2, 3, 4)
    )
    torch.testing.assert_close(
        actual_unrotated, expected_unrotated, atol=2e-6, rtol=2e-5
    )

    optimized = model.cache_processor
    assert isinstance(optimized, GlobalCacheProcessor)
    reference = copy.deepcopy(optimized)
    configure_longshot_full_history_execution(
        optimized, LongshotFullHistoryConfig(enabled=True, max_temp_bytes=1024)
    )
    actual = longshot_module.select_kv_norm_history(optimized, cache)
    expected = longshot_module.select_kv_norm_history(reference, cache)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.parametrize("unrope_keys", [False, True])
def test_windowed_kv_norm_scores_indices_and_selected_history_match_reference(
    unrope_keys: bool,
) -> None:
    processor = _processor(
        iteration_mode="internal",
        iterations=1,
        update_stream="kv",
        unrope_keys=unrope_keys,
    )
    configure_longshot_full_history_execution(
        processor, LongshotFullHistoryConfig(enabled=True, max_temp_bytes=1024)
    )
    cache, _, _ = _cache(requires_grad=True)
    reference_cache, _, _ = _cache(requires_grad=True)

    actual_merged, actual_scores, actual_indices = (
        longshot_module.select_kv_norm_history(processor, cache)
    )
    expected_merged, expected_scores, expected_indices = (
        _reference_kv_norm_history(processor, reference_cache)
    )

    assert torch.equal(actual_scores, expected_scores)
    assert torch.equal(actual_indices, expected_indices)
    assert torch.equal(actual_merged, expected_merged)


@pytest.mark.parametrize(
    ("iteration_mode", "iterations"),
    [("internal", 1), ("internal", 2), ("external", 2)],
)
@pytest.mark.parametrize("update_stream", ["kv", "v"])
@pytest.mark.parametrize("unrope_keys", [False, True])
def test_windowed_kv_norm_fast_path_matches_reference_recurrence(
    iteration_mode: str,
    iterations: int,
    update_stream: str,
    unrope_keys: bool,
) -> None:
    torch.manual_seed(79)
    optimized = _processor(
        iteration_mode=iteration_mode,
        iterations=iterations,
        update_stream=update_stream,
        unrope_keys=unrope_keys,
    )
    configure_longshot_full_history_execution(
        optimized, LongshotFullHistoryConfig(enabled=True, max_temp_bytes=1024)
    )
    reference = copy.deepcopy(optimized)
    optimized_state = {
        name: value.clone() for name, value in optimized.state_dict().items()
    }
    reference_state = {
        name: value.clone() for name, value in reference.state_dict().items()
    }

    optimized_result = _run_processor(optimized)
    reference_result = _run_processor(
        reference, selection_fn=_reference_kv_norm_history
    )

    assert torch.equal(optimized_result.history[2], reference_result.history[2])
    assert torch.equal(optimized_result.history[0], reference_result.history[0])
    assert len(optimized_result.selection_trace) == len(
        reference_result.selection_trace
    )
    for actual, expected in zip(
        optimized_result.selection_trace,
        reference_result.selection_trace,
        strict=True,
    ):
        assert torch.equal(actual, expected)
    assert torch.equal(
        optimized_result.output.keys, reference_result.output.keys
    )
    assert torch.equal(
        optimized_result.output.values, reference_result.output.values
    )
    assert torch.equal(optimized_result.key_grad, reference_result.key_grad)
    assert torch.equal(optimized_result.value_grad, reference_result.value_grad)
    assert (
        optimized_result.iteration_metrics == reference_result.iteration_metrics
    )

    for optimized_parameter, reference_parameter in zip(
        optimized.parameters(), reference.parameters(), strict=True
    ):
        assert (optimized_parameter.grad is None) == (
            reference_parameter.grad is None
        )
        if optimized_parameter.grad is not None:
            assert reference_parameter.grad is not None
            assert torch.equal(
                optimized_parameter.grad, reference_parameter.grad
            )
    assert _state_dict_equal(optimized.state_dict(), optimized_state)
    assert _state_dict_equal(reference.state_dict(), reference_state)
    assert optimized.state_dict().keys() == reference.state_dict().keys()


def test_default_windowed_kv_norm_path_remains_eager() -> None:
    processor = _processor(
        iteration_mode="internal",
        iterations=1,
        update_stream="v",
        unrope_keys=False,
    )
    cache, _, _ = _cache(requires_grad=False)

    with patch.object(
        longshot_selection,
        "gather_per_cache_head_kv",
        wraps=gather_per_cache_head_kv,
    ) as gather:
        processor(cache)

    gather.assert_called_once()


@pytest.mark.parametrize("unrope_keys", [False, True])
def test_windowed_kv_norm_path_does_not_call_full_history_gather(
    unrope_keys: bool,
) -> None:
    processor = _processor(
        iteration_mode="internal",
        iterations=1,
        update_stream="v",
        unrope_keys=unrope_keys,
    )
    configure_longshot_full_history_execution(
        processor, LongshotFullHistoryConfig(enabled=True, max_temp_bytes=1024)
    )
    cache, _, _ = _cache(requires_grad=False)

    with patch.object(
        longshot_selection,
        "gather_per_cache_head_kv",
        side_effect=AssertionError(
            "windowed kv_norm must gather selected history only"
        ),
    ):
        processor(cache)

    metrics = pop_longshot_full_history_metrics(processor)
    assert metrics["performance/longshot_full_history_events"] == 1.0
    assert (
        metrics[
            "performance/longshot_full_history_selected_history_logical_payload_bytes"
        ]
        < metrics[
            "performance/longshot_full_history_reference_full_history_logical_payload_bytes"
        ]
    )
    assert (
        metrics[
            "performance/longshot_full_history_optimized_score_estimated_fp32_working_set_bound_bytes"
        ]
        < metrics[
            "performance/longshot_full_history_reference_score_estimated_fp32_working_set_bound_bytes"
        ]
    )
    if unrope_keys:
        assert (
            metrics[
                "performance/longshot_full_history_optimized_selected_unrope_estimated_live_bytes"
            ]
            < metrics[
                "performance/longshot_full_history_reference_unrope_estimated_live_bytes"
            ]
        )
        full_rows = 2 * 2 * 2 * 6
        selected_positions = min(
            cache.keys.shape[3],
            processor.config.process_every + processor.config.topk,
        )
        selected_rows = full_rows // 6 * selected_positions
        score_row_bound = unroped_score_estimated_temp_bytes_per_row(
            dtype=torch.float32, head_dim=8, rotary_dim=8
        )
        expected_score_calls = (full_rows + (1024 // score_row_bound) - 1) // (
            1024 // score_row_bound
        )
        assert (
            metrics[
                "performance/longshot_full_history_optimized_score_unrope_rotate_cache_keys_calls"
            ]
            == expected_score_calls
        )
        assert (
            metrics[
                "performance/longshot_full_history_optimized_score_unrope_rotate_cache_keys_rows"
            ]
            == full_rows
        )
        assert (
            metrics[
                "performance/longshot_full_history_optimized_selected_unrope_rotate_cache_keys_calls"
            ]
            == 1.0
        )
        assert (
            metrics[
                "performance/longshot_full_history_optimized_selected_unrope_rotate_cache_keys_rows"
            ]
            == selected_rows
        )

    with pytest.raises(TypeError, match="WindowedCache"):
        longshot_module.select_kv_norm_history(
            processor, cache.to_cache_state()
        )


def test_enabled_full_history_admits_bounded_chunk_local_unrope() -> None:
    processor = _processor(
        iteration_mode="internal",
        iterations=1,
        update_stream="kv",
        unrope_keys=True,
    )
    configure_longshot_full_history_execution(
        processor, LongshotFullHistoryConfig(enabled=True, max_temp_bytes=1024)
    )
    assert (
        longshot_module.select_kv_norm_history(
            processor, _cache(requires_grad=False)[0]
        )[2].shape[-1]
        > 0
    )


def test_full_history_telemetry_ignores_checkpoint_recomputation() -> None:
    processor = _processor(
        iteration_mode="internal",
        iterations=1,
        update_stream="v",
        unrope_keys=False,
    )
    cache, _, _ = _cache(requires_grad=False)
    record_longshot_full_history_event(
        processor,
        keys=cache.keys,
        selected_positions=2,
        max_temp_bytes=1024,
    )
    with cache_processor_recomputation():
        record_longshot_full_history_event(
            processor,
            keys=cache.keys,
            selected_positions=2,
            max_temp_bytes=1024,
        )
        assert pop_longshot_full_history_metrics(processor) == {}
    metrics = pop_longshot_full_history_metrics(processor)
    assert metrics["performance/longshot_full_history_events"] == 1.0


def test_segmented_backward_records_full_history_events_once_per_original_forward() -> (
    None
):
    config = ModelConfig(
        variant="longshot",
        family="smollm",
        sequence_len=8,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        cache_processor=CacheProcessorConfig(
            process_every=2,
            topk=2,
            topk_heads=2,
            score="kv_norm",
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
            encoder_method="query_attention",
            encoder_query_source="final_token",
            gradient_checkpointing=True,
        ),
    )
    model = build_model(config)
    model.init_weights()
    model.train()
    configure_longshot_full_history_execution(
        model, LongshotFullHistoryConfig(enabled=True, max_temp_bytes=1024)
    )
    processor = model.cache_processor
    original_select = longshot_module.select_kv_norm_history
    calls = 0

    def traced(owner: GlobalCacheProcessor, cache: WindowedCache):
        nonlocal calls
        calls += 1
        return original_select(owner, cache)

    with patch.object(
        longshot_module, "select_kv_norm_history", side_effect=traced
    ):
        inputs = torch.randint(0, config.vocab_size, (2, 6))
        targets = torch.randint(0, config.vocab_size, (2, 6))
        row_sums, row_counts = row_loss_sums_and_counts(
            model,
            inputs,
            targets,
            activation_checkpoint_windows=2,
            longshot_gradient_checkpointing=True,
        )
        (row_sums / row_counts).mean().backward()

    metrics = pop_longshot_full_history_metrics(processor)
    assert calls == 4
    assert metrics["performance/longshot_full_history_events"] == 2.0


class _RunResult:
    def __init__(
        self,
        *,
        history: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output: WindowedCache,
        selection_trace: list[torch.Tensor],
        key_grad: torch.Tensor,
        value_grad: torch.Tensor,
        iteration_metrics: dict[str, float],
    ) -> None:
        self.history = history
        self.output = output
        self.selection_trace = selection_trace
        self.key_grad = key_grad
        self.value_grad = value_grad
        self.iteration_metrics = iteration_metrics


def _run_processor(
    processor: GlobalCacheProcessor,
    *,
    selection_fn=None,
) -> _RunResult:
    source_cache, _, _ = _cache(requires_grad=False)
    cache, input_keys, input_values = _cache(requires_grad=True)
    trace: list[torch.Tensor] = []
    selector = longshot_module._select_heads_from_head_scores

    def traced_selector(scores: torch.Tensor, topk: int) -> torch.Tensor:
        selected = selector(scores, topk)
        trace.append(selected.detach().clone())
        return selected

    selection_fn = selection_fn or longshot_module.select_kv_norm_history

    def select_history(owner, current_cache):
        return selection_fn(owner, current_cache)

    with (
        patch.object(
            longshot_module,
            "_select_heads_from_head_scores",
            side_effect=traced_selector,
        ),
        patch.object(
            longshot_module,
            "select_kv_norm_history",
            side_effect=select_history,
        ),
    ):
        history = longshot_module.select_kv_norm_history(
            processor, source_cache
        )
        output = processor(cache)
    loss = (
        output.keys.float().pow(2).mean() + output.values.float().pow(2).mean()
    )
    loss.backward()
    assert input_keys.grad is not None
    assert input_values.grad is not None
    return _RunResult(
        history=history,
        output=output,
        selection_trace=trace,
        key_grad=input_keys.grad,
        value_grad=input_values.grad,
        iteration_metrics=processor.block.pop_iteration_metrics(),
    )


@pytest.mark.parametrize("unrope_keys", [False, True])
def test_selected_history_matches_native_checkpoint_recomputation(
    unrope_keys: bool,
) -> None:
    torch.manual_seed(97)
    optimized = _processor(
        iteration_mode="internal",
        iterations=2,
        update_stream="kv",
        unrope_keys=unrope_keys,
    )
    optimized.config.gradient_checkpointing = True
    optimized.train()
    configure_longshot_full_history_execution(
        optimized, LongshotFullHistoryConfig(enabled=True, max_temp_bytes=1024)
    )
    reference = copy.deepcopy(optimized)
    actual = _run_processor(optimized)
    expected = _run_processor(
        reference, selection_fn=_reference_kv_norm_history
    )

    assert torch.equal(actual.history[2], expected.history[2])
    assert torch.equal(actual.output.keys, expected.output.keys)
    assert torch.equal(actual.output.values, expected.output.values)
    assert torch.equal(actual.key_grad, expected.key_grad)
    assert torch.equal(actual.value_grad, expected.value_grad)
    for actual_parameter, expected_parameter in zip(
        optimized.parameters(), reference.parameters(), strict=True
    ):
        if actual_parameter.grad is not None:
            assert expected_parameter.grad is not None
            assert torch.equal(actual_parameter.grad, expected_parameter.grad)


def _reference_kv_norm_history(
    self: GlobalCacheProcessor,
    cache: WindowedCache,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_indices = expand_cache_head_indices(
        cache.all_indices(), self.num_kv_cache_heads
    )
    merged_all = gather_per_cache_head_kv(
        cache,
        all_indices,
        unrope_keys=self.config.unrope_keys,
        rope_theta=self.rope_theta,
        rotary_style=self.rotary_style,
        rotary_dim=self.rotary_dim,
    )
    scores = chunked_fp32_squared_l2(merged_all)
    indices = select_per_cache_head_kv_norm_indices_from_scores(
        scores,
        self.config.process_every,
        self.config.topk,
    )
    index = indices[..., None].expand(-1, -1, -1, merged_all.shape[-1])
    return torch.gather(merged_all, dim=2, index=index), scores, indices


def _processor(
    *,
    iteration_mode: str,
    iterations: int,
    update_stream: str,
    unrope_keys: bool,
) -> GlobalCacheProcessor:
    config = ModelConfig(
        variant="longshot",
        family="smollm",
        sequence_len=16,
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        cache_processor=CacheProcessorConfig(
            process_every=2,
            topk=2,
            topk_heads=2,
            score="kv_norm",
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_heads=2,
            num_kv_heads=2,
            encoder_method="query_attention",
            encoder_query_source="final_token",
            update_stream=update_stream,  # type: ignore[arg-type]
            gate_init=0.5,
            iterations=iterations,
            iteration_mode=iteration_mode,  # type: ignore[arg-type]
            internal_iteration_score="kv_norm",
            unrope_keys=unrope_keys,
            gradient_checkpointing=False,
        ),
    )
    return GlobalCacheProcessor(config)


def _cache(
    *, requires_grad: bool
) -> tuple[WindowedCache, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(83)
    keys = torch.randn(2, 2, 2, 6, 8, generator=generator).requires_grad_(
        requires_grad
    )
    values = torch.randn(2, 2, 2, 6, 8, generator=generator).requires_grad_(
        requires_grad
    )
    return (
        WindowedCache(
            keys=keys,
            values=values,
            scores=torch.randn(2, 6, generator=generator),
        ),
        keys,
        values,
    )


def _state_dict_equal(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> bool:
    return actual.keys() == expected.keys() and all(
        torch.equal(actual[name], expected[name]) for name in actual
    )
