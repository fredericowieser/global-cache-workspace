from __future__ import annotations

import copy
from types import MethodType
from unittest.mock import patch

import pytest
import torch

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models.cache_processors import longshot as longshot_module
from gcw.models.cache_processors.longshot import GlobalCacheProcessor
from gcw.models.cache_processors.ops import (
    expand_cache_head_indices,
    gather_per_cache_head_kv,
    select_per_cache_head_kv_norm_indices_from_scores,
)
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.performance.config import LongshotFullHistoryConfig
from gcw.performance.longshot_cache import (
    configure_longshot_full_history_execution,
    gather_windowed_per_cache_head_kv,
    pop_longshot_full_history_metrics,
)
from gcw.performance.longshot_scores import (
    chunked_fp32_concatenated_kv_squared_l2,
    chunked_fp32_squared_l2,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_layer_major_kv_scores_match_identity_gather_forward_backward(dtype: torch.dtype) -> None:
    torch.manual_seed(71)
    keys = torch.randn(3, 2, 2, 7, 4, dtype=dtype, requires_grad=True)
    values = torch.randn(3, 2, 2, 7, 4, dtype=dtype, requires_grad=True)
    cache = WindowedCache(keys=keys, values=values, scores=torch.randn(2, 7))
    all_indices = expand_cache_head_indices(cache.all_indices(), 6)
    identity_merged = gather_per_cache_head_kv(cache, all_indices)

    expected = chunked_fp32_squared_l2(identity_merged, max_temp_bytes=8 * 4)
    actual = chunked_fp32_concatenated_kv_squared_l2(keys, values, max_temp_bytes=8 * 4)

    assert torch.equal(actual, expected)
    upstream = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, (keys, values), upstream, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, (keys, values), upstream)
    assert torch.equal(actual_grads[0], expected_grads[0])
    assert torch.equal(actual_grads[1], expected_grads[1])


def test_layer_major_kv_score_saves_sources_not_full_merged_history() -> None:
    keys = torch.randn(3, 2, 2, 7, 4, dtype=torch.bfloat16, requires_grad=True)
    values = torch.randn(3, 2, 2, 7, 4, dtype=torch.bfloat16, requires_grad=True)
    saved: list[torch.Tensor] = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: (saved.append(tensor) or tensor),
        lambda tensor: tensor,
    ):
        chunked_fp32_concatenated_kv_squared_l2(keys, values, max_temp_bytes=8 * 4)

    assert len(saved) == 2
    assert saved[0].data_ptr() == keys.data_ptr()
    assert saved[1].data_ptr() == values.data_ptr()
    assert all(tensor.shape != (2, 6, 7, 8) for tensor in saved)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_direct_selected_kv_gather_matches_flatten_then_gather(dtype: torch.dtype) -> None:
    torch.manual_seed(73)
    keys = torch.randn(3, 2, 2, 7, 4, dtype=dtype, requires_grad=True)
    values = torch.randn(3, 2, 2, 7, 4, dtype=dtype, requires_grad=True)
    indices = torch.randint(0, 7, (2, 6, 3))
    cache = WindowedCache(keys=keys, values=values, scores=torch.randn(2, 7))

    expected = gather_per_cache_head_kv(cache, indices)
    selected_keys, selected_values = gather_windowed_per_cache_head_kv(keys, values, indices)
    actual = torch.cat([selected_keys, selected_values], dim=-1)

    assert torch.equal(actual, expected)
    upstream = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, (keys, values), upstream, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, (keys, values), upstream)
    assert torch.equal(actual_grads[0], expected_grads[0])
    assert torch.equal(actual_grads[1], expected_grads[1])


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

    actual_merged, actual_scores, actual_indices = processor._select_kv_norm_history(cache)
    expected_merged, expected_scores, expected_indices = _reference_kv_norm_history(
        processor, reference_cache
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
    object.__setattr__(reference, "_select_kv_norm_history", MethodType(_reference_kv_norm_history, reference))
    optimized_state = {name: value.clone() for name, value in optimized.state_dict().items()}
    reference_state = {name: value.clone() for name, value in reference.state_dict().items()}

    optimized_result = _run_processor(optimized)
    reference_result = _run_processor(reference)

    assert torch.equal(optimized_result.source.indices, reference_result.source.indices)
    assert torch.equal(optimized_result.source.selected_heads, reference_result.source.selected_heads)
    assert torch.equal(optimized_result.source.latest_indices, reference_result.source.latest_indices)
    assert torch.equal(optimized_result.source.merged, reference_result.source.merged)
    assert len(optimized_result.selection_trace) == len(reference_result.selection_trace)
    for actual, expected in zip(optimized_result.selection_trace, reference_result.selection_trace, strict=True):
        assert torch.equal(actual, expected)
    assert torch.equal(optimized_result.output.keys, reference_result.output.keys)
    assert torch.equal(optimized_result.output.values, reference_result.output.values)
    assert torch.equal(optimized_result.key_grad, reference_result.key_grad)
    assert torch.equal(optimized_result.value_grad, reference_result.value_grad)
    assert optimized_result.iteration_metrics == reference_result.iteration_metrics

    for optimized_parameter, reference_parameter in zip(
        optimized.parameters(), reference.parameters(), strict=True
    ):
        assert (optimized_parameter.grad is None) == (reference_parameter.grad is None)
        if optimized_parameter.grad is not None:
            assert reference_parameter.grad is not None
            assert torch.equal(optimized_parameter.grad, reference_parameter.grad)
    assert _state_dict_equal(optimized.state_dict(), optimized_state)
    assert _state_dict_equal(reference.state_dict(), reference_state)
    assert optimized.state_dict().keys() == reference.state_dict().keys()


def test_default_windowed_kv_norm_path_remains_eager() -> None:
    processor = _processor(
        iteration_mode="internal", iterations=1, update_stream="v", unrope_keys=False
    )
    cache, _, _ = _cache(requires_grad=False)

    with patch.object(longshot_module, "gather_per_cache_head_kv", wraps=gather_per_cache_head_kv) as gather:
        processor.encode_source(cache)

    gather.assert_called_once()


def test_windowed_kv_norm_path_does_not_call_full_history_gather() -> None:
    processor = _processor(iteration_mode="internal", iterations=1, update_stream="v", unrope_keys=True)
    configure_longshot_full_history_execution(
        processor, LongshotFullHistoryConfig(enabled=True, max_temp_bytes=1024)
    )
    cache, _, _ = _cache(requires_grad=False)

    with patch.object(
        longshot_module,
        "gather_per_cache_head_kv",
        side_effect=AssertionError("windowed kv_norm must gather selected history only"),
    ):
        source = processor.encode_source(cache)
        processor(cache)

    assert source.indices.shape[-1] > 0
    metrics = pop_longshot_full_history_metrics(processor)
    assert metrics["performance/longshot_full_history_events"] == 2.0
    assert (
        metrics["performance/longshot_full_history_selected_history_gather_bytes"]
        < metrics["performance/longshot_full_history_reference_full_history_gather_bytes"]
    )
    assert (
        metrics["performance/longshot_full_history_optimized_score_fp32_temporary_bound_bytes"]
        < metrics["performance/longshot_full_history_reference_score_fp32_temporary_bound_bytes"]
    )

    with pytest.raises(TypeError, match="WindowedCache"):
        processor._select_kv_norm_history(cache.to_cache_state())


class _RunResult:
    def __init__(
        self,
        *,
        source,
        output: WindowedCache,
        selection_trace: list[torch.Tensor],
        key_grad: torch.Tensor,
        value_grad: torch.Tensor,
        iteration_metrics: dict[str, float],
    ) -> None:
        self.source = source
        self.output = output
        self.selection_trace = selection_trace
        self.key_grad = key_grad
        self.value_grad = value_grad
        self.iteration_metrics = iteration_metrics


def _run_processor(processor: GlobalCacheProcessor) -> _RunResult:
    source_cache, _, _ = _cache(requires_grad=False)
    cache, input_keys, input_values = _cache(requires_grad=True)
    trace: list[torch.Tensor] = []
    selector = longshot_module._select_heads_from_head_scores

    def traced_selector(scores: torch.Tensor, topk: int) -> torch.Tensor:
        selected = selector(scores, topk)
        trace.append(selected.detach().clone())
        return selected

    with patch.object(longshot_module, "_select_heads_from_head_scores", side_effect=traced_selector):
        source = processor.encode_source(source_cache)
        output = processor(cache)
    loss = output.keys.float().pow(2).mean() + output.values.float().pow(2).mean()
    loss.backward()
    assert input_keys.grad is not None
    assert input_values.grad is not None
    return _RunResult(
        source=source,
        output=output,
        selection_trace=trace,
        key_grad=input_keys.grad,
        value_grad=input_values.grad,
        iteration_metrics=processor.block.pop_iteration_metrics(),
    )


def test_selected_history_matches_native_checkpoint_recomputation() -> None:
    torch.manual_seed(97)
    optimized = _processor(
        iteration_mode="internal", iterations=2, update_stream="kv", unrope_keys=False
    )
    optimized.config.gradient_checkpointing = True
    optimized.train()
    configure_longshot_full_history_execution(
        optimized, LongshotFullHistoryConfig(enabled=True, max_temp_bytes=1024)
    )
    reference = copy.deepcopy(optimized)
    object.__setattr__(
        reference, "_select_kv_norm_history", MethodType(_reference_kv_norm_history, reference)
    )

    actual = _run_processor(optimized)
    expected = _run_processor(reference)

    assert torch.equal(actual.source.indices, expected.source.indices)
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
    all_indices = expand_cache_head_indices(cache.all_indices(), self.num_kv_cache_heads)
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


def _cache(*, requires_grad: bool) -> tuple[WindowedCache, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(83)
    keys = torch.randn(2, 2, 2, 6, 8, generator=generator).requires_grad_(requires_grad)
    values = torch.randn(2, 2, 2, 6, 8, generator=generator).requires_grad_(requires_grad)
    return WindowedCache(keys=keys, values=values, scores=torch.randn(2, 6, generator=generator)), keys, values


def _state_dict_equal(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> bool:
    return actual.keys() == expected.keys() and all(
        torch.equal(actual[name], expected[name]) for name in actual
    )
