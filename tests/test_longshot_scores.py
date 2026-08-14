from __future__ import annotations

import copy

import pytest
import torch

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models.cache_processors import longshot as longshot_module
from gcw.models.cache_processors.longshot import GlobalCacheProcessor
from gcw.models.cache_processors.longshot_scores import (
    chunked_fp32_l2_norm,
    chunked_fp32_squared_l2,
    eager_fp32_l2_norm,
    eager_fp32_squared_l2,
)
from gcw.models.cache_processors.windowed import WindowedCache


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("score", ["squared_l2", "l2_norm"])
def test_chunked_longshot_scores_match_eager_forward_backward(dtype: torch.dtype, score: str) -> None:
    torch.manual_seed(41)
    x = torch.randn(2, 3, 5, 7, dtype=dtype, requires_grad=True)
    if score == "squared_l2":
        actual = chunked_fp32_squared_l2(x, max_temp_bytes=7 * 4)
        expected = eager_fp32_squared_l2(x)
    else:
        actual = chunked_fp32_l2_norm(x, max_temp_bytes=5 * 7 * 4)
        expected = eager_fp32_l2_norm(x)

    assert torch.equal(actual, expected)
    upstream = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, x, upstream, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, x, upstream)[0]
    assert torch.equal(actual_grad, expected_grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_longshot_score_selection_matches_eager_on_near_ties(dtype: torch.dtype) -> None:
    one = torch.tensor(1.0, dtype=dtype)
    up = torch.nextafter(one, torch.tensor(float("inf"), dtype=dtype))
    down = torch.nextafter(one, torch.tensor(float("-inf"), dtype=dtype))
    base = torch.stack(
        [
            torch.stack([one, one, one, one]),
            torch.stack([one, one, one, up]),
            torch.stack([one, one, up, one]),
            torch.stack([one, up, one, one]),
            torch.stack([up, one, one, one]),
            torch.stack([down, up, one, one]),
        ]
    ).view(1, 2, 3, 4)
    actual = chunked_fp32_squared_l2(base, max_temp_bytes=4 * 4)
    expected = eager_fp32_squared_l2(base)

    assert torch.equal(actual, expected)
    assert torch.equal(torch.topk(actual.flatten(), k=4).indices, torch.topk(expected.flatten(), k=4).indices)

    actual_norm = chunked_fp32_l2_norm(base, max_temp_bytes=3 * 4 * 4)
    expected_norm = eager_fp32_l2_norm(base)
    assert torch.equal(actual_norm, expected_norm)
    assert torch.equal(torch.topk(actual_norm, k=1).indices, torch.topk(expected_norm, k=1).indices)


def test_chunked_longshot_score_saves_source_not_full_fp32_conversion() -> None:
    x = torch.randn(4, 8, 16, 32, dtype=torch.bfloat16, requires_grad=True)

    def saved_by(function):
        saved: list[torch.Tensor] = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: (saved.append(tensor) or tensor),
            lambda tensor: tensor,
        ):
            function(x)
        return saved

    chunked_saved = saved_by(lambda value: chunked_fp32_squared_l2(value, max_temp_bytes=4 * 32 * 4))
    eager_saved = saved_by(eager_fp32_squared_l2)

    assert len(chunked_saved) == 1
    assert chunked_saved[0].data_ptr() == x.data_ptr()
    assert chunked_saved[0].dtype == torch.bfloat16
    assert any(tensor.dtype == torch.float32 and tensor.numel() == x.numel() for tensor in eager_saved)


def test_chunked_longshot_score_bounds_each_profiled_fp32_temporary() -> None:
    max_temp_bytes = 16 * 1024
    x = torch.randn(8, 16, 32, 64, dtype=torch.bfloat16)

    def allocation_sizes(function) -> list[int]:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=True,
        ) as profile:
            with torch.no_grad():
                function()
        return [
            event.cpu_memory_usage
            for event in profile.events()
            if event.name in {"aten::_to_copy", "aten::pow"} and event.cpu_memory_usage > 0
        ]

    chunked_allocations = allocation_sizes(
        lambda: chunked_fp32_squared_l2(x, max_temp_bytes=max_temp_bytes)
    )
    eager_allocations = allocation_sizes(lambda: eager_fp32_squared_l2(x))

    assert chunked_allocations
    assert max(chunked_allocations) <= max_temp_bytes
    assert max(eager_allocations) == x.numel() * torch.empty((), dtype=torch.float32).element_size()


@pytest.mark.parametrize("score", ["attn_l2", "kv_norm"])
@pytest.mark.parametrize("encoder_method", ["linear", "query_attention"])
def test_longshot_modes_match_eager_forward_backward_and_selection(
    monkeypatch: pytest.MonkeyPatch,
    score: str,
    encoder_method: str,
) -> None:
    torch.manual_seed(43)
    optimized = _processor(score=score, encoder_method=encoder_method)
    reference = copy.deepcopy(optimized)
    optimized_cache = _cache()
    reference_cache = _clone_cache(optimized_cache)

    optimized_source = optimized.encode_source(optimized_cache)
    optimized_output = optimized(optimized_cache)
    optimized_loss = optimized_output.values.float().pow(2).mean()
    optimized_loss.backward()

    monkeypatch.setattr(longshot_module, "chunked_fp32_squared_l2", eager_fp32_squared_l2)
    monkeypatch.setattr(longshot_module, "chunked_fp32_l2_norm", eager_fp32_l2_norm)
    reference_source = reference.encode_source(reference_cache)
    reference_output = reference(reference_cache)
    reference_loss = reference_output.values.float().pow(2).mean()
    reference_loss.backward()

    assert torch.equal(optimized_source.indices, reference_source.indices)
    assert torch.equal(optimized_source.selected_heads, reference_source.selected_heads)
    assert torch.equal(optimized_output.keys, reference_output.keys)
    assert torch.equal(optimized_output.values, reference_output.values)
    for optimized_parameter, reference_parameter in zip(
        optimized.parameters(),
        reference.parameters(),
        strict=True,
    ):
        assert optimized_parameter.grad is not None
        assert reference_parameter.grad is not None
        assert torch.equal(optimized_parameter.grad, reference_parameter.grad)


def _processor(*, score: str, encoder_method: str) -> GlobalCacheProcessor:
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
            score=score,  # type: ignore[arg-type]
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            encoder_method=encoder_method,  # type: ignore[arg-type]
            encoder_query_source="final_token",
            update_stream="v",
            gate_init=0.5,
            gradient_checkpointing=False,
        ),
    )
    return GlobalCacheProcessor(config)


def _cache() -> WindowedCache:
    return WindowedCache(
        keys=torch.randn(2, 2, 2, 6, 8),
        values=torch.randn(2, 2, 2, 6, 8),
        scores=torch.randn(2, 6),
    )


def _clone_cache(cache: WindowedCache) -> WindowedCache:
    return WindowedCache(
        keys=cache.keys.clone(),
        values=cache.values.clone(),
        scores=cache.scores.clone(),
    )
