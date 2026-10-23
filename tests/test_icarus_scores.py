from __future__ import annotations

import copy
from functools import partial

import pytest
import torch

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models.cache_processors import icarus
from gcw.models.cache_processors.ops import (
    gather_per_cache_head_scores,
    select_heads_from_per_cache_head_scores,
    select_per_cache_head_kv_norm_indices_from_scores,
)
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.performance.longshot_scores import (
    chunked_fp32_l2_norm,
    chunked_fp32_squared_l2,
)


@pytest.mark.parametrize("score", ["attn_l2", "kv_norm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("update_stream", ["kv", "v"])
@pytest.mark.parametrize(
    "unrope_keys,rotary_style,partial_rotary_factor",
    [(False, "ppt", 1.0), (True, "ppt", 1.0), (True, "llama", 0.5)],
)
def test_icarus_chunked_scores_preserve_selection_outputs_and_gradients(
    monkeypatch,
    score,
    dtype,
    update_stream,
    unrope_keys,
    rotary_style,
    partial_rotary_factor,
):
    torch.manual_seed(43)
    optimized = _processor(
        score,
        update_stream=update_stream,
        unrope_keys=unrope_keys,
        per_cache_head_projections=update_stream == "v",
        include_self_head_kv=update_stream == "v",
        gradient_checkpointing=update_stream == "v",
        rotary_style=rotary_style,
        partial_rotary_factor=partial_rotary_factor,
    ).to(dtype=dtype)
    reference = copy.deepcopy(optimized)
    keys = torch.randn(2, 2, 2, 6, 8, dtype=dtype)
    values = torch.randn_like(keys)
    scores = torch.randn(2, 6)
    runs = []

    for processor, bounded in ((optimized, True), (reference, False)):
        with monkeypatch.context() as patch:
            patch.setattr(
                icarus,
                "chunked_fp32_squared_l2",
                (
                    partial(chunked_fp32_squared_l2, max_temp_bytes=256)
                    if bounded
                    else lambda x: x.float().pow(2).sum(dim=-1)
                ),
            )
            patch.setattr(
                icarus,
                "chunked_fp32_l2_norm",
                (
                    partial(chunked_fp32_l2_norm, max_temp_bytes=256)
                    if bounded
                    else lambda x: x.float().flatten(start_dim=2).norm(dim=-1)
                ),
            )
            source_keys = keys.clone().requires_grad_()
            source_values = values.clone().requires_grad_()
            cache = WindowedCache(source_keys, source_values, scores.clone())
            selections = []

            def capture_selection(_module, args):
                selections.append((args[1].clone(), args[2].clone()))

            handle = processor.block.register_forward_pre_hook(
                capture_selection
            )
            output = processor(cache)
            handle.remove()
            loss = (
                output.keys.float().square().mean()
                + output.values.float().square().mean()
            )
            loss.backward()
            runs.append(
                (output, selections, source_keys.grad, source_values.grad)
            )

    actual, expected = runs
    assert torch.equal(actual[0].keys, expected[0].keys)
    assert torch.equal(actual[0].values, expected[0].values)
    assert len(actual[1]) == len(expected[1]) == 1
    for actual_selection, expected_selection in zip(
        actual[1][0], expected[1][0], strict=True
    ):
        assert torch.equal(actual_selection, expected_selection)
    for actual_grad, expected_grad in zip(
        actual[2:], expected[2:], strict=True
    ):
        assert actual_grad is not None and expected_grad is not None
        assert torch.equal(actual_grad, expected_grad)
    for (name, parameter), (reference_name, reference_parameter) in zip(
        optimized.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == reference_name
        assert parameter.grad is not None
        assert reference_parameter.grad is not None
        assert torch.equal(parameter.grad, reference_parameter.grad), name


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_icarus_chunked_selection_preserves_ties_and_near_ties(
    monkeypatch, dtype
):
    one = torch.tensor(1.0, dtype=dtype)
    up = torch.nextafter(one, torch.tensor(float("inf"), dtype=dtype))
    down = torch.nextafter(one, torch.tensor(float("-inf"), dtype=dtype))
    merged = torch.ones(2, 4, 6, 16, dtype=dtype)
    merged[0, 0, 0, -1] = up
    merged[0, 1, 1, -1] = down
    merged[1, 2, 2, -1] = up
    processor = _processor("kv_norm")
    monkeypatch.setattr(
        icarus,
        "chunked_fp32_squared_l2",
        partial(chunked_fp32_squared_l2, max_temp_bytes=64),
    )
    monkeypatch.setattr(
        icarus,
        "chunked_fp32_l2_norm",
        partial(chunked_fp32_l2_norm, max_temp_bytes=6 * 16 * 4),
    )

    indices, heads = processor._select_indices_and_heads(None, merged, False)
    scores = merged.float().pow(2).sum(dim=-1)
    expected_indices = select_per_cache_head_kv_norm_indices_from_scores(
        scores, processor.config.process_every, processor.config.topk
    )
    expected_heads = select_heads_from_per_cache_head_scores(
        gather_per_cache_head_scores(scores, expected_indices),
        processor.config.topk_heads,
    )
    assert torch.equal(indices, expected_indices)
    assert torch.equal(heads, expected_heads)

    norm_heads = icarus._select_heads_from_merged(merged, topk=2)
    expected_norm_heads = (
        merged.float()
        .flatten(start_dim=2)
        .norm(dim=-1)
        .topk(k=2, dim=-1)
        .indices.sort(dim=-1)
        .values
    )
    assert torch.equal(norm_heads, expected_norm_heads)


@pytest.mark.parametrize("score", ["attn_l2", "kv_norm"])
def test_icarus_scoring_bounds_fp32_temporaries(monkeypatch, score):
    processor = _processor(score)
    merged = torch.randn(2, 4, 256, 32, dtype=torch.bfloat16)
    budget = 32 * 1024
    monkeypatch.setattr(
        icarus,
        "chunked_fp32_squared_l2",
        partial(chunked_fp32_squared_l2, max_temp_bytes=budget),
    )
    monkeypatch.setattr(
        icarus,
        "chunked_fp32_l2_norm",
        partial(chunked_fp32_l2_norm, max_temp_bytes=budget),
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
    ) as profile:
        with torch.no_grad():
            if score == "kv_norm":
                processor._select_indices_and_heads(None, merged, False)
            else:
                icarus._select_heads_from_merged(merged, topk=2)
    allocations = [
        event.cpu_memory_usage
        for event in profile.events()
        if event.name in {"aten::_to_copy", "aten::pow"}
        and event.cpu_memory_usage > 0
    ]
    assert merged.numel() * 4 > budget
    assert len(allocations) > 1
    assert max(allocations) <= budget


def _processor(
    score,
    *,
    rotary_style="ppt",
    partial_rotary_factor=1.0,
    **processor_options,
):
    return icarus.GlobalCacheProcessor(
        ModelConfig(
            variant="icarus",
            hidden_size=32,
            intermediate_size=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            rotary_style=rotary_style,
            partial_rotary_factor=partial_rotary_factor,
            cache_processor=CacheProcessorConfig(
                score=score,
                process_every=2,
                topk=2,
                topk_heads=2,
                hidden_size=16,
                intermediate_size=32,
                num_heads=4,
                num_kv_heads=2,
                gate_init=0.5,
                **processor_options,
            ),
        )
    )
