from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from test_gemma4_longshot import mixed_config
from test_icarus_scores import _processor
from gcw.models.cache_processors.icarus import GlobalCacheProcessor
from gcw.models.cache_processors.ops import (
    scatter_cache_layers,
    scatter_cache_values,
    rotate_cache_keys,
)
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import (
    WindowedCache,
    GroupedWindowedCache,
)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("storage", ["list", "windowed", "static"])
@pytest.mark.parametrize("stream", ["kv", "v"])
@pytest.mark.parametrize(
    "score,select_all",
    [("kv_norm", False), ("attn_l2", False), ("kv_norm", True)],
)
def test_every_gathered_position_is_written(
    monkeypatch, mixed, storage, stream, score, select_all
):
    torch.manual_seed(101)
    if mixed:
        cfg = replace(
            mixed_config(update_stream=stream, score=score), variant="icarus"
        )
        processor = GlobalCacheProcessor(cfg)
    else:
        processor = _processor(score, update_stream=stream)
        cfg = processor._model_config
    original = CacheState(
        [torch.randn(2, h, 9, d) for h, d in cfg.stored_cache_shapes],
        [torch.randn(2, h, 9, d) for h, d in cfg.stored_cache_shapes],
        torch.randn(2, 9),
    )
    cache = deepcopy(original)
    if storage == "windowed":
        cache = (
            GroupedWindowedCache.from_cache_state(cache)
            if mixed
            else WindowedCache.from_lists(
                cache.keys, cache.values, cache.scores
            )
        )
    elif storage == "static":
        cache = StaticCacheState(
            [torch.zeros(2, h, 12, d) for h, d in cfg.stored_cache_shapes],
            [torch.zeros(2, h, 12, d) for h, d in cfg.stored_cache_shapes],
            torch.zeros(2, 12),
            9,
        )
        for i in range(cache.num_layers):
            cache.keys[i][:, :, :9] = original.keys[i]
            cache.values[i][:, :, :9] = original.values[i]
        cache.score_storage[:, :9] = original.scores
    expected = deepcopy(original)
    touched = []

    def record(x, positions, layers):
        result = x[..., -x.shape[-1] // (2 if stream == "v" else 1) :] + 3
        offset = 0
        for layer in layers:
            heads = expected.keys[layer].shape[1]
            width = expected.keys[layer].shape[-1]
            for b in range(2):
                for h in range(heads):
                    for j, pos in enumerate(positions[b, offset + h].tolist()):
                        touched.append(pos)
                        if stream == "kv":
                            expected.keys[layer][b, h, pos] = result[
                                b, offset + h, j, :width
                            ]
                        expected.values[layer][b, h, pos] = result[
                            b, offset + h, j, -width:
                        ]
            offset += heads
        return result

    if mixed:

        def grouped(histories, positions, selected):
            return tuple(
                record(x, pos, group.layers)
                for x, pos, group in zip(
                    histories, positions, processor.shape_groups, strict=True
                )
            )

        monkeypatch.setattr(processor.block, "forward_grouped", grouped)
    else:
        monkeypatch.setattr(
            processor.block,
            "forward",
            lambda x, positions, selected, **kwargs: record(
                x, positions, range(cache.num_layers)
            ),
        )
    pointers = (
        [x.data_ptr() for x in cache.keys + cache.values]
        if storage == "static"
        else None
    )
    with torch.no_grad():
        actual = processor(cache, select_all=select_all)
    assert min(touched) < 8
    for layer in range(cache.num_layers):
        for a, b in zip(actual.layer(layer), expected.layer(layer)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    if storage == "static":
        assert pointers == [x.data_ptr() for x in cache.keys + cache.values]
        assert cache.length == 9


@pytest.mark.parametrize("storage", ["list", "windowed"])
@pytest.mark.parametrize("stream", ["kv", "v"])
def test_per_head_scatter_gradients(storage, stream):
    torch.manual_seed(102)
    k = torch.randn(2, 2, 6, 4, requires_grad=True)
    v = torch.randn(2, 2, 6, 4, requires_grad=True)
    indices = torch.tensor([[[0, 3, 5], [1, 2, 5]], [[2, 4, 5], [0, 1, 5]]])
    delta = torch.randn(2, 2, 3, 4, requires_grad=True)
    cache = CacheState([k], [v], torch.zeros(2, 6))
    if storage == "windowed":
        cache = WindowedCache.from_lists(cache.keys, cache.values, cache.scores)
    updates = torch.gather(v, 2, indices[..., None].expand_as(delta)) + delta
    if stream == "kv":
        scatter_cache_layers(
            cache, indices, updates.unsqueeze(0), updates.unsqueeze(0)
        )
    else:
        scatter_cache_values(cache, indices, updates.unsqueeze(0))
    actual_k, actual_v = cache.layer(0)
    expected_v = v.clone()
    expected_k = k.clone()
    for b in range(2):
        for h in range(2):
            for j, t in enumerate(indices[b, h]):
                expected_v[b, h, t] = updates[b, h, j]
                if stream == "kv":
                    expected_k[b, h, t] = updates[b, h, j]
    actual_grad = torch.autograd.grad(
        actual_k.square().sum() + actual_v.square().sum(),
        (k, v, delta),
        retain_graph=True,
        allow_unused=True,
    )
    expected_grad = torch.autograd.grad(
        expected_k.square().sum() + expected_v.square().sum(),
        (k, v, delta),
        allow_unused=True,
    )
    for a, b in zip(actual_grad, expected_grad):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert delta.grad is None
    assert actual_grad[-1].abs().sum() > 0


@pytest.mark.parametrize("partial", [0.5, 1.0])
def test_inverse_rope_reapplied_at_each_heads_positions(monkeypatch, partial):
    processor = _processor(
        "kv_norm",
        unrope_keys=True,
        rotary_style="llama",
        partial_rotary_factor=partial,
    )
    keys = torch.randn(2, 2, 2, 9, 8)
    cache = WindowedCache(
        keys.clone(), torch.randn_like(keys), torch.randn(2, 9)
    )
    captured = []

    def update(x, indices, selected, **kwargs):
        captured.append(indices)
        return x + 1

    monkeypatch.setattr(processor.block, "forward", update)
    actual = processor(cache)
    indices = captured[0]
    delta = rotate_cache_keys(
        torch.ones(2, 4, indices.shape[-1], 8),
        indices,
        theta=processor.rope_theta,
        rotary_style=processor.rotary_style,
        rotary_dim=processor.rotary_dim,
    )
    expected = keys.clone()
    for b in range(2):
        for c in range(4):
            for j, t in enumerate(indices[b, c]):
                expected[c // 2, b, c % 2, t] += delta[b, c, j]
    torch.testing.assert_close(actual.keys, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("steps", [0, 1])
@pytest.mark.parametrize("select_all", [False, True])
def test_short_and_empty_history(steps, select_all):
    processor = _processor("kv_norm")
    keys = torch.randn(2, 2, 2, steps, 8)
    cache = WindowedCache(keys, torch.randn_like(keys), torch.zeros(2, steps))
    result = processor(cache, select_all=select_all)
    assert result.keys.shape == keys.shape
    assert result.seq_len == steps


@pytest.mark.parametrize("stream", ["kv", "v"])
def test_shared_indices_match_per_head_scatter(stream):
    keys = torch.randn(2, 2, 2, 6, 4)
    values = torch.randn_like(keys)
    indices = torch.tensor([[0, 2, 5], [1, 3, 5]])
    updates = torch.randn(2, 2, 2, 3, 4)
    a = WindowedCache(keys.clone(), values.clone(), torch.zeros(2, 6))
    b = deepcopy(a)
    expanded = indices[:, None].expand(-1, 4, -1)
    if stream == "kv":
        scatter_cache_layers(a, indices, updates, updates)
        scatter_cache_layers(b, expanded, updates, updates)
    else:
        scatter_cache_values(a, indices, updates)
        scatter_cache_values(b, expanded, updates)
    torch.testing.assert_close(a.keys, b.keys, atol=0, rtol=0)
    torch.testing.assert_close(a.values, b.values, atol=0, rtol=0)
