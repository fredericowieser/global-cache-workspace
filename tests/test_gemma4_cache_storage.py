from copy import deepcopy
from dataclasses import replace
from types import MethodType

import pytest
import torch

from gcw.models import build_model
from gcw.models.backbones.transformer.cache import (
    _flatten_windowed_cache,
    _unflatten_windowed_cache,
)
from gcw.models.cache_processors.longshot import GlobalCacheProcessor
from gcw.models.cache_processors.ops import gather_per_cache_head_kv
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import (
    GroupedWindowedCache,
    WindowedCache,
)
from gcw.performance.longshot_scores import native_cache_squared_l2
from test_gemma4_longshot import mixed_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("storage", ["list", "windowed", "static"])
def test_native_selection_matches_full_history_reference(dtype, storage):
    torch.manual_seed(81)
    keys = [torch.randn(2, 3, 17, 8, dtype=dtype) for _ in range(2)]
    values = [torch.randn_like(k) for k in keys]
    cache = CacheState(keys, values, torch.randn(2, 17))
    if storage == "windowed":
        cache = WindowedCache.from_cache_state(cache)
    elif storage == "static":
        cache = StaticCacheState.allocate(
            batch_size=2,
            num_layers=2,
            num_kv_heads=3,
            max_seq_len=23,
            head_dim=8,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        for i in range(2):
            cache.keys[i].fill_(float("nan"))
            cache.values[i].fill_(float("nan"))
            cache.write_layer(i, 0, keys[i], values[i])
        cache.write_scores(0, torch.randn(2, 17))
    merged = torch.cat((torch.cat(keys, 1), torch.cat(values, 1)), -1)
    expected_scores = merged.float().pow(2).sum(-1)
    scores = native_cache_squared_l2(cache, max_temp_bytes=256)
    torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
    indices = scores.topk(4, dim=-1).indices.sort(dim=-1).values
    expected = merged.gather(2, indices[..., None].expand(-1, -1, -1, 16))
    actual = gather_per_cache_head_kv(cache, indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("storage", ["list", "windowed"])
def test_native_gather_gradient_parity(storage):
    torch.manual_seed(82)
    source = torch.randn(2, 2, 2, 13, 4, requires_grad=True)
    reference = source.detach().clone().requires_grad_()
    indices = torch.randint(0, 13, (2, 4, 5))
    cache = CacheState(
        list(source.unbind()), list((source * 2).unbind()), torch.zeros(2, 13)
    )
    if storage == "windowed":
        cache = WindowedCache.from_cache_state(cache)
    actual = gather_per_cache_head_kv(cache, indices)
    flat = reference.permute(1, 0, 2, 3, 4).reshape(2, 4, 13, 4)
    merged = torch.cat((flat, flat * 2), -1)
    expected = merged.gather(2, indices[..., None].expand(-1, -1, -1, 8))
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(source.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize("subset", [None, [1, 3], [0, 2], [0, 3]])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("chunks", [[1] * 13, [3, 2, 6, 1, 5]])
@torch.no_grad()
def test_static_native_caches_match_full_history_attention(
    subset, shared, chunks
):
    torch.manual_seed(83)
    cfg = replace(
        mixed_config(out_proj_init="normal"),
        processor_layer_indices=subset,
        num_layers=6 if shared else 4,
        num_kv_shared_layers=2 if shared else 0,
        layer_types=["sliding_attention", "full_attention"]
        * (3 if shared else 2),
    )
    model = build_model(cfg).eval()
    reference = deepcopy(model)
    for layer in reference.backbone.layers:
        layer.self_attn._attention_view = MethodType(
            lambda self, k, v, start, offset: (k, v, offset), layer.self_attn
        )
    tokens = torch.randint(1, 128, (2, sum(chunks)))
    cache = model.backbone.allocate_generation_cache(
        2, sum(chunks) + 3, tokens.device
    )
    storage = (
        cache.keys
        + cache.values
        + [
            tensor
            for state in cache.attention_states.values()
            for tensor in (state.keys, state.values)
        ]
    )
    pointers = [tensor.data_ptr() for tensor in storage]
    for tensor in storage:
        tensor.fill_(float("nan"))
    dynamic, start = None, 0
    for steps in chunks:
        if start:
            cache = model.cache_processor(cache)
            dynamic = reference.cache_processor(dynamic)
        actual, cache = model.backbone.forward_chunk_static(
            tokens[:, start : start + steps], start, cache
        )
        expected, dynamic = reference.backbone.forward_chunk(
            tokens[:, start : start + steps], start, dynamic
        )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        start += steps
        assert cache.length == start
        for layer in range(cache.num_layers):
            for first, second in zip(
                cache.layer(layer), dynamic.layer(layer), strict=True
            ):
                torch.testing.assert_close(first, second, atol=2e-5, rtol=2e-5)
        for layer, state in cache.attention_states.items():
            full_k, full_v = dynamic.attention_states[layer]
            if state.window is None:
                assert state.keys.shape[2] == cache.capacity
                observed = (
                    state.keys[:, :, :start],
                    state.values[:, :, :start],
                )
                expected_state = (full_k, full_v)
            else:
                assert state.keys.shape[2] == cfg.sliding_window
                first = max(0, start - state.window)
                positions = torch.arange(first, start) % state.keys.shape[2]
                observed = (
                    state.keys.index_select(2, positions),
                    state.values.index_select(2, positions),
                )
                expected_state = (full_k[:, :, first:], full_v[:, :, first:])
            for first, second in zip(observed, expected_state, strict=True):
                torch.testing.assert_close(first, second, atol=2e-5, rtol=2e-5)
        assert pointers == [
            tensor.data_ptr()
            for tensor in cache.keys
            + cache.values
            + [
                tensor
                for state in cache.attention_states.values()
                for tensor in (state.keys, state.values)
            ]
        ]


@pytest.mark.parametrize("subset", [None, [1, 3]])
@pytest.mark.parametrize("checkpoint_windows", [0, 2])
@pytest.mark.parametrize("truncation", [0, 2])
def test_list_training_matches_packed_loss_and_gradients(
    monkeypatch, subset, checkpoint_windows, truncation
):
    torch.manual_seed(84)
    cfg = replace(
        mixed_config(out_proj_init="normal"), processor_layer_indices=subset
    )
    model = build_model(cfg).train()
    reference = deepcopy(model)
    for layer in reference.backbone.layers:
        layer.self_attn._attention_view = MethodType(
            lambda self, k, v, start, offset: (k, v, offset), layer.self_attn
        )
    tokens = torch.randint(1, 128, (2, 13))
    targets = torch.randint(1, 128, (2, 13))
    kwargs = dict(
        activation_checkpoint_windows=checkpoint_windows,
        truncated_bptt_windows=truncation,
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            GlobalCacheProcessor,
            "recurrent_cache_format",
            property(lambda self: "windowed"),
        )
        expected = reference(tokens, targets, **kwargs)
        expected.backward()

    def forbid(*args, **kwargs):
        raise AssertionError(
            "default Gemma training must not pack full histories"
        )

    monkeypatch.setattr(GroupedWindowedCache, "from_cache_state", forbid)
    monkeypatch.setattr(WindowedCache, "from_lists", forbid)
    actual = model(tokens, targets, **kwargs)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for (name, parameter), (ref_name, ref_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == ref_name
        if parameter.grad is None:
            assert ref_parameter.grad is None
        else:
            torch.testing.assert_close(
                parameter.grad, ref_parameter.grad, atol=2e-6, rtol=2e-4
            )


def test_list_checkpoint_boundary_preserves_native_storage_and_optional_state():
    source = torch.randn(2, 2, 7, 4, requires_grad=True)
    cache = CacheState(
        [source * 2, source[:, :1] * 3],
        [source * 4, source[:, :1] * 5],
        source.sum((1, 3)),
        linear_states={3: (source * 6, None), 5: (source * 7, source.sum(1))},
        attention_states={2: (source * 8, source * 9)},
    )
    tensors, layout = _flatten_windowed_cache(cache)
    restored = _unflatten_windowed_cache(tensors, layout)
    roundtrip, _ = _flatten_windowed_cache(restored)
    assert isinstance(restored, CacheState)
    assert all(
        before is after
        for before, after in zip(tensors, roundtrip, strict=True)
    )
    detached, _ = _flatten_windowed_cache(restored.detach())
    assert all(not tensor.requires_grad for tensor in detached)
    sum(tensor.sum() for tensor in roundtrip).backward()
    assert source.grad is not None


def test_static_capacity_and_append_validation():
    backbone = build_model(
        replace(mixed_config(), processor_layer_indices=[1, 3])
    ).backbone
    cache = backbone.allocate_generation_cache(2, 3, torch.device("cpu"))
    assert all(
        state.keys.shape[2] == 3 for state in cache.attention_states.values()
    )
    with torch.no_grad():
        backbone.forward_chunk_static(
            torch.ones(2, 2, dtype=torch.long), 0, cache
        )
        for start, steps in [(0, 1), (3, 1), (2, 2)]:
            with pytest.raises(
                ValueError, match="contiguous append within capacity"
            ):
                backbone.forward_chunk_static(
                    torch.ones(2, steps, dtype=torch.long), start, cache
                )


@pytest.mark.parametrize("subset", [None, [1, 3]])
@pytest.mark.parametrize("score", ["kv_norm", "attn_l2"])
def test_default_processor_gathers_only_selected_history(
    monkeypatch, subset, score
):
    import gcw.models.cache_processors.longshot as longshot
    import gcw.models.cache_processors.longshot_selection as selection

    cfg = replace(
        mixed_config(score=score, out_proj_init="normal"),
        processor_layer_indices=subset,
    )
    cache = CacheState(
        [
            torch.randn(2, heads, 17, dim)
            for heads, dim in cfg.stored_cache_shapes
        ],
        [
            torch.randn(2, heads, 17, dim)
            for heads, dim in cfg.stored_cache_shapes
        ],
        torch.randn(2, 17),
    )
    calls = []

    def gather(cache, indices, **kwargs):
        assert (
            indices.shape[-1]
            == cfg.cache_processor.process_every + cfg.cache_processor.topk
        )
        calls.append(indices.shape)
        return gather_per_cache_head_kv(cache, indices, **kwargs)

    monkeypatch.setattr(longshot, "gather_per_cache_head_kv", gather)
    monkeypatch.setattr(selection, "gather_per_cache_head_kv", gather)
    GlobalCacheProcessor(cfg)(cache)
    assert calls


def test_native_scores_bound_feature_tiles(monkeypatch):
    import gcw.performance.longshot_scores as scoring

    keys = [torch.randn(3, 5, 37, 8) for _ in range(2)]
    cache = CacheState(
        keys, [torch.randn_like(k) for k in keys], torch.zeros(3, 37)
    )
    oracle, tiles = scoring.eager_fp32_squared_l2, []

    def score(tile):
        assert tile.numel() * 4 <= 512
        tiles.append(tile.numel())
        return oracle(tile)

    monkeypatch.setattr(scoring, "eager_fp32_squared_l2", score)
    actual = native_cache_squared_l2(cache, max_temp_bytes=512)
    expected = torch.cat(
        [
            oracle(torch.cat(pair, -1))
            for pair in zip(cache.keys, cache.values, strict=True)
        ],
        1,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(tiles) > 2


@torch.no_grad()
def test_selected_sliding_archive_remains_retrievable_after_wraps():
    cfg = replace(mixed_config(), processor_layer_indices=[0, 2])
    model = build_model(cfg).eval()
    cache = model.backbone.allocate_generation_cache(2, 19, torch.device("cpu"))
    tokens = torch.randint(1, 128, (2, 18))
    for start in range(0, 18, 3):
        model.backbone.forward_chunk_static(
            tokens[:, start : start + 3], start, cache
        )
    for keys in cache.keys:
        keys[:, :, 0] = 100
    scores = native_cache_squared_l2(cache)
    assert torch.equal(scores.argmax(-1), torch.zeros_like(scores.argmax(-1)))
    indices = scores.topk(1, dim=-1).indices
    gathered = gather_per_cache_head_kv(cache, indices)
    assert torch.all(gathered[..., :4] == 100)
    assert cache.seq_len > cfg.sliding_window


@pytest.mark.parametrize("encoder", ["linear", "query_attention"])
@pytest.mark.parametrize("stream", ["kv", "v"])
@pytest.mark.parametrize(
    "routing", ["magnitude", "random", "learned_contextual"]
)
@pytest.mark.parametrize("per_head", [False, True])
@torch.no_grad()
def test_static_processor_matches_packed_across_feature_combinations(
    encoder, stream, routing, per_head
):
    torch.manual_seed(85)
    cfg = mixed_config(
        encoder_method=encoder,
        update_stream=stream,
        head_selection_method=routing,
        per_cache_head_projections=per_head,
        iterations=2,
        iteration_mode="internal" if per_head else "external",
        update_gate_type="contextual_vector" if per_head else "static_scalar",
        out_proj_init="normal",
    )
    model = build_model(cfg).eval()
    source = CacheState(
        [
            torch.randn(2, heads, 11, dim)
            for heads, dim in cfg.stored_cache_shapes
        ],
        [
            torch.randn(2, heads, 11, dim)
            for heads, dim in cfg.stored_cache_shapes
        ],
        torch.randn(2, 11),
    )
    packed = GroupedWindowedCache.from_cache_state(source)
    static = model.backbone.allocate_generation_cache(
        2, 17, torch.device("cpu")
    )
    for i in range(static.num_layers):
        static.write_layer(i, 0, *source.layer(i))
    static.write_scores(0, source.scores)
    torch.manual_seed(86)
    expected = model.cache_processor(packed)
    torch.manual_seed(86)
    actual = model.cache_processor(static)
    for i in range(static.num_layers):
        for index, (first, second) in enumerate(
            zip(actual.layer(i), expected.layer(i), strict=True)
        ):
            torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
            original = source.layer(i)[index]
            assert torch.equal(first[:, :, :-1], original[:, :, :-1])
            if stream == "v" and index == 0:
                assert torch.equal(first, original)
