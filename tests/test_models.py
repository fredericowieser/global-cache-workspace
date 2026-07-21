import functools
import types

import pytest
import torch
import torch.nn as nn

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models import build_model
from gcw.models import icarus as icarus_module
from gcw.models import longshot as longshot_module
from gcw.models.cache_state import CacheState, StaticCacheState
from gcw.models.cache_processor_ops import (
    expand_cache_head_indices,
    gather_per_cache_head_scores,
    gather_per_cache_head_kv,
    select_heads_from_per_cache_head_scores,
    select_per_cache_head_kv_norm_indices_from_scores,
)
from gcw.models.cache_processor import CacheProcessor, _merge_kv, _split_kv
from gcw.models.icarus import CacheProcessor as IcarusCacheProcessor
from gcw.models.icarus import GlobalCacheProcessor as IcarusGlobalCacheProcessor
from gcw.models.icarus import Icarus, _gather_cache_heads, _select_heads_from_merged
from gcw.models.icarus import ProcessorAttention as IcarusProcessorAttention
from gcw.models.longshot import GlobalCacheProcessor as LongshotGlobalCacheProcessor
from gcw.models.longshot import CacheProcessor as LongshotCacheProcessor
from gcw.models.longshot import ProcessorAttention as LongshotProcessorAttention
from gcw.models.longshot import QueryAttentionTimeEncoder
from gcw.models.longshot import Longshot
from gcw.models.transformer import _matches_stop_sequence, apply_rotary
from gcw.models.windowed_cache import WindowedCache


def tiny_config(variant="tf", **processor_overrides):
    cp = None
    if variant in {"bt", "bgt", "icarus", "longshot"}:
        cp = CacheProcessorConfig(
            process_every=4,
            topk=2,
            topk_heads=2,
            hidden_size=8,
            intermediate_size=16,
            num_heads=1,
            num_kv_heads=1,
            **processor_overrides,
        )
    return ModelConfig(
        variant=variant,
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        sequence_len=32,
        tie_embeddings=True,
        cache_processor=cp,
    )


def test_tiny_models_forward_backward():
    for variant in ("tf", "bt", "bgt", "icarus"):
        model = build_model(tiny_config(variant))
        model.init_weights()
        x = torch.randint(0, 128, (2, 9))
        y = torch.randint(0, 128, (2, 9))
        loss = model(x, y)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_tiny_icarus_uses_icarus_processor():
    model = build_model(tiny_config("icarus"))
    assert isinstance(model, Icarus)
    assert isinstance(model.cache_processor, IcarusGlobalCacheProcessor)
    x = torch.randn(2, 4, 6, 16)
    positions = torch.arange(6).view(1, 1, 6).expand(2, 4, -1)
    selected_heads = torch.tensor([[0, 2], [1, 3]])
    assert model.cache_processor.block(x, positions, selected_heads).shape == (2, 4, 6, 16)

    value_only = build_model(tiny_config("icarus", update_stream="v"))
    assert value_only.cache_processor.block.input_dim == 16
    assert value_only.cache_processor.block.output_dim == 8
    assert value_only.cache_processor.block(x, positions, selected_heads).shape == (2, 4, 6, 8)

    model.init_weights()
    tokens = torch.randint(0, 128, (2, 9))
    loss = model(tokens, tokens)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.cache_processor.parameters())


def test_tiny_longshot_uses_longshot_processor_and_configured_encoder():
    linear = build_model(tiny_config("longshot", encoder_seq_len=6))
    assert isinstance(linear, Longshot)
    assert isinstance(linear.cache_processor, LongshotGlobalCacheProcessor)
    assert linear.cache_processor.block.encoder_method == "linear"
    x = torch.randn(2, 4, 6, 16)
    selected_heads = torch.tensor([[0, 2], [1, 3]])
    out = linear.cache_processor.block(x, selected_heads)
    assert out.shape == (2, 4, 16)

    value_only = build_model(tiny_config("longshot", encoder_seq_len=6, update_stream="v"))
    value_out = value_only.cache_processor.block(x, selected_heads)
    assert value_out.shape == (2, 4, 8)

    query = build_model(tiny_config("longshot", encoder_method="query_attention"))
    assert query.cache_processor.block.encoder_method == "query_attention"
    assert query.cache_processor.block.encoder_query_source == "learned"
    assert isinstance(query.cache_processor.block.time_encoder, QueryAttentionTimeEncoder)
    assert query.cache_processor.block.time_encoder.query.shape == (4, 16)
    query.init_weights()
    tokens = torch.randint(0, 128, (2, 9))
    loss = query(tokens, tokens)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in query.cache_processor.parameters())

    final_query = build_model(
        tiny_config("longshot", encoder_method="query_attention", encoder_query_source="final_token")
    )
    assert final_query.cache_processor.block.encoder_query_source == "final_token"
    assert isinstance(final_query.cache_processor.block.time_encoder, QueryAttentionTimeEncoder)
    assert hasattr(final_query.cache_processor.block.time_encoder, "query_proj")
    assert not hasattr(final_query.cache_processor.block.time_encoder, "query")
    final_query.init_weights()
    loss = final_query(tokens, tokens)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in final_query.cache_processor.parameters())


def test_icarus_gradient_checkpointing_is_config_gated(monkeypatch):
    calls = []

    def fake_checkpoint(fn, *args, **kwargs):
        calls.append(kwargs)
        return fn(*args)

    def build_processor(gradient_checkpointing: bool) -> IcarusCacheProcessor:
        config = CacheProcessorConfig(
            topk_heads=1,
            hidden_size=8,
            intermediate_size=16,
            num_layers=2,
            num_heads=1,
            num_kv_heads=1,
            gradient_checkpointing=gradient_checkpointing,
        )
        return IcarusCacheProcessor(16, 2, config).train()

    monkeypatch.setattr(icarus_module, "checkpoint", fake_checkpoint)
    x = torch.randn(1, 2, 3, 16, requires_grad=True)
    positions = torch.arange(3).view(1, 1, 3).expand(1, 2, -1)
    selected_heads = torch.tensor([[0]])

    build_processor(False).compute_update(x, positions, selected_heads).sum().backward()
    assert calls == []

    build_processor(True).compute_update(x, positions, selected_heads).sum().backward()
    assert calls == [{"use_reentrant": False}, {"use_reentrant": False}]


def test_longshot_iterative_checkpointing_includes_score_collection(monkeypatch):
    calls = []

    def fake_checkpoint(fn, *args, **kwargs):
        calls.append(kwargs)
        return fn(*args)

    config = CacheProcessorConfig(
        topk_heads=1,
        hidden_size=8,
        intermediate_size=16,
        num_layers=2,
        num_heads=1,
        num_kv_heads=1,
        encoder_seq_len=3,
        iterations=2,
        gradient_checkpointing=True,
    )
    block = LongshotCacheProcessor(16, 2, config).train()
    monkeypatch.setattr(longshot_module, "checkpoint", fake_checkpoint)
    x = torch.randn(1, 2, 3, 16, requires_grad=True)
    selected_heads = torch.tensor([[0]])

    block.compute_update(x, selected_heads).sum().backward()

    assert calls == [
        {"use_reentrant": False},
        {"use_reentrant": False},
        {"use_reentrant": False},
        {"use_reentrant": False},
    ]


def test_tiny_models_bf16_autocast_forward_backward():
    for variant in ("tf", "bt", "bgt", "icarus"):
        model = build_model(tiny_config(variant))
        model.init_weights()
        x = torch.randint(0, 128, (2, 9))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = model(x, x)
        loss.backward()
        assert torch.isfinite(loss)


def test_freeze_backbone_leaves_processor_trainable():
    for variant in ("bt", "icarus"):
        model = build_model(tiny_config(variant))
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        x = torch.randint(0, 128, (2, 9))
        loss = model(x, x)
        loss.backward()
        assert not any(p.grad is not None for p in model.backbone.parameters())
        assert any(p.grad is not None for p in model.cache_processor.parameters())


def test_bt_bgt_share_backbone_shape():
    bt = tiny_config("bt")
    bgt = tiny_config("bgt")
    assert bt.num_layers == bgt.num_layers
    assert bt.hidden_size == bgt.hidden_size
    assert bt.intermediate_size == bgt.intermediate_size


def test_rotary_matches_hf_rotate_half_convention():
    x = torch.randn(2, 3, 4, 8)
    cos = torch.randn(1, 3, 1, 4)
    sin = torch.randn(1, 3, 1, 4)
    x1, x2 = x[..., :4], x[..., 4:]
    expected = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    assert torch.allclose(apply_rotary(x, cos, sin), expected)


def test_chunked_attention_matches_one_shot_attention():
    torch.manual_seed(2)
    model = build_model(tiny_config("tf")).eval()
    model.init_weights()
    x = torch.randint(0, 128, (2, 9))
    with torch.no_grad():
        one_shot, _ = model.forward_windowed(x, process_every=64)
        chunked, _ = model.forward_windowed(x, process_every=4)
    assert torch.allclose(chunked, one_shot, atol=1e-5, rtol=1e-5)


def test_generate_default_uses_windowed_prefill_cadence():
    model = build_model(tiny_config("tf")).eval()
    model.init_weights()
    calls = []

    def processor(cache, **kwargs):
        calls.append((cache.seq_len, bool(kwargs.get("select_all", False))))
        return cache

    x = torch.randint(0, 128, (1, 9))
    model.generate(x, max_new_tokens=0, processor=processor, process_every=4)
    assert calls == [(4, False), (8, False)]


def test_generate_legacy_bottleneck_prefill_processes_full_cache_once():
    model = build_model(tiny_config("tf")).eval()
    model.init_weights()
    calls = []

    def processor(cache, **kwargs):
        calls.append((cache.seq_len, bool(kwargs.get("select_all", False))))
        return cache

    x = torch.randint(0, 128, (1, 9))
    model.generate(
        x,
        max_new_tokens=5,
        processor=processor,
        process_every=4,
        process_prefill_windows=False,
        process_after_prefill=True,
        relative_generation_cadence=True,
    )
    assert calls == [(9, True), (13, False)]


def test_generate_fast_matches_reference_generate():
    for variant in ("tf", "bt", "bgt", "icarus"):
        torch.manual_seed(3)
        model = build_model(tiny_config(variant)).eval()
        model.init_weights()
        x = torch.randint(0, 128, (2, 9))
        with torch.no_grad():
            reference = model.generate(x, max_new_tokens=6)
            fast = model.generate_fast(x, max_new_tokens=6)
        assert torch.equal(fast, reference)


def test_generate_fast_legacy_bottleneck_cadence_matches_reference():
    model = build_model(tiny_config("tf")).eval()
    model.init_weights()
    calls = []

    def processor(cache, **kwargs):
        calls.append((cache.seq_len, bool(kwargs.get("select_all", False))))
        return cache

    x = torch.randint(0, 128, (1, 9))
    model.generate_fast(
        x,
        max_new_tokens=5,
        processor=processor,
        process_every=4,
        process_prefill_windows=False,
        process_after_prefill=True,
        relative_generation_cadence=True,
    )
    assert calls == [(9, True), (13, False)]


def test_stop_sequence_suffix_matching():
    tokens = torch.tensor([[1, 2, 3], [1, 2, 4]])
    matched = _matches_stop_sequence(tokens, [[2, 3]])
    assert matched.tolist() == [True, False]


def test_per_layer_cache_processor_matches_reference_loop():
    model = build_model(tiny_config("bt"))
    model.init_weights()
    cache = _random_cache(model.config)
    expected = _clone_cache(cache)
    actual = _clone_cache(cache)

    _reference_per_layer_processor(model.cache_processor, expected)
    model.cache_processor(actual)

    _assert_cache_close(actual, expected)
    _assert_unselected_unchanged(cache, actual, cache.select_indices(4, 2))


def test_per_layer_cache_processor_matches_windowed_cache_path():
    model = build_model(tiny_config("bt"))
    model.init_weights()
    cache = _random_cache(model.config)
    expected = _clone_cache(cache)
    actual = WindowedCache.from_cache_state(_clone_cache(cache))

    model.cache_processor(expected)
    model.cache_processor(actual)

    _assert_cache_close(actual.to_cache_state(), expected)
    _assert_unselected_unchanged(cache, actual.to_cache_state(), cache.select_indices(4, 2))


def test_global_cache_processor_matches_reference_loop():
    model = build_model(tiny_config("bgt"))
    model.init_weights()
    cache = _random_cache(model.config)
    expected = _clone_cache(cache)
    actual = _clone_cache(cache)

    _reference_global_processor(model.cache_processor, expected)
    model.cache_processor(actual)

    _assert_cache_close(actual, expected)
    _assert_unselected_unchanged(cache, actual, cache.select_indices(4, 2))


def test_global_cache_processor_matches_windowed_cache_path():
    model = build_model(tiny_config("bgt"))
    model.init_weights()
    cache = _random_cache(model.config)
    expected = _clone_cache(cache)
    actual = WindowedCache.from_cache_state(_clone_cache(cache))

    model.cache_processor(expected)
    model.cache_processor(actual)

    _assert_cache_close(actual.to_cache_state(), expected)
    _assert_unselected_unchanged(cache, actual.to_cache_state(), cache.select_indices(4, 2))


def test_icarus_selects_top_heads_from_gathered_kv_norms():
    parts = [torch.zeros(1, 3, 2, 4) for _ in range(2)]
    parts[0][0, 2] = 10.0
    parts[1][0, 1] = 9.0
    merged = torch.cat(parts, dim=1)

    assert _select_heads_from_merged(merged, topk=2).tolist() == [[2, 4]]


def test_icarus_gathers_cache_heads_per_batch():
    x = torch.arange(2 * 4 * 3 * 8).view(2, 4, 3, 8)
    selected = torch.tensor([[1, 3], [0, 2]])

    gathered = _gather_cache_heads(x, selected)

    assert torch.equal(gathered[0], x[0, [1, 3]])
    assert torch.equal(gathered[1], x[1, [0, 2]])


def test_per_cache_head_kv_norm_indices_select_independent_history():
    keys = [torch.zeros(1, 2, 6, 3)]
    values = [torch.zeros(1, 2, 6, 3)]
    keys[0][0, 0, 1] = 10.0
    values[0][0, 1, 2] = 9.0
    cache = CacheState(keys, values, torch.zeros(1, 6))

    all_indices = expand_cache_head_indices(cache.all_indices(), cache.num_layers * keys[0].shape[1])
    scores = gather_per_cache_head_kv(cache, all_indices).float().pow(2).sum(dim=-1)
    indices = select_per_cache_head_kv_norm_indices_from_scores(scores, process_every=2, topk=1)

    assert indices.tolist() == [[[1, 4, 5], [2, 4, 5]]]


def test_score_based_cache_head_selection_matches_merged_kv_norms():
    cache = _random_cache(tiny_config("icarus"))
    all_indices = expand_cache_head_indices(cache.all_indices(), cache.num_layers * cache.keys[0].shape[1])
    full_merged = gather_per_cache_head_kv(cache, all_indices)
    scores = full_merged.float().pow(2).sum(dim=-1)
    indices = select_per_cache_head_kv_norm_indices_from_scores(scores, process_every=4, topk=2)

    selected_scores = gather_per_cache_head_scores(scores, indices)
    merged = gather_per_cache_head_kv(cache, indices)

    assert torch.equal(
        select_heads_from_per_cache_head_scores(selected_scores, topk=2),
        _select_heads_from_merged(merged, topk=2),
    )


def test_icarus_and_longshot_pass_per_head_positions_to_block():
    for variant in ("icarus", "longshot"):
        model = build_model(tiny_config(variant, score="kv_norm"))
        cache = _random_cache(model.config)
        for layer_idx in range(model.config.num_layers):
            cache.keys[layer_idx].zero_()
            cache.values[layer_idx].zero_()
        cache.keys[0][0, 0, 0] = 10.0
        cache.keys[0][0, 1, 1] = 10.0
        cache.keys[1][0, 0, 1] = 10.0
        cache.keys[1][0, 1, 2] = 10.0
        calls = []

        def forward(self, *args):
            if variant == "icarus":
                x, positions, selected_heads = args
                calls.append((positions.clone(), selected_heads.clone()))
                return x
            x, selected_heads = args
            calls.append((x.clone(), selected_heads.clone()))
            return x[:, :, -1, :]

        model.cache_processor.block.forward = types.MethodType(forward, model.cache_processor.block)
        model.cache_processor(cache)

        assert len(calls) == 1
        if variant == "icarus":
            positions = calls[0][0]
            assert positions.shape == (2, 4, 6)
            assert not torch.equal(positions[0, 0], positions[0, 3])
        else:
            merged = calls[0][0]
            assert merged.shape == (2, 4, 6, 16)


def test_icarus_and_longshot_default_attn_l2_uses_shared_token_positions():
    for variant in ("icarus", "longshot"):
        model = build_model(tiny_config(variant))
        cache = _random_cache(model.config)
        calls = []

        def forward(self, *args):
            if variant == "icarus":
                x, positions, selected_heads = args
                calls.append((positions.clone(), selected_heads.clone()))
                return x
            x, selected_heads = args
            calls.append((x.clone(), selected_heads.clone()))
            return x[:, :, -1, :]

        model.cache_processor.block.forward = types.MethodType(forward, model.cache_processor.block)
        model.cache_processor(cache)

        expected = expand_cache_head_indices(cache.select_indices(4, 2), model.cache_processor.num_cache_heads)
        if variant == "icarus":
            assert torch.equal(calls[0][0], expected)
        else:
            expected_merged = gather_per_cache_head_kv(cache, expected)
            assert torch.equal(calls[0][0], expected_merged)


def test_icarus_cache_processor_matches_windowed_cache_path_and_updates_latest_only():
    model = build_model(tiny_config("icarus"))
    model.init_weights()
    cache = _random_cache(model.config)
    expected = _clone_cache(cache)
    actual = WindowedCache.from_cache_state(_clone_cache(cache))
    selected = cache.select_indices(4, 2)

    model.cache_processor(expected)
    model.cache_processor(actual)

    actual_state = actual.to_cache_state()
    _assert_cache_close(actual_state, expected)
    _assert_cache_positions_unchanged(cache, actual_state, _not_latest_mask(cache, selected))


def test_icarus_cache_processor_matches_static_cache_path():
    model = build_model(tiny_config("icarus"))
    model.init_weights()
    cache = _random_cache(model.config)
    expected = _clone_cache(cache)
    static = _static_cache_from_state(cache)

    model.cache_processor(expected)
    model.cache_processor(static)

    _assert_cache_close(_cache_state_from_static(static), expected)


def test_icarus_value_only_updates_only_latest_values():
    model = build_model(tiny_config("icarus", update_stream="v", gate_mode="direct", gate_init=1.0))
    model.init_weights()
    cache = _random_cache(model.config)
    actual = _clone_cache(cache)
    indices = cache.select_indices(4, 2)

    model.cache_processor(actual)

    _assert_keys_unchanged(cache, actual)
    _assert_cache_positions_unchanged(cache, actual, _not_latest_mask(cache, indices))
    value_delta = sum((new_value - old_value).abs().sum() for old_value, new_value in zip(cache.values, actual.values))
    assert value_delta > 0


def test_icarus_value_only_matches_windowed_cache_path():
    model = build_model(tiny_config("icarus", update_stream="v", gate_mode="direct", gate_init=1.0))
    model.init_weights()
    cache = _random_cache(model.config)
    expected = _clone_cache(cache)
    actual = WindowedCache.from_cache_state(_clone_cache(cache))

    model.cache_processor(expected)
    model.cache_processor(actual)

    actual_state = actual.to_cache_state()
    _assert_cache_close(actual_state, expected)
    _assert_keys_unchanged(cache, actual_state)


def test_icarus_value_only_matches_static_cache_path():
    model = build_model(tiny_config("icarus", update_stream="v", gate_mode="direct", gate_init=1.0))
    model.init_weights()
    cache = _random_cache(model.config)
    expected = _clone_cache(cache)
    static = _static_cache_from_state(cache)

    model.cache_processor(expected)
    model.cache_processor(static)

    actual = _cache_state_from_static(static)
    _assert_cache_close(actual, expected)
    _assert_keys_unchanged(cache, actual)


def test_icarus_select_all_uses_all_indices_and_heads():
    model = build_model(tiny_config("icarus"))
    cache = _random_cache(model.config)
    calls = []

    def forward(self, x, positions, selected_heads):
        calls.append((positions.clone(), selected_heads.clone()))
        return x

    model.cache_processor.block.forward = types.MethodType(forward, model.cache_processor.block)
    model.cache_processor(cache, select_all=True)

    assert len(calls) == 1
    expected_indices = cache.all_indices()[:, None, :].expand(-1, model.cache_processor.num_cache_heads, -1)
    assert torch.equal(calls[0][0], expected_indices)
    assert torch.equal(calls[0][1], cache.all_heads())


def test_icarus_rejects_unsupported_cache_processor_modes():
    model = build_model(tiny_config("icarus", update_stream="v"))
    assert model.cache_processor.block.input_dim == 16
    assert model.cache_processor.block.output_dim == 8
    with pytest.raises(ValueError, match="processor_mode"):
        build_model(tiny_config("icarus", processor_mode="workspace_broadcast"))


def test_longshot_rejects_unsupported_cache_processor_modes():
    model = build_model(tiny_config("longshot", update_stream="v"))
    assert model.cache_processor.block.input_dim == 16
    assert model.cache_processor.block.output_dim == 8
    with pytest.raises(ValueError, match="processor_mode"):
        build_model(tiny_config("longshot", processor_mode="workspace_broadcast"))


def test_bottleneck_gradients_match_reference_loop():
    for variant in ("bt", "bgt"):
        torch.manual_seed(4)
        fast = build_model(tiny_config(variant))
        fast.init_weights()
        ref = build_model(tiny_config(variant))
        ref.load_state_dict(fast.state_dict())
        _patch_reference_processor(ref)
        ref.backbone.forward_windowed_loss = types.MethodType(_reference_windowed_loss, ref.backbone)

        x = torch.randint(0, 128, (2, 9))
        y = torch.randint(0, 128, (2, 9))
        fast_loss = fast(x, y)
        ref_loss = ref(x, y)
        fast_loss.backward()
        ref_loss.backward()

        assert torch.allclose(fast_loss, ref_loss, atol=1e-5, rtol=1e-5)
        for (fast_name, fast_param), (ref_name, ref_param) in zip(
            fast.cache_processor.named_parameters(),
            ref.cache_processor.named_parameters(),
        ):
            assert fast_name == ref_name
            assert fast_param.grad is not None
            assert ref_param.grad is not None
            assert torch.allclose(fast_param.grad, ref_param.grad, atol=2e-5, rtol=2e-4), fast_name


def test_value_only_gradients_match_reference_loop():
    for variant in ("bt", "bgt"):
        torch.manual_seed(8)
        fast = build_model(tiny_config(variant, update_stream="v"))
        fast.init_weights()
        ref = build_model(tiny_config(variant, update_stream="v"))
        ref.load_state_dict(fast.state_dict())
        _patch_reference_processor(ref)
        ref.backbone.forward_windowed_loss = types.MethodType(_reference_windowed_loss, ref.backbone)

        x = torch.randint(0, 128, (2, 9))
        y = torch.randint(0, 128, (2, 9))
        fast_loss = fast(x, y)
        ref_loss = ref(x, y)
        fast_loss.backward()
        ref_loss.backward()

        assert torch.allclose(fast_loss, ref_loss, atol=1e-5, rtol=1e-5)
        for (fast_name, fast_param), (ref_name, ref_param) in zip(
            fast.cache_processor.named_parameters(),
            ref.cache_processor.named_parameters(),
        ):
            assert fast_name == ref_name
            assert fast_param.grad is not None
            assert ref_param.grad is not None
            assert torch.allclose(fast_param.grad, ref_param.grad, atol=2e-5, rtol=2e-4), fast_name


def test_cache_processor_select_all_matches_reference_loop():
    for variant in ("bt", "bgt"):
        model = build_model(tiny_config(variant))
        model.init_weights()
        cache = _random_cache(model.config)
        expected = _clone_cache(cache)
        actual = _clone_cache(cache)

        if variant == "bt":
            _reference_per_layer_processor(model.cache_processor, expected, select_all=True)
        else:
            _reference_global_processor(model.cache_processor, expected, select_all=True)
        model.cache_processor(actual, select_all=True)

        _assert_cache_close(actual, expected)


def test_bottleneck_state_dict_keys_stay_compatible():
    bt_keys = build_model(tiny_config("bt")).state_dict().keys()
    bgt_keys = build_model(tiny_config("bgt")).state_dict().keys()

    assert "cache_processor.blocks.0.layers.0.gate_proj.weight" in bt_keys
    assert "cache_processor.blocks.0.layers.0.attn.q_proj.bias" in bt_keys
    assert "cache_processor.block.layers.0.gate_proj.weight" in bgt_keys
    assert "cache_processor.block.layers.0.attn.q_proj.bias" in bgt_keys
    assert "cache_processor.salience_logits" not in bgt_keys


def test_cache_processor_experimental_defaults_preserve_current_behavior():
    cp = CacheProcessorConfig()
    assert cp.topk_heads == 32
    assert cp.per_cache_head_projections is False
    assert cp.gradient_checkpointing is False
    assert cp.update_stream == "kv"
    assert cp.out_proj_init == "normal"
    assert cp.gate_mode == "sigmoid"
    assert cp.processor_mode == "standard"
    assert cp.score == "attn_l2"
    assert cp.writer_head_fraction == 1.0
    assert cp.salience_temperature == 1.0
    assert cp.unrope_keys is False
    assert cp.include_self_head_kv is False
    assert cp.iterations == 1
    assert cp.iteration_mode == "internal"
    assert cp.internal_iteration_score == "attn_l2"
    assert cp.stream_factor == 2


def test_cache_processor_iterations_must_be_positive():
    with pytest.raises(ValueError, match="iterations"):
        CacheProcessorConfig(iterations=0)


def test_cache_processor_topk_heads_must_be_positive():
    with pytest.raises(ValueError, match="topk_heads"):
        CacheProcessorConfig(topk_heads=0)


def test_cache_processor_iteration_mode_must_be_known():
    with pytest.raises(ValueError, match="iteration_mode"):
        CacheProcessorConfig(iteration_mode="outer")


def test_cache_processor_internal_iteration_score_must_be_known():
    with pytest.raises(ValueError, match="internal_iteration_score"):
        CacheProcessorConfig(internal_iteration_score="attention")


def test_cache_processor_internal_iteration_score_accepts_kv_norm():
    cp = CacheProcessorConfig(internal_iteration_score="kv_norm")
    assert cp.internal_iteration_score == "kv_norm"


def test_self_head_kv_attention_is_config_gated_at_init():
    default_config = CacheProcessorConfig(
        hidden_size=8,
        intermediate_size=16,
        num_heads=1,
        num_kv_heads=1,
        include_self_head_kv=False,
    )
    enabled_config = CacheProcessorConfig(
        hidden_size=8,
        intermediate_size=16,
        num_heads=1,
        num_kv_heads=1,
        include_self_head_kv=True,
    )

    for attention_cls in (IcarusProcessorAttention, LongshotProcessorAttention):
        default_attn = attention_cls(default_config)
        enabled_attn = attention_cls(enabled_config)

        assert default_attn.forward.__func__ is default_attn.forward_selected_head_kv.__func__
        assert enabled_attn.forward.__func__ is enabled_attn.forward_with_self_head_kv.__func__


def test_longshot_cache_head_attention_score_forward_is_config_gated_at_init():
    default_config = CacheProcessorConfig(
        hidden_size=8,
        intermediate_size=16,
        num_heads=1,
        num_kv_heads=1,
        include_self_head_kv=False,
        internal_iteration_score="cache_head_attention",
    )
    enabled_config = CacheProcessorConfig(
        hidden_size=8,
        intermediate_size=16,
        num_heads=1,
        num_kv_heads=1,
        include_self_head_kv=True,
        internal_iteration_score="cache_head_attention",
    )

    default_attn = LongshotProcessorAttention(default_config)
    enabled_attn = LongshotProcessorAttention(enabled_config)

    assert default_attn.forward.__func__ is default_attn.forward_selected_head_kv.__func__
    assert isinstance(default_attn.forward_with_iteration_scores, functools.partial)
    assert default_attn.forward_with_iteration_scores.func.__func__ is default_attn.forward_with_cache_head_attention_scores.__func__
    assert default_attn.forward_with_iteration_scores.keywords == {"include_self_head_kv": False}
    assert enabled_attn.forward.__func__ is enabled_attn.forward_with_self_head_kv.__func__
    assert isinstance(enabled_attn.forward_with_iteration_scores, functools.partial)
    assert enabled_attn.forward_with_iteration_scores.func.__func__ is enabled_attn.forward_with_cache_head_attention_scores.__func__
    assert enabled_attn.forward_with_iteration_scores.keywords == {"include_self_head_kv": True}


def test_longshot_kv_norm_score_forward_is_config_gated_at_init():
    default_config = CacheProcessorConfig(
        hidden_size=8,
        intermediate_size=16,
        num_heads=1,
        num_kv_heads=1,
        include_self_head_kv=False,
        internal_iteration_score="kv_norm",
    )
    enabled_config = CacheProcessorConfig(
        hidden_size=8,
        intermediate_size=16,
        num_heads=1,
        num_kv_heads=1,
        include_self_head_kv=True,
        internal_iteration_score="kv_norm",
    )

    default_attn = LongshotProcessorAttention(default_config)
    enabled_attn = LongshotProcessorAttention(enabled_config)

    assert default_attn.forward.__func__ is default_attn.forward_selected_head_kv.__func__
    assert isinstance(default_attn.forward_with_iteration_scores, functools.partial)
    assert default_attn.forward_with_iteration_scores.func.__func__ is default_attn.forward_selected_head_kv.__func__
    assert default_attn.forward_with_iteration_scores.keywords == {"return_attn_scores": True}
    assert enabled_attn.forward.__func__ is enabled_attn.forward_with_self_head_kv.__func__
    assert isinstance(enabled_attn.forward_with_iteration_scores, functools.partial)
    assert enabled_attn.forward_with_iteration_scores.func.__func__ is enabled_attn.forward_with_self_head_kv.__func__
    assert enabled_attn.forward_with_iteration_scores.keywords == {"return_attn_scores": True}


def test_self_head_kv_attention_can_be_enabled_from_config_dict():
    cfg = ModelConfig(
        variant="icarus",
        cache_processor={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 1,
            "num_kv_heads": 1,
            "include_self_head_kv": True,
        },
    )

    assert cfg.cache_processor.include_self_head_kv is True


def test_cache_processor_forward_is_residual_scaled_update():
    torch.manual_seed(10)
    block = CacheProcessor(
        6,
        CacheProcessorConfig(hidden_size=8, intermediate_size=16, num_heads=1, num_kv_heads=1),
    )
    x = torch.randn(2, 5, 6)
    positions = torch.arange(5).unsqueeze(0).expand(2, -1)

    assert torch.allclose(block(x, positions), x + block.scaled_update(x, positions))


def test_longshot_iterative_processor_reselects_heads_between_iterations():
    class RecordingLayer(nn.Module):
        def __init__(self, scores: torch.Tensor):
            super().__init__()
            self.register_buffer("scores", scores)
            self.calls = []

        def forward(self, x, selected_heads, return_attn_scores=False):
            self.calls.append(selected_heads.detach().clone())
            x = x + 1.0
            if return_attn_scores:
                return x, self.scores[None, :].expand(x.shape[0], -1)
            return x

    config = CacheProcessorConfig(
        process_every=4,
        topk=2,
        topk_heads=2,
        hidden_size=8,
        intermediate_size=16,
        num_heads=1,
        num_kv_heads=1,
        encoder_seq_len=3,
        iterations=3,
    )
    block = LongshotCacheProcessor(16, 4, config)
    layer = RecordingLayer(torch.tensor([0.0, 4.0, 1.0, 3.0]))
    block.layers = nn.ModuleList([layer])
    x = torch.randn(1, 4, 3, 16)

    out = block.compute_update(x, torch.tensor([[0, 2]]))

    assert out.shape == (1, 4, 16)
    assert len(layer.calls) == 3
    assert torch.equal(layer.calls[0], torch.tensor([[0, 2]]))
    assert torch.equal(layer.calls[1], torch.tensor([[1, 3]]))
    assert torch.equal(layer.calls[2], torch.tensor([[1, 3]]))
    assert block.pop_iteration_metrics() == {
        "cache_processor/selected_head_overlap_after_iter_1": 0.0,
        "cache_processor/selected_head_overlap_after_iter_2": 1.0,
    }
    assert block.pop_iteration_metrics() == {}


def test_longshot_iterative_processor_can_reselect_from_cache_head_attention_scores():
    class RecordingAttention(nn.Module):
        def __init__(self, scores: torch.Tensor):
            super().__init__()
            self.register_buffer("scores", scores)
            self.calls = []

        def forward(self, x, selected_heads, return_attn_scores=False):
            assert return_attn_scores is False
            self.calls.append((selected_heads.detach().clone(), return_attn_scores))
            return torch.zeros_like(x)

        def forward_with_iteration_scores(self, x, selected_heads):
            self.calls.append((selected_heads.detach().clone(), True))
            x = torch.zeros_like(x)
            return x, self.scores[None, :].expand(x.shape[0], -1)

    config = CacheProcessorConfig(
        process_every=4,
        topk=2,
        topk_heads=2,
        hidden_size=8,
        intermediate_size=16,
        num_heads=1,
        num_kv_heads=1,
        encoder_seq_len=3,
        iterations=2,
        include_self_head_kv=True,
        internal_iteration_score="cache_head_attention",
    )
    block = LongshotCacheProcessor(16, 4, config)
    attention = RecordingAttention(torch.tensor([0.0, 4.0, 1.0, 3.0]))
    block.layers[0].attn = attention
    x = torch.randn(1, 4, 3, 16)

    out = block.compute_update(x, torch.tensor([[0, 2]]))

    assert out.shape == (1, 4, 16)
    assert len(attention.calls) == 2
    assert torch.equal(attention.calls[0][0], torch.tensor([[0, 2]]))
    assert attention.calls[0][1] is True
    assert torch.equal(attention.calls[1][0], torch.tensor([[1, 3]]))
    assert attention.calls[1][1] is False
    assert block.pop_iteration_metrics() == {"cache_processor/selected_head_overlap_after_iter_1": 0.0}


def test_longshot_external_iteration_repeats_forward_once_and_preserves_select_all():
    model_config = tiny_config("longshot", iterations=3, iteration_mode="external")
    processor = LongshotGlobalCacheProcessor(model_config)
    cache = _random_cache(model_config)
    calls = []
    selected_heads = torch.tensor([[0, 1], [1, 2]])

    def fake_forward_once(self, cache, *, select_all=False):
        calls.append(select_all)
        return cache, selected_heads

    processor._forward_once = types.MethodType(fake_forward_once, processor)

    assert processor(cache, select_all=True) is cache
    assert calls == [True, True, True]


def test_longshot_external_iteration_runs_one_block_iteration_per_global_call():
    class RecordingLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x, selected_heads, return_attn_scores=False):
            self.calls += 1
            assert return_attn_scores is False
            return x

    model_config = tiny_config("longshot", iterations=3, iteration_mode="external")
    processor = LongshotGlobalCacheProcessor(model_config)
    layer = RecordingLayer()
    processor.block.layers = nn.ModuleList([layer])
    cache = _random_cache(model_config)

    processor(cache)

    assert layer.calls == 3


def test_longshot_external_iteration_logs_selected_head_overlap_between_calls():
    processor = LongshotGlobalCacheProcessor(tiny_config("longshot", iterations=3, iteration_mode="external"))
    cache = _random_cache(tiny_config("longshot"))
    selections = iter(
        [
            torch.tensor([[0, 1], [0, 2]]),
            torch.tensor([[1, 2], [2, 3]]),
            torch.tensor([[2, 3], [0, 3]]),
        ]
    )

    def fake_forward_once(self, cache, *, select_all=False):
        return cache, next(selections)

    processor._forward_once = types.MethodType(fake_forward_once, processor)

    processor(cache)

    assert processor.block.pop_iteration_metrics() == {
        "cache_processor/selected_head_overlap_after_iter_1": 0.5,
        "cache_processor/selected_head_overlap_after_iter_2": 0.5,
    }


def test_longshot_external_single_iteration_logs_no_selected_head_overlap():
    processor = LongshotGlobalCacheProcessor(tiny_config("longshot", iterations=1, iteration_mode="external"))
    cache = _random_cache(tiny_config("longshot"))

    def fake_forward_once(self, cache, *, select_all=False):
        return cache, torch.tensor([[0, 1], [1, 2]])

    processor._forward_once = types.MethodType(fake_forward_once, processor)

    processor(cache)

    assert processor.block.pop_iteration_metrics() == {}


def test_zero_out_direct_gate_starts_as_identity():
    for variant in ("bt", "bgt"):
        model = build_model(tiny_config(variant, out_proj_init="zero", gate_mode="direct", gate_init=1.0))
        cache = _random_cache(model.config)
        actual = _clone_cache(cache)

        model.cache_processor(actual, select_all=True)

        _assert_cache_close(actual, cache)


def test_unrope_global_processors_preserve_rope_space_with_zero_update():
    for variant in ("icarus", "longshot"):
        model = build_model(
            tiny_config(
                variant,
                out_proj_init="zero",
                gate_mode="direct",
                gate_init=1.0,
                encoder_seq_len=7,
                unrope_keys=True,
            )
        )
        cache = _random_cache(model.config)
        actual = _clone_cache(cache)

        model.cache_processor(actual, select_all=True)

        _assert_cache_close(actual, cache)


def test_direct_gate_remains_in_gradient_graph_with_zero_out_init():
    for variant in ("bt", "bgt"):
        torch.manual_seed(9)
        model = build_model(tiny_config(variant, out_proj_init="zero", gate_mode="direct", gate_init=1.0))
        x = torch.randint(0, 128, (2, 9))
        loss = model(x, x)
        loss.backward()

        gates = [param.grad for name, param in model.cache_processor.named_parameters() if name.endswith("gate")]
        assert gates
        assert all(grad is not None and torch.isfinite(grad).all() for grad in gates)


def test_value_only_cache_processor_matches_reference_loop():
    for variant in ("bt", "bgt"):
        model = build_model(tiny_config(variant, update_stream="v"))
        model.init_weights()
        cache = _random_cache(model.config)
        expected = _clone_cache(cache)
        actual = _clone_cache(cache)

        if variant == "bt":
            _reference_per_layer_processor(model.cache_processor, expected)
        else:
            _reference_global_processor(model.cache_processor, expected)
        model.cache_processor(actual)

        _assert_cache_close(actual, expected)
        _assert_keys_unchanged(cache, actual)
        _assert_unselected_unchanged(cache, actual, cache.select_indices(4, 2))


def test_value_only_cache_processor_matches_windowed_cache_path():
    for variant in ("bt", "bgt"):
        model = build_model(tiny_config(variant, update_stream="v"))
        model.init_weights()
        cache = _random_cache(model.config)
        expected = _clone_cache(cache)
        actual = WindowedCache.from_cache_state(_clone_cache(cache))

        model.cache_processor(expected)
        model.cache_processor(actual)

        actual_state = actual.to_cache_state()
        _assert_cache_close(actual_state, expected)
        _assert_keys_unchanged(cache, actual_state)


def test_longshot_value_only_updates_only_latest_values():
    model = build_model(tiny_config("longshot", update_stream="v", gate_mode="direct", gate_init=1.0))
    model.init_weights()
    cache = _random_cache(model.config)
    actual = _clone_cache(cache)
    indices = cache.select_indices(4, 2)

    model.cache_processor(actual)

    _assert_keys_unchanged(cache, actual)
    _assert_cache_positions_unchanged(cache, actual, _not_latest_mask(cache, indices))
    value_delta = sum((new_value - old_value).abs().sum() for old_value, new_value in zip(cache.values, actual.values))
    assert value_delta > 0


def test_longshot_value_only_matches_windowed_cache_path():
    model = build_model(tiny_config("longshot", update_stream="v", gate_mode="direct", gate_init=1.0))
    model.init_weights()
    cache = _random_cache(model.config)
    expected = _clone_cache(cache)
    actual = WindowedCache.from_cache_state(_clone_cache(cache))

    model.cache_processor(expected)
    model.cache_processor(actual)

    actual_state = actual.to_cache_state()
    _assert_cache_close(actual_state, expected)
    _assert_keys_unchanged(cache, actual_state)


def test_longshot_value_only_matches_static_cache_path():
    model = build_model(tiny_config("longshot", update_stream="v", gate_mode="direct", gate_init=1.0))
    model.init_weights()
    cache = _random_cache(model.config)
    expected = _clone_cache(cache)
    static = _static_cache_from_state(cache)

    model.cache_processor(expected)
    model.cache_processor(static)

    actual = _cache_state_from_static(static)
    _assert_cache_close(actual, expected)
    _assert_keys_unchanged(cache, actual)


def test_experimental_value_only_forward_backward():
    for variant in ("bt", "bgt"):
        model = build_model(
            tiny_config(
                variant,
                update_stream="v",
                out_proj_init="zero",
                gate_mode="direct",
                gate_init=1.0,
            )
        )
        x = torch.randint(0, 128, (2, 9))
        loss = model(x, x)
        loss.backward()

        assert torch.isfinite(loss)
        assert any(p.grad is not None for p in model.cache_processor.parameters())


def test_icarus_value_only_forward_backward():
    model = build_model(
        tiny_config(
            "icarus",
            update_stream="v",
            out_proj_init="zero",
            gate_mode="direct",
            gate_init=1.0,
        )
    )
    x = torch.randint(0, 128, (2, 9))
    loss = model(x, x)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.cache_processor.parameters())


def test_longshot_value_only_forward_backward():
    model = build_model(
        tiny_config(
            "longshot",
            update_stream="v",
            out_proj_init="zero",
            gate_mode="direct",
            gate_init=1.0,
        )
    )
    x = torch.randint(0, 128, (2, 9))
    loss = model(x, x)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.cache_processor.parameters())


def test_workspace_broadcast_zero_out_starts_as_identity():
    model = build_model(tiny_config("bgt", **_workspace_overrides(out_proj_init="zero")))
    cache = _random_cache(model.config)
    actual = _clone_cache(cache)

    model.cache_processor(actual, select_all=True)

    _assert_cache_close(actual, cache)
    assert "cache_processor.salience_logits" in model.state_dict()


def test_workspace_broadcast_processor_matches_reference_loop():
    torch.manual_seed(11)
    model = build_model(tiny_config("bgt", **_workspace_overrides(out_proj_init="normal")))
    cache = _random_cache(model.config)
    expected = _clone_cache(cache)
    actual = _clone_cache(cache)
    indices = _reference_workspace_indices(model.cache_processor, cache)

    _reference_workspace_processor(model.cache_processor, expected)
    model.cache_processor(actual)

    _assert_cache_close(actual, expected)
    _assert_keys_unchanged(cache, actual)
    _assert_unselected_unchanged(cache, actual, indices)


def test_workspace_broadcast_matches_windowed_cache_path():
    torch.manual_seed(12)
    model = build_model(tiny_config("bgt", **_workspace_overrides(out_proj_init="normal")))
    cache = _random_cache(model.config)
    expected = _clone_cache(cache)
    actual = WindowedCache.from_cache_state(_clone_cache(cache))

    model.cache_processor(expected)
    model.cache_processor(actual)

    actual_state = actual.to_cache_state()
    _assert_cache_close(actual_state, expected)
    _assert_keys_unchanged(cache, actual_state)


def test_workspace_broadcast_salience_logits_receive_gradients():
    torch.manual_seed(13)
    model = build_model(tiny_config("bgt", **_workspace_overrides(out_proj_init="normal")))
    x = torch.randint(0, 128, (2, 9))
    loss = model(x, x)
    loss.backward()

    grad = model.cache_processor.salience_logits.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_workspace_broadcast_is_bgt_only():
    with pytest.raises(ValueError, match="BGT-only"):
        build_model(tiny_config("bt", **_workspace_overrides()))


def test_head_mask_broadcast_zero_out_starts_as_identity():
    model = build_model(tiny_config("bgt", **_head_mask_overrides(out_proj_init="zero")))
    cache = _random_cache(model.config)
    actual = _clone_cache(cache)

    model.cache_processor(actual, select_all=True)

    _assert_cache_close(actual, cache)
    assert "cache_processor.salience_logits" in model.state_dict()


def test_head_mask_broadcast_processor_matches_reference_loop():
    torch.manual_seed(14)
    model = build_model(tiny_config("bgt", **_head_mask_overrides(out_proj_init="normal")))
    cache = _random_cache(model.config)
    expected = _clone_cache(cache)
    actual = _clone_cache(cache)
    indices = _reference_workspace_indices(model.cache_processor, cache)

    _reference_head_mask_processor(model.cache_processor, expected)
    model.cache_processor(actual)

    _assert_cache_close(actual, expected)
    _assert_unselected_unchanged(cache, actual, indices)


def test_head_mask_broadcast_matches_windowed_cache_path():
    torch.manual_seed(15)
    model = build_model(tiny_config("bgt", **_head_mask_overrides(out_proj_init="normal")))
    cache = _random_cache(model.config)
    expected = _clone_cache(cache)
    actual = WindowedCache.from_cache_state(_clone_cache(cache))

    model.cache_processor(expected)
    model.cache_processor(actual)

    actual_state = actual.to_cache_state()
    _assert_cache_close(actual_state, expected)


def test_head_mask_broadcast_salience_logits_receive_gradients():
    torch.manual_seed(16)
    model = build_model(tiny_config("bgt", **_head_mask_overrides(out_proj_init="normal")))
    x = torch.randint(0, 128, (2, 9))
    loss = model(x, x)
    loss.backward()

    grad = model.cache_processor.salience_logits.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_head_mask_broadcast_rejects_invalid_variants_and_streams():
    with pytest.raises(ValueError, match="BGT-only"):
        build_model(tiny_config("bt", **_head_mask_overrides()))
    with pytest.raises(ValueError, match="processor_mode"):
        build_model(tiny_config("icarus", **_head_mask_overrides()))
    with pytest.raises(ValueError, match="update_stream='kv'"):
        build_model(tiny_config("bgt", **_head_mask_overrides(update_stream="v")))
    with pytest.raises(ValueError, match="score='kv_norm'"):
        build_model(tiny_config("bgt", **_head_mask_overrides(score="attn_l2")))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_bottleneck_forward_backward_smoke():
    for variant in ("bt", "bgt", "icarus"):
        torch.manual_seed(5)
        model = build_model(tiny_config(variant)).cuda()
        model.init_weights()
        x = torch.randint(0, 128, (2, 9), device="cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, x)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None for p in model.cache_processor.parameters())
        assert torch.cuda.max_memory_allocated() < 2 * 1024**3


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_experimental_cache_processor_smoke():
    cases = [
        (
            "bt",
            {
                "update_stream": "v",
                "out_proj_init": "zero",
                "gate_mode": "direct",
                "gate_init": 1.0,
            },
        ),
        (
            "bgt",
            {
                "update_stream": "v",
                "out_proj_init": "zero",
                "gate_mode": "direct",
                "gate_init": 1.0,
            },
        ),
        ("bgt", _workspace_overrides(out_proj_init="zero")),
        ("bgt", _head_mask_overrides(out_proj_init="zero")),
    ]
    for variant, overrides in cases:
        torch.manual_seed(6)
        model = build_model(tiny_config(variant, **overrides)).cuda()
        x = torch.randint(0, 128, (2, 9), device="cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, x)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None for p in model.cache_processor.parameters())
        assert torch.cuda.max_memory_allocated() < 2 * 1024**3


def _random_cache(config: ModelConfig) -> CacheState:
    gen = torch.Generator().manual_seed(3)
    keys = [
        torch.randn(2, config.num_kv_heads, 7, int(config.head_dim), generator=gen)
        for _ in range(config.num_layers)
    ]
    values = [
        torch.randn(2, config.num_kv_heads, 7, int(config.head_dim), generator=gen)
        for _ in range(config.num_layers)
    ]
    scores = torch.tensor([[0.1, 2.0, 0.3, 1.5, 0.4, 0.5, 0.6], [1.9, 0.2, 1.7, 0.4, 0.5, 0.6, 0.7]])
    return CacheState(keys, values, scores)


def _workspace_overrides(**overrides):
    base = {
        "processor_mode": "workspace_broadcast",
        "score": "kv_norm",
        "update_stream": "v",
        "gate_mode": "direct",
        "gate_init": 1.0,
        "writer_head_fraction": 0.5,
    }
    base.update(overrides)
    return base


def _head_mask_overrides(**overrides):
    base = {
        "processor_mode": "head_mask_broadcast",
        "score": "kv_norm",
        "update_stream": "kv",
        "writer_head_fraction": 0.5,
    }
    base.update(overrides)
    return base


def _clone_cache(cache: CacheState) -> CacheState:
    return CacheState([k.clone() for k in cache.keys], [v.clone() for v in cache.values], cache.scores.clone())


def _static_cache_from_state(cache: CacheState) -> StaticCacheState:
    return StaticCacheState([k.clone() for k in cache.keys], [v.clone() for v in cache.values], cache.scores.clone(), cache.seq_len)


def _cache_state_from_static(cache: StaticCacheState) -> CacheState:
    keys = [cache.layer_view(i, cache.length)[0].clone() for i in range(cache.num_layers)]
    values = [cache.layer_view(i, cache.length)[1].clone() for i in range(cache.num_layers)]
    return CacheState(keys, values, cache.scores.clone())


def _reference_per_layer_processor(processor, cache: CacheState, *, select_all: bool = False) -> CacheState:
    indices = cache.all_indices() if select_all else cache.select_indices(processor.config.process_every, processor.config.topk)
    for layer_idx, block in enumerate(processor.blocks):
        keys, values = cache.gather_layer(layer_idx, indices)
        if processor.config.update_stream == "v":
            updated = block(_merge_values(values), indices)
            cache.values[layer_idx] = _scatter_values(cache.values[layer_idx], indices, _split_values(updated, values.shape[1], values.shape[3]))
            continue
        updated = block(_merge_kv(keys, values), indices)
        new_keys, new_values = _split_kv(updated, keys.shape[1], keys.shape[3])
        cache.scatter_layer(layer_idx, indices, new_keys, new_values)
    return cache


def _reference_global_processor(processor, cache: CacheState, *, select_all: bool = False) -> CacheState:
    indices = cache.all_indices() if select_all else cache.select_indices(processor.config.process_every, processor.config.topk)
    parts = []
    shapes = []
    for layer_idx in range(processor.num_layers):
        keys, values = cache.gather_layer(layer_idx, indices)
        shapes.append((keys.shape[1], keys.shape[3]))
        if processor.config.update_stream == "v":
            parts.append(_merge_values(values))
        else:
            parts.append(_merge_kv(keys, values))
    updated = processor.block(torch.cat(parts, dim=-1), indices)
    for layer_idx, part in enumerate(torch.split(updated, [p.shape[-1] for p in parts], dim=-1)):
        if processor.config.update_stream == "v":
            cache.values[layer_idx] = _scatter_values(cache.values[layer_idx], indices, _split_values(part, *shapes[layer_idx]))
        else:
            keys, values = _split_kv(part, *shapes[layer_idx])
            cache.scatter_layer(layer_idx, indices, keys, values)
    return cache


def _reference_workspace_processor(processor, cache: CacheState, *, select_all: bool = False) -> CacheState:
    indices = cache.all_indices() if select_all else _reference_workspace_indices(processor, cache)
    values = torch.stack(cache.values, dim=0)
    scores = _reference_kv_scores(processor, cache)
    value_index = indices[None, :, None, :, None].expand(values.shape[0], -1, values.shape[2], -1, values.shape[4])
    score_index = indices[None, :, None, :].expand(scores.shape[0], -1, scores.shape[2], -1)
    selected_values = torch.gather(values, 3, value_index)
    selected_scores = torch.gather(scores, 3, score_index)
    writer_values = selected_values * _reference_writer_gate(processor, selected_scores).to(selected_values.dtype)
    writer_values = writer_values * _reference_token_gate(processor, selected_scores).to(selected_values.dtype)
    delta = processor.block.scaled_update(_merge_global_values(writer_values), indices)
    broadcast = _split_global_values(delta, processor.num_layers, processor.num_kv_heads, processor.head_dim)
    for layer_idx in range(processor.num_layers):
        cache.values[layer_idx] = _scatter_values(cache.values[layer_idx], indices, selected_values[layer_idx] + broadcast[layer_idx])
    return cache


def _reference_head_mask_processor(processor, cache: CacheState, *, select_all: bool = False) -> CacheState:
    indices = cache.all_indices() if select_all else _reference_workspace_indices(processor, cache)
    keys = torch.stack(cache.keys, dim=0)
    values = torch.stack(cache.values, dim=0)
    scores = _reference_kv_scores(processor, cache)
    value_index = indices[None, :, None, :, None].expand(keys.shape[0], -1, keys.shape[2], -1, keys.shape[4])
    score_index = indices[None, :, None, :].expand(scores.shape[0], -1, scores.shape[2], -1)
    selected_keys = torch.gather(keys, 3, value_index)
    selected_values = torch.gather(values, 3, value_index)
    selected_scores = torch.gather(scores, 3, score_index)
    gate = _reference_writer_gate(processor, selected_scores) * _reference_token_gate(processor, selected_scores)
    parts = [
        _merge_kv(selected_keys[layer_idx] * gate[layer_idx].to(keys.dtype), selected_values[layer_idx] * gate[layer_idx].to(values.dtype))
        for layer_idx in range(processor.num_layers)
    ]
    delta = processor.block.scaled_update(torch.cat(parts, dim=-1), indices)
    for layer_idx, part in enumerate(torch.split(delta, [p.shape[-1] for p in parts], dim=-1)):
        key_delta, value_delta = _split_kv(part, selected_keys.shape[2], selected_keys.shape[4])
        cache.scatter_layer(layer_idx, indices, selected_keys[layer_idx] + key_delta, selected_values[layer_idx] + value_delta)
    return cache


def _reference_workspace_indices(processor, cache: CacheState) -> torch.Tensor:
    if cache.scores.shape[1] == 0:
        return torch.empty(cache.scores.shape[0], 0, dtype=torch.long, device=cache.scores.device)
    return _select_l2(_reference_kv_scores(processor, cache).amax(dim=(0, 2)), processor.config)


def _reference_kv_scores(processor, cache: CacheState) -> torch.Tensor:
    keys = torch.stack(cache.keys, dim=0)
    values = torch.stack(cache.values, dim=0)
    weights = torch.softmax(processor.salience_logits.float(), dim=0)
    return weights[0] * keys.float().pow(2).mean(dim=-1).sqrt() + weights[1] * values.float().pow(2).mean(dim=-1).sqrt()


def _select_l2(scores: torch.Tensor, config: CacheProcessorConfig) -> torch.Tensor:
    batch, seq = scores.shape
    recent = min(config.process_every, seq)
    hist_len = seq - recent
    full = torch.arange(seq, device=scores.device).unsqueeze(0).expand(batch, -1)
    recent_idx = full[:, hist_len:]
    if hist_len > 0 and config.topk > 0:
        k = min(config.topk, hist_len)
        _, hist_idx = torch.topk(scores[:, :hist_len].float(), k=k, dim=-1)
        selected = torch.cat([hist_idx, recent_idx], dim=1)
    else:
        selected = recent_idx
    return selected.sort(dim=1).values


def _reference_writer_gate(processor, selected_scores: torch.Tensor) -> torch.Tensor:
    import math

    layers, _, heads, _ = selected_scores.shape
    head_scores = selected_scores.mean(dim=(1, 3))
    writer_count = max(1, math.ceil(layers * heads * processor.config.writer_head_fraction))
    top = torch.topk(head_scores.flatten(), k=writer_count).indices
    mask = torch.zeros_like(head_scores.flatten(), dtype=torch.bool)
    mask.scatter_(0, top, True)
    logits = (head_scores.flatten() / processor.config.salience_temperature).masked_fill(~mask, torch.finfo(head_scores.dtype).min)
    return (torch.softmax(logits, dim=0).view(layers, heads) * writer_count)[:, None, :, None, None]


def _reference_token_gate(processor, selected_scores: torch.Tensor) -> torch.Tensor:
    token_scores = selected_scores.amax(dim=(0, 2))
    gate = torch.softmax(token_scores / processor.config.salience_temperature, dim=-1) * selected_scores.shape[-1]
    return gate[None, :, None, :, None]


def _patch_reference_processor(model) -> None:
    if model.config.variant == "bt":
        model.cache_processor.forward = types.MethodType(_reference_per_layer_processor, model.cache_processor)
    else:
        model.cache_processor.forward = types.MethodType(_reference_global_processor, model.cache_processor)


def _reference_windowed_loss(
    self,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    processor,
    process_every: int,
    loss_reduction: str = "mean",
) -> torch.Tensor:
    cache = CacheState([], [], input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32))
    loss_sum = input_ids.new_zeros((), dtype=torch.float32)
    token_count = input_ids.new_zeros((), dtype=torch.float32)
    for start in range(0, input_ids.shape[1], process_every):
        if start > 0:
            cache = processor(cache)
        end = start + process_every
        chunk_logits, cache = self.forward_chunk(input_ids[:, start:end], start, cache, logits=True)
        chunk_targets = targets[:, start:end].reshape(-1).clone()
        valid = chunk_targets >= 0
        chunk_targets[~valid] = -100
        loss_sum = loss_sum + torch.nn.functional.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.shape[-1]),
            chunk_targets,
            ignore_index=-100,
            reduction="sum",
        )
        token_count = token_count + valid.sum().to(torch.float32)
    if loss_reduction == "sum":
        return loss_sum
    if loss_reduction == "none":
        raise ValueError("windowed loss does not support reduction='none'")
    return loss_sum / token_count.clamp_min(1.0)


def _assert_cache_close(actual: CacheState, expected: CacheState) -> None:
    for actual_key, expected_key in zip(actual.keys, expected.keys):
        assert torch.allclose(actual_key, expected_key, atol=1e-5, rtol=1e-5)
    for actual_value, expected_value in zip(actual.values, expected.values):
        assert torch.allclose(actual_value, expected_value, atol=1e-5, rtol=1e-5)


def _merge_values(values: torch.Tensor) -> torch.Tensor:
    batch, heads, seq, dim = values.shape
    return values.permute(0, 2, 1, 3).reshape(batch, seq, heads * dim)


def _split_values(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    batch, seq, _ = x.shape
    return x.view(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def _merge_global_values(values: torch.Tensor) -> torch.Tensor:
    layers, batch, heads, seq, dim = values.shape
    return values.permute(1, 3, 0, 2, 4).reshape(batch, seq, layers * heads * dim)


def _split_global_values(x: torch.Tensor, num_layers: int, num_heads: int, head_dim: int) -> torch.Tensor:
    batch, seq, _ = x.shape
    return x.view(batch, seq, num_layers, num_heads, head_dim).permute(2, 0, 3, 1, 4).contiguous()


def _scatter_values(old_values: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    index = indices[:, None, :, None].expand(-1, old_values.shape[1], -1, old_values.shape[3])
    return old_values.scatter(2, index, values.to(old_values.dtype))


def _assert_keys_unchanged(original: CacheState, actual: CacheState) -> None:
    for old_key, new_key in zip(original.keys, actual.keys):
        assert torch.equal(old_key, new_key)


def _assert_unselected_unchanged(original: CacheState, actual: CacheState, indices: torch.Tensor) -> None:
    mask = torch.ones(original.scores.shape, dtype=torch.bool)
    mask.scatter_(1, indices.cpu(), False)
    _assert_cache_positions_unchanged(original, actual, mask)


def _not_latest_mask(cache: CacheState, indices: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(cache.scores.shape, dtype=torch.bool)
    latest = indices[:, -1:] if indices.ndim == 2 else indices[:, :1, -1]
    mask.scatter_(1, latest.cpu(), False)
    return mask


def _assert_cache_positions_unchanged(original: CacheState, actual: CacheState, mask: torch.Tensor) -> None:
    for old_key, new_key, old_value, new_value in zip(original.keys, actual.keys, original.values, actual.values):
        for batch_idx in range(mask.shape[0]):
            assert torch.equal(old_key[batch_idx, :, mask[batch_idx], :], new_key[batch_idx, :, mask[batch_idx], :])
            assert torch.equal(old_value[batch_idx, :, mask[batch_idx], :], new_value[batch_idx, :, mask[batch_idx], :])
