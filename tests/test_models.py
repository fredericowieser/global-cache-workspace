import types

import pytest
import torch

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models import build_model
from gcw.models.cache_state import CacheState
from gcw.models.cache_processor import _merge_kv, _split_kv
from gcw.models.transformer import apply_rotary


def tiny_config(variant="tf"):
    cp = None
    if variant in {"bt", "bgt"}:
        cp = CacheProcessorConfig(process_every=4, topk=2, hidden_size=8, intermediate_size=16, num_heads=1, num_kv_heads=1)
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
    for variant in ("tf", "bt", "bgt"):
        model = build_model(tiny_config(variant))
        model.init_weights()
        x = torch.randint(0, 128, (2, 9))
        y = torch.randint(0, 128, (2, 9))
        loss = model(x, y)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_tiny_models_bf16_autocast_forward_backward():
    for variant in ("tf", "bt", "bgt"):
        model = build_model(tiny_config(variant))
        model.init_weights()
        x = torch.randint(0, 128, (2, 9))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = model(x, x)
        loss.backward()
        assert torch.isfinite(loss)


def test_freeze_backbone_leaves_processor_trainable():
    model = build_model(tiny_config("bt"))
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


def test_bottleneck_gradients_match_reference_loop():
    for variant in ("bt", "bgt"):
        torch.manual_seed(4)
        fast = build_model(tiny_config(variant))
        fast.init_weights()
        ref = build_model(tiny_config(variant))
        ref.load_state_dict(fast.state_dict())
        _patch_reference_processor(ref)

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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_bottleneck_forward_backward_smoke():
    for variant in ("bt", "bgt"):
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


def _clone_cache(cache: CacheState) -> CacheState:
    return CacheState([k.clone() for k in cache.keys], [v.clone() for v in cache.values], cache.scores.clone())


def _reference_per_layer_processor(processor, cache: CacheState, *, select_all: bool = False) -> CacheState:
    indices = cache.all_indices() if select_all else cache.select_indices(processor.config.process_every, processor.config.topk)
    for layer_idx, block in enumerate(processor.blocks):
        keys, values = cache.gather_layer(layer_idx, indices)
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
        parts.append(_merge_kv(keys, values))
    updated = processor.block(torch.cat(parts, dim=-1), indices)
    for layer_idx, part in enumerate(torch.split(updated, [p.shape[-1] for p in parts], dim=-1)):
        keys, values = _split_kv(part, *shapes[layer_idx])
        cache.scatter_layer(layer_idx, indices, keys, values)
    return cache


def _patch_reference_processor(model) -> None:
    if model.config.variant == "bt":
        model.cache_processor.forward = types.MethodType(_reference_per_layer_processor, model.cache_processor)
    else:
        model.cache_processor.forward = types.MethodType(_reference_global_processor, model.cache_processor)


def _assert_cache_close(actual: CacheState, expected: CacheState) -> None:
    for actual_key, expected_key in zip(actual.keys, expected.keys):
        assert torch.allclose(actual_key, expected_key, atol=1e-5, rtol=1e-5)
    for actual_value, expected_value in zip(actual.values, expected.values):
        assert torch.allclose(actual_value, expected_value, atol=1e-5, rtol=1e-5)


def _assert_unselected_unchanged(original: CacheState, actual: CacheState, indices: torch.Tensor) -> None:
    mask = torch.ones(original.scores.shape, dtype=torch.bool)
    mask.scatter_(1, indices.cpu(), False)
    for old_key, new_key, old_value, new_value in zip(original.keys, actual.keys, original.values, actual.values):
        for batch_idx in range(mask.shape[0]):
            assert torch.equal(old_key[batch_idx, :, mask[batch_idx], :], new_key[batch_idx, :, mask[batch_idx], :])
            assert torch.equal(old_value[batch_idx, :, mask[batch_idx], :], new_value[batch_idx, :, mask[batch_idx], :])
