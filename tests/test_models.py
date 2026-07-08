import torch

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models import build_model
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
