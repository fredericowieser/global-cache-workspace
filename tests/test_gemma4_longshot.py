from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from gcw.architecture import resolve_model
from gcw.checkpoint import (
    load_backbone_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from gcw.config import (
    CacheProcessorConfig,
    DiffusionProcessorConfig,
    FlowProcessorConfig,
    ModelConfig,
    TrainConfig,
    to_dict,
)
from gcw.convert import config_from_hf, load_hf_state
from gcw.models import build_model
from gcw.models.backbones.transformer.cache import (
    _flatten_windowed_cache,
    _unflatten_windowed_cache,
)
from gcw.models.cache_processors.state import CacheState
from gcw.models.cache_processors.update_metrics import UPDATE_BATCH, UpdateBatch
from gcw.models.cache_processors.windowed import GroupedWindowedCache


def mixed_config(**processor):
    return ModelConfig(
        family="gemma4",
        variant="longshot",
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        global_head_dim=8,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        sequence_len=32,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"] * 2,
        hidden_act="gelu_pytorch_tanh",
        final_logit_softcapping=30.0,
        rope_parameters={
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        },
        cache_processor=CacheProcessorConfig(
            **{
                "process_every": 3,
                "topk": 2,
                "topk_heads": 2,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_layers": 1,
                "num_heads": 2,
                "num_kv_heads": 1,
                "encoder_method": "query_attention",
                "score": "kv_norm",
                "head_score_normalization": "rms",
                "unrope_keys": False,
                **processor,
            }
        ),
    )


def test_mixed_gemma_matches_transformers():
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    hf_config = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        global_head_dim=8,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=128,
        max_position_embeddings=32,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"] * 2,
        final_logit_softcapping=30.0,
        rope_parameters=mixed_config().rope_parameters,
    )
    torch.manual_seed(31)
    hf = Gemma4ForCausalLM(hf_config).eval()
    native = build_model(config_from_hf(hf_config, family="gemma4")).eval()
    load_hf_state(native, hf.state_dict(), family="gemma4")
    tokens = torch.randint(1, 128, (2, 11))
    with torch.no_grad():
        expected = hf(tokens, use_cache=False).logits
        torch.testing.assert_close(
            native(tokens), expected, atol=1e-5, rtol=1e-5
        )
        for cache_format in ("list", "windowed"):
            cache, chunks = None, []
            for start in range(0, 11, 3):
                logits, cache = native.forward_chunk(
                    tokens[:, start : start + 3],
                    start,
                    cache,
                    cache_format=cache_format,
                )
                chunks.append(logits)
            torch.testing.assert_close(
                torch.cat(chunks, 1), expected, atol=1e-5, rtol=1e-5
            )


def test_grouped_cache_roundtrip_and_subset():
    cfg = replace(mixed_config(), processor_layer_indices=[0, 1, 3])
    model = build_model(cfg)
    model.init_weights()
    _, cache = model.backbone.forward_chunk(
        torch.randint(1, 128, (2, 7)), 0, cache_format="windowed"
    )
    assert isinstance(cache, GroupedWindowedCache)
    assert cache.layer_groups == ((1, 2), (0,))
    assert set(cache.attention_states) == {2}
    tensors, layout = _flatten_windowed_cache(cache)
    restored = _unflatten_windowed_cache(tensors, layout)
    restored = GroupedWindowedCache.from_cache_state(restored.to_cache_state())
    for i in range(cache.num_layers):
        for actual, expected in zip(
            restored.layer(i), cache.layer(i), strict=True
        ):
            torch.testing.assert_close(actual, expected)
    assert sum(group.keys.numel() for group in cache.groups) == sum(
        k.numel() for k in cache.to_cache_state().keys
    )
    detached = restored.detach()
    assert not any(
        t.requires_grad for t in _flatten_windowed_cache(detached)[0]
    )


@pytest.mark.parametrize("encoder", ["linear", "query_attention"])
@pytest.mark.parametrize("per_head", [False, True])
@pytest.mark.parametrize("stream", ["kv", "v"])
def test_mixed_updates_and_gradients(encoder, per_head, stream):
    torch.manual_seed(32)
    cfg = mixed_config(
        encoder_method=encoder,
        per_cache_head_projections=per_head,
        update_stream=stream,
        out_proj_init="normal",
    )
    model = build_model(cfg)
    model.init_weights()
    model.backbone.requires_grad_(False)
    tokens = torch.randint(1, 128, (2, 11))
    loss = model(tokens[:, :-1], tokens[:, 1:])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.backbone.parameters())
    for projection in model.cache_processor.block.projections.values():
        assert projection.in_proj.weight.grad.count_nonzero()
        assert projection.out_proj.weight.grad.count_nonzero()
        assert torch.isfinite(projection.out_proj.weight.grad).all()
    with torch.no_grad():
        _, cache = model.backbone.forward_chunk(tokens[:, :7], 0)
        before = deepcopy(cache)
        updated = model.cache_processor(cache)
    for i in range(4):
        old_k, old_v = before.layer(i)
        key, value = updated.layer(i)
        torch.testing.assert_close(
            value[:, :, :-1], old_v[:, :, :-1], atol=0, rtol=0
        )
        assert not torch.equal(value[:, :, -1], old_v[:, :, -1])
        if stream == "v":
            torch.testing.assert_close(key, old_k, atol=0, rtol=0)
        else:
            torch.testing.assert_close(
                key[:, :, :-1], old_k[:, :, :-1], atol=0, rtol=0
            )


@pytest.mark.parametrize("stream", ["kv", "v"])
@pytest.mark.parametrize("checkpoint_windows", [0, 2])
def test_mixed_bf16_forward_backward(stream, checkpoint_windows):
    torch.manual_seed(36)
    model = build_model(
        mixed_config(out_proj_init="normal", update_stream=stream)
    ).to(dtype=torch.bfloat16)
    model.init_weights()
    model.backbone.requires_grad_(False)
    tokens = torch.randint(1, 128, (2, 13))
    loss = model(
        tokens[:, :-1],
        tokens[:, 1:],
        activation_checkpoint_windows=checkpoint_windows,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.backbone.parameters())
    for projection in model.cache_processor.block.projections.values():
        for weight in (projection.in_proj.weight, projection.out_proj.weight):
            assert weight.dtype == torch.bfloat16
            assert weight.grad is not None
            assert torch.isfinite(weight.grad).all()
            assert weight.grad.count_nonzero()


@pytest.mark.parametrize("score", ["kv_norm", "attn_l2"])
@pytest.mark.parametrize("subset", [None, [1, 3]])
@pytest.mark.parametrize(
    "routing,select_all",
    [
        ("random", False),
        ("learned_contextual", False),
        ("magnitude", True),
        ("random", True),
        ("learned_contextual", True),
    ],
)
def test_unused_head_magnitude_is_skipped(
    monkeypatch, score, subset, routing, select_all
):
    import gcw.models.cache_processors.longshot as longshot

    cfg = replace(
        mixed_config(
            score=score,
            head_selection_method=routing,
            out_proj_init="normal",
        ),
        processor_layer_indices=subset,
    )
    processor = longshot.GlobalCacheProcessor(cfg).eval()
    cache = CacheState(
        [
            torch.randn(2, heads, 8, dim)
            for heads, dim in cfg.stored_cache_shapes
        ],
        [
            torch.randn(2, heads, 8, dim)
            for heads, dim in cfg.stored_cache_shapes
        ],
        torch.rand(2, 8),
    )
    torch.manual_seed(37)
    with torch.no_grad():
        expected = processor(deepcopy(cache), select_all=select_all)

    def unused(*args, **kwargs):
        raise AssertionError("unused head magnitudes were computed")

    monkeypatch.setattr(longshot, "chunked_fp32_l2_norm", unused)
    monkeypatch.setattr(longshot, "gather_per_cache_head_scores", unused)
    monkeypatch.setattr(processor, "_select_combined_heads", unused)
    torch.manual_seed(37)
    with torch.no_grad():
        actual = processor(deepcopy(cache), select_all=select_all)
    for index in range(cache.num_layers):
        for first, second in zip(
            actual.layer(index), expected.layer(index), strict=True
        ):
            torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("truncation", [0, 2])
def test_mixed_checkpoint_loss_and_gradient_parity(truncation):
    torch.manual_seed(33)
    cfg = mixed_config(out_proj_init="normal")
    TrainConfig(model=cfg, activation_checkpoint_windows=2, sequence_len=12)
    eager = build_model(cfg)
    eager.init_weights()
    eager.backbone.requires_grad_(False)
    checkpointed = deepcopy(eager)
    tokens = torch.randint(1, 128, (2, 13))
    losses = []
    for model, windows in ((eager, 0), (checkpointed, 2)):
        loss = model(
            tokens[:, :-1],
            tokens[:, 1:],
            activation_checkpoint_windows=windows,
            truncated_bptt_windows=truncation,
        )
        loss.backward()
        losses.append(loss)
    torch.testing.assert_close(*losses)
    for (name, first), (_, second) in zip(
        eager.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        if first.grad is not None:
            torch.testing.assert_close(
                first.grad, second.grad, atol=1e-6, rtol=1e-4, msg=name
            )


@pytest.mark.parametrize(
    "routing", ["magnitude", "random", "learned_contextual"]
)
@pytest.mark.parametrize("iteration_mode", ["internal", "external"])
def test_mixed_routing_generation_and_metrics(routing, iteration_mode):
    model = build_model(
        mixed_config(
            head_selection_method=routing,
            iterations=2,
            iteration_mode=iteration_mode,
            out_proj_init="zero",
        )
    ).eval()
    model.init_weights()
    tokens = torch.randint(1, 128, (2, 5))
    metrics = UpdateBatch(2, "cpu")
    token = UPDATE_BATCH.set(metrics)
    try:
        with torch.no_grad():
            expected = model.backbone.generate(tokens, max_new_tokens=3)
            torch.manual_seed(42)
            regular = model.generate(tokens, max_new_tokens=3)
            torch.manual_seed(42)
            fast = model.generate_fast(tokens, max_new_tokens=3)
            torch.testing.assert_close(fast, regular)
            torch.testing.assert_close(fast, expected)
            stopped = model.generate_fast(
                tokens, max_new_tokens=3, eos_token_id=fast[:, 5].tolist()
            )
            assert stopped.shape[1] == 6
    finally:
        UPDATE_BATCH.reset(token)
    assert metrics.totals


@pytest.mark.parametrize("score", ["kv_norm", "attn_l2"])
def test_width_normalized_magnitude(monkeypatch, score):
    model = build_model(mixed_config(score=score))
    processor = model.cache_processor
    keys = [
        torch.ones(1, heads, 5, dim)
        for heads, dim in model.config.stored_cache_shapes
    ]
    cache = CacheState(keys, [k.clone() for k in keys], torch.ones(1, 5))
    observed = []
    original = processor._select_combined_heads

    def capture(**kwargs):
        observed.append(kwargs["kv_head_scores"].clone())
        return original(**kwargs)

    monkeypatch.setattr(processor, "_select_combined_heads", capture)
    processor(deepcopy(cache))
    torch.testing.assert_close(
        observed[0], observed[0][:, :1].expand_as(observed[0])
    )
    processor.config.head_score_normalization = "none"
    processor(deepcopy(cache))
    assert observed[1][0, 2] > observed[1][0, 0]


def test_mixed_checkpoint_roundtrip(tmp_path):
    cfg = mixed_config(out_proj_init="normal")
    model = build_model(cfg).eval()
    model.init_weights()
    save_checkpoint(
        tmp_path / "backbone", model.backbone, model.backbone.config
    )
    save_checkpoint(
        tmp_path / "processor",
        model,
        cfg,
        {"backbone_checkpoint": str(tmp_path / "backbone")},
        cache_processor_only=True,
    )
    restored, _, _ = load_checkpoint(tmp_path / "processor")
    load_backbone_checkpoint(restored, tmp_path / "backbone")
    restored.eval()
    tokens = torch.randint(1, 128, (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(
            restored.generate_fast(tokens, 3), model.generate_fast(tokens, 3)
        )


def test_mixed_admission_and_legacy_defaults():
    from gcw.performance.config import LongshotFullHistoryConfig
    from gcw.performance.longshot_cache import (
        configure_longshot_full_history_execution,
    )

    assert "head_score_normalization" not in to_dict(CacheProcessorConfig())
    cfg = mixed_config()
    assert resolve_model(cfg).report.compatible
    for variant in ("bt", "bgt", "icarus"):
        with pytest.raises(ValueError, match="uniform stored KV"):
            build_model(
                replace(
                    cfg,
                    variant=variant,
                    cache_processor=replace(
                        cfg.cache_processor, head_score_normalization="none"
                    ),
                )
            )
    with pytest.raises(ValueError, match="unrope_keys"):
        build_model(mixed_config(unrope_keys=True))
    with pytest.raises(ValueError, match="reference cache movement"):
        TrainConfig(
            model=cfg, performance={"triton_cache_movement": {"enabled": True}}
        )
    with pytest.raises(ValueError, match="uniform cache shapes"):
        configure_longshot_full_history_execution(
            build_model(cfg), LongshotFullHistoryConfig(enabled=True)
        )


@pytest.mark.parametrize(
    "gate", ["static_scalar", "contextual_scalar", "contextual_vector"]
)
@pytest.mark.parametrize("encoder", ["linear", "query_attention"])
@pytest.mark.parametrize("per_head", [False, True])
def test_mixed_parameter_accounting(gate, encoder, per_head):
    from gcw.params import processor_params

    cfg = mixed_config(
        update_gate_type=gate,
        encoder_method=encoder,
        per_cache_head_projections=per_head,
    )
    model = build_model(cfg)
    assert processor_params(cfg) == sum(
        p.numel() for p in model.cache_processor.parameters()
    )


@pytest.mark.parametrize("refinement", ["flow", "diffusion", "scheduled"])
def test_mixed_refinement_and_update_observations(refinement):
    from gcw.params import processor_params

    cfg = mixed_config(out_proj_init="zero", update_stream="v")
    if refinement == "flow":
        cfg = replace(cfg, flow_processor=FlowProcessorConfig(num_steps=2))
    else:
        cfg = replace(
            cfg,
            diffusion_processor=DiffusionProcessorConfig(
                num_steps=2,
                noise_scale=0.1 if refinement == "scheduled" else 0,
                training_method=(
                    "scheduled" if refinement == "scheduled" else "fixed"
                ),
            ),
        )
    model = build_model(cfg).eval()
    model.init_weights()
    assert processor_params(cfg) == sum(
        p.numel() for p in model.cache_processor.parameters()
    )
    metrics = UpdateBatch(1, "cpu")
    token = UPDATE_BATCH.set(metrics)
    try:
        with torch.no_grad():
            output = model.generate_fast(torch.tensor([[1, 2, 3, 4, 5]]), 3)
    finally:
        UPDATE_BATCH.reset(token)
    assert output.shape == (1, 8)
    assert metrics.totals


@pytest.mark.parametrize("per_head", [False, True])
@pytest.mark.parametrize(
    "encoder,query,stream,score,expected",
    [
        ("linear", "learned", "kv", "kv_norm", 28596),
        ("query_attention", "learned", "kv", "kv_norm", 22452),
        ("query_attention", "final_token", "kv", "kv_norm", 25524),
        ("query_attention", "final_token", "v", "kv_norm", 24436),
        ("query_attention", "final_token", "kv", "attn_l2", 26500),
    ],
)
def test_mixed_flops_hand_count(
    encoder, query, stream, score, expected, per_head
):
    from gcw.flops import processor_forward_flops_per_sequence

    cfg = mixed_config(
        encoder_method=encoder,
        encoder_query_source=query,
        update_stream=stream,
        score=score,
        topk_heads=6,
        per_cache_head_projections=per_head,
    )
    # Two calls encode 3 then 5 positions for four width-8 and two width-16
    # KV streams. Each shared layer costs 6,912 projection/MLP FLOPs and
    # 1,152 attention FLOPs. KV adapters cost 2,048 (query) or 6,144 (linear),
    # plus 66 gate operations. Query pooling costs 256 per position, with
    # another 1,536 per call for final-token queries. Value-only decoding
    # saves 512 projection and 32 gate operations per call. Head scoring
    # costs 6 (KV sum) or 128 (L2) per position, under the estimator's convention.
    assert processor_forward_flops_per_sequence(cfg, 7) == expected


def test_31b_processor_shapes_and_flops():
    from gcw.flops import processor_forward_flops_per_sequence
    from gcw.models.cache_processors.longshot import GlobalCacheProcessor
    from gcw.params import processor_params

    base = replace(
        mixed_config(
            process_every=64,
            topk=32,
            hidden_size=1920,
            intermediate_size=7680,
            num_layers=7,
            num_heads=15,
            num_kv_heads=15,
            encoder_query_source="final_token",
        ),
        hidden_size=5376,
        intermediate_size=21504,
        num_layers=60,
        num_heads=32,
        num_kv_heads=16,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=4,
        sequence_len=262144,
        sliding_window=1024,
        layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 10,
    )
    for layers, heads, expected_count in (
        (list(range(5, 60, 6)), 40, 417931649),
        (list(range(60)), 840, 420163330),
    ):
        cfg = replace(base, processor_layer_indices=layers)
        assert resolve_model(cfg).report.compatible
        assert cfg.cache_num_heads == heads
        with torch.device("meta"):
            processor = GlobalCacheProcessor(cfg)
        count = sum(p.numel() for p in processor.parameters())
        assert count == processor_params(cfg) == expected_count
        assert processor_forward_flops_per_sequence(cfg, 4096) > 0
    cfg = mixed_config(score="attn_l2")
    smaller = replace(cfg, global_head_dim=4)
    assert processor_forward_flops_per_sequence(
        cfg, 12
    ) > processor_forward_flops_per_sequence(smaller, 12)
