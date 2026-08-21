import types
from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from gcw.architecture import _config_payload
from gcw.config import (
    CacheProcessorConfig,
    FlowProcessorConfig,
    LoRAConfig,
    ModelConfig,
    gemma4_default_layer_types,
)
from gcw.models import build_model
from gcw.models.backbones import gemma4 as gemma4_module
from gcw.models.backbones.gemma4 import _gemma4_attention
from gcw.models.cache_processors import icarus as icarus_module
from gcw.models.cache_processors import longshot as longshot_module
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.ops import (
    expand_cache_head_indices,
    gather_per_cache_head_scores,
    gather_per_cache_head_kv,
    rotate_cache_keys,
    select_heads_from_per_cache_head_scores,
    select_per_cache_head_kv_norm_indices_from_scores,
)
from gcw.models.cache_processors.standard import (
    CacheProcessor,
    _merge_kv,
    _split_kv,
)
from gcw.models.model import LanguageModel
from gcw.models.cache_processors.icarus import (
    CacheProcessor as IcarusCacheProcessor,
)
from gcw.models.cache_processors.icarus import (
    GlobalCacheProcessor as IcarusGlobalCacheProcessor,
)
from gcw.models.cache_processors.icarus import (
    _gather_cache_heads,
    _select_heads_from_merged,
)
from gcw.models.cache_processors.icarus import (
    ProcessorAttention as IcarusProcessorAttention,
)
from gcw.models.cache_processors.longshot import (
    GlobalCacheProcessor as LongshotGlobalCacheProcessor,
)
from gcw.models.cache_processors.longshot import (
    CacheProcessor as LongshotCacheProcessor,
)
from gcw.models.cache_processors.longshot import (
    ProcessorAttention as LongshotProcessorAttention,
)
from gcw.models.cache_processors.longshot import QueryAttentionTimeEncoder
from gcw.models.cache_processors.longshot_layers import (
    _gather_cache_heads as _longshot_gather_cache_heads,
)
from gcw.models.lora import LoRALinear
from gcw.models.backbones.qwen3 import Qwen35GatedDeltaDynamics
from gcw.models.backbones.transformer import (
    _matches_stop_sequence,
    apply_rotary,
)
from gcw.models.cache_processors.windowed import WindowedCache


def test_icarus_and_longshot_share_exact_tensor_operations_without_changing_symbols():
    for name in (
        "_scatter_latest",
        "_scatter_latest_values",
        "_split_merged",
        "_split_merged_values",
    ):
        assert getattr(icarus_module, name) is getattr(longshot_module, name)


def test_processed_models_share_one_explicit_composition_wrapper():
    for variant in ("bt", "bgt", "icarus", "longshot"):
        model = build_model(tiny_config(variant))
        assert isinstance(model, LanguageModel)
        keys = tuple(model.state_dict())
        assert any(key.startswith("backbone.") for key in keys)
        assert any(key.startswith("cache_processor.") for key in keys)


def test_transformer_loss_reduction_none_returns_per_token_cross_entropy():
    torch.manual_seed(123)
    model = build_model(tiny_config("tf"))
    model.init_weights()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 7))
    targets = input_ids.roll(shifts=-1, dims=1)
    targets[:, -1] = -1

    per_token = model(input_ids, targets, loss_reduction="none")
    logits = model(input_ids)
    expected_targets = targets.reshape(-1).clone()
    expected_targets[expected_targets < 0] = -100
    expected = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        expected_targets,
        ignore_index=-100,
        reduction="none",
    ).reshape_as(targets)

    assert per_token.shape == targets.shape
    torch.testing.assert_close(per_token, expected)


def test_cache_window_loss_reduction_none_returns_per_token_cross_entropy():
    torch.manual_seed(124)
    model = build_model(tiny_config("bgt"))
    model.init_weights()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 9))
    targets = input_ids.roll(shifts=-1, dims=1)
    targets[:, -1] = -1

    per_token = model(input_ids, targets, loss_reduction="none")
    logits = model(input_ids)
    expected_targets = targets.reshape(-1).clone()
    expected_targets[expected_targets < 0] = -100
    expected = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        expected_targets,
        ignore_index=-100,
        reduction="none",
    ).reshape_as(targets)

    assert per_token.shape == targets.shape
    torch.testing.assert_close(per_token, expected)


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


def tiny_qwen35_moe_config(variant="tf", **processor_overrides):
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
        family="qwen3_5_moe",
        vocab_size=128,
        hidden_size=16,
        intermediate_size=8,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=32,
        tie_embeddings=False,
        qk_norm=True,
        rotary_style="llama",
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        processor_layer_indices=[1, 3],
        partial_rotary_factor=0.5,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=8,
        cache_processor=cp,
    )


def tiny_qwen35_config(variant="tf", **processor_overrides):
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
        family="qwen3_5",
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=32,
        tie_embeddings=False,
        qk_norm=True,
        rotary_style="llama",
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        processor_layer_indices=[1, 3],
        partial_rotary_factor=0.5,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        cache_processor=cp,
    )


def tiny_gemma4_config(
    variant="tf", *, shared=False, global_head_dim=8, **processor_overrides
):
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
        family="gemma4",
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=32,
        tie_embeddings=True,
        rotary_style="llama",
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        sliding_window=4,
        global_head_dim=global_head_dim,
        num_kv_shared_layers=2 if shared else 0,
        hidden_size_per_layer_input=4,
        vocab_size_per_layer_input=128,
        final_logit_softcapping=30.0,
        hidden_act="gelu_pytorch_tanh",
        cache_processor=cp,
    )


def test_tiny_models_forward_backward():
    for variant in ("tf", "bt", "bgt", "icarus", "longshot"):
        model = build_model(tiny_config(variant))
        model.init_weights()
        x = torch.randint(0, 128, (2, 9))
        y = torch.randint(0, 128, (2, 9))
        loss = model(x, y)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )


def test_tiny_qwen35_forward_backward_and_cache_layers():
    model = build_model(tiny_qwen35_config())
    model.init_weights()
    x = torch.randint(0, 128, (2, 9))
    loss = model(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.config.cache_layer_indices == [1, 3]
    assert model.config.cache_num_heads == 2
    assert any(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )


def test_tiny_gemma4_forward_backward_and_cache_layers():
    model = build_model(tiny_gemma4_config())
    model.init_weights()
    x = torch.randint(0, 128, (2, 9))
    loss = model(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.config.cache_layer_indices == [0, 1, 2, 3]
    assert model.config.cache_num_heads == 4
    assert any(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )


def test_gemma4_attention_uses_explicit_unit_scale():
    torch.manual_seed(13)
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)

    out = _gemma4_attention(
        q, k, v, None, False, groups=1, scale=1.0, try_flash=True
    )
    expected = torch.matmul(
        torch.softmax(torch.matmul(q, k.transpose(2, 3)), dim=-1), v
    )
    default_scaled = F.scaled_dot_product_attention(q, k, v)

    assert torch.allclose(out, expected.transpose(1, 2), atol=1e-6, rtol=1e-6)
    assert not torch.allclose(
        out, default_scaled.transpose(1, 2), atol=1e-6, rtol=1e-6
    )


def test_gemma4_full_attention_uses_causal_bias_without_dense_mask(monkeypatch):
    calls = []

    def fake_flash_sdpa(
        q,
        k,
        v,
        *,
        attn_mask=None,
        is_causal=False,
        enable_gqa=False,
        scale=None,
        try_flash=True,
        **kwargs,
    ):
        calls.append(
            {
                "attn_mask": attn_mask,
                "is_causal": is_causal,
                "enable_gqa": enable_gqa,
                "scale": scale,
                "try_flash": try_flash,
            }
        )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
            scale=scale,
        )

    monkeypatch.setattr(gemma4_module, "flash_sdpa", fake_flash_sdpa)
    model = build_model(
        ModelConfig(
            family="gemma4",
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            sequence_len=32,
            tie_embeddings=True,
            rotary_style="llama",
            layer_types=["full_attention"],
            global_head_dim=8,
            hidden_size_per_layer_input=0,
        )
    ).eval()
    model.init_weights()

    with torch.no_grad():
        model(torch.randint(0, 128, (1, 5)))

    assert len(calls) == 1
    assert calls[0]["attn_mask"] is None
    assert calls[0]["is_causal"] is True
    assert calls[0]["try_flash"] is True
    assert calls[0]["scale"] == 1.0


def test_gemma4_sliding_attention_skips_forced_flash_for_dense_mask(
    monkeypatch,
):
    calls = []

    def fake_flash_sdpa(
        q,
        k,
        v,
        *,
        attn_mask=None,
        is_causal=False,
        enable_gqa=False,
        scale=None,
        try_flash=True,
        **kwargs,
    ):
        calls.append(
            {
                "attn_mask": attn_mask,
                "is_causal": is_causal,
                "try_flash": try_flash,
            }
        )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
            scale=scale,
        )

    monkeypatch.setattr(gemma4_module, "flash_sdpa", fake_flash_sdpa)
    model = build_model(
        ModelConfig(
            family="gemma4",
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            sequence_len=32,
            tie_embeddings=True,
            rotary_style="llama",
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=4,
            hidden_size_per_layer_input=0,
        )
    ).eval()
    model.init_weights()

    with torch.no_grad():
        model(torch.randint(0, 128, (1, 6)))

    assert len(calls) == 2
    assert calls[0]["attn_mask"] is not None
    assert calls[0]["is_causal"] is False
    assert calls[0]["try_flash"] is False


def test_gemma4_default_layer_types_uses_shared_pattern():
    cfg = ModelConfig(
        family="gemma4",
        num_layers=7,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
    )

    assert cfg.layer_types == gemma4_default_layer_types(7)
    assert cfg.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "full_attention",
    ]


def test_gemma4_global_kv_heads_only_apply_to_alternative_attention():
    cfg = tiny_gemma4_config(global_head_dim=16)
    cfg.num_global_key_value_heads = 2
    cfg.attention_k_eq_v = False

    assert cfg.kv_heads_for_layer(1) == cfg.num_kv_heads
    assert cfg.head_dim_for_layer(1) == 16

    cfg.attention_k_eq_v = True
    assert cfg.kv_heads_for_layer(1) == 2


def test_tiny_gemma4_static_chunk_matches_dynamic_chunk():
    torch.manual_seed(11)
    model = build_model(tiny_gemma4_config()).eval()
    model.init_weights()
    x = torch.randint(1, 128, (2, 5))
    cache = StaticCacheState.allocate(
        batch_size=x.shape[0],
        num_layers=model.config.cache_num_layers,
        num_kv_heads=model.config.num_kv_heads,
        max_seq_len=8,
        head_dim=int(model.config.head_dim),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    with torch.no_grad():
        dynamic_logits, _ = model.forward_chunk(x, 0)
        static_logits, _ = model.forward_chunk_static(x, 0, cache)

    assert torch.allclose(static_logits, dynamic_logits, atol=1e-5, rtol=1e-5)


def test_tiny_gemma4_shared_kv_tracks_owner_layers_only():
    torch.manual_seed(12)
    model = build_model(tiny_gemma4_config(shared=True)).eval()
    model.init_weights()
    x = torch.randint(0, 128, (2, 7))

    with torch.no_grad():
        logits, cache = model.forward_chunk(x, 0)

    assert logits.shape == (2, 7, 128)
    assert model.config.cache_layer_indices == [0, 1]
    assert cache.num_layers == 2
    assert len(cache.keys) == 2


def test_gemma4_processors_require_uniform_stored_cache_shapes():
    cfg = tiny_gemma4_config("bgt", global_head_dim=16)
    with pytest.raises(ValueError, match="uniform stored KV shapes"):
        build_model(cfg)


def test_gemma4_processors_build_for_uniform_cache_shapes():
    for variant in ("bt", "bgt", "icarus", "longshot"):
        model = build_model(tiny_gemma4_config(variant, encoder_seq_len=6))
        assert model.config.cache_num_layers == 4


def test_gemma4_final_logit_softcapping_bounds_logits():
    model = build_model(tiny_gemma4_config()).eval()
    hidden = torch.full((1, 1, model.config.hidden_size), 100.0)
    with torch.no_grad():
        model.embed_tokens.weight.fill_(100.0)
        logits = model.logits_from_hidden(hidden)
    assert logits.abs().max() <= model.config.final_logit_softcapping


def test_tiny_qwen35_static_chunk_matches_dynamic_chunk():
    torch.manual_seed(7)
    model = build_model(tiny_qwen35_config()).eval()
    model.init_weights()
    x = torch.randint(1, 128, (2, 5))
    cache = StaticCacheState.allocate(
        batch_size=x.shape[0],
        num_layers=model.config.cache_num_layers,
        num_kv_heads=model.config.num_kv_heads,
        max_seq_len=8,
        head_dim=int(model.config.head_dim),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    with torch.no_grad():
        dynamic_logits, _ = model.forward_chunk(x, 0)
        static_logits, _ = model.forward_chunk_static(x, 0, cache)

    assert torch.allclose(static_logits, dynamic_logits, atol=1e-5, rtol=1e-5)


def test_qwen35_chunked_cache_tracks_full_attention_layers_only():
    torch.manual_seed(5)
    model = build_model(tiny_qwen35_config()).eval()
    model.init_weights()
    x = torch.randint(0, 128, (2, 9))
    with torch.no_grad():
        logits, cache = model.forward_windowed(x, process_every=4)
    assert logits.shape == (2, 9, 128)
    assert cache.num_layers == 2
    assert len(cache.keys) == 2
    assert cache.keys[0].shape == (2, 1, 9, 8)
    assert cache.linear_states is not None
    assert sorted(cache.linear_states) == [0, 2]


def test_qwen35_processors_use_full_attention_cache_heads_only():
    for variant in ("bt", "bgt", "icarus", "longshot"):
        model = build_model(tiny_qwen35_config(variant, encoder_seq_len=6))
        assert model.config.cache_num_layers == 2
        if variant == "bt":
            assert len(model.cache_processor.blocks) == 2
        else:
            assert model.cache_processor.num_cache_heads == 2


def test_qwen35_longshot_linear_state_heads_are_opt_in_and_mapped_to_following_kv_layers():
    model = build_model(
        tiny_qwen35_config(
            "longshot",
            encoder_method="query_attention",
            encoder_query_source="final_token",
            linear_state_cache=True,
        )
    )

    assert model.cache_processor.num_kv_cache_heads == 2
    assert model.cache_processor.num_linear_state_heads == 4
    assert model.cache_processor.num_cache_heads == 6
    assert [
        (spec.layer_idx, spec.following_cache_idx, spec.head_count)
        for spec in model.cache_processor.linear_state_specs
    ] == [
        (0, 0, 2),
        (2, 1, 2),
    ]
    query_projs = model.cache_processor.block.linear_state_adapter.query_projs
    assert list(query_projs) == ["global"]
    assert query_projs["global"].out_features == 4
    assert (
        query_projs["global"].in_features
        == model.config.cache_processor.hidden_size
    )

    shared_following = build_model(
        replace(
            tiny_qwen35_config(
                "longshot",
                encoder_method="query_attention",
                encoder_query_source="final_token",
                linear_state_cache=True,
                linear_state_query_mode="shared",
            ),
            layer_types=[
                "linear_attention",
                "linear_attention",
                "full_attention",
                "full_attention",
            ],
            processor_layer_indices=[2, 3],
        )
    )
    assert [
        (spec.layer_idx, spec.following_cache_idx, spec.head_count)
        for spec in shared_following.cache_processor.linear_state_specs
    ] == [
        (0, 0, 2),
        (1, 0, 2),
    ]
    assert list(
        shared_following.cache_processor.block.linear_state_adapter.query_projs
    ) == ["cache:0"]

    per_linear_state = build_model(
        tiny_qwen35_config(
            "longshot",
            encoder_method="query_attention",
            encoder_query_source="final_token",
            linear_state_cache=True,
            linear_state_query_mode="per_linear_state",
        )
    )
    per_linear_projs = (
        per_linear_state.cache_processor.block.linear_state_adapter.query_projs
    )
    assert list(per_linear_projs) == ["layer:0", "layer:2"]
    assert per_linear_projs["layer:0"].out_features == 8

    global_query = build_model(
        tiny_qwen35_config(
            "longshot",
            encoder_method="query_attention",
            encoder_query_source="final_token",
            linear_state_cache=True,
            linear_state_query_mode="global",
        )
    )
    assert list(
        global_query.cache_processor.block.linear_state_adapter.query_projs
    ) == ["global"]

    per_head = build_model(
        tiny_qwen35_config(
            "longshot",
            encoder_method="query_attention",
            encoder_query_source="final_token",
            linear_state_cache=True,
            per_cache_head_projections=True,
        )
    )
    assert (
        per_head.cache_processor.block.in_proj.weight.shape[0]
        == per_head.cache_processor.num_kv_cache_heads
    )
    assert (
        per_head.cache_processor.block.out_proj.weight.shape[0]
        == per_head.cache_processor.num_kv_cache_heads
    )
    assert (
        per_head.cache_processor.block.time_encoder.query_proj.weight.shape[0]
        == per_head.cache_processor.num_kv_cache_heads
    )


def test_qwen35_longshot_linear_state_head_selection_is_split_from_kv_heads():
    cfg = tiny_qwen35_config(
        "longshot",
        topk_linear_heads=2,
        encoder_method="query_attention",
        encoder_query_source="final_token",
        linear_state_cache=True,
    )
    model = build_model(
        replace(cfg, cache_processor=replace(cfg.cache_processor, topk_heads=1))
    )
    processor = model.cache_processor
    linear_inputs = longshot_module.LinearStateInputs(
        features=torch.tensor([[[0.0], [2.0], [5.0], [3.0]]]),
        specs=[],
    )

    selected = processor._select_combined_heads(
        batch=1,
        kv_head_scores=torch.tensor([[1.0, 10.0]]),
        linear_inputs=linear_inputs,
        select_all=False,
        device=torch.device("cpu"),
    )

    assert torch.equal(selected, torch.tensor([[1, 4, 5]]))

    no_linear_cfg = tiny_qwen35_config(
        "longshot",
        topk_linear_heads=0,
        encoder_method="query_attention",
        encoder_query_source="final_token",
        linear_state_cache=True,
    )
    no_linear = build_model(
        replace(
            no_linear_cfg,
            cache_processor=replace(
                no_linear_cfg.cache_processor, topk_heads=1
            ),
        )
    )
    selected = no_linear.cache_processor._select_combined_heads(
        batch=1,
        kv_head_scores=torch.tensor([[1.0, 10.0]]),
        linear_inputs=linear_inputs,
        select_all=False,
        device=torch.device("cpu"),
    )

    assert torch.equal(selected, torch.tensor([[1]]))


def test_qwen35_longshot_internal_reselection_uses_one_budget_across_kv_and_linear_heads():
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

    base = tiny_qwen35_config(
        "longshot",
        encoder_method="query_attention",
        encoder_query_source="final_token",
        linear_state_cache=True,
    )
    assert base.cache_processor is not None
    config = replace(
        base,
        cache_processor=replace(
            base.cache_processor,
            iterations=2,
            topk_heads=1,
            topk_linear_heads=2,
        ),
    )
    model = build_model(config)
    processor = model.cache_processor
    block = processor.block
    layer = RecordingLayer(torch.tensor([10.0, 1.0, 100.0, 90.0, 80.0, 70.0]))
    block.layers = nn.ModuleList([layer])
    merged = torch.randn(
        1, processor.num_kv_cache_heads, 3, 2 * processor.head_dim
    )
    kv_hidden = torch.randn(
        1, processor.num_kv_cache_heads, config.cache_processor.hidden_size
    )
    linear_features = torch.randn(1, processor.num_linear_state_heads, 8)

    block.forward_with_linear_state_features(
        merged,
        kv_hidden,
        linear_features,
        torch.tensor([[0, 2, 3]]),
    )

    assert len(layer.calls) == 2
    assert torch.equal(layer.calls[0], torch.tensor([[0, 2, 3]]))
    assert torch.equal(layer.calls[1], torch.tensor([[2, 3, 4]]))


@pytest.mark.parametrize(
    "linear_state_update_mode", ["outer_product", "gated_delta"]
)
def test_qwen35_longshot_linear_state_processor_updates_recurrent_state_only(
    linear_state_update_mode,
):
    torch.manual_seed(11)
    model = build_model(
        tiny_qwen35_config(
            "longshot",
            encoder_method="query_attention",
            encoder_query_source="final_token",
            linear_state_cache=True,
            linear_state_update_mode=linear_state_update_mode,
            gate_mode="direct",
            gate_init=1.0,
        )
    ).eval()
    model.init_weights()
    x = torch.randint(0, 128, (2, 5))
    with torch.no_grad():
        _, cache = model.backbone.forward_chunk(x, 0)
        before = {
            layer_idx: (conv.clone(), recurrent.clone())
            for layer_idx, (conv, recurrent) in cache.linear_states.items()
        }
        model.cache_processor(cache, select_all=True)

    assert cache.linear_states is not None
    for layer_idx, (old_conv, old_recurrent) in before.items():
        new_conv, new_recurrent = cache.linear_states[layer_idx]
        assert torch.equal(new_conv, old_conv)
        assert not torch.allclose(new_recurrent, old_recurrent)


@pytest.mark.parametrize(
    "linear_state_update_mode", ["outer_product", "gated_delta"]
)
def test_qwen35_longshot_linear_state_delta_gate_can_be_disabled(
    linear_state_update_mode,
):
    changed_by_gate_setting = {}
    for gate_linear_state_delta in (False, True):
        torch.manual_seed(13)
        model = build_model(
            tiny_qwen35_config(
                "longshot",
                encoder_method="query_attention",
                encoder_query_source="final_token",
                linear_state_cache=True,
                linear_state_update_mode=linear_state_update_mode,
                gate_mode="direct",
                gate_init=0.0,
                gate_linear_state_delta=gate_linear_state_delta,
            )
        ).eval()
        if linear_state_update_mode == "gated_delta":
            assert model.config.cache_processor.gate_linear_state_delta is False
        model.init_weights()
        x = torch.randint(0, 128, (2, 5))
        with torch.no_grad():
            _, cache = model.backbone.forward_chunk(x, 0)
            before = {
                layer_idx: recurrent.clone()
                for layer_idx, (_, recurrent) in cache.linear_states.items()
            }
            model.cache_processor(cache, select_all=True)

        changed_by_gate_setting[gate_linear_state_delta] = any(
            not torch.allclose(cache.linear_states[layer_idx][1], old_recurrent)
            for layer_idx, old_recurrent in before.items()
        )

    assert changed_by_gate_setting[False]
    if linear_state_update_mode == "outer_product":
        assert not changed_by_gate_setting[True]
    else:
        assert changed_by_gate_setting[True]


def test_qwen35_longshot_shares_backbone_gated_delta_dynamics():
    model = build_model(
        tiny_qwen35_config(
            "longshot",
            encoder_method="query_attention",
            encoder_query_source="final_token",
            linear_state_cache=True,
            linear_state_update_mode="gated_delta",
        )
    )

    for (
        layer_idx,
        dynamics,
    ) in model.cache_processor.linear_state_dynamics.items():
        layer = model.backbone.layers[layer_idx]
        assert dynamics is layer.linear_attn.dynamics


def test_qwen35_longshot_gated_delta_synthetic_shapes_and_layer_controls(
    monkeypatch,
):
    cfg = tiny_qwen35_config(
        "longshot",
        encoder_method="query_attention",
        encoder_query_source="final_token",
        linear_state_cache=True,
        linear_state_update_mode="gated_delta",
        gate_mode="direct",
        gate_init=1.0,
    )
    dynamics = Qwen35GatedDeltaDynamics(2)
    with torch.no_grad():
        dynamics.A_log.copy_(torch.tensor([0.0, 1.0]))
        dynamics.dt_bias.copy_(torch.tensor([0.5, -0.25]))
    processor = LongshotGlobalCacheProcessor(
        cfg, linear_state_dynamics={0: dynamics, 2: dynamics}
    )
    spec = processor.linear_state_specs[0]
    conv = torch.randn(1, 10, 2)
    recurrent = torch.zeros(1, 2, 4, 4)
    cache = CacheState(
        keys=[],
        values=[],
        scores=torch.zeros(1, 0),
        linear_states={spec.layer_idx: (conv, recurrent)},
    )
    update = torch.randn(1, spec.head_count, 10)
    records = {}

    def fake_gated_delta_rule(query, key, value, *, g, beta, initial_state):
        records["query"] = query
        records["key"] = key
        records["value"] = value
        records["g"] = g
        records["beta"] = beta
        records["initial_state"] = initial_state
        final_state = (
            initial_state
            + torch.einsum("bshk,bshv->bhkv", key.float(), value.float())
            + g.mean().to(initial_state.dtype)
        )
        return value, final_state.to(initial_state.dtype)

    from gcw.models.backbones import qwen3 as qwen3_backbone_module

    monkeypatch.setattr(
        qwen3_backbone_module, "qwen35_gated_delta_rule", fake_gated_delta_rule
    )

    processor._scatter_linear_state_updates(cache, update, [spec])

    assert records["query"].shape == (1, 1, 2, 4)
    assert records["key"].shape == (1, 1, 2, 4)
    assert records["value"].shape == (1, 1, 2, 4)
    assert records["initial_state"] is recurrent
    assert torch.equal(cache.linear_states[spec.layer_idx][0], conv)
    assert not torch.allclose(cache.linear_states[spec.layer_idx][1], recurrent)

    cache.linear_states[spec.layer_idx][1].sum().backward()
    assert dynamics.A_log.grad is not None
    assert dynamics.dt_bias.grad is not None

    expected_beta = torch.sigmoid(update[..., 8]).unsqueeze(1)
    expected_g = (
        -dynamics.A_log.float().exp()
        * F.softplus(update[..., 9].float() + dynamics.dt_bias)
    ).unsqueeze(1)
    assert torch.allclose(records["beta"], expected_beta)
    assert torch.allclose(records["g"], expected_g)


@pytest.mark.parametrize(
    "linear_state_update_mode", ["outer_product", "gated_delta"]
)
def test_qwen35_longshot_linear_state_adapter_receives_gradients(
    linear_state_update_mode,
):
    torch.manual_seed(17)
    model = build_model(
        tiny_qwen35_config(
            "longshot",
            encoder_method="query_attention",
            encoder_query_source="final_token",
            linear_state_cache=True,
            linear_state_update_mode=linear_state_update_mode,
        )
    )
    model.init_weights()
    x = torch.randint(0, 128, (2, 9))

    loss = model(x, x)
    loss.backward()

    adapter_grads = [
        param.grad
        for name, param in model.cache_processor.named_parameters()
        if name.startswith("block.linear_state_adapter")
    ]
    assert adapter_grads
    assert any(
        grad is not None and torch.isfinite(grad).all()
        for grad in adapter_grads
    )


def test_qwen35_longshot_linear_state_processor_composes_with_flow_and_external_iterations():
    torch.manual_seed(19)
    base = tiny_qwen35_config(
        "longshot",
        encoder_method="query_attention",
        encoder_query_source="final_token",
        linear_state_cache=True,
    )
    assert base.cache_processor is not None
    config = replace(
        base,
        cache_processor=replace(
            base.cache_processor, iterations=2, iteration_mode="external"
        ),
        flow_processor=FlowProcessorConfig(num_steps=2),
    )
    model = build_model(config)
    model.init_weights()
    x = torch.randint(0, 128, (2, 9))

    loss = model(x, x)
    loss.backward()

    flow = model.cache_processor.block.flow
    assert flow is not None
    assert torch.isfinite(loss)
    assert (
        flow.flow_gate.grad is not None
        and torch.isfinite(flow.flow_gate.grad).all()
    )
    adapter_grads = [
        param.grad
        for name, param in model.cache_processor.named_parameters()
        if name.startswith("block.linear_state_adapter")
    ]
    assert adapter_grads
    assert all(
        grad is not None and torch.isfinite(grad).all()
        for grad in adapter_grads
    )


def test_tiny_qwen35_moe_forward_backward_and_cache_layers():
    model = build_model(tiny_qwen35_moe_config())
    model.init_weights()
    x = torch.randint(0, 128, (2, 9))
    loss = model(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.config.cache_layer_indices == [1, 3]
    assert model.config.cache_num_heads == 2
    assert any(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )


def test_tiny_qwen35_moe_linear_attention_reports_effective_backends():
    model = build_model(tiny_qwen35_moe_config())
    linear_layer = model.layers[0].linear_attn
    backends = linear_layer.effective_backends(torch.device("cpu"))

    assert backends["causal_conv"] == "torch-conv1d"
    assert backends["gated_delta"] == "transformers-torch"


def test_tiny_qwen35_moe_experts_use_transformers_grouped_mm_backend(
    monkeypatch,
):
    pytest.importorskip("transformers.integrations.moe")
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    calls = []

    def fake_grouped_mm(module, hidden_states, top_k_index, top_k_weights):
        calls.append(
            (
                module.config._experts_implementation,
                top_k_index.shape,
                top_k_weights.shape,
            )
        )
        return torch.zeros_like(hidden_states)

    monkeypatch.setitem(ALL_EXPERTS_FUNCTIONS, "grouped_mm", fake_grouped_mm)
    model = build_model(tiny_qwen35_moe_config())
    flat = torch.randn(3, model.config.hidden_size)
    selected = torch.tensor([[0, 1], [1, 2], [2, 3]])
    weights = torch.full((3, 2), 0.5)

    out = model.layers[1].mlp.experts(flat, selected, weights)

    assert out.shape == flat.shape
    assert calls == [("grouped_mm", selected.shape, weights.shape)]


def test_tiny_qwen35_moe_router_aux_loss_is_added_to_mean_training_loss():
    model = build_model(tiny_qwen35_moe_config())
    model.init_weights()
    x = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 10]])
    y = x.clone()
    y[:, -2:] = -100

    model.config.router_aux_loss_coef = 0.0
    ce_loss = model(x, y)
    model.config.router_aux_loss_coef = 0.1
    loss_with_aux = model(x, y)

    assert loss_with_aux > ce_loss


def test_qwen35_router_aux_mask_changes_real_load_balancing_loss():
    from gcw.models.backbones.qwen3 import qwen35_moe_load_balancing_loss

    router_logits = (
        torch.tensor(
            [
                [8.0, -8.0],
                [-8.0, 8.0],
                [8.0, -8.0],
                [8.0, -8.0],
            ]
        ),
    )
    attention_mask = torch.tensor([[True, True, False, False]])
    masked = qwen35_moe_load_balancing_loss(
        router_logits,
        num_experts=2,
        top_k=1,
        attention_mask=attention_mask,
    )
    unmasked = qwen35_moe_load_balancing_loss(
        router_logits,
        num_experts=2,
        top_k=1,
    )

    assert not torch.allclose(masked, unmasked)


def test_qwen35_bottleneck_windowed_loss_adds_masked_router_aux(monkeypatch):
    from gcw.models.backbones import qwen3 as qwen3_module

    config = tiny_qwen35_moe_config("bgt")
    config.router_aux_loss_coef = 0.1
    model = build_model(config)
    model.init_weights()
    x = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0], [5, 6, 7, 8, 9, 10, 11, 12]])
    y = x.clone()
    y[0, 4:] = -100
    attention_mask = y >= 0
    model.backbone.config.router_aux_loss_coef = 0.0
    ce_loss = model(x, y, attention_mask=attention_mask)
    captured = {}

    def fake_aux(router_logits, *, num_experts, top_k, attention_mask=None):
        captured["router_shapes"] = [
            tuple(logits.shape) for logits in router_logits
        ]
        captured["attention_mask"] = attention_mask
        return torch.ones((), device=router_logits[0].device)

    monkeypatch.setattr(
        qwen3_module, "qwen35_moe_load_balancing_loss", fake_aux
    )
    model.backbone.config.router_aux_loss_coef = 0.1
    loss_with_aux = model(x, y, attention_mask=attention_mask)

    assert torch.allclose(loss_with_aux, ce_loss + 0.1)
    assert captured["attention_mask"] is attention_mask
    assert (
        captured["router_shapes"]
        == [(x.numel(), config.num_experts)] * config.num_layers
    )


def test_tiny_qwen35_moe_router_aux_uses_explicit_attention_mask(monkeypatch):
    from gcw.models.backbones import qwen3 as qwen3_module

    cfg = ModelConfig(
        variant="tf",
        family="qwen3_5_moe",
        vocab_size=32,
        hidden_size=8,
        intermediate_size=4,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=8,
        tie_embeddings=False,
        qk_norm=True,
        rotary_style="llama",
        layer_types=["full_attention"],
        processor_layer_indices=[0],
        partial_rotary_factor=0.5,
        moe_intermediate_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=4,
        router_aux_loss_coef=0.1,
    )
    captured = {}

    def fake_aux(router_logits, *, num_experts, top_k, attention_mask=None):
        captured["attention_mask"] = attention_mask
        return torch.zeros((), device=router_logits[0].device)

    monkeypatch.setattr(
        qwen3_module, "qwen35_moe_load_balancing_loss", fake_aux
    )
    model = build_model(cfg)
    model.init_weights()
    x = torch.tensor([[0, 2, 3, 0]])
    y = torch.tensor([[-100, 2, 3, -100]])
    attention_mask = torch.tensor([[True, True, True, False]])

    model(x, y, attention_mask=attention_mask)

    assert captured["attention_mask"] is attention_mask


def test_tiny_qwen35_moe_static_chunk_matches_dynamic_chunk():
    torch.manual_seed(7)
    model = build_model(tiny_qwen35_moe_config()).eval()
    model.init_weights()
    x = torch.randint(1, 128, (2, 5))
    cache = StaticCacheState.allocate(
        batch_size=x.shape[0],
        num_layers=model.config.cache_num_layers,
        num_kv_heads=model.config.num_kv_heads,
        max_seq_len=8,
        head_dim=int(model.config.head_dim),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    with torch.no_grad():
        dynamic_logits, _ = model.forward_chunk(x, 0)
        static_logits, _ = model.forward_chunk_static(x, 0, cache)

    assert torch.allclose(static_logits, dynamic_logits, atol=1e-5, rtol=1e-5)


def test_qwen35_moe_chunked_cache_tracks_full_attention_layers_only():
    torch.manual_seed(5)
    model = build_model(tiny_qwen35_moe_config()).eval()
    model.init_weights()
    x = torch.randint(0, 128, (2, 9))
    with torch.no_grad():
        logits, cache = model.forward_windowed(x, process_every=4)
    assert logits.shape == (2, 9, 128)
    assert cache.num_layers == 2
    assert len(cache.keys) == 2
    assert cache.keys[0].shape == (2, 1, 9, 8)
    assert cache.linear_states is not None
    assert sorted(cache.linear_states) == [0, 2]


def test_qwen35_moe_processors_use_full_attention_cache_heads_only():
    for variant in ("bt", "bgt", "icarus", "longshot"):
        model = build_model(tiny_qwen35_moe_config(variant, encoder_seq_len=6))
        assert model.config.cache_num_layers == 2
        if variant == "bt":
            assert len(model.cache_processor.blocks) == 2
        else:
            assert model.cache_processor.num_cache_heads == 2


def test_tiny_icarus_uses_icarus_processor():
    model = build_model(tiny_config("icarus"))
    assert isinstance(model, LanguageModel)
    assert isinstance(model.cache_processor, IcarusGlobalCacheProcessor)
    x = torch.randn(2, 4, 6, 16)
    positions = torch.arange(6).view(1, 1, 6).expand(2, 4, -1)
    selected_heads = torch.tensor([[0, 2], [1, 3]])
    assert model.cache_processor.block(x, positions, selected_heads).shape == (
        2,
        4,
        6,
        16,
    )

    value_only = build_model(tiny_config("icarus", update_stream="v"))
    assert value_only.cache_processor.block.input_dim == 16
    assert value_only.cache_processor.block.output_dim == 8
    assert value_only.cache_processor.block(
        x, positions, selected_heads
    ).shape == (2, 4, 6, 8)

    model.init_weights()
    tokens = torch.randint(0, 128, (2, 9))
    loss = model(tokens, tokens)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.cache_processor.parameters())


def test_tiny_longshot_uses_longshot_processor_and_configured_encoder():
    linear = build_model(tiny_config("longshot", encoder_seq_len=6))
    assert isinstance(linear, LanguageModel)
    assert isinstance(linear.cache_processor, LongshotGlobalCacheProcessor)
    assert linear.cache_processor.block.encoder_method == "linear"
    x = torch.randn(2, 4, 6, 16)
    selected_heads = torch.tensor([[0, 2], [1, 3]])
    out = linear.cache_processor.block(x, selected_heads)
    assert out.shape == (2, 4, 16)

    value_only = build_model(
        tiny_config("longshot", encoder_seq_len=6, update_stream="v")
    )
    value_out = value_only.cache_processor.block(x, selected_heads)
    assert value_out.shape == (2, 4, 8)

    query = build_model(
        tiny_config("longshot", encoder_method="query_attention")
    )
    assert query.cache_processor.block.encoder_method == "query_attention"
    assert query.cache_processor.block.encoder_query_source == "learned"
    assert isinstance(
        query.cache_processor.block.time_encoder, QueryAttentionTimeEncoder
    )
    assert query.cache_processor.block.time_encoder.query.shape == (4, 16)
    query.init_weights()
    tokens = torch.randint(0, 128, (2, 9))
    loss = query(tokens, tokens)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in query.cache_processor.parameters())

    final_query = build_model(
        tiny_config(
            "longshot",
            encoder_method="query_attention",
            encoder_query_source="final_token",
        )
    )
    assert (
        final_query.cache_processor.block.encoder_query_source == "final_token"
    )
    assert isinstance(
        final_query.cache_processor.block.time_encoder,
        QueryAttentionTimeEncoder,
    )
    assert hasattr(final_query.cache_processor.block.time_encoder, "query_proj")
    assert not hasattr(final_query.cache_processor.block.time_encoder, "query")
    final_query.init_weights()
    loss = final_query(tokens, tokens)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        p.grad is not None for p in final_query.cache_processor.parameters()
    )


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

    build_processor(False).compute_update(
        x, positions, selected_heads
    ).sum().backward()
    assert calls == []

    build_processor(True).compute_update(
        x, positions, selected_heads
    ).sum().backward()
    assert calls == [{"use_reentrant": False}, {"use_reentrant": False}]


def test_longshot_iterative_checkpointing_includes_score_collection(
    monkeypatch,
):
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


def test_lora_starts_as_exact_backbone_and_trains_only_adapters():
    torch.manual_seed(41)
    base_config = tiny_config("tf")
    base = build_model(base_config)
    base.init_weights()
    adapted_config = replace(base_config, lora=LoRAConfig(rank=4, alpha=8.0))
    adapted = build_model(adapted_config)
    adapted.init_weights()
    adapted.load_state_dict(base.state_dict(), strict=False)
    x = torch.randint(0, 128, (2, 9))

    assert torch.equal(adapted(x), base(x))
    adapters = [
        module for module in adapted.modules() if isinstance(module, LoRALinear)
    ]
    assert adapters
    assert all(torch.count_nonzero(module.lora_b) == 0 for module in adapters)


def test_tiny_models_bf16_autocast_forward_backward():
    for variant in ("tf", "bt", "bgt", "icarus", "longshot"):
        model = build_model(tiny_config(variant))
        model.init_weights()
        x = torch.randint(0, 128, (2, 9))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = model(x, x)
        loss.backward()
        assert torch.isfinite(loss)


def test_frozen_backbone_leaves_processor_trainable():
    for variant in ("bt", "icarus", "longshot"):
        model = build_model(tiny_config(variant))
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        x = torch.randint(0, 128, (2, 9))
        loss = model(x, x)
        loss.backward()
        assert not any(p.grad is not None for p in model.backbone.parameters())
        assert any(
            p.grad is not None for p in model.cache_processor.parameters()
        )


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


def test_generate_default_keeps_absolute_decoding_cadence():
    model = build_model(tiny_config("tf")).eval()
    model.init_weights()
    calls = []

    def processor(cache, **kwargs):
        calls.append((cache.seq_len, bool(kwargs.get("select_all", False))))
        return cache

    x = torch.randint(0, 128, (1, 9))
    model.generate(x, max_new_tokens=5, processor=processor, process_every=4)

    assert calls == [(4, False), (8, False), (12, False)]


@pytest.mark.parametrize("method_name", ["generate", "generate_fast"])
def test_generation_process_every_overrides_only_decoding_cadence(method_name):
    model = build_model(tiny_config("bgt", generation_process_every=2)).eval()
    model.init_weights()
    calls = []

    def processor_forward(self, cache, *, select_all=False):
        calls.append((cache.seq_len, select_all))
        return cache

    model.cache_processor.forward = types.MethodType(
        processor_forward, model.cache_processor
    )
    x = torch.randint(0, 128, (1, 9))
    getattr(model, method_name)(x, max_new_tokens=5)

    assert model.cache_processor.config.process_every == 4
    assert calls == [
        (4, False),
        (8, False),
        (9, False),
        (11, False),
        (13, False),
    ]


def test_generation_process_every_preserves_relative_prefill_select_all():
    config = replace(
        tiny_config("bgt", generation_process_every=2),
        processor_generation_mode="post_prefill_relative",
    )
    model = build_model(config).eval()
    model.init_weights()
    calls = []

    def processor_forward(self, cache, *, select_all=False):
        calls.append((cache.seq_len, select_all))
        return cache

    model.cache_processor.forward = types.MethodType(
        processor_forward, model.cache_processor
    )
    x = torch.randint(0, 128, (1, 9))
    model.generate(x, max_new_tokens=5)

    assert calls == [(9, True), (11, False), (13, False)]


def test_gemma4_nonuniform_fast_generation_uses_generation_cadence():
    model = build_model(tiny_gemma4_config(global_head_dim=16)).eval()
    model.init_weights()
    calls = []

    def processor(cache, **kwargs):
        calls.append((cache.seq_len, bool(kwargs.get("select_all", False))))
        return cache

    x = torch.randint(1, 128, (1, 9))
    model.generate_fast(
        x,
        max_new_tokens=5,
        processor=processor,
        process_every=4,
        generation_process_every=2,
    )

    assert calls == [
        (4, False),
        (8, False),
        (9, False),
        (11, False),
        (13, False),
    ]


def test_generate_fast_matches_reference_generate():
    for variant in ("tf", "bt", "bgt", "icarus", "longshot"):
        torch.manual_seed(3)
        model = build_model(tiny_config(variant)).eval()
        model.init_weights()
        x = torch.randint(0, 128, (2, 9))
        with torch.no_grad():
            reference = model.generate(x, max_new_tokens=6)
            fast = model.generate_fast(x, max_new_tokens=6)
        assert torch.equal(fast, reference)


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
    _assert_unselected_unchanged(
        cache, actual.to_cache_state(), cache.select_indices(4, 2)
    )


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
    _assert_unselected_unchanged(
        cache, actual.to_cache_state(), cache.select_indices(4, 2)
    )


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

    all_indices = expand_cache_head_indices(
        cache.all_indices(), cache.num_layers * keys[0].shape[1]
    )
    scores = (
        gather_per_cache_head_kv(cache, all_indices).float().pow(2).sum(dim=-1)
    )
    indices = select_per_cache_head_kv_norm_indices_from_scores(
        scores, process_every=2, topk=1
    )

    assert indices.tolist() == [[[1, 4, 5], [2, 4, 5]]]


def test_score_based_cache_head_selection_matches_merged_kv_norms():
    cache = _random_cache(tiny_config("icarus"))
    all_indices = expand_cache_head_indices(
        cache.all_indices(), cache.num_layers * cache.keys[0].shape[1]
    )
    full_merged = gather_per_cache_head_kv(cache, all_indices)
    scores = full_merged.float().pow(2).sum(dim=-1)
    indices = select_per_cache_head_kv_norm_indices_from_scores(
        scores, process_every=4, topk=2
    )

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

        model.cache_processor.block.forward = types.MethodType(
            forward, model.cache_processor.block
        )
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

        model.cache_processor.block.forward = types.MethodType(
            forward, model.cache_processor.block
        )
        model.cache_processor(cache)

        expected = expand_cache_head_indices(
            cache.select_indices(4, 2), model.cache_processor.num_cache_heads
        )
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
    _assert_cache_positions_unchanged(
        cache, actual_state, _not_latest_mask(cache, selected)
    )


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
    model = build_model(
        tiny_config(
            "icarus", update_stream="v", gate_mode="direct", gate_init=1.0
        )
    )
    model.init_weights()
    cache = _random_cache(model.config)
    actual = _clone_cache(cache)
    indices = cache.select_indices(4, 2)

    model.cache_processor(actual)

    _assert_keys_unchanged(cache, actual)
    _assert_cache_positions_unchanged(
        cache, actual, _not_latest_mask(cache, indices)
    )
    value_delta = sum(
        (new_value - old_value).abs().sum()
        for old_value, new_value in zip(cache.values, actual.values)
    )
    assert value_delta > 0


def test_icarus_value_only_matches_windowed_cache_path():
    model = build_model(
        tiny_config(
            "icarus", update_stream="v", gate_mode="direct", gate_init=1.0
        )
    )
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
    model = build_model(
        tiny_config(
            "icarus", update_stream="v", gate_mode="direct", gate_init=1.0
        )
    )
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

    model.cache_processor.block.forward = types.MethodType(
        forward, model.cache_processor.block
    )
    model.cache_processor(cache, select_all=True)

    assert len(calls) == 1
    expected_indices = cache.all_indices()[:, None, :].expand(
        -1, model.cache_processor.num_cache_heads, -1
    )
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
        build_model(
            tiny_config("longshot", processor_mode="workspace_broadcast")
        )


def test_bottleneck_gradients_match_reference_loop():
    for variant in ("bt", "bgt"):
        torch.manual_seed(4)
        fast = build_model(tiny_config(variant))
        fast.init_weights()
        ref = build_model(tiny_config(variant))
        ref.load_state_dict(fast.state_dict())
        _patch_reference_processor(ref)
        ref.backbone.forward_windowed_loss = types.MethodType(
            _reference_windowed_loss, ref.backbone
        )

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
            assert torch.allclose(
                fast_param.grad, ref_param.grad, atol=2e-5, rtol=2e-4
            ), fast_name


def test_value_only_gradients_match_reference_loop():
    for variant in ("bt", "bgt"):
        torch.manual_seed(8)
        fast = build_model(tiny_config(variant, update_stream="v"))
        fast.init_weights()
        ref = build_model(tiny_config(variant, update_stream="v"))
        ref.load_state_dict(fast.state_dict())
        _patch_reference_processor(ref)
        ref.backbone.forward_windowed_loss = types.MethodType(
            _reference_windowed_loss, ref.backbone
        )

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
            assert torch.allclose(
                fast_param.grad, ref_param.grad, atol=2e-5, rtol=2e-4
            ), fast_name


def test_cache_processor_select_all_matches_reference_loop():
    for variant in ("bt", "bgt"):
        model = build_model(tiny_config(variant))
        model.init_weights()
        cache = _random_cache(model.config)
        expected = _clone_cache(cache)
        actual = _clone_cache(cache)

        if variant == "bt":
            _reference_per_layer_processor(
                model.cache_processor, expected, select_all=True
            )
        else:
            _reference_global_processor(
                model.cache_processor, expected, select_all=True
            )
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
    assert cp.generation_process_every is None
    assert "generation_process_every" not in _config_payload(cp)
    assert cp.topk_heads == 32
    assert cp.topk_linear_heads is None
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
    assert cp.linear_state_cache is False
    assert cp.linear_state_query_mode == "global"
    assert cp.gate_linear_state_delta is True
    assert cp.stream_factor == 2


def test_generation_process_every_is_in_nondefault_architecture_payload():
    payload = _config_payload(CacheProcessorConfig(generation_process_every=1))
    assert payload["generation_process_every"] == 1


def test_cache_processor_iterations_must_be_positive():
    with pytest.raises(ValueError, match="iterations"):
        CacheProcessorConfig(iterations=0)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_cache_processor_rejects_invalid_generation_process_every(value):
    with pytest.raises(ValueError, match="generation_process_every"):
        CacheProcessorConfig(generation_process_every=value)


def test_cache_processor_topk_heads_must_be_positive():
    with pytest.raises(ValueError, match="topk_heads"):
        CacheProcessorConfig(topk_heads=0)


def test_cache_processor_topk_linear_heads_must_be_non_negative():
    with pytest.raises(ValueError, match="topk_linear_heads"):
        CacheProcessorConfig(topk_linear_heads=-1)


def test_cache_processor_iteration_mode_must_be_known():
    with pytest.raises(ValueError, match="iteration_mode"):
        CacheProcessorConfig(iteration_mode="outer")


def test_cache_processor_internal_iteration_score_must_be_known():
    with pytest.raises(ValueError, match="internal_iteration_score"):
        CacheProcessorConfig(internal_iteration_score="attention")


def test_cache_processor_internal_iteration_score_accepts_kv_norm():
    cp = CacheProcessorConfig(internal_iteration_score="kv_norm")
    assert cp.internal_iteration_score == "kv_norm"


def test_longshot_linear_state_cache_validates_supported_combinations():
    with pytest.raises(ValueError, match="variant='longshot'"):
        ModelConfig(
            variant="bgt",
            family="qwen3_5",
            num_layers=1,
            layer_types=["linear_attention"],
            cache_processor=CacheProcessorConfig(linear_state_cache=True),
        )
    with pytest.raises(ValueError, match="encoder_method='query_attention'"):
        tiny_qwen35_config("longshot", linear_state_cache=True)
    with pytest.raises(ValueError, match="encoder_query_source='final_token'"):
        tiny_qwen35_config(
            "longshot",
            encoder_method="query_attention",
            linear_state_cache=True,
        )
    with pytest.raises(ValueError, match="update_stream='kv'"):
        tiny_qwen35_config(
            "longshot",
            encoder_method="query_attention",
            encoder_query_source="final_token",
            update_stream="v",
            linear_state_cache=True,
        )
    with pytest.raises(ValueError, match="no following full-attention"):
        build_model(
            ModelConfig(
                variant="longshot",
                family="qwen3_5",
                vocab_size=128,
                hidden_size=16,
                intermediate_size=32,
                num_layers=2,
                num_heads=2,
                num_kv_heads=1,
                head_dim=8,
                sequence_len=32,
                layer_types=["full_attention", "linear_attention"],
                processor_layer_indices=[0],
                linear_key_head_dim=4,
                linear_value_head_dim=4,
                linear_num_key_heads=1,
                linear_num_value_heads=2,
                cache_processor=CacheProcessorConfig(
                    hidden_size=8,
                    intermediate_size=16,
                    num_heads=1,
                    num_kv_heads=1,
                    encoder_method="query_attention",
                    encoder_query_source="final_token",
                    linear_state_cache=True,
                ),
            )
        )


@pytest.mark.parametrize("variant", ["icarus", "longshot"])
@pytest.mark.parametrize(
    ("head_dim", "partial_rotary_factor"),
    [(3, 1.0), (8, 0.01)],
)
def test_unrope_processor_requires_positive_even_partial_rotary_width(
    variant,
    head_dim: int,
    partial_rotary_factor: float,
):
    with pytest.raises(ValueError, match="positive even rotary width"):
        ModelConfig(
            variant=variant,
            family="qwen3_5",
            vocab_size=32,
            hidden_size=2 * head_dim,
            intermediate_size=4 * head_dim,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=head_dim,
            sequence_len=8,
            layer_types=["linear_attention", "full_attention"],
            processor_layer_indices=[1],
            partial_rotary_factor=partial_rotary_factor,
            cache_processor=CacheProcessorConfig(
                process_every=2,
                topk=1,
                topk_heads=1,
                hidden_size=4,
                intermediate_size=8,
                num_heads=1,
                num_kv_heads=1,
                unrope_keys=True,
            ),
        )


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

        if isinstance(default_attn, LongshotProcessorAttention):
            assert default_attn.include_self_head_kv is False
            assert enabled_attn.include_self_head_kv is True
        else:
            assert (
                default_attn.forward.__func__
                is default_attn.forward_selected_head_kv.__func__
            )
            assert (
                enabled_attn.forward.__func__
                is enabled_attn.forward_with_self_head_kv.__func__
            )


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

    assert default_attn.include_self_head_kv is False
    assert default_attn.internal_iteration_score == "cache_head_attention"
    assert enabled_attn.include_self_head_kv is True
    assert enabled_attn.internal_iteration_score == "cache_head_attention"


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

    assert default_attn.include_self_head_kv is False
    assert default_attn.internal_iteration_score == "kv_norm"
    assert enabled_attn.include_self_head_kv is True
    assert enabled_attn.internal_iteration_score == "kv_norm"


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
        CacheProcessorConfig(
            hidden_size=8, intermediate_size=16, num_heads=1, num_kv_heads=1
        ),
    )
    x = torch.randn(2, 5, 6)
    positions = torch.arange(5).unsqueeze(0).expand(2, -1)

    assert torch.allclose(
        block(x, positions), x + block.scaled_update(x, positions)
    )


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
            self.calls.append(
                (selected_heads.detach().clone(), return_attn_scores)
            )
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
    assert block.pop_iteration_metrics() == {
        "cache_processor/selected_head_overlap_after_iter_1": 0.0
    }


def test_longshot_external_iteration_repeats_forward_once_and_preserves_select_all():
    model_config = tiny_config(
        "longshot", iterations=3, iteration_mode="external"
    )
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

    model_config = tiny_config(
        "longshot", iterations=3, iteration_mode="external"
    )
    processor = LongshotGlobalCacheProcessor(model_config)
    layer = RecordingLayer()
    processor.block.layers = nn.ModuleList([layer])
    cache = _random_cache(model_config)

    processor(cache)

    assert layer.calls == 3


def test_longshot_external_iteration_logs_selected_head_overlap_between_calls():
    processor = LongshotGlobalCacheProcessor(
        tiny_config("longshot", iterations=3, iteration_mode="external")
    )
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
    processor = LongshotGlobalCacheProcessor(
        tiny_config("longshot", iterations=1, iteration_mode="external")
    )
    cache = _random_cache(tiny_config("longshot"))

    def fake_forward_once(self, cache, *, select_all=False):
        return cache, torch.tensor([[0, 1], [1, 2]])

    processor._forward_once = types.MethodType(fake_forward_once, processor)

    processor(cache)

    assert processor.block.pop_iteration_metrics() == {}


def test_zero_out_direct_gate_starts_as_identity():
    for variant in ("bt", "bgt"):
        model = build_model(
            tiny_config(
                variant, out_proj_init="zero", gate_mode="direct", gate_init=1.0
            )
        )
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


def test_rotate_cache_keys_preserves_partial_rotary_tail():
    keys = torch.randn(2, 3, 4, 8)
    indices = torch.tensor(
        [
            [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
            [[3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
        ]
    )

    rotated = rotate_cache_keys(
        keys, indices, theta=10000.0, rotary_style="llama", rotary_dim=4
    )
    restored = rotate_cache_keys(
        rotated,
        indices,
        theta=10000.0,
        rotary_style="llama",
        inverse=True,
        rotary_dim=4,
    )

    assert torch.equal(rotated[..., 4:], keys[..., 4:])
    assert torch.allclose(restored, keys, atol=1e-6)


def test_direct_gate_remains_in_gradient_graph_with_zero_out_init():
    for variant in ("bt", "bgt"):
        torch.manual_seed(9)
        model = build_model(
            tiny_config(
                variant, out_proj_init="zero", gate_mode="direct", gate_init=1.0
            )
        )
        x = torch.randint(0, 128, (2, 9))
        loss = model(x, x)
        loss.backward()

        gates = [
            param.grad
            for name, param in model.cache_processor.named_parameters()
            if name.endswith("gate")
        ]
        assert gates
        assert all(
            grad is not None and torch.isfinite(grad).all() for grad in gates
        )


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
    model = build_model(
        tiny_config(
            "longshot", update_stream="v", gate_mode="direct", gate_init=1.0
        )
    )
    model.init_weights()
    cache = _random_cache(model.config)
    actual = _clone_cache(cache)
    indices = cache.select_indices(4, 2)

    model.cache_processor(actual)

    _assert_keys_unchanged(cache, actual)
    _assert_cache_positions_unchanged(
        cache, actual, _not_latest_mask(cache, indices)
    )
    value_delta = sum(
        (new_value - old_value).abs().sum()
        for old_value, new_value in zip(cache.values, actual.values)
    )
    assert value_delta > 0


def test_longshot_value_only_matches_windowed_cache_path():
    model = build_model(
        tiny_config(
            "longshot", update_stream="v", gate_mode="direct", gate_init=1.0
        )
    )
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
    model = build_model(
        tiny_config(
            "longshot", update_stream="v", gate_mode="direct", gate_init=1.0
        )
    )
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
        assert any(
            p.grad is not None for p in model.cache_processor.parameters()
        )


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
    model = build_model(
        tiny_config("bgt", **_workspace_overrides(out_proj_init="zero"))
    )
    cache = _random_cache(model.config)
    actual = _clone_cache(cache)

    model.cache_processor(actual, select_all=True)

    _assert_cache_close(actual, cache)
    assert "cache_processor.salience_logits" in model.state_dict()


def test_workspace_broadcast_processor_matches_reference_loop():
    torch.manual_seed(11)
    model = build_model(
        tiny_config("bgt", **_workspace_overrides(out_proj_init="normal"))
    )
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
    model = build_model(
        tiny_config("bgt", **_workspace_overrides(out_proj_init="normal"))
    )
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
    model = build_model(
        tiny_config("bgt", **_workspace_overrides(out_proj_init="normal"))
    )
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
    model = build_model(
        tiny_config("bgt", **_head_mask_overrides(out_proj_init="zero"))
    )
    cache = _random_cache(model.config)
    actual = _clone_cache(cache)

    model.cache_processor(actual, select_all=True)

    _assert_cache_close(actual, cache)
    assert "cache_processor.salience_logits" in model.state_dict()


def test_head_mask_broadcast_processor_matches_reference_loop():
    torch.manual_seed(14)
    model = build_model(
        tiny_config("bgt", **_head_mask_overrides(out_proj_init="normal"))
    )
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
    model = build_model(
        tiny_config("bgt", **_head_mask_overrides(out_proj_init="normal"))
    )
    cache = _random_cache(model.config)
    expected = _clone_cache(cache)
    actual = WindowedCache.from_cache_state(_clone_cache(cache))

    model.cache_processor(expected)
    model.cache_processor(actual)

    actual_state = actual.to_cache_state()
    _assert_cache_close(actual_state, expected)


def test_head_mask_broadcast_salience_logits_receive_gradients():
    torch.manual_seed(16)
    model = build_model(
        tiny_config("bgt", **_head_mask_overrides(out_proj_init="normal"))
    )
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
        build_model(
            tiny_config("bgt", **_head_mask_overrides(update_stream="v"))
        )
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
        assert any(
            p.grad is not None for p in model.cache_processor.parameters()
        )
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
        assert any(
            p.grad is not None for p in model.cache_processor.parameters()
        )
        assert torch.cuda.max_memory_allocated() < 2 * 1024**3


def _random_cache(config: ModelConfig) -> CacheState:
    gen = torch.Generator().manual_seed(3)
    keys = [
        torch.randn(
            2, config.num_kv_heads, 7, int(config.head_dim), generator=gen
        )
        for _ in range(config.num_layers)
    ]
    values = [
        torch.randn(
            2, config.num_kv_heads, 7, int(config.head_dim), generator=gen
        )
        for _ in range(config.num_layers)
    ]
    scores = torch.tensor(
        [
            [0.1, 2.0, 0.3, 1.5, 0.4, 0.5, 0.6],
            [1.9, 0.2, 1.7, 0.4, 0.5, 0.6, 0.7],
        ]
    )
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
    return CacheState(
        [k.clone() for k in cache.keys],
        [v.clone() for v in cache.values],
        cache.scores.clone(),
    )


def _static_cache_from_state(cache: CacheState) -> StaticCacheState:
    return StaticCacheState(
        [k.clone() for k in cache.keys],
        [v.clone() for v in cache.values],
        cache.scores.clone(),
        cache.seq_len,
    )


def _cache_state_from_static(cache: StaticCacheState) -> CacheState:
    keys = [
        cache.layer_view(i, cache.length)[0].clone()
        for i in range(cache.num_layers)
    ]
    values = [
        cache.layer_view(i, cache.length)[1].clone()
        for i in range(cache.num_layers)
    ]
    return CacheState(keys, values, cache.scores.clone())


def _reference_per_layer_processor(
    processor, cache: CacheState, *, select_all: bool = False
) -> CacheState:
    indices = (
        cache.all_indices()
        if select_all
        else cache.select_indices(
            processor.config.process_every, processor.config.topk
        )
    )
    for layer_idx, block in enumerate(processor.blocks):
        keys, values = cache.gather_layer(layer_idx, indices)
        if processor.config.update_stream == "v":
            updated = block(_merge_values(values), indices)
            cache.values[layer_idx] = _scatter_values(
                cache.values[layer_idx],
                indices,
                _split_values(updated, values.shape[1], values.shape[3]),
            )
            continue
        updated = block(_merge_kv(keys, values), indices)
        new_keys, new_values = _split_kv(updated, keys.shape[1], keys.shape[3])
        cache.scatter_layer(layer_idx, indices, new_keys, new_values)
    return cache


def _reference_global_processor(
    processor, cache: CacheState, *, select_all: bool = False
) -> CacheState:
    indices = (
        cache.all_indices()
        if select_all
        else cache.select_indices(
            processor.config.process_every, processor.config.topk
        )
    )
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
    for layer_idx, part in enumerate(
        torch.split(updated, [p.shape[-1] for p in parts], dim=-1)
    ):
        if processor.config.update_stream == "v":
            cache.values[layer_idx] = _scatter_values(
                cache.values[layer_idx],
                indices,
                _split_values(part, *shapes[layer_idx]),
            )
        else:
            keys, values = _split_kv(part, *shapes[layer_idx])
            cache.scatter_layer(layer_idx, indices, keys, values)
    return cache


def _reference_workspace_processor(
    processor, cache: CacheState, *, select_all: bool = False
) -> CacheState:
    indices = (
        cache.all_indices()
        if select_all
        else _reference_workspace_indices(processor, cache)
    )
    values = torch.stack(cache.values, dim=0)
    scores = _reference_kv_scores(processor, cache)
    value_index = indices[None, :, None, :, None].expand(
        values.shape[0], -1, values.shape[2], -1, values.shape[4]
    )
    score_index = indices[None, :, None, :].expand(
        scores.shape[0], -1, scores.shape[2], -1
    )
    selected_values = torch.gather(values, 3, value_index)
    selected_scores = torch.gather(scores, 3, score_index)
    writer_values = selected_values * _reference_writer_gate(
        processor, selected_scores
    ).to(selected_values.dtype)
    writer_values = writer_values * _reference_token_gate(
        processor, selected_scores
    ).to(selected_values.dtype)
    delta = processor.block.scaled_update(
        _merge_global_values(writer_values), indices
    )
    broadcast = _split_global_values(
        delta, processor.num_layers, processor.num_kv_heads, processor.head_dim
    )
    for layer_idx in range(processor.num_layers):
        cache.values[layer_idx] = _scatter_values(
            cache.values[layer_idx],
            indices,
            selected_values[layer_idx] + broadcast[layer_idx],
        )
    return cache


def _reference_head_mask_processor(
    processor, cache: CacheState, *, select_all: bool = False
) -> CacheState:
    indices = (
        cache.all_indices()
        if select_all
        else _reference_workspace_indices(processor, cache)
    )
    keys = torch.stack(cache.keys, dim=0)
    values = torch.stack(cache.values, dim=0)
    scores = _reference_kv_scores(processor, cache)
    value_index = indices[None, :, None, :, None].expand(
        keys.shape[0], -1, keys.shape[2], -1, keys.shape[4]
    )
    score_index = indices[None, :, None, :].expand(
        scores.shape[0], -1, scores.shape[2], -1
    )
    selected_keys = torch.gather(keys, 3, value_index)
    selected_values = torch.gather(values, 3, value_index)
    selected_scores = torch.gather(scores, 3, score_index)
    gate = _reference_writer_gate(
        processor, selected_scores
    ) * _reference_token_gate(processor, selected_scores)
    parts = [
        _merge_kv(
            selected_keys[layer_idx] * gate[layer_idx].to(keys.dtype),
            selected_values[layer_idx] * gate[layer_idx].to(values.dtype),
        )
        for layer_idx in range(processor.num_layers)
    ]
    delta = processor.block.scaled_update(torch.cat(parts, dim=-1), indices)
    for layer_idx, part in enumerate(
        torch.split(delta, [p.shape[-1] for p in parts], dim=-1)
    ):
        key_delta, value_delta = _split_kv(
            part, selected_keys.shape[2], selected_keys.shape[4]
        )
        cache.scatter_layer(
            layer_idx,
            indices,
            selected_keys[layer_idx] + key_delta,
            selected_values[layer_idx] + value_delta,
        )
    return cache


def _reference_workspace_indices(processor, cache: CacheState) -> torch.Tensor:
    if cache.scores.shape[1] == 0:
        return torch.empty(
            cache.scores.shape[0],
            0,
            dtype=torch.long,
            device=cache.scores.device,
        )
    return _select_l2(
        _reference_kv_scores(processor, cache).amax(dim=(0, 2)),
        processor.config,
    )


def _reference_kv_scores(processor, cache: CacheState) -> torch.Tensor:
    keys = torch.stack(cache.keys, dim=0)
    values = torch.stack(cache.values, dim=0)
    weights = torch.softmax(processor.salience_logits.float(), dim=0)
    return (
        weights[0] * keys.float().pow(2).mean(dim=-1).sqrt()
        + weights[1] * values.float().pow(2).mean(dim=-1).sqrt()
    )


def _select_l2(
    scores: torch.Tensor, config: CacheProcessorConfig
) -> torch.Tensor:
    batch, seq = scores.shape
    recent = min(config.process_every, seq)
    hist_len = seq - recent
    full = (
        torch.arange(seq, device=scores.device).unsqueeze(0).expand(batch, -1)
    )
    recent_idx = full[:, hist_len:]
    if hist_len > 0 and config.topk > 0:
        k = min(config.topk, hist_len)
        _, hist_idx = torch.topk(scores[:, :hist_len].float(), k=k, dim=-1)
        selected = torch.cat([hist_idx, recent_idx], dim=1)
    else:
        selected = recent_idx
    return selected.sort(dim=1).values


def _reference_writer_gate(
    processor, selected_scores: torch.Tensor
) -> torch.Tensor:
    import math

    layers, _, heads, _ = selected_scores.shape
    head_scores = selected_scores.mean(dim=(1, 3))
    writer_count = max(
        1, math.ceil(layers * heads * processor.config.writer_head_fraction)
    )
    top = torch.topk(head_scores.flatten(), k=writer_count).indices
    mask = torch.zeros_like(head_scores.flatten(), dtype=torch.bool)
    mask.scatter_(0, top, True)
    logits = (
        head_scores.flatten() / processor.config.salience_temperature
    ).masked_fill(~mask, torch.finfo(head_scores.dtype).min)
    return (torch.softmax(logits, dim=0).view(layers, heads) * writer_count)[
        :, None, :, None, None
    ]


def _reference_token_gate(
    processor, selected_scores: torch.Tensor
) -> torch.Tensor:
    token_scores = selected_scores.amax(dim=(0, 2))
    gate = (
        torch.softmax(
            token_scores / processor.config.salience_temperature, dim=-1
        )
        * selected_scores.shape[-1]
    )
    return gate[None, :, None, :, None]


def _patch_reference_processor(model) -> None:
    if model.config.variant == "bt":
        model.cache_processor.forward = types.MethodType(
            _reference_per_layer_processor, model.cache_processor
        )
        model.cache_processor.prepare_forward_execution = types.MethodType(
            lambda self: nullcontext(self), model.cache_processor
        )
    else:
        model.cache_processor.forward = types.MethodType(
            _reference_global_processor, model.cache_processor
        )


def _reference_windowed_loss(
    self,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    processor,
    process_every: int,
    loss_reduction: str = "mean",
    attention_mask: torch.Tensor | None = None,
    loss_backend: str = "reference",
) -> torch.Tensor:
    cache = CacheState(
        [], [], input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32)
    )
    loss_sum = input_ids.new_zeros((), dtype=torch.float32)
    token_count = input_ids.new_zeros((), dtype=torch.float32)
    for start in range(0, input_ids.shape[1], process_every):
        if start > 0:
            cache = processor(cache)
        end = start + process_every
        chunk_logits, cache = self.forward_chunk(
            input_ids[:, start:end], start, cache, logits=True
        )
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
        assert torch.allclose(
            actual_value, expected_value, atol=1e-5, rtol=1e-5
        )


def _merge_values(values: torch.Tensor) -> torch.Tensor:
    batch, heads, seq, dim = values.shape
    return values.permute(0, 2, 1, 3).reshape(batch, seq, heads * dim)


def _split_values(
    x: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    batch, seq, _ = x.shape
    return (
        x.view(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    )


def _merge_global_values(values: torch.Tensor) -> torch.Tensor:
    layers, batch, heads, seq, dim = values.shape
    return values.permute(1, 3, 0, 2, 4).reshape(
        batch, seq, layers * heads * dim
    )


def _split_global_values(
    x: torch.Tensor, num_layers: int, num_heads: int, head_dim: int
) -> torch.Tensor:
    batch, seq, _ = x.shape
    return (
        x.view(batch, seq, num_layers, num_heads, head_dim)
        .permute(2, 0, 3, 1, 4)
        .contiguous()
    )


def _scatter_values(
    old_values: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    index = indices[:, None, :, None].expand(
        -1, old_values.shape[1], -1, old_values.shape[3]
    )
    return old_values.scatter(2, index, values.to(old_values.dtype))


def _assert_keys_unchanged(original: CacheState, actual: CacheState) -> None:
    for old_key, new_key in zip(original.keys, actual.keys):
        assert torch.equal(old_key, new_key)


def _assert_unselected_unchanged(
    original: CacheState, actual: CacheState, indices: torch.Tensor
) -> None:
    mask = torch.ones(original.scores.shape, dtype=torch.bool)
    mask.scatter_(1, indices.cpu(), False)
    _assert_cache_positions_unchanged(original, actual, mask)


def _not_latest_mask(cache: CacheState, indices: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(cache.scores.shape, dtype=torch.bool)
    latest = indices[:, -1:] if indices.ndim == 2 else indices[:, :1, -1]
    mask.scatter_(1, latest.cpu(), False)
    return mask


def _assert_cache_positions_unchanged(
    original: CacheState, actual: CacheState, mask: torch.Tensor
) -> None:
    for old_key, new_key, old_value, new_value in zip(
        original.keys, actual.keys, original.values, actual.values
    ):
        for batch_idx in range(mask.shape[0]):
            assert torch.equal(
                old_key[batch_idx, :, mask[batch_idx], :],
                new_key[batch_idx, :, mask[batch_idx], :],
            )
            assert torch.equal(
                old_value[batch_idx, :, mask[batch_idx], :],
                new_value[batch_idx, :, mask[batch_idx], :],
            )


def test_qwen35_longshot_gated_delta_rejects_a_head_count_the_dynamics_cannot_gate():
    cfg = tiny_qwen35_config(
        "longshot",
        encoder_method="query_attention",
        encoder_query_source="final_token",
        linear_state_cache=True,
        linear_state_update_mode="gated_delta",
    )
    mismatched = Qwen35GatedDeltaDynamics(1)
    processor = LongshotGlobalCacheProcessor(
        cfg, linear_state_dynamics={0: mismatched, 2: mismatched}
    )
    key_dim, value_dim = processor.block.linear_state_update_dims()
    update = torch.randn(1, 2, key_dim + value_dim + 2)
    recurrent = torch.zeros(1, 2, key_dim, value_dim)

    with pytest.raises(
        ValueError, match="head count 2 does not match layer 0 value heads 1"
    ):
        processor._gated_delta_linear_state_update(0, update, recurrent)


def test_qwen35_longshot_gated_delta_rejects_construction_without_matching_dynamics():
    cfg = tiny_qwen35_config(
        "longshot",
        encoder_method="query_attention",
        encoder_query_source="final_token",
        linear_state_cache=True,
        linear_state_update_mode="gated_delta",
    )

    with pytest.raises(
        ValueError,
        match="require matching Qwen3.5 linear-attention modules for layers",
    ):
        LongshotGlobalCacheProcessor(cfg, linear_state_dynamics={})


def test_gather_cache_heads_rejects_shapes_it_cannot_gather():
    x = torch.randn(2, 6, 8)
    selected = torch.tensor([[0, 2], [1, 3]])

    with pytest.raises(ValueError, match="selected_heads must have shape"):
        _longshot_gather_cache_heads(x, selected[None])
    with pytest.raises(ValueError, match="x must have shape"):
        _longshot_gather_cache_heads(x[None], selected)
