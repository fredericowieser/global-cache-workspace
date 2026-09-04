from dataclasses import replace

from gcw.config import (
    CacheProcessorConfig,
    LongshotSelectorTrainingConfig,
    ModelConfig,
)
from gcw.flops import (
    backbone_flops_per_token,
    processor_forward_flops_per_sequence,
    total_training_flops,
    training_flops_per_token,
)
from gcw.params import fit_processor_config, smollm360m_backbone


def test_training_flops_include_processor_work():
    tf = replace(
        smollm360m_backbone(num_layers=2, intermediate_size=256),
        hidden_size=128,
        num_heads=2,
        num_kv_heads=1,
    )
    bt, _ = fit_processor_config(tf, "bt", fraction=0.05)
    bgt, _ = fit_processor_config(tf, "bgt", fraction=0.05)
    icarus, _ = fit_processor_config(tf, "icarus", fraction=0.05)
    assert processor_forward_flops_per_sequence(tf, 128) == 0.0
    assert processor_forward_flops_per_sequence(bt, 128) > 0.0
    assert processor_forward_flops_per_sequence(bgt, 128) > 0.0
    assert processor_forward_flops_per_sequence(icarus, 128) > 0.0
    assert training_flops_per_token(bt, 128) > training_flops_per_token(tf, 128)
    assert (
        total_training_flops(bgt, tokens=1024, seq_len=128)
        == training_flops_per_token(bgt, 128) * 1024
    )


def test_longshot_selector_training_flops_follow_routing_phase():
    backbone = replace(
        smollm360m_backbone(num_layers=2),
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        num_kv_heads=2,
    )
    config = replace(
        backbone,
        variant="longshot",
        cache_processor=CacheProcessorConfig(
            process_every=8,
            topk=0,
            topk_heads=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            num_kv_heads=1,
            iterations=2,
            head_selection_method="learned_contextual",
            internal_iteration_score="learned_contextual",
        ),
    )
    routing = LongshotSelectorTrainingConfig()
    hard = processor_forward_flops_per_sequence(config, 24)
    warmup = processor_forward_flops_per_sequence(
        config,
        24,
        selector_training=routing,
        training_progress=0.0,
    )
    soft = processor_forward_flops_per_sequence(
        config,
        24,
        selector_training=routing,
        training_progress=0.2,
    )
    transitioned_hard = processor_forward_flops_per_sequence(
        config,
        24,
        selector_training=routing,
        training_progress=0.9,
    )

    assert soft > warmup
    assert soft > hard
    assert warmup != hard
    assert transitioned_hard == hard


def test_processor_layer_subset_does_not_reduce_backbone_attention_flops():
    backbone = replace(
        smollm360m_backbone(num_layers=4),
        hidden_size=128,
        num_heads=2,
        num_kv_heads=1,
    )
    cp = CacheProcessorConfig(
        process_every=32,
        topk=8,
        hidden_size=64,
        intermediate_size=256,
        num_heads=1,
        num_kv_heads=1,
    )
    all_layers = replace(backbone, variant="bgt", cache_processor=cp)
    subset = replace(all_layers, processor_layer_indices=[0])

    assert backbone_flops_per_token(subset, 128) == backbone_flops_per_token(
        all_layers, 128
    )
    assert processor_forward_flops_per_sequence(
        subset, 128
    ) < processor_forward_flops_per_sequence(all_layers, 128)


def test_non_qwen_layer_types_do_not_change_backbone_attention_flops():
    dense = ModelConfig(
        family="smollm",
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
    )
    misleading = replace(
        dense, layer_types=["full_attention", "linear_attention"]
    )

    assert backbone_flops_per_token(misleading, 16) == backbone_flops_per_token(
        dense, 16
    )


def test_qwen35_moe_flops_count_active_routed_experts_per_token():
    top1 = ModelConfig(
        family="qwen3_5_moe",
        vocab_size=128,
        hidden_size=16,
        intermediate_size=8,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        layer_types=["linear_attention"],
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=1,
    )
    top4 = replace(top1, num_experts_per_tok=4)
    per_expert_matmuls = (
        2 * top1.moe_intermediate_size * top1.hidden_size
        + top1.hidden_size * top1.moe_intermediate_size
    )

    assert (
        training_flops_per_token(top4, 16) - training_flops_per_token(top1, 16)
        == 3 * 2 * 3 * per_expert_matmuls
    )


def test_value_only_processor_flops_use_smaller_cache_stream():
    backbone = replace(
        smollm360m_backbone(num_layers=2),
        hidden_size=128,
        num_heads=2,
        num_kv_heads=1,
    )
    kv_cp = CacheProcessorConfig(
        process_every=32,
        topk=8,
        hidden_size=64,
        intermediate_size=256,
        num_heads=1,
        num_kv_heads=1,
    )
    kv = replace(backbone, variant="bgt", cache_processor=kv_cp)
    v = replace(
        backbone,
        variant="bgt",
        cache_processor=replace(kv_cp, update_stream="v"),
    )

    assert processor_forward_flops_per_sequence(
        v, 128
    ) < processor_forward_flops_per_sequence(kv, 128)


def test_longshot_value_only_writeback_flops_keep_kv_input_but_reduce_output():
    backbone = replace(
        smollm360m_backbone(num_layers=2),
        hidden_size=128,
        num_heads=2,
        num_kv_heads=1,
    )
    kv_cp = CacheProcessorConfig(
        process_every=32,
        topk=8,
        hidden_size=64,
        intermediate_size=256,
        num_heads=1,
        num_kv_heads=1,
    )
    kv = replace(backbone, variant="longshot", cache_processor=kv_cp)
    v = replace(
        backbone,
        variant="longshot",
        cache_processor=replace(kv_cp, update_stream="v"),
    )

    assert processor_forward_flops_per_sequence(
        v, 128
    ) < processor_forward_flops_per_sequence(kv, 128)


def test_longshot_processor_flops_scale_with_iterations():
    backbone = replace(
        smollm360m_backbone(num_layers=2),
        hidden_size=128,
        num_heads=2,
        num_kv_heads=1,
    )
    cp = CacheProcessorConfig(
        process_every=32,
        topk=8,
        topk_heads=1,
        hidden_size=64,
        intermediate_size=256,
        num_heads=1,
        num_kv_heads=1,
    )
    once = replace(backbone, variant="longshot", cache_processor=cp)
    twice = replace(
        backbone, variant="longshot", cache_processor=replace(cp, iterations=2)
    )
    twice_learned = replace(
        twice,
        cache_processor=replace(
            twice.cache_processor,
            internal_iteration_score="learned_contextual",
        ),
    )
    twice_kv_norm = replace(
        twice,
        cache_processor=replace(
            twice.cache_processor, internal_iteration_score="kv_norm"
        ),
    )
    twice_cache_head_attention = replace(
        twice,
        cache_processor=replace(
            twice.cache_processor,
            internal_iteration_score="cache_head_attention",
        ),
    )
    twice_external = replace(
        backbone,
        variant="longshot",
        cache_processor=replace(cp, iterations=2, iteration_mode="external"),
    )

    once_flops = processor_forward_flops_per_sequence(once, 128)
    internal_flops = processor_forward_flops_per_sequence(twice, 128)
    external_flops = processor_forward_flops_per_sequence(twice_external, 128)

    assert once_flops < internal_flops < 2 * once_flops
    for scored in (twice_learned, twice_kv_norm, twice_cache_head_attention):
        assert (
            processor_forward_flops_per_sequence(scored, 128) > internal_flops
        )
    assert external_flops == 2 * once_flops


def test_qwen35_longshot_linear_state_flops_are_counted():
    cp = CacheProcessorConfig(
        process_every=4,
        topk=2,
        topk_heads=2,
        hidden_size=8,
        intermediate_size=16,
        num_heads=1,
        num_kv_heads=1,
        encoder_method="query_attention",
        encoder_query_source="final_token",
    )
    cfg = ModelConfig(
        variant="longshot",
        family="qwen3_5",
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=32,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        processor_layer_indices=[1, 3],
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        cache_processor=cp,
    )
    enabled = replace(cfg, cache_processor=replace(cp, linear_state_cache=True))
    per_linear_state = replace(
        enabled,
        cache_processor=replace(
            enabled.cache_processor, linear_state_query_mode="per_linear_state"
        ),
    )
    global_query = replace(
        enabled,
        cache_processor=replace(
            enabled.cache_processor, linear_state_query_mode="global"
        ),
    )
    shared = replace(
        enabled,
        cache_processor=replace(
            enabled.cache_processor, linear_state_query_mode="shared"
        ),
    )
    unselected = replace(
        enabled,
        cache_processor=replace(enabled.cache_processor, topk_linear_heads=0),
    )
    gated = replace(
        enabled,
        cache_processor=replace(
            enabled.cache_processor, linear_state_update_mode="gated_delta"
        ),
    )
    learned = replace(
        enabled,
        cache_processor=replace(
            enabled.cache_processor,
            head_selection_method="learned_contextual",
        ),
    )

    assert processor_forward_flops_per_sequence(
        enabled, 16
    ) > processor_forward_flops_per_sequence(cfg, 16)
    assert processor_forward_flops_per_sequence(
        unselected, 16
    ) < processor_forward_flops_per_sequence(enabled, 16)
    assert processor_forward_flops_per_sequence(
        global_query, 16
    ) == processor_forward_flops_per_sequence(enabled, 16)
    assert processor_forward_flops_per_sequence(
        shared, 16
    ) == processor_forward_flops_per_sequence(enabled, 16)
    assert processor_forward_flops_per_sequence(
        shared, 16
    ) < processor_forward_flops_per_sequence(per_linear_state, 16)
    assert processor_forward_flops_per_sequence(
        gated, 16
    ) > processor_forward_flops_per_sequence(enabled, 16)
    assert processor_forward_flops_per_sequence(
        learned, 16
    ) > processor_forward_flops_per_sequence(enabled, 16)


def test_icarus_value_only_writeback_flops_keep_kv_input_but_reduce_output():
    backbone = replace(
        smollm360m_backbone(num_layers=2),
        hidden_size=128,
        num_heads=2,
        num_kv_heads=1,
    )
    kv_cp = CacheProcessorConfig(
        process_every=32,
        topk=8,
        hidden_size=64,
        intermediate_size=256,
        num_heads=1,
        num_kv_heads=1,
    )
    kv = replace(backbone, variant="icarus", cache_processor=kv_cp)
    v = replace(
        backbone,
        variant="icarus",
        cache_processor=replace(kv_cp, update_stream="v"),
    )

    assert processor_forward_flops_per_sequence(
        v, 128
    ) < processor_forward_flops_per_sequence(kv, 128)


def test_self_head_kv_processor_flops_include_appended_self_head():
    backbone = replace(
        smollm360m_backbone(num_layers=2),
        hidden_size=128,
        num_heads=2,
        num_kv_heads=1,
    )
    cp = CacheProcessorConfig(
        process_every=32,
        topk=8,
        topk_heads=1,
        hidden_size=64,
        intermediate_size=256,
        num_heads=1,
        num_kv_heads=1,
    )

    for variant in ("icarus", "longshot"):
        base = replace(backbone, variant=variant, cache_processor=cp)
        self_head = replace(
            backbone,
            variant=variant,
            cache_processor=replace(cp, include_self_head_kv=True),
        )

        assert processor_forward_flops_per_sequence(
            self_head, 128
        ) > processor_forward_flops_per_sequence(base, 128)


def test_self_head_kv_flops_include_slot_when_all_heads_are_selected():
    backbone = replace(
        smollm360m_backbone(num_layers=2),
        hidden_size=128,
        num_heads=2,
        num_kv_heads=1,
    )
    cp = CacheProcessorConfig(
        process_every=32,
        topk=8,
        topk_heads=2,
        hidden_size=64,
        intermediate_size=256,
        num_heads=1,
        num_kv_heads=1,
    )

    for variant in ("icarus", "longshot"):
        base = replace(backbone, variant=variant, cache_processor=cp)
        self_head = replace(
            backbone,
            variant=variant,
            cache_processor=replace(cp, include_self_head_kv=True),
        )
        assert processor_forward_flops_per_sequence(
            self_head, 128
        ) > processor_forward_flops_per_sequence(base, 128)


def test_qwen35_processor_subset_does_not_reduce_backbone_attention_flops():
    processor = CacheProcessorConfig(
        hidden_size=16,
        intermediate_size=32,
        num_heads=1,
        num_kv_heads=1,
    )
    all_layers = ModelConfig(
        variant="bgt",
        family="qwen3_5",
        sequence_len=128,
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        layer_types=[
            "full_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ],
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        cache_processor=processor,
        processor_layer_indices=[0, 2],
    )
    subset = replace(all_layers, processor_layer_indices=[0])

    assert backbone_flops_per_token(subset, 128) == backbone_flops_per_token(
        all_layers, 128
    )
