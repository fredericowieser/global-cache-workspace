from dataclasses import replace

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.flops import processor_forward_flops_per_sequence, total_training_flops, training_flops_per_token
from gcw.params import fit_processor_config, smollm360m_backbone


def test_training_flops_include_processor_work():
    tf = replace(smollm360m_backbone(num_layers=2, intermediate_size=256), hidden_size=128, num_heads=2, num_kv_heads=1)
    bt, _ = fit_processor_config(tf, "bt", fraction=0.05)
    bgt, _ = fit_processor_config(tf, "bgt", fraction=0.05)
    icarus, _ = fit_processor_config(tf, "icarus", fraction=0.05)
    assert processor_forward_flops_per_sequence(tf, 128) == 0.0
    assert processor_forward_flops_per_sequence(bt, 128) > 0.0
    assert processor_forward_flops_per_sequence(bgt, 128) > 0.0
    assert processor_forward_flops_per_sequence(icarus, 128) > 0.0
    assert training_flops_per_token(bt, 128) > training_flops_per_token(tf, 128)
    assert total_training_flops(bgt, tokens=1024, seq_len=128) == training_flops_per_token(bgt, 128) * 1024


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
    per_expert_matmuls = 2 * top1.moe_intermediate_size * top1.hidden_size + top1.hidden_size * top1.moe_intermediate_size

    assert training_flops_per_token(top4, 16) - training_flops_per_token(top1, 16) == 3 * 2 * 3 * per_expert_matmuls


def test_value_only_processor_flops_use_smaller_cache_stream():
    backbone = replace(smollm360m_backbone(num_layers=2), hidden_size=128, num_heads=2, num_kv_heads=1)
    kv_cp = CacheProcessorConfig(process_every=32, topk=8, hidden_size=64, intermediate_size=256, num_heads=1, num_kv_heads=1)
    kv = replace(backbone, variant="bgt", cache_processor=kv_cp)
    v = replace(backbone, variant="bgt", cache_processor=replace(kv_cp, update_stream="v"))

    assert processor_forward_flops_per_sequence(v, 128) < processor_forward_flops_per_sequence(kv, 128)


def test_longshot_value_only_writeback_flops_keep_kv_input_but_reduce_output():
    backbone = replace(smollm360m_backbone(num_layers=2), hidden_size=128, num_heads=2, num_kv_heads=1)
    kv_cp = CacheProcessorConfig(process_every=32, topk=8, hidden_size=64, intermediate_size=256, num_heads=1, num_kv_heads=1)
    kv = replace(backbone, variant="longshot", cache_processor=kv_cp)
    v = replace(backbone, variant="longshot", cache_processor=replace(kv_cp, update_stream="v"))

    assert processor_forward_flops_per_sequence(v, 128) < processor_forward_flops_per_sequence(kv, 128)


def test_longshot_processor_flops_scale_with_iterations():
    backbone = replace(smollm360m_backbone(num_layers=2), hidden_size=128, num_heads=2, num_kv_heads=1)
    cp = CacheProcessorConfig(
        process_every=32,
        topk=8,
        hidden_size=64,
        intermediate_size=256,
        num_heads=1,
        num_kv_heads=1,
    )
    once = replace(backbone, variant="longshot", cache_processor=cp)
    twice = replace(backbone, variant="longshot", cache_processor=replace(cp, iterations=2))
    twice_external = replace(
        backbone,
        variant="longshot",
        cache_processor=replace(cp, iterations=2, iteration_mode="external"),
    )

    assert processor_forward_flops_per_sequence(twice, 128) == 2 * processor_forward_flops_per_sequence(once, 128)
    assert processor_forward_flops_per_sequence(twice_external, 128) == 2 * processor_forward_flops_per_sequence(
        once, 128
    )


def test_icarus_value_only_writeback_flops_keep_kv_input_but_reduce_output():
    backbone = replace(smollm360m_backbone(num_layers=2), hidden_size=128, num_heads=2, num_kv_heads=1)
    kv_cp = CacheProcessorConfig(process_every=32, topk=8, hidden_size=64, intermediate_size=256, num_heads=1, num_kv_heads=1)
    kv = replace(backbone, variant="icarus", cache_processor=kv_cp)
    v = replace(backbone, variant="icarus", cache_processor=replace(kv_cp, update_stream="v"))

    assert processor_forward_flops_per_sequence(v, 128) < processor_forward_flops_per_sequence(kv, 128)


def test_self_head_kv_processor_flops_include_appended_self_head():
    backbone = replace(smollm360m_backbone(num_layers=2), hidden_size=128, num_heads=2, num_kv_heads=1)
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
        self_head = replace(backbone, variant=variant, cache_processor=replace(cp, include_self_head_kv=True))

        assert processor_forward_flops_per_sequence(self_head, 128) > processor_forward_flops_per_sequence(base, 128)
