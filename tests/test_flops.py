from dataclasses import replace

from gcw.config import CacheProcessorConfig
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


def test_value_only_processor_flops_use_smaller_cache_stream():
    backbone = replace(smollm360m_backbone(num_layers=2), hidden_size=128, num_heads=2, num_kv_heads=1)
    kv_cp = CacheProcessorConfig(process_every=32, topk=8, hidden_size=64, intermediate_size=256, num_heads=1, num_kv_heads=1)
    kv = replace(backbone, variant="bgt", cache_processor=kv_cp)
    v = replace(backbone, variant="bgt", cache_processor=replace(kv_cp, update_stream="v"))

    assert processor_forward_flops_per_sequence(v, 128) < processor_forward_flops_per_sequence(kv, 128)
