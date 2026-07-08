from dataclasses import replace

from gcw.config import CacheProcessorConfig
from gcw.models.transformer import estimate_non_embedding_params
from gcw.params import fit_matched_total_backbone, fit_processor_config, processor_params, smollm360m_backbone, total_non_embedding_params


def test_smol_bt_bgt_share_backbone_and_stay_under_five_percent():
    backbone = smollm360m_backbone()
    budget = 0.05 * estimate_non_embedding_params(backbone)
    bt, bt_meta = fit_processor_config(backbone, "bt", fraction=0.05)
    bgt, bgt_meta = fit_processor_config(backbone, "bgt", fraction=0.05)
    assert bt.num_layers == bgt.num_layers == backbone.num_layers
    assert bt.hidden_size == bgt.hidden_size == backbone.hidden_size
    assert processor_params(bt) <= budget
    assert processor_params(bgt) <= budget
    assert bt_meta["processor_fraction"] <= 0.05
    assert bgt_meta["processor_fraction"] <= 0.05


def test_matched_total_uses_one_backbone_under_tf_target():
    target = estimate_non_embedding_params(smollm360m_backbone())
    backbone, _ = fit_matched_total_backbone(target, fraction=0.05)
    bt, bt_meta = fit_processor_config(backbone, "bt", fraction=0.05)
    bgt, bgt_meta = fit_processor_config(backbone, "bgt", fraction=0.05)
    assert bt.num_layers == bgt.num_layers == backbone.num_layers
    assert bt.hidden_size == bgt.hidden_size == backbone.hidden_size
    assert bt.intermediate_size == bgt.intermediate_size == backbone.intermediate_size
    assert bt_meta["processor_fraction"] <= 0.05
    assert bgt_meta["processor_fraction"] <= 0.05
    assert total_non_embedding_params(bt) <= target
    assert total_non_embedding_params(bgt) <= target


def test_value_only_processor_params_use_smaller_cache_stream():
    backbone = replace(smollm360m_backbone(num_layers=2), hidden_size=128, num_heads=2, num_kv_heads=1)
    kv_cp = CacheProcessorConfig(hidden_size=64, intermediate_size=256, num_heads=1, num_kv_heads=1)
    kv = replace(backbone, variant="bt", cache_processor=kv_cp)
    v = replace(backbone, variant="bt", cache_processor=replace(kv_cp, update_stream="v"))

    assert processor_params(v) < processor_params(kv)


def test_bgt_kv_norm_salience_adds_two_params():
    backbone = replace(smollm360m_backbone(num_layers=2), hidden_size=128, num_heads=2, num_kv_heads=1)
    cp = CacheProcessorConfig(hidden_size=64, intermediate_size=256, num_heads=1, num_kv_heads=1, update_stream="v")
    standard = replace(backbone, variant="bgt", cache_processor=cp)
    kv_norm = replace(backbone, variant="bgt", cache_processor=replace(cp, score="kv_norm"))

    assert processor_params(kv_norm) == processor_params(standard) + 2
