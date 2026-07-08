from dataclasses import replace

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models import build_model
from gcw.models.transformer import estimate_non_embedding_params
from gcw.params import (
    fit_matched_total_backbone,
    fit_processor_config,
    processor_params,
    smollm360m_backbone,
    total_non_embedding_params,
    write_family_configs,
)


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


def test_icarus_processor_params_are_counted():
    backbone = smollm360m_backbone(num_layers=2, intermediate_size=256)
    icarus, meta = fit_processor_config(backbone, "icarus", fraction=0.05)
    assert icarus.cache_processor is not None
    assert processor_params(icarus) > 0
    assert meta["processor_params"] == processor_params(icarus)


def test_icarus_per_cache_head_projection_params_match_model():
    cp = CacheProcessorConfig(
        hidden_size=8,
        intermediate_size=16,
        num_heads=1,
        num_kv_heads=1,
        per_cache_head_projections=True,
    )
    cfg = ModelConfig(
        variant="icarus",
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        cache_processor=cp,
    )
    shared = replace(cfg, cache_processor=replace(cp, per_cache_head_projections=False))
    per_head = build_model(cfg)

    assert processor_params(cfg) == sum(p.numel() for p in per_head.cache_processor.parameters())
    assert processor_params(cfg) > processor_params(shared)


def test_fit_processor_config_applies_icarus_overrides():
    backbone = smollm360m_backbone(num_layers=2, intermediate_size=256)
    cfg, meta = fit_processor_config(
        backbone,
        "icarus",
        fraction=0.05,
        cache_processor_overrides={"per_cache_head_projections": True},
    )
    assert cfg.cache_processor is not None
    assert cfg.cache_processor.per_cache_head_projections is True
    assert cfg.cache_processor.topk_heads == 32
    assert meta["processor_params"] == processor_params(cfg)
    assert meta["processor_fraction"] <= 0.05


def test_generated_family_configs_include_icarus(tmp_path):
    table = write_family_configs(tmp_path)
    assert table["icarus"]["model"]["variant"] == "icarus"
    assert table["icarus"]["model"]["cache_processor"]["topk_heads"] == 32
    assert table["icarus_per_cache_head"]["model"]["variant"] == "icarus"
    assert table["icarus_per_cache_head"]["model"]["cache_processor"]["per_cache_head_projections"] is True
    assert table["icarus_per_cache_head"]["model"]["cache_processor"]["topk_heads"] == 32
    assert (tmp_path / "smollm360m_icarus.yaml").exists()
    assert (tmp_path / "smollm360m_icarus_per_cache_head.yaml").exists()


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
