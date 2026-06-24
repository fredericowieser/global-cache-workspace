from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from gcw.config import CacheProcessorConfig, ModelConfig, dump_yaml, to_dict
from gcw.models.transformer import estimate_non_embedding_params


def processor_params(config: ModelConfig) -> int:
    if config.variant in {"bt", "bgt", "icarus"}:
        return estimate_processor_params(config)
    return 0


def estimate_processor_params(config: ModelConfig) -> int:
    if config.cache_processor is None:
        return 0
    per_layer_dim = config.cache_processor.stream_factor * config.num_kv_heads * int(config.head_dim)
    if config.variant == "bt":
        return config.num_layers * _processor_block_params(per_layer_dim, config.cache_processor)
    if config.variant == "bgt":
        return _processor_block_params(config.num_layers * per_layer_dim, config.cache_processor) + _salience_params(config)
    if config.variant == "icarus":
        projection_count = config.num_layers * config.num_kv_heads if config.cache_processor.per_cache_head_projections else 1
        return _processor_block_params(2 * int(config.head_dim), config.cache_processor, projection_count=projection_count)
    return 0


def _salience_params(config: ModelConfig) -> int:
    if config.cache_processor is not None and config.cache_processor.score == "kv_norm":
        return 2
    return 0


def total_non_embedding_params(config: ModelConfig) -> int:
    return estimate_non_embedding_params(config) + processor_params(config)


def _processor_block_params(input_dim: int, config: CacheProcessorConfig, *, projection_count: int = 1) -> int:
    head_dim = config.hidden_size // config.num_heads
    q_dim = config.num_heads * head_dim
    kv_dim = config.num_kv_heads * head_dim
    attn = (
        config.hidden_size * q_dim
        + q_dim
        + config.hidden_size * kv_dim
        + kv_dim
        + config.hidden_size * kv_dim
        + kv_dim
        + q_dim * config.hidden_size
    )
    mlp = 3 * config.hidden_size * config.intermediate_size
    layer = attn + mlp + 2 * config.hidden_size
    in_proj = projection_count * (input_dim * config.hidden_size + config.hidden_size)
    out_proj = projection_count * (config.hidden_size * input_dim + input_dim)
    return input_dim + in_proj + config.num_layers * layer + config.hidden_size + out_proj + 1


def fit_processor_config(
    backbone: ModelConfig,
    variant: str,
    fraction: float = 0.05,
    *,
    cache_processor_overrides: dict[str, Any] | None = None,
) -> tuple[ModelConfig, dict]:
    cache_processor_overrides = cache_processor_overrides or {}
    target = round(estimate_non_embedding_params(backbone) * fraction)
    best: tuple[int, ModelConfig] | None = None
    heads_choices = list(range(1, 17))
    hidden_choices = list(range(32, 2049 if variant == "icarus" else 1025, 32))
    for hidden in hidden_choices:
        for heads in heads_choices:
            if hidden % heads:
                continue
            if hidden // heads not in {32, 64, 128}:
                continue
            cp = CacheProcessorConfig(**_processor_config_kwargs(hidden, heads, variant, cache_processor_overrides))
            cfg = replace(backbone, variant=variant, cache_processor=cp)
            params = estimate_processor_params(cfg)
            if params <= target and (best is None or params > best[0]):
                best = (params, cfg)
    if best is None:
        cp = CacheProcessorConfig(**_processor_config_kwargs(32, 1, variant, cache_processor_overrides))
        cfg = replace(backbone, variant=variant, cache_processor=cp)
        best = (estimate_processor_params(cfg), cfg)
    params, cfg = best
    backbone_params = estimate_non_embedding_params(backbone)
    meta = {
        "backbone_non_embedding_params": backbone_params,
        "processor_params": params,
        "total_non_embedding_params": backbone_params + params,
        "processor_fraction": params / max(1, backbone_params),
        "target_processor_fraction": fraction,
        "target_processor_params": target,
    }
    return cfg, meta


def _processor_config_kwargs(hidden: int, heads: int, variant: str, overrides: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "hidden_size": hidden,
        "intermediate_size": hidden * 4,
        "num_heads": heads,
        "num_kv_heads": heads,
        "process_every": 64,
        "topk": 32,
    }
    if variant == "icarus":
        kwargs["topk_heads"] = 32
    kwargs.update(overrides)
    return kwargs


def _write_variant_config(
    out: Path,
    table: dict,
    *,
    table_key: str,
    file_variant: str,
    cfg: ModelConfig,
    meta: dict,
    target_non_embedding: int,
    regime: Literal["added_on_top", "matched_total"],
) -> None:
    meta = {
        **meta,
        "target_non_embedding_params": target_non_embedding,
        "delta_from_target": meta["total_non_embedding_params"] - target_non_embedding,
        "regime": regime,
    }
    table[table_key] = {"model": to_dict(cfg), "metadata": meta}
    suffix = f"smollm360m_{file_variant}" if regime == "added_on_top" else f"smollm360m_matched_{file_variant}"
    dump_yaml(out / f"{suffix}.yaml", {"model": cfg, "metadata": meta})


def smollm360m_backbone(num_layers: int = 32, intermediate_size: int = 2560) -> ModelConfig:
    return ModelConfig(
        variant="tf",
        family="smollm",
        vocab_size=49152,
        hidden_size=960,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_heads=15,
        num_kv_heads=5,
        head_dim=64,
        sequence_len=2048,
        tie_embeddings=True,
        rotary_style="ppt",
    )


def fit_matched_total_backbone(
    target_non_embedding: int,
    fraction: float = 0.05,
    *,
    base: ModelConfig | None = None,
    tune_intermediate: bool = False,
) -> tuple[ModelConfig, dict]:
    base = base or smollm360m_backbone()
    best: tuple[tuple[float, ...], ModelConfig, dict] | None = None
    intermediate_choices = range(2048, 4097, 32) if tune_intermediate else [base.intermediate_size]
    for num_layers in range(16, base.num_layers + 1):
        for intermediate_size in intermediate_choices:
            backbone = replace(base, num_layers=num_layers, intermediate_size=intermediate_size)
            bt, bt_meta = fit_processor_config(backbone, "bt", fraction=fraction)
            bgt, bgt_meta = fit_processor_config(backbone, "bgt", fraction=fraction)
            bt_total = total_non_embedding_params(bt)
            bgt_total = total_non_embedding_params(bgt)
            if bt_total > target_non_embedding or bgt_total > target_non_embedding:
                continue
            max_gap = max(target_non_embedding - bt_total, target_non_embedding - bgt_total)
            mean_gap = ((target_non_embedding - bt_total) + (target_non_embedding - bgt_total)) / 2
            frac_gap = abs(bt_meta["processor_fraction"] - fraction) + abs(bgt_meta["processor_fraction"] - fraction)
            score = (max_gap, mean_gap, frac_gap, -estimate_non_embedding_params(backbone))
            meta = {
                "target_non_embedding_params": target_non_embedding,
                "target_processor_fraction": fraction,
                "bt": bt_meta,
                "bgt": bgt_meta,
            }
            if best is None or score < best[0]:
                best = (score, backbone, meta)
    if best is None:
        raise ValueError("no matched-total backbone fits the target")
    _, backbone, meta = best
    return backbone, meta


def write_family_configs(
    out: str | Path,
    family: str = "smollm360m",
    fraction: float = 0.05,
    *,
    regime: Literal["added_on_top", "matched_total"] = "added_on_top",
    target_non_embedding: int | None = None,
) -> dict:
    if family != "smollm360m":
        raise ValueError("only smollm360m is built in; pass explicit configs for others")
    tf_backbone = smollm360m_backbone()
    target_non_embedding = target_non_embedding or estimate_non_embedding_params(tf_backbone)
    if regime == "matched_total":
        backbone, match_meta = fit_matched_total_backbone(target_non_embedding, fraction=fraction, base=tf_backbone)
    else:
        backbone, match_meta = tf_backbone, {}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = {
        "metadata": {"regime": regime, **match_meta},
        "tf": {
            "model": to_dict(tf_backbone),
            "metadata": {
                "backbone_non_embedding_params": estimate_non_embedding_params(tf_backbone),
                "total_non_embedding_params": estimate_non_embedding_params(tf_backbone),
            },
        },
    }
    dump_yaml(out / "smollm360m_tf.yaml", {"model": tf_backbone, "metadata": table["tf"]["metadata"]})
    for variant in ("bt", "bgt", "icarus"):
        cfg, meta = fit_processor_config(backbone, variant, fraction=fraction)
        _write_variant_config(
            out,
            table,
            table_key=variant,
            file_variant=variant,
            cfg=cfg,
            meta=meta,
            target_non_embedding=target_non_embedding,
            regime=regime,
        )
    cfg, meta = fit_processor_config(
        backbone,
        "icarus",
        fraction=fraction,
        cache_processor_overrides={"per_cache_head_projections": True},
    )
    _write_variant_config(
        out,
        table,
        table_key="icarus_per_cache_head",
        file_variant="icarus_per_cache_head",
        cfg=cfg,
        meta=meta,
        target_non_embedding=target_non_embedding,
        regime=regime,
    )
    return table
