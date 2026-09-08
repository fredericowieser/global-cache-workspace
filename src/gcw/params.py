from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from gcw.config import (
    CacheProcessorConfig,
    ModelConfig,
    Variant,
    dump_yaml,
    linear_attention_following_cache_indices,
    to_dict,
)
from gcw.models.backbones.transformer import estimate_non_embedding_params
from gcw.models.lora import LoRALinear


def trainable_adapter_params(model) -> int:
    return sum(
        module.lora_a.numel() + module.lora_b.numel()
        for module in model.modules()
        if isinstance(module, LoRALinear)
    )


def processor_params(config: ModelConfig) -> int:
    if config.variant in {"bt", "bgt", "icarus", "longshot"}:
        return estimate_processor_params(config)
    return 0


def _head_dim(config: ModelConfig) -> int:
    if config.head_dim is None:
        raise ValueError(
            "model head_dim must be resolved before parameter accounting"
        )
    return config.head_dim


def estimate_processor_params(config: ModelConfig) -> int:
    if config.cache_processor is None:
        return 0
    head_dim = _head_dim(config)
    cache_layers = config.cache_num_layers
    per_layer_dim = (
        config.cache_processor.stream_factor * config.num_kv_heads * head_dim
    )
    if config.variant == "bt":
        return cache_layers * _processor_block_params(
            per_layer_dim, config.cache_processor
        )
    scheduled_diffusion = (
        config.diffusion_processor is not None
        and config.diffusion_processor.training_method == "scheduled"
    )
    flow_params = (
        1
        if config.flow_processor is not None
        or (config.diffusion_processor is not None and not scheduled_diffusion)
        else 0
    )
    write_gate_adjustment = (
        -1
        if scheduled_diffusion
        or config.cache_processor.output_scale_mode == "fixed_one"
        else 0
    )
    if config.variant == "bgt":
        return (
            _processor_block_params(
                cache_layers * per_layer_dim, config.cache_processor
            )
            + _salience_params(config)
            + flow_params
            + write_gate_adjustment
        )
    if config.variant == "icarus":
        projection_count = (
            config.cache_num_heads
            if config.cache_processor.per_cache_head_projections
            else 1
        )
        input_dim = 2 * head_dim
        out_proj_dim = (
            head_dim
            if config.cache_processor.update_stream == "v"
            else input_dim
        )
        return (
            _processor_block_params_with_dims(
                input_norm_dim=input_dim,
                in_proj_dim=input_dim,
                out_proj_dim=out_proj_dim,
                config=config.cache_processor,
                projection_count=projection_count,
            )
            + flow_params
        )
    if config.variant == "longshot":
        return (
            _longshot_processor_params(config)
            + flow_params
            + write_gate_adjustment
        )
    return 0


def _salience_params(config: ModelConfig) -> int:
    if (
        config.cache_processor is not None
        and config.cache_processor.score == "kv_norm"
    ):
        return 2
    return 0


def total_non_embedding_params(config: ModelConfig) -> int:
    return estimate_non_embedding_params(config) + processor_params(config)


def _processor_block_params(
    input_dim: int, config: CacheProcessorConfig, *, projection_count: int = 1
) -> int:
    return _processor_block_params_with_dims(
        input_norm_dim=input_dim,
        in_proj_dim=input_dim,
        out_proj_dim=input_dim,
        config=config,
        projection_count=projection_count,
    )


def _processor_block_params_with_dims(
    *,
    input_norm_dim: int,
    in_proj_dim: int,
    out_proj_dim: int,
    config: CacheProcessorConfig,
    projection_count: int = 1,
) -> int:
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
    in_proj = projection_count * (
        in_proj_dim * config.hidden_size + config.hidden_size
    )
    out_proj = projection_count * (
        config.hidden_size * out_proj_dim + out_proj_dim
    )
    return (
        input_norm_dim
        + in_proj
        + config.num_layers * layer
        + config.hidden_size
        + out_proj
        + 1
    )


def _longshot_processor_params(config: ModelConfig) -> int:
    cp = config.cache_processor
    if cp is None:
        return 0
    input_dim = 2 * _head_dim(config)
    cache_heads = config.cache_num_heads
    projection_count = cache_heads if cp.per_cache_head_projections else 1
    in_proj_dim = (
        input_dim
        if cp.encoder_method == "query_attention"
        else input_dim * _longshot_encoder_seq_len(cp)
    )
    out_proj_dim = _head_dim(config) if cp.update_stream == "v" else input_dim
    query = _longshot_query_params(input_dim, cache_heads, cp)
    linear_heads = _longshot_linear_state_heads(config)
    return (
        query
        + _longshot_linear_state_params(config)
        + _longshot_head_selector_params(cp, cache_heads + linear_heads)
        + _processor_block_params_with_dims(
            input_norm_dim=input_dim,
            in_proj_dim=in_proj_dim,
            out_proj_dim=out_proj_dim,
            config=cp,
            projection_count=projection_count,
        )
    )


def _longshot_head_selector_params(
    config: CacheProcessorConfig, cache_heads: int
) -> int:
    if (
        config.head_selection_method != "learned_contextual"
        and config.internal_iteration_score != "learned_contextual"
    ):
        return 0
    return (
        config.hidden_size
        + 2 * config.hidden_size * config.head_selector_hidden_size
        + (cache_heads if config.head_selector_use_identity_bias else 0)
    )


def _longshot_query_params(
    input_dim: int, cache_heads: int, config: CacheProcessorConfig
) -> int:
    if config.encoder_method != "query_attention":
        return 0
    if config.encoder_query_source == "learned":
        return cache_heads * input_dim
    if config.encoder_query_source == "final_token":
        projection_count = (
            cache_heads if config.per_cache_head_projections else 1
        )
        return projection_count * (input_dim * input_dim + input_dim)
    raise ValueError(
        f"unsupported cache_processor.encoder_query_source: {config.encoder_query_source}"
    )


def _longshot_encoder_seq_len(config: CacheProcessorConfig) -> int:
    return config.encoder_seq_len or config.process_every + max(config.topk, 0)


def _longshot_linear_state_heads(config: ModelConfig) -> int:
    cp = config.cache_processor
    if cp is None or not cp.linear_state_cache:
        return 0
    return len(linear_attention_following_cache_indices(config)) * int(
        config.linear_num_value_heads or config.num_heads
    )


def _longshot_linear_state_params(config: ModelConfig) -> int:
    cp = config.cache_processor
    if cp is None or not cp.linear_state_cache:
        return 0
    layer_pairs = linear_attention_following_cache_indices(config)
    if not layer_pairs:
        return 0
    layer_count = len(layer_pairs)
    linear_key_dim = int(config.linear_key_head_dim or _head_dim(config))
    linear_value_dim = int(config.linear_value_head_dim or linear_key_dim)
    linear_value_heads = int(config.linear_num_value_heads or config.num_heads)
    context_dim = cp.hidden_size
    if cp.linear_state_query_mode == "per_linear_state":
        query_count = layer_count
        query_out = linear_value_heads * linear_key_dim
    elif cp.linear_state_query_mode == "shared":
        query_count = len(
            {following_cache_idx for _, following_cache_idx in layer_pairs}
        )
        query_out = linear_key_dim
    elif cp.linear_state_query_mode == "global":
        query_count = 1
        query_out = linear_key_dim
    else:
        raise ValueError(
            f"unsupported cache_processor.linear_state_query_mode: {cp.linear_state_query_mode}"
        )
    query = query_count * (context_dim * query_out + query_out)
    feature_dim = linear_key_dim + linear_value_dim
    output_dim = feature_dim + (
        2 if cp.linear_state_update_mode == "gated_delta" else 0
    )
    query_context_norm = context_dim
    input_adapter = feature_dim + feature_dim * cp.hidden_size + cp.hidden_size
    output_adapter = cp.hidden_size * output_dim + output_dim
    return query + query_context_norm + input_adapter + output_adapter


def fit_processor_config(
    backbone: ModelConfig,
    variant: Variant,
    fraction: float = 0.05,
    *,
    cache_processor_overrides: dict[str, Any] | None = None,
) -> tuple[ModelConfig, dict]:
    cache_processor_overrides = cache_processor_overrides or {}
    target = round(estimate_non_embedding_params(backbone) * fraction)
    best: tuple[int, ModelConfig] | None = None
    heads_choices = list(range(1, 17))
    hidden_choices = list(
        range(32, 2049 if variant in {"icarus", "longshot"} else 1025, 32)
    )
    for hidden in hidden_choices:
        for heads in heads_choices:
            if hidden % heads:
                continue
            if hidden // heads not in {32, 64, 128}:
                continue
            cp = CacheProcessorConfig(
                **_processor_config_kwargs(
                    hidden, heads, variant, cache_processor_overrides
                )
            )
            cfg = replace(backbone, variant=variant, cache_processor=cp)
            params = estimate_processor_params(cfg)
            if params <= target and (best is None or params > best[0]):
                best = (params, cfg)
    if best is None:
        cp = CacheProcessorConfig(
            **_processor_config_kwargs(
                32, 1, variant, cache_processor_overrides
            )
        )
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


def _processor_config_kwargs(
    hidden: int, heads: int, variant: Variant, overrides: dict[str, Any]
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "hidden_size": hidden,
        "intermediate_size": hidden * 4,
        "num_heads": heads,
        "num_kv_heads": heads,
        "process_every": 64,
        "topk": 32,
    }
    if variant in {"icarus", "longshot"}:
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
        "delta_from_target": meta["total_non_embedding_params"]
        - target_non_embedding,
        "regime": regime,
    }
    table[table_key] = {"model": to_dict(cfg), "metadata": meta}
    suffix = (
        f"smollm360m_{file_variant}"
        if regime == "added_on_top"
        else f"smollm360m_matched_{file_variant}"
    )
    dump_yaml(out / f"{suffix}.yaml", {"model": cfg, "metadata": meta})


def smollm360m_backbone(
    num_layers: int = 32, intermediate_size: int = 2560
) -> ModelConfig:
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
    intermediate_choices = (
        range(2048, 4097, 32) if tune_intermediate else [base.intermediate_size]
    )
    for num_layers in range(16, base.num_layers + 1):
        for intermediate_size in intermediate_choices:
            backbone = replace(
                base, num_layers=num_layers, intermediate_size=intermediate_size
            )
            bt, bt_meta = fit_processor_config(
                backbone, "bt", fraction=fraction
            )
            bgt, bgt_meta = fit_processor_config(
                backbone, "bgt", fraction=fraction
            )
            bt_total = total_non_embedding_params(bt)
            bgt_total = total_non_embedding_params(bgt)
            if (
                bt_total > target_non_embedding
                or bgt_total > target_non_embedding
            ):
                continue
            max_gap = max(
                target_non_embedding - bt_total,
                target_non_embedding - bgt_total,
            )
            mean_gap = (
                (target_non_embedding - bt_total)
                + (target_non_embedding - bgt_total)
            ) / 2
            frac_gap = abs(bt_meta["processor_fraction"] - fraction) + abs(
                bgt_meta["processor_fraction"] - fraction
            )
            score = (
                max_gap,
                mean_gap,
                frac_gap,
                -estimate_non_embedding_params(backbone),
            )
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
        raise ValueError(
            "only smollm360m is built in; pass explicit configs for others"
        )
    tf_backbone = smollm360m_backbone()
    target_non_embedding = (
        target_non_embedding or estimate_non_embedding_params(tf_backbone)
    )
    if regime == "matched_total":
        backbone, match_meta = fit_matched_total_backbone(
            target_non_embedding, fraction=fraction, base=tf_backbone
        )
    else:
        backbone, match_meta = tf_backbone, {}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = {
        "metadata": {"regime": regime, **match_meta},
        "tf": {
            "model": to_dict(tf_backbone),
            "metadata": {
                "backbone_non_embedding_params": estimate_non_embedding_params(
                    tf_backbone
                ),
                "total_non_embedding_params": estimate_non_embedding_params(
                    tf_backbone
                ),
            },
        },
    }
    dump_yaml(
        out / "smollm360m_tf.yaml",
        {"model": tf_backbone, "metadata": table["tf"]["metadata"]},
    )
    for variant in ("bt", "bgt", "icarus", "longshot"):
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
    cfg, meta = fit_processor_config(
        backbone,
        "longshot",
        fraction=fraction,
        cache_processor_overrides={"encoder_method": "query_attention"},
    )
    _write_variant_config(
        out,
        table,
        table_key="longshot_query_attention",
        file_variant="longshot_query_attention",
        cfg=cfg,
        meta=meta,
        target_non_embedding=target_non_embedding,
        regime=regime,
    )
    cfg, meta = fit_processor_config(
        backbone,
        "longshot",
        fraction=fraction,
        cache_processor_overrides={
            "encoder_method": "query_attention",
            "per_cache_head_projections": True,
        },
    )
    _write_variant_config(
        out,
        table,
        table_key="longshot_query_attention_per_cache_head",
        file_variant="longshot_query_attention_per_cache_head",
        cfg=cfg,
        meta=meta,
        target_non_embedding=target_non_embedding,
        regime=regime,
    )
    return table
