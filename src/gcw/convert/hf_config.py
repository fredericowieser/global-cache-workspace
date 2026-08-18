from __future__ import annotations

from typing import Any, cast
from gcw.config import Family, ModelConfig, Variant, gemma4_default_layer_types

ALIASES = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama3-70b": "meta-llama/Meta-Llama-3-70B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama3.1-70b": "meta-llama/Llama-3.1-70B",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B",
}


def _validated_family(value: str) -> Family:
    if value not in {"smollm", "llama", "qwen3", "qwen3_5", "qwen3_5_moe", "gemma4"}:
        raise ValueError(f"unsupported model family: {value}")
    return cast(Family, value)


def _validated_variant(value: str) -> Variant:
    if value not in {"tf", "bt", "bgt", "icarus", "longshot"}:
        raise ValueError(f"unsupported model variant: {value}")
    return cast(Variant, value)


def config_from_hf(hf_cfg, family: str, variant: str = "tf") -> ModelConfig:
    model_family = _validated_family(family)
    model_variant = _validated_variant(variant)
    hf_text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
    layer_types = getattr(hf_text_cfg, "layer_types", None)
    if model_family == "qwen3" and layer_types and any(t != "full_attention" for t in layer_types):
        raise NotImplementedError("Qwen3 sliding attention is not implemented in gcw yet.")
    if model_family == "qwen3_5":
        return _qwen35_config_from_hf_text(hf_text_cfg, model_variant)
    if model_family == "qwen3_5_moe":
        return _qwen35_moe_config_from_hf_text(hf_text_cfg, model_variant)
    if model_family == "gemma4":
        return _gemma4_config_from_hf_text(hf_text_cfg, model_variant)
    heads = int(hf_text_cfg.num_attention_heads)
    hidden = int(hf_text_cfg.hidden_size)
    head_dim = int(getattr(hf_text_cfg, "head_dim", hidden // heads))
    rope_theta = getattr(hf_text_cfg, "rope_theta", None)
    if rope_theta is None:
        rope_theta = getattr(hf_text_cfg, "rope_parameters", {}).get("rope_theta", 10000.0)
    return ModelConfig(
        variant=model_variant,
        family=model_family,
        sequence_len=int(getattr(hf_text_cfg, "max_position_embeddings", 2048)),
        vocab_size=int(hf_text_cfg.vocab_size),
        hidden_size=hidden,
        intermediate_size=int(hf_text_cfg.intermediate_size),
        num_layers=int(hf_text_cfg.num_hidden_layers),
        num_heads=heads,
        num_kv_heads=int(getattr(hf_text_cfg, "num_key_value_heads", heads)),
        head_dim=head_dim,
        rope_theta=float(rope_theta),
        norm_eps=float(getattr(hf_text_cfg, "rms_norm_eps", 1e-5)),
        tie_embeddings=bool(getattr(hf_text_cfg, "tie_word_embeddings", False)),
        qk_norm=model_family == "qwen3",
        rotary_style="llama",
    )


def _qwen35_shared_config_kwargs(
    hf_cfg,
    variant: Variant,
    family: Family,
    intermediate_size: int,
) -> dict[str, Any]:
    heads = int(hf_cfg.num_attention_heads)
    hidden = int(hf_cfg.hidden_size)
    head_dim = int(getattr(hf_cfg, "head_dim", hidden // heads))
    rope = getattr(hf_cfg, "rope_parameters", {}) or {}
    layer_types = list(getattr(hf_cfg, "layer_types"))
    processor_layer_indices = [i for i, layer_type in enumerate(layer_types) if layer_type == "full_attention"]
    return dict(
        variant=variant,
        family=family,
        sequence_len=int(getattr(hf_cfg, "max_position_embeddings", 2048)),
        vocab_size=int(hf_cfg.vocab_size),
        hidden_size=hidden,
        intermediate_size=intermediate_size,
        num_layers=int(hf_cfg.num_hidden_layers),
        num_heads=heads,
        num_kv_heads=int(getattr(hf_cfg, "num_key_value_heads", heads)),
        head_dim=head_dim,
        rope_theta=float(rope.get("rope_theta", getattr(hf_cfg, "rope_theta", 10000.0))),
        norm_eps=float(getattr(hf_cfg, "rms_norm_eps", 1e-6)),
        tie_embeddings=bool(getattr(hf_cfg, "tie_word_embeddings", False)),
        qk_norm=True,
        rotary_style="llama",
        layer_types=layer_types,
        processor_layer_indices=processor_layer_indices,
        partial_rotary_factor=float(getattr(hf_cfg, "partial_rotary_factor", rope.get("partial_rotary_factor", 1.0))),
        linear_conv_kernel_dim=int(getattr(hf_cfg, "linear_conv_kernel_dim", 4)),
        linear_key_head_dim=int(getattr(hf_cfg, "linear_key_head_dim", head_dim)),
        linear_value_head_dim=int(getattr(hf_cfg, "linear_value_head_dim", head_dim)),
        linear_num_key_heads=int(getattr(hf_cfg, "linear_num_key_heads", getattr(hf_cfg, "num_key_value_heads", heads))),
        linear_num_value_heads=int(getattr(hf_cfg, "linear_num_value_heads", heads)),
        hidden_act=str(getattr(hf_cfg, "hidden_act", "silu")),
    )


def _qwen35_config_from_hf_text(hf_cfg, variant: Variant) -> ModelConfig:
    return ModelConfig(
        **_qwen35_shared_config_kwargs(
            hf_cfg,
            variant,
            "qwen3_5",
            int(getattr(hf_cfg, "intermediate_size")),
        )
    )


def _qwen35_moe_config_from_hf_text(hf_cfg, variant: Variant) -> ModelConfig:
    return ModelConfig(
        **_qwen35_shared_config_kwargs(
            hf_cfg,
            variant,
            "qwen3_5_moe",
            int(getattr(hf_cfg, "moe_intermediate_size", getattr(hf_cfg, "intermediate_size", 0))),
        ),
        moe_intermediate_size=int(getattr(hf_cfg, "moe_intermediate_size")),
        num_experts=int(getattr(hf_cfg, "num_experts")),
        num_experts_per_tok=int(getattr(hf_cfg, "num_experts_per_tok")),
        shared_expert_intermediate_size=int(getattr(hf_cfg, "shared_expert_intermediate_size")),
        router_aux_loss_coef=float(getattr(hf_cfg, "router_aux_loss_coef", 0.0)),
        output_router_logits=bool(getattr(hf_cfg, "output_router_logits", False)),
    )


def _gemma4_config_from_hf_text(hf_cfg, variant: Variant) -> ModelConfig:
    if bool(getattr(hf_cfg, "enable_moe_block", False)):
        raise NotImplementedError("Gemma4 MoE blocks are not implemented in gcw yet.")
    if bool(getattr(hf_cfg, "attention_bias", False)):
        raise NotImplementedError("Gemma4 attention_bias=True is not implemented in gcw yet.")
    num_layers = int(hf_cfg.num_hidden_layers)
    layer_types = list(getattr(hf_cfg, "layer_types", None) or gemma4_default_layer_types(num_layers))
    if layer_types[-1] != "full_attention":
        layer_types[-1] = "full_attention"
    heads = int(hf_cfg.num_attention_heads)
    hidden = int(hf_cfg.hidden_size)
    head_dim = int(getattr(hf_cfg, "head_dim", hidden // heads))
    rope_parameters = getattr(hf_cfg, "rope_parameters", None)
    if rope_parameters is None:
        rope_parameters = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
        }
    cfg = ModelConfig(
        variant=variant,
        family="gemma4",
        sequence_len=int(getattr(hf_cfg, "max_position_embeddings", 2048)),
        vocab_size=int(hf_cfg.vocab_size),
        hidden_size=hidden,
        intermediate_size=int(hf_cfg.intermediate_size),
        num_layers=num_layers,
        num_heads=heads,
        num_kv_heads=int(getattr(hf_cfg, "num_key_value_heads", heads)),
        head_dim=head_dim,
        rope_theta=float(rope_parameters.get("sliding_attention", {}).get("rope_theta", 10000.0)),
        norm_eps=float(getattr(hf_cfg, "rms_norm_eps", 1e-6)),
        tie_embeddings=bool(getattr(hf_cfg, "tie_word_embeddings", True)),
        qk_norm=True,
        rotary_style="llama",
        layer_types=layer_types,
        sliding_window=int(getattr(hf_cfg, "sliding_window", 512)),
        num_global_key_value_heads=(
            None
            if getattr(hf_cfg, "num_global_key_value_heads", None) is None
            else int(getattr(hf_cfg, "num_global_key_value_heads"))
        ),
        global_head_dim=int(getattr(hf_cfg, "global_head_dim", head_dim)),
        attention_k_eq_v=bool(getattr(hf_cfg, "attention_k_eq_v", False)),
        num_kv_shared_layers=int(getattr(hf_cfg, "num_kv_shared_layers", 0)),
        hidden_size_per_layer_input=int(getattr(hf_cfg, "hidden_size_per_layer_input", 0) or 0),
        vocab_size_per_layer_input=int(getattr(hf_cfg, "vocab_size_per_layer_input", getattr(hf_cfg, "vocab_size"))),
        final_logit_softcapping=getattr(hf_cfg, "final_logit_softcapping", None),
        rope_parameters=rope_parameters,
        hidden_act=str(getattr(hf_cfg, "hidden_activation", getattr(hf_cfg, "hidden_act", "gelu_pytorch_tanh"))),
        use_double_wide_mlp=bool(getattr(hf_cfg, "use_double_wide_mlp", False)),
    )
    cfg.processor_layer_indices = cfg.cache_layer_indices
    return cfg
