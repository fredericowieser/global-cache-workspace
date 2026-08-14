from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import torch
from safetensors.torch import load_file

from gcw.bottleneck_specs import BottleneckSpec, bottleneck_spec
from gcw.checkpoint import save_checkpoint
from gcw.config import (
    CacheProcessorConfig,
    ModelConfig,
    gemma4_default_layer_types,
)
from gcw.models import build_model

ALIASES = {
    "llama3.1-7b": "meta-llama/Meta-Llama-3.1-8B",
    "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B",
}


def convert_hf(
    model_id: str,
    out: str | Path,
    family: str = "llama",
    variant: str = "tf",
    dtype: str = "bfloat16",
) -> None:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ImportError(
            "Install the convert extra: uv sync --extra convert"
        ) from exc

    model_id = ALIASES.get(model_id.lower(), model_id)
    hf_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    cfg = config_from_hf(hf_cfg, family=family, variant=variant)
    if variant in {"bt", "bgt", "icarus", "longshot"}:
        cfg = replace(cfg, cache_processor=CacheProcessorConfig())
    torch_dtype = _resolve_dtype(dtype)
    hf = _load_hf_model(model_id, torch_dtype=torch_dtype, family=family)
    model_dtype = _infer_model_dtype(hf)
    native = build_model(cfg).to(dtype=model_dtype)
    native.init_weights()
    load_hf_state(native, hf.state_dict(), family=family)
    save_checkpoint(
        out,
        native,
        cfg,
        {
            "source": model_id,
            "source_format": "hf",
            "family": family,
            "dtype": str(model_dtype).removeprefix("torch."),
        },
    )


def convert_ppt(
    checkpoint: str | Path,
    meta: str | Path,
    out: str | Path,
    dtype: str = "bfloat16",
) -> None:
    checkpoint = Path(checkpoint)
    meta = Path(meta)
    raw_meta = json.loads(meta.read_text())
    cfg = config_from_ppt_meta(raw_meta)
    torch_dtype = _resolve_dtype(dtype)
    if torch_dtype == "auto":
        torch_dtype = torch.bfloat16
    model = build_model(cfg).to(dtype=torch_dtype)
    model.init_weights()
    state = torch.load(checkpoint, map_location="cpu")
    load_ppt_state(model, state)
    save_checkpoint(
        out,
        model,
        cfg,
        {
            "source": str(checkpoint),
            "source_format": "ppt",
            "meta": str(meta),
            "family": "smollm",
            "dtype": str(torch_dtype).removeprefix("torch."),
            "step": raw_meta.get("step"),
        },
    )


def convert_bottleneck(
    checkpoint: str | Path,
    out: str | Path,
    variant: str,
    dtype: str = "auto",
    source: str | None = None,
    tokenizer_source: str | None = None,
    backbone: str = "llama-3.2-1b",
    sequence_len: int = 8192,
    legacy_generate: bool = False,
) -> None:
    if variant not in {"bt", "bgt"}:
        raise ValueError(
            "bottleneck conversion supports only bt and bgt variants"
        )
    if sequence_len <= 0:
        raise ValueError("sequence_len must be positive")
    checkpoint = _model_file(checkpoint)
    old_state = load_file(str(checkpoint), device="cpu")
    torch_dtype = _resolve_dtype(dtype)
    if torch_dtype == "auto":
        torch_dtype = _infer_state_dtype(old_state) or torch.bfloat16
    cfg = config_from_bottleneck(
        backbone,
        variant,
        sequence_len=sequence_len,
        legacy_generate=legacy_generate,
        legacy_vocab=True,
    )
    model = build_model(cfg).to(dtype=torch_dtype)
    model.init_weights()
    load_bottleneck_state(model, old_state, variant=variant)
    spec = _bottleneck_spec(backbone)
    source = source or spec.source
    tokenizer_source = tokenizer_source or (
        str(checkpoint.parent)
        if (checkpoint.parent / "tokenizer.json").exists()
        else source
    )
    save_checkpoint(
        out,
        model,
        cfg,
        {
            "source": tokenizer_source,
            "backbone_source": source,
            "backbone": backbone,
            "source_format": "bottleneck_llm",
            "checkpoint": str(checkpoint),
            "family": spec.family,
            "variant": variant,
            "dtype": str(torch_dtype).removeprefix("torch."),
            "step": _step_from_path(checkpoint),
            "legacy_generate": legacy_generate,
            "sequence_len": sequence_len,
        },
    )


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    try:
        return getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(
            f"unsupported dtype {dtype!r}; use auto, bfloat16, float16, or float32"
        ) from exc


def _infer_model_dtype(model: torch.nn.Module) -> torch.dtype:
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def _infer_state_dtype(state: dict[str, torch.Tensor]) -> torch.dtype | None:
    for tensor in state.values():
        if tensor.is_floating_point():
            return tensor.dtype
    return None


def _model_file(path: str | Path) -> Path:
    path = Path(path)
    return path / "model.safetensors" if path.is_dir() else path


def _step_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        if part.startswith("checkpoint-"):
            try:
                return int(part.removeprefix("checkpoint-"))
            except ValueError:
                return None
    return None


def _load_hf_model(model_id: str, *, torch_dtype, family: str):
    from transformers import AutoModelForCausalLM

    if family == "qwen3_5_moe":
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError:
            AutoModelForMultimodalLM = None
        if AutoModelForMultimodalLM is not None:
            return AutoModelForMultimodalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, trust_remote_code=False
            )
    if family == "gemma4":
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError:
            AutoModelForMultimodalLM = None
        if AutoModelForMultimodalLM is not None:
            try:
                return AutoModelForMultimodalLM.from_pretrained(
                    model_id, torch_dtype=torch_dtype, trust_remote_code=False
                )
            except ValueError:
                pass
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, trust_remote_code=False
    )


def config_from_hf(hf_cfg, family: str, variant: str = "tf") -> ModelConfig:
    hf_text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
    layer_types = getattr(hf_text_cfg, "layer_types", None)
    if (
        family == "qwen3"
        and layer_types
        and any(t != "full_attention" for t in layer_types)
    ):
        raise NotImplementedError(
            "Qwen3 sliding attention is not implemented in gcw yet."
        )
    if family == "qwen3_5":
        return _qwen35_config_from_hf_text(hf_text_cfg, variant)
    if family == "qwen3_5_moe":
        return _qwen35_moe_config_from_hf_text(hf_text_cfg, variant)
    if family == "gemma4":
        return _gemma4_config_from_hf_text(hf_text_cfg, variant)
    heads = int(hf_text_cfg.num_attention_heads)
    hidden = int(hf_text_cfg.hidden_size)
    head_dim = int(getattr(hf_text_cfg, "head_dim", hidden // heads))
    rope_theta = getattr(hf_text_cfg, "rope_theta", None)
    if rope_theta is None:
        rope_theta = getattr(hf_text_cfg, "rope_parameters", {}).get(
            "rope_theta", 10000.0
        )
    return ModelConfig(
        variant=variant,
        family=family,
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
        qk_norm=family == "qwen3",
        rotary_style="llama",
    )


def _qwen35_shared_config_kwargs(
    hf_cfg, variant: str, family: str, intermediate_size: int
) -> dict:
    heads = int(hf_cfg.num_attention_heads)
    hidden = int(hf_cfg.hidden_size)
    head_dim = int(getattr(hf_cfg, "head_dim", hidden // heads))
    rope = getattr(hf_cfg, "rope_parameters", {}) or {}
    layer_types = list(getattr(hf_cfg, "layer_types"))
    processor_layer_indices = [
        i
        for i, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    ]
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
        rope_theta=float(
            rope.get("rope_theta", getattr(hf_cfg, "rope_theta", 10000.0))
        ),
        norm_eps=float(getattr(hf_cfg, "rms_norm_eps", 1e-6)),
        tie_embeddings=bool(getattr(hf_cfg, "tie_word_embeddings", False)),
        qk_norm=True,
        rotary_style="llama",
        layer_types=layer_types,
        processor_layer_indices=processor_layer_indices,
        partial_rotary_factor=float(
            getattr(
                hf_cfg,
                "partial_rotary_factor",
                rope.get("partial_rotary_factor", 1.0),
            )
        ),
        linear_conv_kernel_dim=int(
            getattr(hf_cfg, "linear_conv_kernel_dim", 4)
        ),
        linear_key_head_dim=int(
            getattr(hf_cfg, "linear_key_head_dim", head_dim)
        ),
        linear_value_head_dim=int(
            getattr(hf_cfg, "linear_value_head_dim", head_dim)
        ),
        linear_num_key_heads=int(
            getattr(
                hf_cfg,
                "linear_num_key_heads",
                getattr(hf_cfg, "num_key_value_heads", heads),
            )
        ),
        linear_num_value_heads=int(
            getattr(hf_cfg, "linear_num_value_heads", heads)
        ),
        hidden_act=str(getattr(hf_cfg, "hidden_act", "silu")),
    )


def _qwen35_config_from_hf_text(hf_cfg, variant: str) -> ModelConfig:
    return ModelConfig(
        **_qwen35_shared_config_kwargs(
            hf_cfg,
            variant,
            "qwen3_5",
            int(getattr(hf_cfg, "intermediate_size")),
        )
    )


def _qwen35_moe_config_from_hf_text(hf_cfg, variant: str) -> ModelConfig:
    return ModelConfig(
        **_qwen35_shared_config_kwargs(
            hf_cfg,
            variant,
            "qwen3_5_moe",
            int(
                getattr(
                    hf_cfg,
                    "moe_intermediate_size",
                    getattr(hf_cfg, "intermediate_size", 0),
                )
            ),
        ),
        moe_intermediate_size=int(getattr(hf_cfg, "moe_intermediate_size")),
        num_experts=int(getattr(hf_cfg, "num_experts")),
        num_experts_per_tok=int(getattr(hf_cfg, "num_experts_per_tok")),
        shared_expert_intermediate_size=int(
            getattr(hf_cfg, "shared_expert_intermediate_size")
        ),
        router_aux_loss_coef=float(
            getattr(hf_cfg, "router_aux_loss_coef", 0.0)
        ),
        output_router_logits=bool(
            getattr(hf_cfg, "output_router_logits", False)
        ),
    )


def _gemma4_config_from_hf_text(hf_cfg, variant: str) -> ModelConfig:
    if bool(getattr(hf_cfg, "enable_moe_block", False)):
        raise NotImplementedError(
            "Gemma4 MoE blocks are not implemented in gcw yet."
        )
    if bool(getattr(hf_cfg, "attention_bias", False)):
        raise NotImplementedError(
            "Gemma4 attention_bias=True is not implemented in gcw yet."
        )
    num_layers = int(hf_cfg.num_hidden_layers)
    layer_types = list(
        getattr(hf_cfg, "layer_types", None)
        or gemma4_default_layer_types(num_layers)
    )
    if layer_types[-1] != "full_attention":
        layer_types[-1] = "full_attention"
    heads = int(hf_cfg.num_attention_heads)
    hidden = int(hf_cfg.hidden_size)
    head_dim = int(getattr(hf_cfg, "head_dim", hidden // heads))
    rope_parameters = getattr(hf_cfg, "rope_parameters", None)
    if rope_parameters is None:
        rope_parameters = {
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
            },
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
        rope_theta=float(
            rope_parameters.get("sliding_attention", {}).get(
                "rope_theta", 10000.0
            )
        ),
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
        hidden_size_per_layer_input=int(
            getattr(hf_cfg, "hidden_size_per_layer_input", 0) or 0
        ),
        vocab_size_per_layer_input=int(
            getattr(
                hf_cfg,
                "vocab_size_per_layer_input",
                getattr(hf_cfg, "vocab_size"),
            )
        ),
        final_logit_softcapping=getattr(
            hf_cfg, "final_logit_softcapping", None
        ),
        rope_parameters=rope_parameters,
        hidden_act=str(
            getattr(
                hf_cfg,
                "hidden_activation",
                getattr(hf_cfg, "hidden_act", "gelu_pytorch_tanh"),
            )
        ),
        use_double_wide_mlp=bool(getattr(hf_cfg, "use_double_wide_mlp", False)),
    )
    cfg.processor_layer_indices = cfg.cache_layer_indices
    return cfg


def config_from_bottleneck(
    backbone: str,
    variant: str,
    sequence_len: int = 8192,
    legacy_generate: bool = False,
    legacy_vocab: bool = False,
) -> ModelConfig:
    if variant not in {"bt", "bgt"}:
        raise ValueError("bottleneck configs support only bt and bgt variants")
    spec = _bottleneck_spec(backbone)
    vocab_size = (
        spec.legacy_vocab_size
        if legacy_vocab and spec.legacy_vocab_size is not None
        else spec.vocab_size
    )
    hidden = spec.bt_hidden_size if variant == "bt" else spec.bgt_hidden_size
    intermediate = (
        spec.bt_intermediate_size
        if variant == "bt"
        else spec.bgt_intermediate_size
    )
    return ModelConfig(
        variant=variant,
        family=spec.family,
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        num_kv_heads=spec.num_kv_heads,
        head_dim=spec.head_dim,
        rope_theta=spec.rope_theta,
        norm_eps=spec.norm_eps,
        tie_embeddings=spec.tie_embeddings,
        qk_norm=spec.qk_norm,
        rotary_style="llama",
        legacy_bottleneck_generate=legacy_generate,
        cache_processor=CacheProcessorConfig(
            process_every=64,
            topk=32,
            score="attn_l2",
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_layers=1,
            num_heads=16,
            num_kv_heads=16,
            norm_eps=1e-6,
            rope_theta=10000.0,
            gate_init=-5.0,
            attention_dropout=0.0,
        ),
    )


def config_from_bottleneck_llama1b(
    variant: str,
    sequence_len: int = 8192,
    legacy_generate: bool = False,
) -> ModelConfig:
    return config_from_bottleneck(
        "llama-3.2-1b",
        variant,
        sequence_len=sequence_len,
        legacy_generate=legacy_generate,
    )


def _bottleneck_spec(backbone: str) -> BottleneckSpec:
    return bottleneck_spec(backbone)


def config_from_ppt_meta(meta: dict) -> ModelConfig:
    model_config = meta["model_config"]
    return ModelConfig(
        variant="tf",
        family="smollm",
        sequence_len=int(model_config["sequence_len"]),
        vocab_size=int(model_config["vocab_size"]),
        hidden_size=int(model_config["n_embd"]),
        intermediate_size=int(model_config["intermediate_size"]),
        num_layers=int(model_config["n_layer"]),
        num_heads=int(model_config["n_head"]),
        num_kv_heads=int(model_config["n_kv_head"]),
        head_dim=int(model_config["head_dim"]),
        rope_theta=float(model_config.get("rope_theta", 10000.0)),
        norm_eps=float(model_config.get("norm_eps", 1e-5)),
        tie_embeddings=bool(model_config.get("tie_embeddings", True)),
        rotary_style="ppt",
    )


def load_bottleneck_state(
    native, old_state: dict[str, torch.Tensor], variant: str
) -> None:
    target = native.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    for old_key, tensor in old_state.items():
        key = _map_bottleneck_key(old_key, variant)
        if key is None:
            unmapped.append(old_key)
            continue
        if key not in target:
            if key == "backbone.lm_head.weight":
                continue
            unmapped.append(old_key)
            continue
        if target[key].shape != tensor.shape:
            padded = _pad_vocab_tensor(key, target[key], tensor)
            if padded is not None:
                mapped[key] = padded
                continue
            raise ValueError(
                f"shape mismatch for {key}: native {tuple(target[key].shape)} "
                f"vs bottleneck {tuple(tensor.shape)} from {old_key}"
            )
        mapped[key] = tensor.to(dtype=target[key].dtype)
    missing = sorted(set(target) - set(mapped))
    extra = sorted(set(mapped) - set(target))
    if missing or extra or unmapped:
        raise KeyError(
            "bottleneck mapping mismatch "
            f"missing={missing[:8]} extra={extra[:8]} unmapped={unmapped[:8]}"
        )
    native.load_state_dict(mapped, strict=True)


def _pad_vocab_tensor(
    key: str, expected: torch.Tensor, tensor: torch.Tensor
) -> torch.Tensor | None:
    if key not in {"backbone.embed_tokens.weight", "backbone.lm_head.weight"}:
        return None
    if expected.ndim != tensor.ndim or expected.ndim != 2:
        return None
    if (
        expected.shape[0] < tensor.shape[0]
        or expected.shape[1:] != tensor.shape[1:]
    ):
        return None
    padded = expected.clone()
    padded[: tensor.shape[0]] = tensor.to(dtype=expected.dtype)
    return padded


def _map_bottleneck_key(key: str, variant: str) -> str | None:
    if key.startswith("backbone.model."):
        return "backbone." + key.removeprefix("backbone.model.")
    if key == "backbone.lm_head.weight":
        return key
    if variant == "bt" and key.startswith("cache_processor.processor_blocks."):
        rest = key.removeprefix("cache_processor.processor_blocks.")
        layer, sub = rest.split(".", 1)
        return (
            "cache_processor.blocks."
            + layer
            + "."
            + _map_processor_subkey(sub, "transformer.")
        )
    if variant == "bgt" and key.startswith("cache_processor.unified_block."):
        sub = key.removeprefix("cache_processor.unified_block.")
        return "cache_processor.block." + _map_processor_subkey(
            sub, "transformer.0."
        )
    return None


def _map_processor_subkey(sub: str, transformer_prefix: str) -> str:
    if sub == "g":
        return "gate"
    direct = {
        "kv_in_norm.": "in_norm.",
        "kv_in_proj.": "in_proj.",
        "kv_out_proj.": "out_proj.",
        "out_norm.": "out_norm.",
    }
    for old, new in direct.items():
        if sub.startswith(old):
            return new + sub.removeprefix(old)
    if not sub.startswith(transformer_prefix):
        return sub
    inner = sub.removeprefix(transformer_prefix)
    blocks = {
        "input_layernorm.": "layers.0.input_norm.",
        "post_attention_layernorm.": "layers.0.post_norm.",
        "self_attn.": "layers.0.attn.",
        "mlp.": "layers.0.",
    }
    for old, new in blocks.items():
        if inner.startswith(old):
            return new + inner.removeprefix(old)
    return inner


def load_hf_state(
    native, hf_state: dict[str, torch.Tensor], family: str = "llama"
) -> None:
    target = native.backbone if hasattr(native, "backbone") else native
    if family == "qwen3_5":
        return _load_qwen35_hf_state(target, hf_state)
    if family == "qwen3_5_moe":
        return _load_qwen35_moe_hf_state(target, hf_state)
    if family == "gemma4":
        return _load_gemma4_hf_state(target, hf_state)
    state = target.state_dict()
    mapped = {}
    pairs = {
        "embed_tokens.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for local, remote in pairs.items():
        if local in state and remote in hf_state:
            mapped[local] = hf_state[remote]
    for i in range(target.config.num_layers):
        prefix = f"layers.{i}"
        remote = f"model.layers.{i}"
        names = {
            f"{prefix}.input_layernorm.weight": f"{remote}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight": f"{remote}.post_attention_layernorm.weight",
            f"{prefix}.self_attn.q_proj.weight": f"{remote}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight": f"{remote}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight": f"{remote}.self_attn.v_proj.weight",
            f"{prefix}.self_attn.o_proj.weight": f"{remote}.self_attn.o_proj.weight",
            f"{prefix}.mlp.gate_proj.weight": f"{remote}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight": f"{remote}.mlp.up_proj.weight",
            f"{prefix}.mlp.down_proj.weight": f"{remote}.mlp.down_proj.weight",
        }
        if family == "qwen3":
            names[f"{prefix}.self_attn.q_norm.weight"] = (
                f"{remote}.self_attn.q_norm.weight"
            )
            names[f"{prefix}.self_attn.k_norm.weight"] = (
                f"{remote}.self_attn.k_norm.weight"
            )
        for local, key in names.items():
            if local in state and key in hf_state:
                mapped[local] = hf_state[key]
    merged = dict(state)
    for key, tensor in mapped.items():
        if merged[key].shape == tensor.shape:
            merged[key] = tensor
        elif (
            key in {"embed_tokens.weight", "lm_head.weight"}
            and merged[key].shape[1:] == tensor.shape[1:]
        ):
            padded = merged[key].clone()
            padded[: tensor.shape[0]] = tensor
            merged[key] = padded
        else:
            raise ValueError(
                f"shape mismatch for {key}: native {tuple(merged[key].shape)} vs HF {tuple(tensor.shape)}"
            )
    missing, unexpected = target.load_state_dict(merged, strict=False)
    missing = [m for m in missing if not m.startswith("rotary.")]
    if missing:
        raise KeyError(f"missing HF weights for native keys: {missing[:8]}")
    if unexpected:
        raise KeyError(f"unexpected native keys: {unexpected[:8]}")


def _qwen35_remote_candidates(remote_key: str) -> list[str]:
    return [
        f"model.language_model.{remote_key}",
        f"language_model.{remote_key}",
        remote_key,
    ]


def _map_qwen35_shared_hf_state(
    target, hf_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    state = target.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    direct_pairs = {
        "embed_tokens.weight": [
            "model.language_model.model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ],
        "norm.weight": [
            "model.language_model.model.norm.weight",
            "language_model.model.norm.weight",
            "model.norm.weight",
            "norm.weight",
        ],
        "lm_head.weight": [
            "model.language_model.lm_head.weight",
            "language_model.lm_head.weight",
            "lm_head.weight",
        ],
    }
    for local, remotes in direct_pairs.items():
        if local in state:
            key = _first_existing(hf_state, remotes)
            if key is not None:
                mapped[local] = hf_state[key]
    for i, layer in enumerate(target.layers):
        prefix = f"layers.{i}"
        remote = f"model.layers.{i}"
        pairs = {
            f"{prefix}.input_layernorm.weight": f"{remote}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight": f"{remote}.post_attention_layernorm.weight",
        }
        if layer.layer_type == "full_attention":
            pairs.update(
                {
                    f"{prefix}.self_attn.q_proj.weight": f"{remote}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight": f"{remote}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight": f"{remote}.self_attn.v_proj.weight",
                    f"{prefix}.self_attn.o_proj.weight": f"{remote}.self_attn.o_proj.weight",
                    f"{prefix}.self_attn.q_norm.weight": f"{remote}.self_attn.q_norm.weight",
                    f"{prefix}.self_attn.k_norm.weight": f"{remote}.self_attn.k_norm.weight",
                }
            )
        else:
            pairs.update(
                {
                    f"{prefix}.linear_attn.conv1d.weight": f"{remote}.linear_attn.conv1d.weight",
                    f"{prefix}.linear_attn.dynamics.dt_bias": f"{remote}.linear_attn.dt_bias",
                    f"{prefix}.linear_attn.dynamics.A_log": f"{remote}.linear_attn.A_log",
                    f"{prefix}.linear_attn.norm.weight": f"{remote}.linear_attn.norm.weight",
                    f"{prefix}.linear_attn.out_proj.weight": f"{remote}.linear_attn.out_proj.weight",
                    f"{prefix}.linear_attn.in_proj_qkv.weight": f"{remote}.linear_attn.in_proj_qkv.weight",
                    f"{prefix}.linear_attn.in_proj_z.weight": f"{remote}.linear_attn.in_proj_z.weight",
                    f"{prefix}.linear_attn.in_proj_b.weight": f"{remote}.linear_attn.in_proj_b.weight",
                    f"{prefix}.linear_attn.in_proj_a.weight": f"{remote}.linear_attn.in_proj_a.weight",
                }
            )
        for local, remote_key in pairs.items():
            key = _first_existing(
                hf_state, _qwen35_remote_candidates(remote_key)
            )
            if key is not None:
                mapped[local] = hf_state[key]
    return mapped


def _load_mapped_qwen35_state(
    target, mapped: dict[str, torch.Tensor], missing_message: str
) -> None:
    state = target.state_dict()
    optional = {"lm_head.weight"}
    missing = sorted(set(state) - set(mapped) - optional)
    if missing:
        raise KeyError(f"{missing_message}: {missing[:8]}")
    loadable = {}
    for key, tensor in {**state, **mapped}.items():
        expected = state[key]
        if expected.shape == tensor.shape:
            loadable[key] = tensor.to(dtype=expected.dtype)
            continue
        if (
            key in {"embed_tokens.weight", "lm_head.weight"}
            and expected.ndim == tensor.ndim == 2
            and expected.shape[1:] == tensor.shape[1:]
        ):
            padded = expected.clone()
            padded[: tensor.shape[0]] = tensor.to(dtype=expected.dtype)
            loadable[key] = padded
            continue
        raise ValueError(
            f"shape mismatch for {key}: native {tuple(expected.shape)} vs HF {tuple(tensor.shape)}"
        )
    target.load_state_dict(loadable, strict=True)


def _load_qwen35_hf_state(target, hf_state: dict[str, torch.Tensor]) -> None:
    mapped = _map_qwen35_shared_hf_state(target, hf_state)
    for i in range(target.config.num_layers):
        prefix = f"layers.{i}"
        remote = f"model.layers.{i}.mlp"
        for local, remote_key in {
            f"{prefix}.mlp.gate_proj.weight": f"{remote}.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight": f"{remote}.up_proj.weight",
            f"{prefix}.mlp.down_proj.weight": f"{remote}.down_proj.weight",
        }.items():
            key = _first_existing(
                hf_state, _qwen35_remote_candidates(remote_key)
            )
            if key is not None:
                mapped[local] = hf_state[key]
    _load_mapped_qwen35_state(
        target, mapped, "missing Qwen3.5 HF text weights for native keys"
    )


def _load_qwen35_moe_hf_state(
    target, hf_state: dict[str, torch.Tensor]
) -> None:
    mapped = _map_qwen35_shared_hf_state(target, hf_state)
    for i in range(target.config.num_layers):
        prefix = f"layers.{i}"
        remote = f"model.layers.{i}.mlp"
        for local, remote_key in {
            f"{prefix}.mlp.gate.weight": f"{remote}.gate.weight",
            f"{prefix}.mlp.experts.gate_up_proj": f"{remote}.experts.gate_up_proj",
            f"{prefix}.mlp.experts.down_proj": f"{remote}.experts.down_proj",
            f"{prefix}.mlp.shared_expert.gate_proj.weight": f"{remote}.shared_expert.gate_proj.weight",
            f"{prefix}.mlp.shared_expert.up_proj.weight": f"{remote}.shared_expert.up_proj.weight",
            f"{prefix}.mlp.shared_expert.down_proj.weight": f"{remote}.shared_expert.down_proj.weight",
            f"{prefix}.mlp.shared_expert_gate.weight": f"{remote}.shared_expert_gate.weight",
        }.items():
            key = _first_existing(
                hf_state, _qwen35_remote_candidates(remote_key)
            )
            if key is not None:
                mapped[local] = hf_state[key]
    _load_mapped_qwen35_state(
        target, mapped, "missing Qwen3.5-MoE HF text weights for native keys"
    )


def _gemma4_remote_candidates(remote_key: str) -> list[str]:
    return [
        f"model.language_model.{remote_key}",
        f"language_model.{remote_key}",
        remote_key,
    ]


def _load_gemma4_hf_state(target, hf_state: dict[str, torch.Tensor]) -> None:
    state = target.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    direct_pairs = {
        "embed_tokens.weight": [
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ],
        "norm.weight": ["model.norm.weight", "norm.weight"],
        "lm_head.weight": ["lm_head.weight"],
        "embed_tokens_per_layer.weight": [
            "model.embed_tokens_per_layer.weight",
            "embed_tokens_per_layer.weight",
        ],
        "per_layer_model_projection.weight": [
            "model.per_layer_model_projection.weight",
            "per_layer_model_projection.weight",
        ],
        "per_layer_projection_norm.weight": [
            "model.per_layer_projection_norm.weight",
            "per_layer_projection_norm.weight",
        ],
    }
    for local, remotes in direct_pairs.items():
        if local not in state:
            continue
        key = _first_existing(
            hf_state,
            [
                candidate
                for remote in remotes
                for candidate in _gemma4_remote_candidates(remote)
            ],
        )
        if key is not None:
            mapped[local] = hf_state[key]
    for i, layer in enumerate(target.layers):
        prefix = f"layers.{i}"
        remote = f"model.layers.{i}"
        pairs = {
            f"{prefix}.layer_scalar": f"{remote}.layer_scalar",
            f"{prefix}.input_layernorm.weight": f"{remote}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight": f"{remote}.post_attention_layernorm.weight",
            f"{prefix}.pre_feedforward_layernorm.weight": f"{remote}.pre_feedforward_layernorm.weight",
            f"{prefix}.post_feedforward_layernorm.weight": f"{remote}.post_feedforward_layernorm.weight",
            f"{prefix}.self_attn.q_proj.weight": f"{remote}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.q_norm.weight": f"{remote}.self_attn.q_norm.weight",
            f"{prefix}.self_attn.o_proj.weight": f"{remote}.self_attn.o_proj.weight",
            f"{prefix}.mlp.gate_proj.weight": f"{remote}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight": f"{remote}.mlp.up_proj.weight",
            f"{prefix}.mlp.down_proj.weight": f"{remote}.mlp.down_proj.weight",
        }
        if not layer.self_attn.is_kv_shared_layer:
            pairs[f"{prefix}.self_attn.k_proj.weight"] = (
                f"{remote}.self_attn.k_proj.weight"
            )
            pairs[f"{prefix}.self_attn.k_norm.weight"] = (
                f"{remote}.self_attn.k_norm.weight"
            )
            if f"{prefix}.self_attn.v_proj.weight" in state:
                pairs[f"{prefix}.self_attn.v_proj.weight"] = (
                    f"{remote}.self_attn.v_proj.weight"
                )
        if target.config.hidden_size_per_layer_input:
            pairs.update(
                {
                    f"{prefix}.per_layer_input_gate.weight": f"{remote}.per_layer_input_gate.weight",
                    f"{prefix}.per_layer_projection.weight": f"{remote}.per_layer_projection.weight",
                    f"{prefix}.post_per_layer_input_norm.weight": f"{remote}.post_per_layer_input_norm.weight",
                }
            )
        for local, remote_key in pairs.items():
            if local not in state:
                continue
            key = _first_existing(
                hf_state, _gemma4_remote_candidates(remote_key)
            )
            if key is not None:
                mapped[local] = hf_state[key]
    optional = {
        "lm_head.weight",
        *{f"layers.{i}.layer_scalar" for i in range(len(target.layers))},
    }
    missing = sorted(set(state) - set(mapped) - optional)
    if missing:
        raise KeyError(
            f"missing Gemma4 HF text weights for native keys: {missing[:8]}"
        )
    loadable = {}
    for key, tensor in {**state, **mapped}.items():
        expected = state[key]
        if expected.shape == tensor.shape:
            loadable[key] = tensor.to(dtype=expected.dtype)
            continue
        if (
            key
            in {
                "embed_tokens.weight",
                "lm_head.weight",
                "embed_tokens_per_layer.weight",
            }
            and expected.ndim == tensor.ndim == 2
            and expected.shape[1:] == tensor.shape[1:]
            and expected.shape[0] >= tensor.shape[0]
        ):
            padded = expected.clone()
            padded[: tensor.shape[0]] = tensor.to(dtype=expected.dtype)
            loadable[key] = padded
            continue
        raise ValueError(
            f"shape mismatch for {key}: native {tuple(expected.shape)} vs HF {tuple(tensor.shape)}"
        )
    target.load_state_dict(loadable, strict=True)


def _first_existing(
    state: dict[str, torch.Tensor], keys: list[str]
) -> str | None:
    for key in keys:
        if key in state:
            return key
    return None


def load_ppt_state(native, ppt_state: dict[str, torch.Tensor]) -> None:
    target = native.backbone if hasattr(native, "backbone") else native
    state = target.state_dict()
    ppt_state = {k.removeprefix("_orig_mod."): v for k, v in ppt_state.items()}
    mapped = {
        "embed_tokens.weight": ppt_state["transformer.wte.weight"],
        "norm.weight": ppt_state["transformer.ln_f.weight"],
    }
    if not target.config.tie_embeddings:
        mapped["lm_head.weight"] = ppt_state["lm_head.weight"]
    for i in range(target.config.num_layers):
        prefix = f"layers.{i}"
        remote = f"transformer.h.{i}"
        mapped.update(
            {
                f"{prefix}.input_layernorm.weight": ppt_state[
                    f"{remote}.input_layernorm.weight"
                ],
                f"{prefix}.post_attention_layernorm.weight": ppt_state[
                    f"{remote}.post_attention_layernorm.weight"
                ],
                f"{prefix}.self_attn.q_proj.weight": ppt_state[
                    f"{remote}.attn.q_proj.weight"
                ],
                f"{prefix}.self_attn.k_proj.weight": ppt_state[
                    f"{remote}.attn.k_proj.weight"
                ],
                f"{prefix}.self_attn.v_proj.weight": ppt_state[
                    f"{remote}.attn.v_proj.weight"
                ],
                f"{prefix}.self_attn.o_proj.weight": ppt_state[
                    f"{remote}.attn.o_proj.weight"
                ],
                f"{prefix}.mlp.gate_proj.weight": ppt_state[
                    f"{remote}.mlp.gate_proj.weight"
                ],
                f"{prefix}.mlp.up_proj.weight": ppt_state[
                    f"{remote}.mlp.up_proj.weight"
                ],
                f"{prefix}.mlp.down_proj.weight": ppt_state[
                    f"{remote}.mlp.down_proj.weight"
                ],
            }
        )
    missing = sorted(set(state) - set(mapped))
    extra = sorted(set(mapped) - set(state))
    if missing or extra:
        raise KeyError(
            f"PPT mapping mismatch missing={missing[:8]} extra={extra[:8]}"
        )
    loadable = {}
    for key, tensor in mapped.items():
        tensor = tensor.to(dtype=state[key].dtype)
        if state[key].shape == tensor.shape:
            loadable[key] = tensor
        elif (
            key in {"embed_tokens.weight", "lm_head.weight"}
            and state[key].shape[1:] == tensor.shape[1:]
        ):
            padded = state[key].clone()
            padded[: tensor.shape[0]] = tensor
            loadable[key] = padded
        else:
            raise ValueError(
                f"shape mismatch for {key}: native {tuple(state[key].shape)} vs PPT {tuple(tensor.shape)}"
            )
    target.load_state_dict(loadable, strict=True)
