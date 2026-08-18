from __future__ import annotations

import torch


def _first_existing(state: dict[str, torch.Tensor], keys: list[str]) -> str | None:
    for key in keys:
        if key in state:
            return key
    return None


def load_hf_state(native, hf_state: dict[str, torch.Tensor], family: str = "llama") -> None:
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
            names[f"{prefix}.self_attn.q_norm.weight"] = f"{remote}.self_attn.q_norm.weight"
            names[f"{prefix}.self_attn.k_norm.weight"] = f"{remote}.self_attn.k_norm.weight"
        for local, key in names.items():
            if local in state and key in hf_state:
                mapped[local] = hf_state[key]
    merged = dict(state)
    for key, tensor in mapped.items():
        if merged[key].shape == tensor.shape:
            merged[key] = tensor
        elif key in {"embed_tokens.weight", "lm_head.weight"} and merged[key].shape[1:] == tensor.shape[1:]:
            padded = merged[key].clone()
            padded[: tensor.shape[0]] = tensor
            merged[key] = padded
        else:
            raise ValueError(f"shape mismatch for {key}: native {tuple(merged[key].shape)} vs HF {tuple(tensor.shape)}")
    missing, unexpected = target.load_state_dict(merged, strict=False)
    missing = [m for m in missing if not m.startswith("rotary.")]
    if missing:
        raise KeyError(f"missing HF weights for native keys: {missing[:8]}")
    if unexpected:
        raise KeyError(f"unexpected native keys: {unexpected[:8]}")


def _qwen35_remote_candidates(remote_key: str) -> list[str]:
    return [f"model.language_model.{remote_key}", f"language_model.{remote_key}", remote_key]


def _map_qwen35_shared_hf_state(target, hf_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
        "lm_head.weight": ["model.language_model.lm_head.weight", "language_model.lm_head.weight", "lm_head.weight"],
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
            key = _first_existing(hf_state, _qwen35_remote_candidates(remote_key))
            if key is not None:
                mapped[local] = hf_state[key]
    return mapped


def _load_mapped_qwen35_state(target, mapped: dict[str, torch.Tensor], missing_message: str) -> None:
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
        if key in {"embed_tokens.weight", "lm_head.weight"} and expected.ndim == tensor.ndim == 2 and expected.shape[1:] == tensor.shape[1:]:
            padded = expected.clone()
            padded[: tensor.shape[0]] = tensor.to(dtype=expected.dtype)
            loadable[key] = padded
            continue
        raise ValueError(f"shape mismatch for {key}: native {tuple(expected.shape)} vs HF {tuple(tensor.shape)}")
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
            key = _first_existing(hf_state, _qwen35_remote_candidates(remote_key))
            if key is not None:
                mapped[local] = hf_state[key]
    _load_mapped_qwen35_state(target, mapped, "missing Qwen3.5 HF text weights for native keys")


def _load_qwen35_moe_hf_state(target, hf_state: dict[str, torch.Tensor]) -> None:
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
            key = _first_existing(hf_state, _qwen35_remote_candidates(remote_key))
            if key is not None:
                mapped[local] = hf_state[key]
    _load_mapped_qwen35_state(target, mapped, "missing Qwen3.5-MoE HF text weights for native keys")


def _gemma4_remote_candidates(remote_key: str) -> list[str]:
    return [f"model.language_model.{remote_key}", f"language_model.{remote_key}", remote_key]


def _load_gemma4_hf_state(target, hf_state: dict[str, torch.Tensor]) -> None:
    state = target.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    direct_pairs = {
        "embed_tokens.weight": ["model.embed_tokens.weight", "embed_tokens.weight"],
        "norm.weight": ["model.norm.weight", "norm.weight"],
        "lm_head.weight": ["lm_head.weight"],
        "embed_tokens_per_layer.weight": ["model.embed_tokens_per_layer.weight", "embed_tokens_per_layer.weight"],
        "per_layer_model_projection.weight": [
            "model.per_layer_model_projection.weight",
            "per_layer_model_projection.weight",
        ],
        "per_layer_projection_norm.weight": ["model.per_layer_projection_norm.weight", "per_layer_projection_norm.weight"],
    }
    for local, remotes in direct_pairs.items():
        if local not in state:
            continue
        key = _first_existing(hf_state, [candidate for remote in remotes for candidate in _gemma4_remote_candidates(remote)])
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
            pairs[f"{prefix}.self_attn.k_proj.weight"] = f"{remote}.self_attn.k_proj.weight"
            pairs[f"{prefix}.self_attn.k_norm.weight"] = f"{remote}.self_attn.k_norm.weight"
            if f"{prefix}.self_attn.v_proj.weight" in state:
                pairs[f"{prefix}.self_attn.v_proj.weight"] = f"{remote}.self_attn.v_proj.weight"
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
            key = _first_existing(hf_state, _gemma4_remote_candidates(remote_key))
            if key is not None:
                mapped[local] = hf_state[key]
    optional = {"lm_head.weight", *{f"layers.{i}.layer_scalar" for i in range(len(target.layers))}}
    missing = sorted(set(state) - set(mapped) - optional)
    if missing:
        raise KeyError(f"missing Gemma4 HF text weights for native keys: {missing[:8]}")
    loadable = {}
    for key, tensor in {**state, **mapped}.items():
        expected = state[key]
        if expected.shape == tensor.shape:
            loadable[key] = tensor.to(dtype=expected.dtype)
            continue
        if (
            key in {"embed_tokens.weight", "lm_head.weight", "embed_tokens_per_layer.weight"}
            and expected.ndim == tensor.ndim == 2
            and expected.shape[1:] == tensor.shape[1:]
            and expected.shape[0] >= tensor.shape[0]
        ):
            padded = expected.clone()
            padded[: tensor.shape[0]] = tensor.to(dtype=expected.dtype)
            loadable[key] = padded
            continue
        raise ValueError(f"shape mismatch for {key}: native {tuple(expected.shape)} vs HF {tuple(tensor.shape)}")
    target.load_state_dict(loadable, strict=True)
