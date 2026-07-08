import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.convert import config_from_hf, config_from_ppt_meta, load_bottleneck_state, load_hf_state, load_ppt_state
from gcw.models import build_model


def test_hf_llama_key_mapping_loads_native_backbone():
    cfg = ModelConfig(vocab_size=16, hidden_size=8, intermediate_size=16, num_layers=1, num_heads=2, num_kv_heads=1, head_dim=4, sequence_len=8, tie_embeddings=False)
    model = build_model(cfg)
    hf = {
        "model.embed_tokens.weight": torch.randn(16, 8),
        "model.norm.weight": torch.ones(8),
        "lm_head.weight": torch.randn(16, 8),
        "model.layers.0.input_layernorm.weight": torch.ones(8),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(8),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(4, 8),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(4, 8),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8),
        "model.layers.0.mlp.up_proj.weight": torch.randn(16, 8),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 16),
    }
    load_hf_state(model, hf, family="llama")
    assert torch.equal(model.embed_tokens.weight[:16], hf["model.embed_tokens.weight"])


def test_tiny_llama_logits_match_transformers():
    hf_cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        tie_word_embeddings=False,
        rope_theta=10000.0,
    )
    torch.manual_seed(0)
    hf = LlamaForCausalLM(hf_cfg).eval()
    native = build_model(config_from_hf(hf_cfg, family="llama")).eval()
    load_hf_state(native, hf.state_dict(), family="llama")
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        hf_logits = hf(x).logits
        native_logits = native(x)
    assert torch.allclose(native_logits, hf_logits, atol=1e-5, rtol=1e-5)


def test_tiny_qwen3_logits_match_transformers():
    hf_cfg = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=16,
        tie_word_embeddings=False,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )
    torch.manual_seed(0)
    hf = Qwen3ForCausalLM(hf_cfg).eval()
    native = build_model(config_from_hf(hf_cfg, family="qwen3")).eval()
    load_hf_state(native, hf.state_dict(), family="qwen3")
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        hf_logits = hf(x).logits
        native_logits = native(x)
    assert torch.allclose(native_logits, hf_logits, atol=1e-5, rtol=1e-5)


def test_ppt_state_mapping_loads_native_smollm():
    meta = {
        "model_config": {
            "sequence_len": 8,
            "vocab_size": 16,
            "n_layer": 1,
            "n_head": 2,
            "n_kv_head": 1,
            "head_dim": 4,
            "n_embd": 8,
            "intermediate_size": 16,
            "tie_embeddings": True,
        }
    }
    model = build_model(config_from_ppt_meta(meta))
    ppt = {
        "transformer.wte.weight": torch.randn(16, 8),
        "transformer.ln_f.weight": torch.ones(8),
        "transformer.h.0.input_layernorm.weight": torch.ones(8),
        "transformer.h.0.post_attention_layernorm.weight": torch.ones(8),
        "transformer.h.0.attn.q_proj.weight": torch.randn(8, 8),
        "transformer.h.0.attn.k_proj.weight": torch.randn(4, 8),
        "transformer.h.0.attn.v_proj.weight": torch.randn(4, 8),
        "transformer.h.0.attn.o_proj.weight": torch.randn(8, 8),
        "transformer.h.0.mlp.gate_proj.weight": torch.randn(16, 8),
        "transformer.h.0.mlp.up_proj.weight": torch.randn(16, 8),
        "transformer.h.0.mlp.down_proj.weight": torch.randn(8, 16),
    }
    load_ppt_state(model, ppt)
    assert torch.equal(model.embed_tokens.weight[:16], ppt["transformer.wte.weight"])
    assert model.config.rotary_style == "ppt"


def test_bottleneck_bt_mapping_loads_legacy_checkpoint_keys():
    cfg = _tiny_bottleneck_config("bt")
    model = build_model(cfg)
    old = _legacy_state_from_native(model.state_dict(), "bt")
    load_bottleneck_state(model, old, variant="bt")
    state = model.state_dict()
    assert torch.equal(state["backbone.embed_tokens.weight"], old["backbone.model.embed_tokens.weight"])
    assert torch.equal(
        state["cache_processor.blocks.0.layers.0.attn.q_proj.bias"],
        old["cache_processor.processor_blocks.0.transformer.self_attn.q_proj.bias"],
    )


def test_bottleneck_bgt_mapping_loads_legacy_checkpoint_keys():
    cfg = _tiny_bottleneck_config("bgt")
    model = build_model(cfg)
    old = _legacy_state_from_native(model.state_dict(), "bgt")
    load_bottleneck_state(model, old, variant="bgt")
    state = model.state_dict()
    assert torch.equal(state["backbone.lm_head.weight"], old["backbone.lm_head.weight"])
    assert torch.equal(
        state["cache_processor.block.layers.0.gate_proj.weight"],
        old["cache_processor.unified_block.transformer.0.mlp.gate_proj.weight"],
    )


def _tiny_bottleneck_config(variant: str) -> ModelConfig:
    return ModelConfig(
        variant=variant,
        family="llama",
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        sequence_len=8,
        tie_embeddings=False,
        rotary_style="llama",
        cache_processor=CacheProcessorConfig(
            hidden_size=4,
            intermediate_size=8,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
        ),
    )


def _legacy_state_from_native(native_state: dict[str, torch.Tensor], variant: str) -> dict[str, torch.Tensor]:
    old = {}
    for idx, (key, tensor) in enumerate(native_state.items()):
        old_key = _legacy_key(key, variant)
        value = torch.full(tensor.shape, (idx + 1) / 1000, dtype=tensor.dtype)
        old[old_key] = value
    return old


def _legacy_key(key: str, variant: str) -> str:
    if key.startswith("backbone.embed_tokens."):
        return "backbone.model.embed_tokens." + key.removeprefix("backbone.embed_tokens.")
    if key.startswith("backbone.layers."):
        return "backbone.model.layers." + key.removeprefix("backbone.layers.")
    if key.startswith("backbone.norm."):
        return "backbone.model.norm." + key.removeprefix("backbone.norm.")
    if key.startswith("backbone.lm_head."):
        return key
    if variant == "bt" and key.startswith("cache_processor.blocks."):
        rest = key.removeprefix("cache_processor.blocks.")
        layer, sub = rest.split(".", 1)
        return f"cache_processor.processor_blocks.{layer}.{_legacy_processor_subkey(sub, 'transformer.')}"
    if variant == "bgt" and key.startswith("cache_processor.block."):
        sub = key.removeprefix("cache_processor.block.")
        return "cache_processor.unified_block." + _legacy_processor_subkey(sub, "transformer.0.")
    raise KeyError(key)


def _legacy_processor_subkey(sub: str, transformer_prefix: str) -> str:
    if sub == "gate":
        return "g"
    direct = {
        "in_norm.": "kv_in_norm.",
        "in_proj.": "kv_in_proj.",
        "out_proj.": "kv_out_proj.",
        "out_norm.": "out_norm.",
    }
    for new, old in direct.items():
        if sub.startswith(new):
            return old + sub.removeprefix(new)
    blocks = {
        "layers.0.input_norm.": transformer_prefix + "input_layernorm.",
        "layers.0.post_norm.": transformer_prefix + "post_attention_layernorm.",
        "layers.0.attn.": transformer_prefix + "self_attn.",
        "layers.0.gate_proj.": transformer_prefix + "mlp.gate_proj.",
        "layers.0.up_proj.": transformer_prefix + "mlp.up_proj.",
        "layers.0.down_proj.": transformer_prefix + "mlp.down_proj.",
    }
    for new, old in blocks.items():
        if sub.startswith(new):
            return old + sub.removeprefix(new)
    raise KeyError(sub)
