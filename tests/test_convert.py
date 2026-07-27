import pytest
import torch
from types import SimpleNamespace
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from gcw.checkpoint import load_backbone_checkpoint, load_checkpoint, save_checkpoint
from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.convert import (
    config_from_bottleneck,
    config_from_hf,
    config_from_ppt_meta,
    convert_bottleneck,
    load_bottleneck_state,
    load_hf_state,
    load_ppt_state,
)
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


def test_untied_backbone_loads_lm_head_from_tied_embedding_checkpoint(tmp_path):
    tied = ModelConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        sequence_len=8,
        tie_embeddings=True,
    )
    source = build_model(tied)
    source.init_weights()
    save_checkpoint(tmp_path / "tied", source, tied)

    cp = CacheProcessorConfig(hidden_size=8, intermediate_size=16, num_heads=1, num_kv_heads=1)
    target = build_model(
        ModelConfig(
            variant="bgt",
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=4,
            sequence_len=8,
            tie_embeddings=False,
            cache_processor=cp,
        )
    )

    load_backbone_checkpoint(target, tmp_path / "tied")

    assert torch.equal(target.backbone.lm_head.weight, source.embed_tokens.weight)


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


def test_hf_config_can_build_cache_processor_variants():
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
    for variant in ("icarus", "longshot"):
        cfg = config_from_hf(hf_cfg, family="llama", variant=variant)
        cfg.cache_processor = CacheProcessorConfig()
        model = build_model(cfg)
        assert model.config.variant == variant


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


def test_tiny_qwen35_moe_logits_match_transformers():
    pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    hf_cfg = Qwen3_5MoeTextConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=16,
        tie_word_embeddings=False,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 0.5},
        partial_rotary_factor=0.5,
        layer_types=["full_attention", "linear_attention"],
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        output_router_logits=False,
    )
    torch.manual_seed(0)
    hf = Qwen3_5MoeForCausalLM(hf_cfg).eval()
    native = build_model(config_from_hf(hf_cfg, family="qwen3_5_moe")).eval()
    load_hf_state(native, hf.state_dict(), family="qwen3_5_moe")
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        hf_logits = hf(x, use_cache=False).logits
        native_logits = native(x)
    assert torch.allclose(native_logits, hf_logits.float(), atol=1e-5, rtol=1e-5)


def test_qwen35_moe_hf_text_config_uses_full_attention_processor_layers():
    text = SimpleNamespace(
        num_attention_heads=2,
        hidden_size=16,
        head_dim=8,
        num_key_value_heads=1,
        max_position_embeddings=32,
        vocab_size=128,
        num_hidden_layers=4,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        rope_parameters={"rope_theta": 10000000.0, "partial_rotary_factor": 0.5},
        partial_rotary_factor=0.5,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=8,
        hidden_act="silu",
        router_aux_loss_coef=0.001,
        output_router_logits=False,
    )
    cfg = config_from_hf(SimpleNamespace(text_config=text), family="qwen3_5_moe")

    assert cfg.family == "qwen3_5_moe"
    assert cfg.layer_types == text.layer_types
    assert cfg.processor_layer_indices == [1, 3]
    assert cfg.cache_num_heads == 2
    assert cfg.hidden_act == "silu"


def test_qwen35_moe_hf_mapping_loads_multimodal_language_model_prefix():
    model = build_model(_tiny_qwen35_moe_config())
    hf = _prefixed_state_like(model.state_dict(), "language_model.")

    load_hf_state(model, hf, family="qwen3_5_moe")

    assert torch.equal(model.embed_tokens.weight, hf["language_model.model.embed_tokens.weight"])
    assert torch.equal(
        model.layers[1].self_attn.q_proj.weight,
        hf["language_model.model.layers.1.self_attn.q_proj.weight"],
    )
    assert torch.equal(
        model.layers[0].linear_attn.in_proj_qkv.weight,
        hf["language_model.model.layers.0.linear_attn.in_proj_qkv.weight"],
    )


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


def test_bottleneck_mapping_pads_legacy_vocab_rows():
    cfg = _tiny_bottleneck_config("bgt")
    cfg.vocab_size = 129
    model = build_model(cfg)
    old = _legacy_state_from_native(model.state_dict(), "bgt")
    old["backbone.model.embed_tokens.weight"] = torch.randn(129, cfg.hidden_size)
    old["backbone.lm_head.weight"] = torch.randn(129, cfg.hidden_size)

    load_bottleneck_state(model, old, variant="bgt")
    state = model.state_dict()

    assert torch.equal(state["backbone.embed_tokens.weight"][:129], old["backbone.model.embed_tokens.weight"])
    assert torch.equal(state["backbone.lm_head.weight"][:129], old["backbone.lm_head.weight"])


def test_bottleneck_mapping_ignores_duplicate_lm_head_for_tied_embeddings():
    cfg = _tiny_bottleneck_config("bgt")
    cfg.tie_embeddings = True
    model = build_model(cfg)
    old = _legacy_state_from_native(model.state_dict(), "bgt")
    old["backbone.lm_head.weight"] = torch.randn(cfg.vocab_size, cfg.hidden_size)

    load_bottleneck_state(model, old, variant="bgt")

    assert "backbone.lm_head.weight" not in model.state_dict()


def test_bottleneck_specs_cover_paper_backbones():
    llama = config_from_bottleneck("llama-3.2-1b", "bgt")
    assert llama.sequence_len == 8192
    assert llama.cache_processor.hidden_size == 1536
    assert llama.cache_processor.intermediate_size == 6272
    assert llama.legacy_bottleneck_generate is False

    qwen = config_from_bottleneck("qwen3-0.6b", "bgt")
    assert qwen.family == "qwen3"
    assert qwen.qk_norm
    assert qwen.vocab_size == 151936
    assert qwen.cache_processor.hidden_size == 1344
    assert qwen.cache_processor.intermediate_size == 5632

    qwen_legacy = config_from_bottleneck("qwen3-0.6b", "bgt", legacy_vocab=True)
    assert qwen_legacy.vocab_size == 151669

    bt = config_from_bottleneck("llama-3.1-8b", "bt", sequence_len=4096, legacy_generate=True)
    assert bt.sequence_len == 4096
    assert bt.legacy_bottleneck_generate
    assert bt.cache_processor.hidden_size == 512
    assert bt.cache_processor.intermediate_size == 2240


def test_convert_bottleneck_prefers_local_tokenizer_source(monkeypatch, tmp_path):
    import gcw.convert as convert

    cfg = _tiny_bottleneck_config("bgt")
    cfg.sequence_len = 8192
    cfg.legacy_bottleneck_generate = True
    model = build_model(cfg)
    model.init_weights()
    old = _legacy_state_from_native(model.state_dict(), "bgt")
    monkeypatch.setattr(convert, "config_from_bottleneck", lambda *args, **kwargs: cfg)

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    from safetensors.torch import save_file

    save_file(old, legacy_dir / "model.safetensors")
    (legacy_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    out = tmp_path / "converted"
    convert_bottleneck(
        legacy_dir,
        out,
        variant="bgt",
        backbone="llama-3.2-1b",
        sequence_len=8192,
        legacy_generate=True,
    )

    _, converted_cfg, state = load_checkpoint(out)
    assert converted_cfg.sequence_len == 8192
    assert converted_cfg.legacy_bottleneck_generate
    assert state["source"] == str(legacy_dir)
    assert state["backbone_source"] == "meta-llama/Llama-3.2-1B"
    assert state["sequence_len"] == 8192


def test_convert_bottleneck_allows_public_tokenizer_source(monkeypatch, tmp_path):
    import gcw.convert as convert

    cfg = _tiny_bottleneck_config("bt")
    cfg.sequence_len = 8192
    cfg.legacy_bottleneck_generate = True
    model = build_model(cfg)
    model.init_weights()
    old = _legacy_state_from_native(model.state_dict(), "bt")
    monkeypatch.setattr(convert, "config_from_bottleneck", lambda *args, **kwargs: cfg)

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    from safetensors.torch import save_file

    save_file(old, legacy_dir / "model.safetensors")
    (legacy_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    out = tmp_path / "converted"
    convert_bottleneck(
        legacy_dir,
        out,
        variant="bt",
        backbone="llama-3.2-1b",
        legacy_generate=True,
        tokenizer_source="/mnt/jfs-01/shared/bottleneck_llm/sft/llama-3.2-1b/final_model",
    )

    _, _, state = load_checkpoint(out)
    assert state["source"] == "/mnt/jfs-01/shared/bottleneck_llm/sft/llama-3.2-1b/final_model"


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


def _tiny_qwen35_moe_config() -> ModelConfig:
    return ModelConfig(
        variant="tf",
        family="qwen3_5_moe",
        vocab_size=128,
        hidden_size=16,
        intermediate_size=8,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=32,
        tie_embeddings=False,
        qk_norm=True,
        rotary_style="llama",
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        processor_layer_indices=[1, 3],
        partial_rotary_factor=0.5,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=8,
    )


def _prefixed_state_like(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        if key == "embed_tokens.weight":
            remote = f"{prefix}model.embed_tokens.weight"
        elif key == "norm.weight":
            remote = f"{prefix}model.norm.weight"
        elif key == "lm_head.weight":
            remote = f"{prefix}lm_head.weight"
        else:
            remote = f"{prefix}model.{key}"
        if value.is_floating_point():
            out[remote] = torch.randn_like(value)
        else:
            out[remote] = torch.zeros_like(value)
    return out


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
