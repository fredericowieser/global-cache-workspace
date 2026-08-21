import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from gcw.checkpoint import load_backbone_checkpoint, save_checkpoint
from gcw.config import (
    CacheProcessorConfig,
    ModelConfig,
    gemma4_default_layer_types,
)
from gcw.convert import (
    config_from_hf,
    load_hf_state,
)
from gcw.models import build_model
from gcw.provenance import file_sha256
from scripts.analysis import diagnose_hf_conversion as diagnosis
from scripts.analysis import verify_hf_conversion as parity
from scripts.analysis.verify_hf_conversion import _atomic_write_new


def test_conversion_parity_evidence_is_atomic_and_no_clobber(tmp_path):
    output = tmp_path / "parity.json"
    payload = {"passed": True, "max_abs_error": 0.01}

    _atomic_write_new(output, payload)

    assert __import__("json").loads(output.read_text()) == payload
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _atomic_write_new(output, payload)


def test_conversion_parity_policy_rejects_tampering(tmp_path, monkeypatch):
    hf_path = tmp_path / "hf"
    checkpoint = tmp_path / "checkpoint"
    hf_path.mkdir()
    checkpoint.mkdir()
    source_revision = "a" * 40
    token_ids = [1, 2, 3]
    hf_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    model_config = {
        "variant": "tf",
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "tie_embeddings": True,
    }
    source_state = {"dtype": "bfloat16", "source": str(hf_path.resolve())}
    for name in parity.HF_PARITY_FILES:
        (hf_path / name).write_bytes(f"hf-{name}".encode())
    (hf_path / "config.json").write_text(__import__("json").dumps(hf_config))
    (checkpoint / "CONVERSION.json").write_text("{}")
    (checkpoint / "config.json").write_text(
        __import__("json").dumps(model_config)
    )
    (checkpoint / "model.safetensors").write_bytes(b"native-model")
    (checkpoint / "train_state.json").write_text(
        __import__("json").dumps(source_state)
    )
    monkeypatch.setattr(parity.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(parity.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        parity.torch.cuda,
        "get_device_name",
        lambda device: "NVIDIA H100 80GB HBM3",
    )

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def encode(self, prompt, add_special_tokens):
            assert prompt == parity.CANONICAL_PROMPT
            assert add_special_tokens is True
            return token_ids

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizer),
    )
    payload = {
        "schema_version": parity.SCHEMA_VERSION,
        "kind": parity.EVIDENCE_KIND,
        "policy_id": parity.POLICY_ID,
        "source_revision": source_revision,
        "script_sha256": file_sha256(Path(parity.__file__).resolve()),
        "hf_path": str(hf_path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "canonical_prompt_sha256": __import__("hashlib")
        .sha256(parity.CANONICAL_PROMPT.encode())
        .hexdigest(),
        "max_abs_threshold": parity.MAX_ABS_THRESHOLD,
        "mean_abs_threshold": parity.MEAN_ABS_THRESHOLD,
        "required_dtype": parity.REQUIRED_DTYPE,
        "passed": True,
        "hf_model_sha256": file_sha256(hf_path / "model.safetensors"),
        "checkpoint_model_sha256": file_sha256(
            checkpoint / "model.safetensors"
        ),
        "hf_files": parity._file_manifest(hf_path, parity.HF_PARITY_FILES),
        "checkpoint_files": parity._file_manifest(
            checkpoint, parity.CHECKPOINT_PARITY_FILES
        ),
        "token_ids": token_ids,
        "token_ids_sha256": parity._token_ids_sha256(token_ids),
        "tokens": len(token_ids),
        "logit_shape": [1, len(token_ids), 151936],
        "max_abs_error": 0.1,
        "mean_abs_error": 0.005,
        "top1_match_fraction": 1.0,
        "final_top1_match": True,
        "hf_parameter_dtypes": ["bfloat16"],
        "native_parameter_dtypes": ["bfloat16"],
        "hf_logits_dtype": "bfloat16",
        "native_logits_dtype": "float32",
        "comparison_dtype": "float32",
        "source_state": source_state,
        "model_config": model_config,
        "torch_version": parity.torch.__version__,
        "torch_cuda_version": parity.torch.version.cuda,
        "device": "cuda",
        "gpu_name": "NVIDIA H100 80GB HBM3",
    }

    parity.validate_parity_evidence(
        payload,
        hf_path=hf_path,
        checkpoint=checkpoint,
        source_revision=source_revision,
    )

    tampering = (
        ("max_abs_threshold", float("inf")),
        ("max_abs_error", float("nan")),
        ("mean_abs_error", -0.001),
        ("final_top1_match", False),
        ("native_logits_dtype", "bfloat16"),
        ("comparison_dtype", "bfloat16"),
        ("token_ids_sha256", "0" * 64),
        ("script_sha256", "0" * 64),
        ("source_revision", "b" * 40),
    )
    for key, value in tampering:
        altered = copy.deepcopy(payload)
        altered[key] = value
        with pytest.raises(ValueError):
            parity.validate_parity_evidence(
                altered,
                hf_path=hf_path,
                checkpoint=checkpoint,
                source_revision=source_revision,
            )

    forged_tokens = copy.deepcopy(payload)
    forged_tokens["token_ids"] = [9, 8, 7]
    forged_tokens["token_ids_sha256"] = parity._token_ids_sha256([9, 8, 7])
    with pytest.raises(ValueError, match="current canonical tokenizer"):
        parity.validate_parity_evidence(
            forged_tokens,
            hf_path=hf_path,
            checkpoint=checkpoint,
            source_revision=source_revision,
        )

    incomplete = copy.deepcopy(payload)
    incomplete.pop("logit_shape")
    with pytest.raises(ValueError, match="missing fields"):
        parity.validate_parity_evidence(
            incomplete,
            hf_path=hf_path,
            checkpoint=checkpoint,
            source_revision=source_revision,
        )

    for artifact in (
        hf_path / "config.json",
        hf_path / "tokenizer.json",
        checkpoint / "config.json",
        checkpoint / "train_state.json",
    ):
        original = artifact.read_bytes()
        artifact.write_bytes(original + b"tamper")
        with pytest.raises(ValueError, match="manifest mismatch"):
            parity.validate_parity_evidence(
                payload,
                hf_path=hf_path,
                checkpoint=checkpoint,
                source_revision=source_revision,
            )
        artifact.write_bytes(original)


def test_conversion_diagnostic_removes_harmless_per_position_logit_offsets():
    assert diagnosis.FIXED_PROMPTS[0][1] == parity.CANONICAL_PROMPT
    hf = torch.tensor([[[1.0, 2.0, 4.0], [-2.0, 0.0, 1.0]]])
    native = hf + torch.tensor([[[7.0], [-3.0]]])

    metrics = diagnosis.distribution_metrics(hf, native)

    assert metrics["raw_mean_abs_error"] == pytest.approx(5.0)
    assert metrics["centred_max_abs_error"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["centred_mean_abs_error"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["kl_hf_to_native_mean"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["total_variation_mean"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["top1_match_fraction"] == pytest.approx(1.0)


def test_conversion_diagnostic_kl_is_nonnegative_at_full_qwen_vocab():
    torch.manual_seed(7)
    hf = torch.randn(1, 2, 151_936)
    native = hf + torch.tensor([[[3.0], [-5.0]]])

    metrics = diagnosis.distribution_metrics(hf, native)

    assert metrics["kl_hf_to_native_min"] >= 0.0
    assert metrics["kl_hf_to_native_mean"] == pytest.approx(0.0, abs=1e-12)


def test_conversion_diagnostic_reads_nested_checkpoint_model_config(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model": {"num_layers": 28}}')

    assert diagnosis._checkpoint_model_config(checkpoint) == {"num_layers": 28}


def test_conversion_diagnostic_hidden_cosine_stays_in_mathematical_range():
    torch.manual_seed(0)
    hidden = torch.randn(1, 32, 1024, dtype=torch.bfloat16)

    metrics = diagnosis.hidden_metrics(hidden, hidden)

    assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-15)
    assert -1.0 <= metrics["cosine_similarity"] <= 1.0


def test_conversion_diagnostic_detects_distribution_and_hidden_state_drift():
    hf_logits = torch.tensor([[[2.0, 0.0, -1.0]]])
    native_logits = torch.tensor([[[0.0, 2.0, -1.0]]])
    distribution = diagnosis.distribution_metrics(hf_logits, native_logits)
    hidden = diagnosis.hidden_metrics(
        torch.tensor([[[1.0, 2.0]]]),
        torch.tensor([[[1.0, 3.0]]]),
    )

    assert distribution["kl_hf_to_native_mean"] > 0.0
    assert distribution["total_variation_mean"] > 0.0
    assert distribution["top1_match_fraction"] == pytest.approx(0.0)
    assert hidden["max_abs_error"] == pytest.approx(1.0)
    assert hidden["mean_abs_error"] == pytest.approx(0.5)
    assert hidden["relative_l2_error"] > 0.0
    assert hidden["cosine_similarity"] < 1.0


def test_conversion_parity_execution_failure_is_persisted(
    tmp_path, monkeypatch
):
    output = tmp_path / "failed-parity.json"
    hf_path = tmp_path / "hf"
    checkpoint = tmp_path / "checkpoint"
    hf_path.mkdir()
    checkpoint.mkdir()
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    monkeypatch.setattr(parity.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        parity,
        "verify_hf_conversion",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("load failed")
        ),
    )
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        [
            "verify_hf_conversion.py",
            "--hf-path",
            str(hf_path),
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(output),
        ],
    )

    with pytest.raises(RuntimeError, match="load failed"):
        parity.main()

    payload = __import__("json").loads(output.read_text())
    assert payload["passed"] is False
    assert payload["failure_stage"] == "parity_execution"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "load failed"


def test_hf_llama_key_mapping_loads_native_backbone():
    cfg = ModelConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        sequence_len=8,
        tie_embeddings=False,
    )
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
    assert torch.equal(
        model.embed_tokens.weight[:16], hf["model.embed_tokens.weight"]
    )


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

    cp = CacheProcessorConfig(
        hidden_size=8, intermediate_size=16, num_heads=1, num_kv_heads=1
    )
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

    assert torch.equal(
        target.backbone.lm_head.weight, source.embed_tokens.weight
    )


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
    hf_logits, hf_hidden, hf_dtypes, hf_logits_dtype = (
        diagnosis._capture_forward(hf, x, hf_model=True)
    )
    native_logits, native_hidden, native_dtypes, native_logits_dtype = (
        diagnosis._capture_forward(native, x, hf_model=False)
    )
    assert torch.allclose(native_logits, hf_logits, atol=1e-5, rtol=1e-5)
    assert list(hf_hidden) == ["embedding", "layer_00", "final_norm"]
    assert list(native_hidden) == list(hf_hidden)
    assert hf_dtypes == native_dtypes
    assert hf_logits_dtype == "float32"
    assert native_logits_dtype == "float32"
    for stage in hf_hidden:
        assert torch.allclose(
            native_hidden[stage], hf_hidden[stage], atol=1e-5, rtol=1e-5
        )


def test_tiny_qwen35_logits_match_transformers():
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    import transformers

    config_cls = getattr(transformers, "Qwen3_5TextConfig", None) or getattr(
        transformers, "Qwen3_5Config", None
    )
    model_cls = getattr(transformers, "Qwen3_5ForCausalLM", None)
    if config_cls is None or model_cls is None:
        pytest.skip(
            "installed Transformers does not expose dense Qwen3.5 CausalLM classes"
        )

    try:
        hf_cfg = config_cls(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=16,
            tie_word_embeddings=False,
            rope_parameters={
                "rope_theta": 10000.0,
                "rope_type": "default",
                "partial_rotary_factor": 0.5,
            },
            partial_rotary_factor=0.5,
            layer_types=["full_attention", "linear_attention"],
            linear_conv_kernel_dim=3,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
        )
        torch.manual_seed(0)
        hf = model_cls(hf_cfg).eval()
    except Exception as exc:
        pytest.skip(
            f"installed Transformers cannot instantiate dense Qwen3.5: {exc}"
        )

    native = build_model(config_from_hf(hf_cfg, family="qwen3_5")).eval()
    load_hf_state(native, hf.state_dict(), family="qwen3_5")
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        hf_logits = hf(x, use_cache=False).logits
        native_logits = native(x)
    assert torch.allclose(
        native_logits, hf_logits.float(), atol=1e-5, rtol=1e-5
    )


def test_qwen35_hf_text_config_uses_full_attention_processor_layers_without_moe_fields():
    text = SimpleNamespace(
        num_attention_heads=2,
        hidden_size=16,
        intermediate_size=32,
        head_dim=8,
        num_key_value_heads=1,
        max_position_embeddings=32,
        vocab_size=128,
        num_hidden_layers=4,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        rope_parameters={
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.5,
        },
        partial_rotary_factor=0.5,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        hidden_act="silu",
    )
    cfg = config_from_hf(SimpleNamespace(text_config=text), family="qwen3_5")

    assert cfg.family == "qwen3_5"
    assert cfg.intermediate_size == 32
    assert cfg.layer_types == text.layer_types
    assert cfg.processor_layer_indices == [1, 3]
    assert cfg.cache_num_heads == 2
    assert cfg.num_experts == 0
    assert cfg.moe_intermediate_size is None


def test_qwen35_hf_mapping_loads_language_model_prefix():
    model = build_model(_tiny_qwen35_config())
    hf = _prefixed_state_like(model.state_dict(), "language_model.")

    load_hf_state(model, hf, family="qwen3_5")

    assert torch.equal(
        model.embed_tokens.weight,
        hf["language_model.model.embed_tokens.weight"],
    )
    assert torch.equal(
        model.layers[1].self_attn.q_proj.weight,
        hf["language_model.model.layers.1.self_attn.q_proj.weight"],
    )
    assert torch.equal(
        model.layers[0].linear_attn.in_proj_qkv.weight,
        hf["language_model.model.layers.0.linear_attn.in_proj_qkv.weight"],
    )
    assert torch.equal(
        model.layers[1].mlp.gate_proj.weight,
        hf["language_model.model.layers.1.mlp.gate_proj.weight"],
    )


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
        rope_parameters={
            "rope_theta": 10000.0,
            "rope_type": "default",
            "partial_rotary_factor": 0.5,
        },
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
    assert torch.allclose(
        native_logits, hf_logits.float(), atol=1e-5, rtol=1e-5
    )


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
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        rope_parameters={
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.5,
        },
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
    cfg = config_from_hf(
        SimpleNamespace(text_config=text), family="qwen3_5_moe"
    )

    assert cfg.family == "qwen3_5_moe"
    assert cfg.layer_types == text.layer_types
    assert cfg.processor_layer_indices == [1, 3]
    assert cfg.cache_num_heads == 2
    assert cfg.hidden_act == "silu"


def test_qwen35_moe_hf_mapping_loads_multimodal_language_model_prefix():
    model = build_model(_tiny_qwen35_moe_config())
    hf = _prefixed_state_like(model.state_dict(), "language_model.")

    load_hf_state(model, hf, family="qwen3_5_moe")

    assert torch.equal(
        model.embed_tokens.weight,
        hf["language_model.model.embed_tokens.weight"],
    )
    assert torch.equal(
        model.layers[1].self_attn.q_proj.weight,
        hf["language_model.model.layers.1.self_attn.q_proj.weight"],
    )
    assert torch.equal(
        model.layers[0].linear_attn.in_proj_qkv.weight,
        hf["language_model.model.layers.0.linear_attn.in_proj_qkv.weight"],
    )


def test_tiny_gemma4_logits_match_transformers():
    pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    import transformers

    config_cls = getattr(transformers, "Gemma4TextConfig", None)
    model_cls = getattr(transformers, "Gemma4ForCausalLM", None)
    if config_cls is None or model_cls is None:
        pytest.skip(
            "installed Transformers does not expose Gemma4 CausalLM classes"
        )

    try:
        hf_cfg = config_cls(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=16,
            tie_word_embeddings=False,
            layer_types=["full_attention", "full_attention"],
            sliding_window=4,
            global_head_dim=8,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=32,
            final_logit_softcapping=None,
            rope_parameters={
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
            },
        )
        torch.manual_seed(0)
        hf = model_cls(hf_cfg).eval()
    except Exception as exc:
        pytest.skip(
            f"installed Transformers cannot instantiate dense Gemma4: {exc}"
        )

    native = build_model(config_from_hf(hf_cfg, family="gemma4")).eval()
    load_hf_state(native, hf.state_dict(), family="gemma4")
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        hf_logits = hf(x, use_cache=False).logits
        native_logits = native(x)
    assert torch.allclose(
        native_logits, hf_logits.float(), atol=1e-5, rtol=1e-5
    )


def test_gemma4_hf_text_config_maps_dense_fields_and_cache_layers():
    text = SimpleNamespace(
        num_attention_heads=2,
        hidden_size=16,
        intermediate_size=32,
        head_dim=8,
        num_key_value_heads=1,
        max_position_embeddings=32,
        vocab_size=128,
        num_hidden_layers=4,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        rope_parameters={
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        },
        sliding_window=4,
        global_head_dim=8,
        num_global_key_value_heads=None,
        attention_k_eq_v=False,
        num_kv_shared_layers=2,
        hidden_size_per_layer_input=4,
        vocab_size_per_layer_input=128,
        final_logit_softcapping=30.0,
        hidden_activation="gelu_pytorch_tanh",
        use_double_wide_mlp=False,
        enable_moe_block=False,
        attention_bias=False,
    )
    cfg = config_from_hf(SimpleNamespace(text_config=text), family="gemma4")

    assert cfg.family == "gemma4"
    assert cfg.layer_types == text.layer_types
    assert cfg.cache_layer_indices == [0, 1]
    assert cfg.hidden_size_per_layer_input == 4
    assert cfg.final_logit_softcapping == 30.0
    assert cfg.hidden_act == "gelu_pytorch_tanh"


def test_gemma4_hf_text_config_uses_shared_default_layer_types():
    text = SimpleNamespace(
        num_attention_heads=2,
        hidden_size=16,
        intermediate_size=32,
        head_dim=8,
        num_key_value_heads=1,
        max_position_embeddings=32,
        vocab_size=128,
        num_hidden_layers=7,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        rope_parameters=None,
        sliding_window=4,
        global_head_dim=8,
        num_global_key_value_heads=None,
        attention_k_eq_v=False,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=128,
        final_logit_softcapping=None,
        hidden_activation="gelu_pytorch_tanh",
        use_double_wide_mlp=False,
        enable_moe_block=False,
        attention_bias=False,
    )

    cfg = config_from_hf(SimpleNamespace(text_config=text), family="gemma4")

    assert cfg.layer_types == gemma4_default_layer_types(7)


def test_gemma4_hf_config_rejects_moe():
    text = SimpleNamespace(
        enable_moe_block=True,
        attention_bias=False,
        num_hidden_layers=1,
    )
    with pytest.raises(NotImplementedError, match="Gemma4 MoE"):
        config_from_hf(text, family="gemma4")


def test_gemma4_hf_mapping_loads_language_model_prefix():
    model = build_model(_tiny_gemma4_config())
    hf = _prefixed_state_like(model.state_dict(), "language_model.")

    load_hf_state(model, hf, family="gemma4")

    assert torch.equal(
        model.embed_tokens.weight,
        hf["language_model.model.embed_tokens.weight"],
    )
    assert torch.equal(
        model.layers[1].self_attn.q_proj.weight,
        hf["language_model.model.layers.1.self_attn.q_proj.weight"],
    )
    assert torch.equal(
        model.embed_tokens_per_layer.weight,
        hf["language_model.model.embed_tokens_per_layer.weight"],
    )


def test_gemma4_hf_mapping_defaults_missing_layer_scalar():
    model = build_model(_tiny_gemma4_config())
    hf = _prefixed_state_like(model.state_dict(), "language_model.")
    hf = {
        key: value
        for key, value in hf.items()
        if not key.endswith(".layer_scalar")
    }

    load_hf_state(model, hf, family="gemma4")

    for layer in model.layers:
        assert torch.equal(
            layer.layer_scalar, torch.ones_like(layer.layer_scalar)
        )


def _tiny_qwen35_config() -> ModelConfig:
    return ModelConfig(
        variant="tf",
        family="qwen3_5",
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=32,
        tie_embeddings=False,
        qk_norm=True,
        rotary_style="llama",
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        processor_layer_indices=[1, 3],
        partial_rotary_factor=0.5,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
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
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
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


def _tiny_gemma4_config() -> ModelConfig:
    return ModelConfig(
        variant="tf",
        family="gemma4",
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=32,
        tie_embeddings=True,
        rotary_style="llama",
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        sliding_window=4,
        global_head_dim=8,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=4,
        vocab_size_per_layer_input=128,
        final_logit_softcapping=30.0,
        hidden_act="gelu_pytorch_tanh",
    )


def _prefixed_state_like(
    state: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        if key == "embed_tokens.weight":
            remote = f"{prefix}model.embed_tokens.weight"
        elif key == "norm.weight":
            remote = f"{prefix}model.norm.weight"
        elif key == "lm_head.weight":
            remote = f"{prefix}lm_head.weight"
        else:
            remote_key = key.replace(".linear_attn.dynamics.", ".linear_attn.")
            remote = f"{prefix}model.{remote_key}"
        if value.is_floating_point():
            out[remote] = torch.randn_like(value)
        else:
            out[remote] = torch.zeros_like(value)
    return out
