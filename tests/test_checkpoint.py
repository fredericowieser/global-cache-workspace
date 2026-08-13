import torch
import pytest

from gcw.checkpoint import (
    _model_config_from_checkpoint,
    load_backbone_checkpoint,
    load_checkpoint,
    publish_final_checkpoint_manifest,
    save_checkpoint,
)
from gcw.config import LoRAConfig, ModelConfig
from gcw.evaluation.core import GCWEvalLM
from gcw.evaluation.scoring import HFTokenizerAdapter, tokenizer_for_checkpoint
from gcw.models import build_model


def test_legacy_generation_flag_maps_to_explicit_policy():
    config = _model_config_from_checkpoint(
        {
            "variant": "bgt",
            "family": "qwen3",
            "legacy_bottleneck_generate": True,
        }
    )
    assert config.processor_generation_mode == "post_prefill_relative"


def test_checkpoint_roundtrip(tmp_path):
    cfg = ModelConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, num_kv_heads=1, head_dim=8, sequence_len=16)
    model = build_model(cfg).to(dtype=torch.bfloat16)
    model.init_weights()
    save_checkpoint(tmp_path, model, cfg, {"step": 3})
    loaded, loaded_cfg, state = load_checkpoint(tmp_path)
    x = torch.randint(0, 64, (1, 4))
    assert loaded_cfg.vocab_size == cfg.vocab_size
    assert state["step"] == 3
    assert loaded.embed_tokens.weight.dtype == torch.bfloat16
    assert loaded(x).shape == (1, 4, 64)


def test_backbone_checkpoint_initializes_lora_model_without_overwriting_adapters(tmp_path):
    base_config = ModelConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
    )
    base = build_model(base_config)
    base.init_weights()
    save_checkpoint(tmp_path / "base", base, base_config)
    adapted_config = ModelConfig(**{**base_config.__dict__, "lora": LoRAConfig(rank=2)})
    adapted = build_model(adapted_config)
    adapted.init_weights()

    load_backbone_checkpoint(adapted, tmp_path / "base")

    assert torch.equal(adapted.layers[0].self_attn.q_proj.weight, base.layers[0].self_attn.q_proj.weight)
    assert torch.count_nonzero(adapted.layers[0].self_attn.q_proj.lora_b) == 0


def test_hf_family_eval_requires_tokenizer_source():
    cfg = ModelConfig(
        family="llama",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=16,
    )

    with pytest.raises(ValueError, match="tokenizer"):
        tokenizer_for_checkpoint(cfg, {})


def test_gcw_eval_lm_exposes_tokenizer_name_and_chat_template():
    class FakeTokenizer:
        name_or_path = "qwen-tokenizer"
        eot_id = 0

        def apply_chat_template(self, chat_history, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            return f"{chat_history[0]['role']}:{chat_history[0]['content']}:{add_generation_prompt}"

    lm = GCWEvalLM.__new__(GCWEvalLM)
    lm.tokenizer = FakeTokenizer()

    assert lm.tokenizer_name == "qwen-tokenizer"
    assert lm.apply_chat_template([{"role": "user", "content": "hi"}]) == "user:hi:True"


def test_hf_eval_tokenizer_adapter_exposes_the_training_tokenizer_contract():
    class FakeTokenizer:
        name_or_path = "fake-tokenizer"
        eos_token_id = 7
        pad_token_id = 0
        bos_token_id = 1
        special_tokens_map = {"eos_token": "</s>"}
        chat_template = "{{ messages }}"

        def __len__(self):
            return 32

        def get_vocab(self):
            return {"a": 0, "b": 1}

        def encode(self, text, add_special_tokens=False):
            return [1]

        def decode(self, ids, skip_special_tokens=False):
            return "decoded"

        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

    adapter = HFTokenizerAdapter(FakeTokenizer(), "fake-tokenizer")

    assert adapter.get_vocab_size() == 32
    assert len(adapter.identity_sha256()) == 64
    assert adapter.apply_chat_template([], tokenize=True, add_generation_prompt=True) == [1, 2]



def test_load_checkpoint_preserves_mixed_backbone_and_processor_dtypes(tmp_path):
    from gcw.checkpoint import load_checkpoint, save_checkpoint
    from gcw.config import CacheProcessorConfig, ModelConfig
    from gcw.models import LanguageModel, build_model

    config = ModelConfig(
        variant="bgt",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
        ),
    )
    model = build_model(config)
    assert isinstance(model, LanguageModel)
    model.backbone.to(dtype=torch.bfloat16)
    model.cache_processor.to(dtype=torch.float32)
    model.cache_processor.block.gate.data.fill_(-4.9896)
    save_checkpoint(
        tmp_path / "mixed",
        model,
        config,
        {"dtype": "mixed_bfloat16_backbone_float32_processor"},
    )

    loaded, _, _ = load_checkpoint(tmp_path / "mixed")

    assert isinstance(loaded, LanguageModel)
    assert {p.dtype for p in loaded.backbone.parameters()} == {torch.bfloat16}
    assert {p.dtype for p in loaded.cache_processor.parameters()} == {torch.float32}
    assert loaded.cache_processor.block.gate.item() == pytest.approx(-4.9896)


def test_final_checkpoint_manifest_binds_all_loader_inputs(tmp_path):
    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "initial_processor.safetensors").write_bytes(b"initial")
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "run_config.json").write_text("{}\n")
    (checkpoint / "train_state.json").write_text("{}\n")

    manifest_sha256 = publish_final_checkpoint_manifest(
        checkpoint,
        "a" * 64,
        "b" * 64,
    )

    import hashlib
    import json

    manifest_path = checkpoint / "manifest.json"
    assert manifest_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text())
    assert manifest == {
        "schema": "gcw.final_checkpoint.manifest.v2",
        "files": {
            name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
            for name in (
                "config.json",
                "initial_processor.safetensors",
                "model.safetensors",
                "run_config.json",
                "train_state.json",
            )
        },
        "initial_processor_sha256": "b" * 64,
        "final_processor_sha256": "a" * 64,
    }
