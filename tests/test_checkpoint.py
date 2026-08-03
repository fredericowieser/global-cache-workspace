import torch
import pytest

from gcw.checkpoint import load_checkpoint, save_checkpoint
from gcw.config import ModelConfig
from gcw.eval import GCWEvalLM
from gcw.eval_support import tokenizer_for_checkpoint
from gcw.models import build_model


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



def test_load_checkpoint_preserves_mixed_backbone_and_processor_dtypes(tmp_path):
    from gcw.checkpoint import load_checkpoint, save_checkpoint
    from gcw.config import CacheProcessorConfig, ModelConfig
    from gcw.models import build_model
    from gcw.models.bgt import BGT

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
    assert isinstance(model, BGT)
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

    assert isinstance(loaded, BGT)
    assert {p.dtype for p in loaded.backbone.parameters()} == {torch.bfloat16}
    assert {p.dtype for p in loaded.cache_processor.parameters()} == {torch.float32}
    assert loaded.cache_processor.block.gate.item() == pytest.approx(-4.9896)
