import torch
import pytest

from gcw.checkpoint import load_checkpoint, save_checkpoint
from gcw.config import ModelConfig
from gcw.eval import _tokenizer_for
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
        _tokenizer_for(cfg, {})
