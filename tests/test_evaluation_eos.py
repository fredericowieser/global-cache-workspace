from types import SimpleNamespace
import json

import pytest
import torch

from gcw.config import EvalConfig
from gcw.evaluation.scoring import (
    HFTokenizerAdapter,
    generation_config_for_source,
    generation_token_ids,
    tokenizer_for_checkpoint,
    trim_generated_token_ids,
)


@pytest.mark.parametrize(
    "config,expected",
    [
        ({}, ((9,), 0)),
        ({"eos_token_id": 0}, ((0,), 0)),
        ({"eos_token_id": [9, 7, 9], "pad_token_id": 3}, ((9, 7), 3)),
        ({"eos_token_id": [], "pad_token_id": 0}, ((), 0)),
        ({"eos_token_id": None, "pad_token_id": None}, ((9,), 0)),
    ],
)
def test_generation_metadata_precedence_and_zero(config, expected):
    tokenizer = SimpleNamespace(eos_token_id=9, pad_token_id=0)
    assert generation_token_ids(tokenizer, config) == expected


def test_hf_tokenizer_wrappers_preserve_eos_zero(monkeypatch):
    from transformers import AutoTokenizer
    from gcw.data.tokenizers import TransformersTokenizer

    tokenizer = Tokenizer(eos_token_id=0, pad_token_id=9, vocab_size=10)
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer
    )
    assert TransformersTokenizer("test").eot_id == 0
    assert HFTokenizerAdapter(tokenizer).eot_id == 0


def test_native_tokenizer_and_padding_fallback():
    assert generation_token_ids(SimpleNamespace(eot_id=0)) == ((0,), 0)
    assert generation_token_ids(SimpleNamespace(eos_token_id=9)) == ((9,), 9)


@pytest.mark.parametrize("eos", [True, -1, "9", [9, None], [9, False], [1.0]])
def test_invalid_eos_metadata_fails(eos):
    with pytest.raises(ValueError, match="eos_token_id"):
        generation_token_ids(None, {"eos_token_id": eos})


def test_missing_eos_is_not_a_likelihood_prefix_fallback():
    tokenizer = SimpleNamespace(eos_token_id=None, pad_token_id=0, eot_id=0)
    with pytest.raises(ValueError, match="EOS metadata"):
        generation_token_ids(tokenizer)


@pytest.mark.parametrize("pad", [-1, True, "0"])
def test_invalid_padding_fails(pad):
    with pytest.raises(ValueError, match="pad_token_id"):
        generation_token_ids(None, {"eos_token_id": 9, "pad_token_id": pad})


def test_generation_file_then_text_config_fallback(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"eos_token_id": 7, "pad_token_id": 0}})
    )
    assert generation_config_for_source(tmp_path)["eos_token_id"] == 7
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [7, 9], "pad_token_id": 0})
    )
    assert generation_config_for_source(tmp_path)["eos_token_id"] == [7, 9]


@pytest.mark.parametrize(
    "family", ["llama", "qwen3", "qwen3_5", "qwen3_5_moe", "gemma4"]
)
def test_old_checkpoint_uses_source_generation_metadata(
    tmp_path, monkeypatch, family
):
    from transformers import AutoTokenizer

    calls = []

    def load(source, **kwargs):
        calls.append(source)
        return Tokenizer(eos_token_id=7, pad_token_id=0, vocab_size=10)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [7, 9], "pad_token_id": 0})
    )
    config = SimpleNamespace(family=family)
    tokenizer = tokenizer_for_checkpoint(config, {"source": str(tmp_path)})
    assert generation_token_ids(tokenizer, tokenizer.generation_config) == (
        (7, 9),
        0,
    )
    assert tokenizer.eot_id == 7
    overridden = tokenizer_for_checkpoint(
        config, {"source": "unavailable", "eos_token_id": 8}, source=tmp_path
    )
    assert generation_token_ids(overridden, overridden.generation_config) == (
        (8,),
        0,
    )
    assert calls == [str(tmp_path), tmp_path]


class Tokenizer:
    def __init__(self, eos_token_id=9, pad_token_id=0, vocab_size=12):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size

    def encode(self, text, **kwargs):
        return [4] if text == "STOP" else [1, 2]

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("reference", [False, True])
def test_lm_eval_routes_stop_and_trim_all_eos(monkeypatch, legacy, reference):
    from gcw.evaluation.lm_eval_adapter import GCWEvalLM
    from gcw.evaluation.legacy import LegacyPaper7EvalLM

    class Model(torch.nn.Module):
        config = SimpleNamespace(sequence_len=32, family="smollm", variant="tf")

        def get_device(self):
            return torch.device("cpu")

        def generate(
            self,
            ids,
            max_new_tokens,
            eos_token_id,
            pad_token_id,
            stop_sequences=None,
        ):
            assert eos_token_id == (9, 7)
            assert stop_sequences == (None if reference else [[4]])
            assert pad_token_id == 0
            return torch.cat(
                (ids, torch.tensor([[3, 7, 8, 9]]).expand(len(ids), -1)), dim=1
            )

        generate_fast = generate

    if reference:
        monkeypatch.setenv("GCW_EVAL_REFERENCE_GENERATE", "1")
    else:
        monkeypatch.delenv("GCW_EVAL_REFERENCE_GENERATE", raising=False)
    tokenizer = HFTokenizerAdapter(
        Tokenizer(), generation_config={"eos_token_id": [9, 7]}
    )
    cls = LegacyPaper7EvalLM if legacy else GCWEvalLM
    lm = cls(Model(), tokenizer, EvalConfig(batch_size=2))
    req = SimpleNamespace(
        args=("question", {"max_gen_toks": 8, "until": ["STOP"]})
    )
    assert lm.generate_until([req, req]) == ["3", "3"]
    assert lm.eot_token_id == 9


def test_trim_earliest_eos_or_task_stop():
    assert trim_generated_token_ids([2, 7, 8, 9], [], (9, 7)) == [2]
    assert trim_generated_token_ids([2, 4, 5, 7], [[4, 5]], (9, 7)) == [2]


def tiny_gpt2(*, n_positions, pad_token_id):
    from transformers import GPT2Config, GPT2LMHeadModel

    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=12,
            n_positions=n_positions,
            n_embd=8,
            n_layer=1,
            n_head=1,
            eos_token_id=9,
            pad_token_id=pad_token_id,
        )
    )


def test_hf_backend_uses_explicit_tokenizer_for_missing_ids(
    tmp_path, monkeypatch
):
    from transformers import AutoTokenizer
    from gcw.evaluation.benchmarks import HFBenchmarkBackend

    source = tmp_path / "weights-only"
    tokenizer_source = tmp_path / "tokenizer"
    model = tiny_gpt2(n_positions=16, pad_token_id=None)
    model.save_pretrained(source)
    calls = []

    def load(path, **kwargs):
        calls.append(path)
        return SimpleNamespace(eos_token_id=7, pad_token_id=0)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)
    backend = HFBenchmarkBackend(
        source, torch.device("cpu"), tokenizer_path=tokenizer_source
    )
    assert calls == [tokenizer_source]
    assert backend.eos_ids == (9,)
    assert backend.pad_id == 0


def test_hf_real_generation_pads_rows_after_alternate_eos():
    from gcw.evaluation.lm_eval_adapter import HFCausalLMAdapter

    model = tiny_gpt2(n_positions=32, pad_token_id=0).eval()
    script = iter(([7, 3], [8, 4], [8, 9]))

    def predict(module, args, output):
        output.logits.fill_(-100)
        for row, token in enumerate(next(script)):
            output.logits[row, -1, token] = 100
        return output

    model.register_forward_hook(predict)
    result = HFCausalLMAdapter(model).generate(
        torch.tensor([[1, 2], [1, 2]]),
        max_new_tokens=8,
        eos_token_id=(9, 7),
        pad_token_id=0,
    )
    assert result.tolist() == [[1, 2, 7, 0, 0], [1, 2, 3, 4, 9]]
