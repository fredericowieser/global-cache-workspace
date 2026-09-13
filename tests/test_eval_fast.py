from gcw.evaluation.core import (
    _normalize_until,
    _trim_until,
)
from gcw.evaluation.scoring import (
    tokenizer_for_checkpoint,
    trim_generated_token_ids,
    truncate_context_continuation,
)


def test_trim_until_matches_existing_ordered_stop_semantics():
    assert _trim_until("answer\n\nextra", ("\n\n", "\n")) == "answer"
    assert _trim_until("answer\nextra", ("\n\n", "\n")) == "answer"


def test_token_stop_trimming_uses_earliest_stop_or_eos():
    tokens = [10, 11, 12, 13, 14]
    assert trim_generated_token_ids(tokens, [[12, 13]], None) == [10, 11]
    assert trim_generated_token_ids(tokens, [[13]], 12) == [10, 11]


def test_normalize_until_accepts_string_list_and_none():
    assert _normalize_until(None) == ()
    assert _normalize_until("stop") == ("stop",)
    assert _normalize_until(["a", None, "b"]) == ("a", "b")


def test_native_hf_families_require_original_tokenizer(monkeypatch):
    calls = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append((source, kwargs))
            return type(
                "Tokenizer",
                (),
                {
                    "eos_token_id": 1,
                    "pad_token_id": 1,
                    "__len__": lambda self: 2,
                },
            )()

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        type("Transformers", (), {"AutoTokenizer": AutoTokenizer}),
    )
    for family in ("qwen3_5", "gemma4"):
        tokenizer_for_checkpoint(
            type("Config", (), {"family": family, "vocab_size": 2})(),
            {"source": "model"},
        )
    assert len(calls) == 2


def test_context_continuation_truncation_keeps_at_least_one_context_token():
    ctx, cont = truncate_context_continuation(
        [1, 2, 3, 4], [5, 6], max_length=4
    )
    assert ctx == [2, 3, 4]
    assert cont == [5, 6]

    ctx, cont = truncate_context_continuation(
        [1], [2, 3, 4, 5, 6], max_length=3
    )
    assert ctx == [3]
    assert cont == [4, 5, 6]
