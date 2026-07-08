from types import SimpleNamespace

from gcw.eval import (
    _encode_generate_prompt,
    _legacy_bottleneck_cache_eval,
    _legacy_generate_add_special_tokens,
    _normalize_until,
    _trim_generated_token_ids,
    _trim_until,
    _truncate_context_continuation,
)


def test_trim_until_matches_existing_ordered_stop_semantics():
    assert _trim_until("answer\n\nextra", ("\n\n", "\n")) == "answer"
    assert _trim_until("answer\nextra", ("\n\n", "\n")) == "answer"


def test_token_stop_trimming_uses_earliest_stop_or_eos():
    tokens = [10, 11, 12, 13, 14]
    assert _trim_generated_token_ids(tokens, [[12, 13]], None) == [10, 11]
    assert _trim_generated_token_ids(tokens, [[13]], 12) == [10, 11]


def test_normalize_until_accepts_string_list_and_none():
    assert _normalize_until(None) == ()
    assert _normalize_until("stop") == ("stop",)
    assert _normalize_until(["a", None, "b"]) == ("a", "b")


def test_legacy_bottleneck_generation_prompts_add_special_tokens():
    model = SimpleNamespace(config=SimpleNamespace(legacy_bottleneck_generate=True, family="llama"))
    assert _legacy_generate_add_special_tokens(model)

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            ids = [ord(ch) for ch in text]
            return [1, *ids] if add_special_tokens else ids

    assert _encode_generate_prompt(Tokenizer(), "ab", 8, add_special_tokens=True) == [1, 97, 98]
    assert _encode_generate_prompt(Tokenizer(), "ab", 2, add_special_tokens=True) == [97, 98]


def test_non_legacy_generation_prompts_do_not_add_special_tokens():
    model = SimpleNamespace(config=SimpleNamespace(legacy_bottleneck_generate=False, family="llama"))
    assert not _legacy_generate_add_special_tokens(model)


def test_legacy_bottleneck_loglikelihood_uses_boundary_path_only_for_cp_models():
    model = SimpleNamespace(
        config=SimpleNamespace(legacy_bottleneck_generate=True, variant="bgt"),
        backbone=object(),
        cache_processor=object(),
    )
    assert _legacy_bottleneck_cache_eval(model)
    assert not _legacy_bottleneck_cache_eval(SimpleNamespace(config=SimpleNamespace(legacy_bottleneck_generate=True, variant="tf")))


def test_context_continuation_truncation_keeps_at_least_one_context_token():
    ctx, cont = _truncate_context_continuation([1, 2, 3, 4], [5, 6], max_length=4)
    assert ctx == [2, 3, 4]
    assert cont == [5, 6]

    ctx, cont = _truncate_context_continuation([1], [2, 3, 4, 5, 6], max_length=3)
    assert ctx == [3]
    assert cont == [4, 5, 6]
