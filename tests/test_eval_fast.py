from types import SimpleNamespace

from gcw.eval import (
    _legacy_bottleneck_cache_eval,
    _legacy_generate_add_special_tokens,
    _normalize_until,
    _trim_until,
)
from gcw.eval_support import encode_generate_prompt, trim_generated_token_ids, truncate_context_continuation


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


def test_legacy_bottleneck_generation_prompts_add_special_tokens():
    model = SimpleNamespace(config=SimpleNamespace(legacy_bottleneck_generate=True, family="llama"))
    assert _legacy_generate_add_special_tokens(model)

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            ids = [ord(ch) for ch in text]
            return [1, *ids] if add_special_tokens else ids

    assert encode_generate_prompt(Tokenizer(), "ab", 8, add_special_tokens=True) == [1, 97, 98]
    assert encode_generate_prompt(Tokenizer(), "ab", 2, add_special_tokens=True) == [97, 98]


def test_non_legacy_generation_prompts_do_not_add_special_tokens():
    model = SimpleNamespace(config=SimpleNamespace(legacy_bottleneck_generate=False, family="llama"))
    assert not _legacy_generate_add_special_tokens(model)


def test_legacy_bottleneck_loglikelihood_uses_boundary_path_only_for_cp_models():
    for variant in ("bt", "bgt", "longshot"):
        model = SimpleNamespace(
            config=SimpleNamespace(legacy_bottleneck_generate=True, variant=variant),
            backbone=object(),
            cache_processor=object(),
        )
        assert _legacy_bottleneck_cache_eval(model)
    assert not _legacy_bottleneck_cache_eval(SimpleNamespace(config=SimpleNamespace(legacy_bottleneck_generate=True, variant="tf")))


def test_context_continuation_truncation_keeps_at_least_one_context_token():
    ctx, cont = truncate_context_continuation([1, 2, 3, 4], [5, 6], max_length=4)
    assert ctx == [2, 3, 4]
    assert cont == [5, 6]

    ctx, cont = truncate_context_continuation([1], [2, 3, 4, 5, 6], max_length=3)
    assert ctx == [3]
    assert cont == [4, 5, 6]



def test_opsd_eval_chat_prompt_controls_thinking_without_changing_base():
    from gcw.opsd_eval import build_opsd_eval_prompt

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
            assert tokenize is True
            base = [10, 11, 12]
            if not add_generation_prompt:
                return base
            return [*base, 20, 21] if messages else [20, 21]

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            if text == "<think>\n":
                return [30, 31]
            raise AssertionError(text)

    tokenizer = Tokenizer()
    thinking_off = build_opsd_eval_prompt(tokenizer, "question", enable_thinking=False)
    thinking_on = build_opsd_eval_prompt(tokenizer, "question", enable_thinking=True)

    assert thinking_off.base_ids == thinking_on.base_ids == (10, 11, 12)
    assert thinking_off.trigger_ids == (20, 21)
    assert thinking_on.trigger_ids == (20, 21, 30, 31)



def test_opsd_generate_until_trims_online_stop_with_eot_argument(monkeypatch):
    from types import SimpleNamespace

    from gcw.distill.prompt import ChatPrompt
    from gcw.distill.session import RolloutResult
    from gcw.eval import GCWEvalLM
    from gcw.opsd_eval import OpsdEvalCondition

    class Tokenizer:
        eot_id = 99

        def encode(self, text, add_special_tokens=False):
            assert text == "STOP"
            return [9]

        def decode(self, token_ids):
            return ",".join(str(token_id) for token_id in token_ids)

    fake = SimpleNamespace(
        model=object(),
        tokenizer=Tokenizer(),
        opsd_condition=OpsdEvalCondition(False, False),
        opsd_generation_records=[],
        tok_decode=lambda token_ids: ",".join(str(token_id) for token_id in token_ids),
        eot_token_id=99,
    )
    monkeypatch.setattr(
        "gcw.eval.build_opsd_eval_prompt",
        lambda *args, **kwargs: ChatPrompt(base_ids=(1,), trigger_ids=(2,)),
    )
    monkeypatch.setattr(
        "gcw.eval.greedy_opsd_response",
        lambda *args, **kwargs: RolloutResult(response_ids=(5, 9), event_cache_lengths=()),
    )
    request = SimpleNamespace(args=("question", {"until": ["STOP"], "max_gen_toks": 12}))

    output = GCWEvalLM._generate_until_opsd(fake, [request])

    assert output == ["5"]
