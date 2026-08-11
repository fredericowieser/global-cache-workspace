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



def test_process_branch_generate_until_uses_explicit_branch_contract(monkeypatch):
    import torch

    from gcw.eval import GCWEvalLM, ProcessBranchEvalCondition

    class Tokenizer:
        eot_id = 99

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            if text == "STOP":
                return [9]
            return [ord(character) % 64 for character in text]

        def decode(self, token_ids):
            return ",".join(str(token_id) for token_id in token_ids)

        def get_vocab_size(self):
            return 64

    calls = []

    def fake_generate(model, **kwargs):
        calls.append((model, kwargs))
        return torch.tensor([5, 9, 6])

    monkeypatch.setattr("gcw.eval.generate_process_branch", fake_generate)
    fake = SimpleNamespace(
        model=SimpleNamespace(
            get_device=lambda: torch.device("cpu"),
            config=SimpleNamespace(sequence_len=128),
        ),
        tokenizer=Tokenizer(),
        process_branch_condition=ProcessBranchEvalCondition(
            cycles=1,
            prompt_suffix="suffix",
            answer_prefix="prefix",
        ),
        add_special_tokens_for_generate=False,
        max_length=128,
        tok_decode=lambda token_ids: ",".join(str(token_id) for token_id in token_ids),
        eot_token_id=99,
    )
    request = SimpleNamespace(args=("question", {"until": ["STOP"], "max_gen_toks": 12}))

    output = GCWEvalLM._generate_until_process_branch(fake, [request])

    assert output == ["5"]
    assert len(calls) == 1
    kwargs = calls[0][1]
    assert kwargs["cycles"] == 1
    assert kwargs["max_new_tokens"] == 12
    assert kwargs["answer_prefix_ids"].tolist() == fake.tokenizer.encode("prefix")
    assert kwargs["prompt_ids"].tolist() == fake.tokenizer.encode("question\n\nsuffix")


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


def test_opsd_generate_until_batches_equal_shape_requests(monkeypatch):
    from gcw.distill.prompt import ChatPrompt
    from gcw.distill.session import RolloutResult
    from gcw.eval import GCWEvalLM
    from gcw.opsd_eval import OpsdEvalCondition

    class Tokenizer:
        eot_id = 99

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            assert text == "STOP"
            return [9]

    calls = []
    fake = SimpleNamespace(
        model=object(),
        tokenizer=Tokenizer(),
        opsd_condition=OpsdEvalCondition(False, False, max_response_tokens=2048),
        opsd_generation_records=[],
        batch_size=8,
        tok_decode=lambda token_ids: ",".join(str(token_id) for token_id in token_ids),
        eot_token_id=99,
    )
    monkeypatch.setattr(
        "gcw.eval.build_opsd_eval_prompt",
        lambda *args, **kwargs: ChatPrompt(base_ids=(1,), trigger_ids=(2,)),
    )

    def fake_batch(*args, **kwargs):
        calls.append((args, kwargs))
        return (
            RolloutResult(response_ids=(5, 9), event_cache_lengths=()),
            RolloutResult(response_ids=(6, 9), event_cache_lengths=()),
        )

    monkeypatch.setattr("gcw.eval.greedy_opsd_responses", fake_batch)
    requests = [
        SimpleNamespace(args=("question one", {"until": ["STOP"], "max_gen_toks": 12})),
        SimpleNamespace(args=("question two", {"until": ["STOP"], "max_gen_toks": 12})),
    ]

    output = GCWEvalLM._generate_until_opsd(fake, requests)

    assert output == ["5", "6"]
    assert len(calls) == 1
    assert calls[0][1]["prompts"] == ("question one", "question two")
    assert calls[0][1]["max_response_tokens"] == 2048
    assert all(row["max_response_tokens"] == 2048 for row in fake.opsd_generation_records)
    assert [row["request_index"] for row in fake.opsd_generation_records] == [0, 1]
