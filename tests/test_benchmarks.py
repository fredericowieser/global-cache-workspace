from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from gcw.evaluation.benchmarks import (
    DEFAULT_BATCH_SIZE,
    MAX_NEW_TOKENS,
    BenchmarkConfig,
    apply_qwen_chat_template,
    extract_gsm8k_target,
    extract_lm_eval_flexible,
    evaluate_mode,
    score_lm_eval_flexible,
    score_numeric_response,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [1, 2, 4 if kwargs["enable_thinking"] else 3]


def test_qwen_template_binds_only_requested_thinking_mode():
    tokenizer = FakeTokenizer()
    off = apply_qwen_chat_template(tokenizer, "question", enable_thinking=False)
    on = apply_qwen_chat_template(tokenizer, "question", enable_thinking=True)

    assert off == (1, 2, 3)
    assert on == (1, 2, 4)
    assert [call[1]["enable_thinking"] for call in tokenizer.calls] == [False, True]
    assert all(call[1]["tokenize"] and call[1]["add_generation_prompt"] for call in tokenizer.calls)


def test_gsm8k_target_and_response_use_exact_numeric_values():
    assert extract_gsm8k_target("work\n#### 1,234") == Fraction(1234)
    assert score_numeric_response("The answer is 1,234.", Fraction(1234))
    assert score_numeric_response(r"Final: \boxed{7/2}", Fraction(7, 2))
    assert not score_numeric_response("The answer is 1,235.", Fraction(1234))


def test_lm_eval_flexible_protocol_matches_historical_last_regex():
    response = r"work with 12, then final $18."
    assert extract_lm_eval_flexible(response) == "$18."
    assert score_lm_eval_flexible(response, "work\n#### 18")
    assert not score_lm_eval_flexible("answer 17", "work\n#### 18")


class RecordingBackend:
    identity = {}
    eos_ids = (9,)
    pad_id = 0
    device = torch.device("cpu")

    def __init__(self, supports_left_padding: bool) -> None:
        self.supports_left_padding = supports_left_padding
        self.calls = []

    def generate(self, input_ids, attention_mask, _max_new_tokens):
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        suffix = torch.tensor([[7, 9]] * input_ids.shape[0])
        return torch.cat((input_ids, suffix), dim=1)


class EvaluationTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        width = len(messages[0]["content"])
        return list(range(1, width + 1))

    @staticmethod
    def decode(_tokens, **_kwargs):
        return "7"


def test_hf_evaluation_left_pads_variable_length_prompts():
    backend = RecordingBackend(supports_left_padding=True)
    result = evaluate_mode(
        backend,
        EvaluationTokenizer(),
        [
            {"question": "abc", "answer": "#### 7"},
            {"question": "x", "answer": "#### 7"},
        ],
        mode="thinking_off",
        batch_size=2,
        max_new_tokens=MAX_NEW_TOKENS,
        scoring_protocol="lm_eval_flexible_exact",
    )

    input_ids, attention_mask = backend.calls[0]
    assert input_ids.tolist() == [[0, 0, 1], [1, 2, 3]]
    assert attention_mask.tolist() == [[0, 0, 1], [1, 1, 1]]
    assert result["generation"]["grouping"] == "length_sorted_left_padding"
    assert result["correct"] == 2


def test_native_evaluation_keeps_exact_length_batches():
    backend = RecordingBackend(supports_left_padding=False)
    result = evaluate_mode(
        backend,
        EvaluationTokenizer(),
        [
            {"question": "abc", "answer": "#### 7"},
            {"question": "x", "answer": "#### 7"},
        ],
        mode="thinking_off",
        batch_size=2,
        max_new_tokens=MAX_NEW_TOKENS,
        scoring_protocol="lm_eval_flexible_exact",
    )

    assert len(backend.calls) == 2
    assert all(bool(mask.all()) for _, mask in backend.calls)
    assert result["generation"]["grouping"] == "exact_prompt_token_length"


def test_fast_benchmark_batch_is_the_default():
    assert BenchmarkConfig(model_path="model", output_dir="out").batch_size == DEFAULT_BATCH_SIZE == 128


def test_config_rejects_protocol_drift():
    with pytest.raises(ValueError, match=f"max_new_tokens={MAX_NEW_TOKENS}"):
        BenchmarkConfig(model_path="model", output_dir="out", max_new_tokens=512)
    with pytest.raises(ValueError, match="duplicates"):
        BenchmarkConfig(model_path="model", output_dir="out", modes=("thinking_off", "thinking_off"))
    with pytest.raises(ValueError, match="no GCW processor"):
        BenchmarkConfig(model_path="model", output_dir="out", target_kind="hf", processor_enabled=False)
