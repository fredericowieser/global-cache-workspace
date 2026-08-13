from __future__ import annotations

from fractions import Fraction

import pytest

from gcw.evaluation.benchmarks import (
    MAX_NEW_TOKENS,
    BenchmarkConfig,
    apply_qwen_chat_template,
    extract_gsm8k_target,
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


def test_config_rejects_protocol_drift():
    with pytest.raises(ValueError, match=f"max_new_tokens={MAX_NEW_TOKENS}"):
        BenchmarkConfig(model_path="model", output_dir="out", max_new_tokens=512)
    with pytest.raises(ValueError, match="duplicates"):
        BenchmarkConfig(model_path="model", output_dir="out", modes=("thinking_off", "thinking_off"))
