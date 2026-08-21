from __future__ import annotations

from fractions import Fraction

import torch

from gcw.evaluation.chat_benchmarks import (
    ChatExample,
    evaluate_chat_examples,
    extract_final_choice,
    extract_final_choices,
    format_multiple_choice_prompt,
    format_numeric_prompt,
    run_chat_benchmarks,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        marker = 9 if kwargs["enable_thinking"] else 8
        return [1, marker, len(messages[0]["content"])]

    @staticmethod
    def decode(tokens, **_kwargs):
        marker = tokens[0]
        if marker == 21:
            return "reasoning\nFinal answer: B"
        return "reasoning\nFinal answer: 7/2"


class FakeBackend:
    identity = {"kind": "fake"}
    eos_ids = (99,)
    pad_id = 0
    device = torch.device("cpu")

    def __init__(self, *, supports_left_padding: bool = True):
        self.supports_left_padding = supports_left_padding
        self.calls = []

    def generate(self, input_ids, attention_mask, max_new_tokens):
        self.calls.append(
            (input_ids.clone(), attention_mask.clone(), max_new_tokens)
        )
        markers = torch.tensor([[21, 99], [22, 99]], dtype=torch.long)[
            : input_ids.shape[0]
        ]
        return torch.cat((input_ids, markers), dim=1)


def test_multiple_choice_prompt_is_explicit_and_requires_final_letter():
    prompt = format_multiple_choice_prompt(
        "Question?\nAnswer:", ["first", "second"]
    )
    assert "A. first" in prompt
    assert "B. second" in prompt
    assert "Final answer: X" in prompt
    assert "more than one" not in prompt
    multi_prompt = format_multiple_choice_prompt(
        "Question?", ["first", "second"], allow_multiple=True
    )
    assert "Final answer: A,D" in multi_prompt
    assert not prompt.startswith("Answer:")


def test_final_choice_extraction_requires_an_explicit_final_answer():
    assert extract_final_choice("work\nFinal answer: C", 4) == 2
    assert extract_final_choice(r"work \boxed{B}", 4) == 1
    assert extract_final_choice("I considered A and B", 4) is None
    assert extract_final_choice("Final answer: Z", 4) is None
    assert extract_final_choices("work\nFinal answer: A,D", 4) == (0, 3)
    assert extract_final_choice("work\n**Final answer: D**", 4) == 3
    assert extract_final_choice("work\nFinal answer: C. -3", 4) == 2


def test_numeric_prompt_preserves_the_calibrated_question():
    assert format_numeric_prompt("  What is 1 + 1?  ") == "What is 1 + 1?"


def test_chat_generation_cadence_requires_active_gcw_processor():
    try:
        run_chat_benchmarks(
            tasks=["gsm8k_openmath"],
            mode="thinking_off",
            batch_size=1,
            hf_model="hf-model",
            generation_process_every=1,
        )
    except ValueError as exc:
        assert "only to GCW checkpoints" in str(exc)
    else:
        raise AssertionError("accepted generation cadence for an HF target")


def test_chat_evaluation_scores_choices_numbers_and_protocol_telemetry():
    backend = FakeBackend()
    examples = [
        ChatExample(
            0, "piqa", "piqa", 0, "prompt one", "choice", "B", ("x", "y")
        ),
        ChatExample(
            1, "svamp", "svamp", 0, "prompt two", "numeric", str(Fraction(7, 2))
        ),
    ]
    result = evaluate_chat_examples(
        backend,
        FakeTokenizer(),
        examples,
        mode="thinking_on",
        batch_size=2,
    )

    assert result["protocol"] == "official_chat_template_answer_generation_v1"
    assert result["max_new_tokens"] == 2048
    assert result["tasks"]["piqa"]["accuracy"] == 1.0
    assert result["tasks"]["svamp"]["accuracy"] == 1.0
    assert result["tasks"]["piqa"]["parse_rate"] == 1.0
    assert result["tasks"]["svamp"]["truncation_rate"] == 0.0
    assert backend.calls[0][2] == 2048
