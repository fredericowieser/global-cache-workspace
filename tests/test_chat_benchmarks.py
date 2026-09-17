from __future__ import annotations

from fractions import Fraction
import hashlib
import json

import pytest
import torch

from gcw.config import EvalConfig
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
    tokenizer = None
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


def test_native_chat_requires_tokenizer_before_loading():
    with pytest.raises(ValueError, match="requires tokenizer_path"):
        run_chat_benchmarks(tasks=["piqa"], checkpoint="unused")


def test_chat_generation_cadence_requires_active_gcw_processor():
    try:
        run_chat_benchmarks(
            tasks=["gsm8k_openmath"],
            config=EvalConfig(batch_size=1, apply_chat_template=True),
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
        config=EvalConfig(
            batch_size=2,
            apply_chat_template=True,
            enable_thinking=True,
        ),
    )

    assert result["protocol"] == "official_chat_template_answer_generation_v2"
    assert result["max_new_tokens"] == 4096
    assert result["enable_thinking"] is True
    assert result["tasks"]["piqa"]["accuracy"] == 1.0
    assert result["tasks"]["svamp"]["accuracy"] == 1.0
    assert result["tasks"]["piqa"]["parse_rate"] == 1.0
    assert result["tasks"]["svamp"]["truncation_rate"] == 0.0
    assert result["tasks"]["svamp"]["generation_cap_hits"] == 0
    assert result["rows"][0]["generation_cap_hit"] is False
    assert result["tasks"]["piqa"]["answer_tokens"] == 0
    assert result["tasks"]["piqa"]["thinking_tokens"] == 1
    assert result["tasks"]["piqa"]["thinking_tokens_stddev"] == 0.0
    assert result["tasks"]["piqa"]["thinking_tokens_max"] == 1
    assert result["tasks"]["piqa"]["thinking_tokens_p95"] == 1
    assert result["tasks"]["piqa"]["response_tokens_max"] == 1
    assert result["tasks"]["piqa"]["unterminated_thinking"] == 1
    assert result["tasks"]["piqa"]["thinking_truncation_rate"] == 0.0
    assert result["rows"][0]["thinking_terminated"] is False
    assert result["rows"][0]["thinking_truncated"] is False
    assert backend.calls[0][2] == 4096


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_chat_token_length_distribution_exposes_long_tail(enable_thinking):
    class VariableLengthBackend(FakeBackend):
        def generate(self, input_ids, attention_mask, max_new_tokens):
            suffix = torch.full((input_ids.shape[0], 11), 99, dtype=torch.long)
            for row, length in enumerate((1, 1, 1, 1, 10)):
                suffix[row, :length] = 21
            return torch.cat((input_ids, suffix), dim=1)

    examples = [
        ChatExample(
            index, "piqa", "piqa", index, "prompt", "choice", "B", ("x", "y")
        )
        for index in range(5)
    ]
    result = evaluate_chat_examples(
        VariableLengthBackend(),
        FakeTokenizer(),
        examples,
        config=EvalConfig(
            batch_size=5,
            apply_chat_template=True,
            enable_thinking=enable_thinking,
        ),
    )
    task = result["tasks"]["piqa"]
    assert task["response_tokens_mean"] == 2.8
    assert task["response_tokens_stddev"] == pytest.approx(3.6)
    assert task["response_tokens_max"] == 10
    assert task["response_tokens_p50"] == 1
    assert task["response_tokens_p95"] == 10
    assert task["response_tokens_above_2x_mean_count"] == 1
    assert task["response_tokens_above_2x_mean_rate"] == 0.2
    category = "thinking_tokens" if enable_thinking else "answer_tokens"
    assert task[f"{category}_stddev"] == pytest.approx(3.6)
    assert task[f"{category}_max"] == 10


def test_chat_cap_hits_include_max_length_and_exclude_eos():
    class MixedStopBackend(FakeBackend):
        def generate(self, input_ids, attention_mask, max_new_tokens):
            assert max_new_tokens == 3
            return torch.cat(
                (input_ids, torch.tensor([[21, 22, 23], [21, 99, 99]])), dim=1
            )

    examples = [
        ChatExample(
            index, "piqa", "piqa", index, "prompt", "choice", "B", ("x", "y")
        )
        for index in range(2)
    ]
    result = evaluate_chat_examples(
        MixedStopBackend(),
        FakeTokenizer(),
        examples,
        config=EvalConfig(batch_size=2, apply_chat_template=True),
        max_new_tokens=3,
    )
    assert [row["generation_cap_hit"] for row in result["rows"]] == [
        True,
        False,
    ]
    assert result["tasks"]["piqa"]["generation_cap_hits"] == 1
    assert result["tasks"]["piqa"]["generation_cap_hit_rate"] == 0.5
    assert result["tasks"]["piqa"]["truncated"] == 1


@pytest.mark.parametrize("target", ["hf", "tf", "bgt"])
@pytest.mark.parametrize("requested", [False, True])
def test_chat_processor_metadata_comes_from_backend(
    tmp_path, monkeypatch, target, requested
):
    import gcw.evaluation.chat_benchmarks as chat

    backend = FakeBackend()
    present = target == "bgt"
    backend.identity = {
        "kind": "hf" if target == "hf" else "gcw",
        "processor_requested": requested,
        "processor_present": present,
        "processor_enabled": requested and present,
    }

    def load_backend(path, device, *, processor_enabled, **kwargs):
        assert path == tmp_path
        assert processor_enabled is requested
        return backend

    monkeypatch.setattr(chat, "HFBenchmarkBackend", load_backend)
    monkeypatch.setattr(chat, "GCWBenchmarkBackend", load_backend)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        chat,
        "_load_tokenizer",
        lambda path, tokenizer=None: (FakeTokenizer(), {}),
    )
    monkeypatch.setattr(
        chat,
        "_load_task_examples",
        lambda tasks: (
            [
                ChatExample(
                    0, "piqa", "piqa", 0, "prompt", "choice", "B", ("x", "y")
                )
            ],
            {},
        ),
    )
    result = run_chat_benchmarks(
        tasks=["piqa"],
        config=EvalConfig(batch_size=1, apply_chat_template=True),
        tokenizer_path=tmp_path,
        processor_enabled=requested,
        **{"hf_model" if target == "hf" else "checkpoint": tmp_path},
    )
    assert result["schema"] == "gcw.chat_benchmark.v8"
    for field in (
        "processor_requested",
        "processor_present",
        "processor_enabled",
    ):
        assert result[field] is backend.identity[field]
        assert result[field] is result["target"][field]
    assert result["tasks"]["piqa"]["accuracy"] == 1.0
    assert result["end_to_end_elapsed_seconds"] >= result["elapsed_seconds"]
    digest = result.pop("payload_sha256")
    assert (
        digest
        == hashlib.sha256(
            json.dumps(
                result, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    )


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize(
    "override", ["generation_process_every", "diffusion_eval_solver_steps"]
)
def test_chat_rejects_processor_only_overrides_for_hf(requested, override):
    with pytest.raises(ValueError, match="HF targets|only to GCW"):
        run_chat_benchmarks(
            tasks=["piqa"],
            config=EvalConfig(batch_size=1, apply_chat_template=True),
            hf_model="unused-local-path",
            processor_enabled=requested,
            **{override: 1},
        )
