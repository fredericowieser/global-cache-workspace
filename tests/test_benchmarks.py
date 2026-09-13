from __future__ import annotations

from fractions import Fraction
import hashlib
import json

from gcw.checkpoint import load_checkpoint, save_checkpoint
from gcw.config import CacheProcessorConfig, EvalConfig, ModelConfig
from gcw.models import build_model
from gcw.evaluation.lm_eval_adapter import ProcessorBypass

import pytest
import torch

from gcw.evaluation.benchmarks import (
    DEFAULT_BATCH_SIZE,
    MAX_NEW_TOKENS,
    BenchmarkConfig,
    GCWBenchmarkBackend,
    HFBenchmarkBackend,
    run_gsm8k_reproduction,
    load_benchmark_result,
    apply_qwen_chat_template,
    extract_gsm8k_target,
    extract_lm_eval_flexible,
    evaluate_benchmark,
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
    assert [call[1]["enable_thinking"] for call in tokenizer.calls] == [
        False,
        True,
    ]
    assert all(
        call[1]["tokenize"] and call[1]["add_generation_prompt"]
        for call in tokenizer.calls
    )


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
    result = evaluate_benchmark(
        backend,
        EvaluationTokenizer(),
        [
            {"question": "abc", "answer": "#### 7"},
            {"question": "x", "answer": "#### 7"},
        ],
        config=EvalConfig(batch_size=2, apply_chat_template=True),
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
    result = evaluate_benchmark(
        backend,
        EvaluationTokenizer(),
        [
            {"question": "abc", "answer": "#### 7"},
            {"question": "x", "answer": "#### 7"},
        ],
        config=EvalConfig(batch_size=2, apply_chat_template=True),
        max_new_tokens=MAX_NEW_TOKENS,
        scoring_protocol="lm_eval_flexible_exact",
    )

    assert len(backend.calls) == 2
    assert all(bool(mask.all()) for _, mask in backend.calls)
    assert result["generation"]["grouping"] == "exact_prompt_token_length"


def test_fast_benchmark_batch_is_the_default():
    assert (
        BenchmarkConfig(
            model_path="model", output_dir="out"
        ).eval_config.batch_size
        == DEFAULT_BATCH_SIZE
        == 64
    )


def test_config_rejects_protocol_drift():
    with pytest.raises(ValueError, match=f"max_new_tokens={MAX_NEW_TOKENS}"):
        BenchmarkConfig(
            model_path="model", output_dir="out", max_new_tokens=512
        )
    with pytest.raises(ValueError, match="chat formatting"):
        BenchmarkConfig(
            model_path="model",
            output_dir="out",
            eval_config=EvalConfig(),
        )


def test_historical_matrix_reader_selects_one_boolean_condition(tmp_path):
    path = tmp_path / "historical.json"
    path.write_text(
        json.dumps(
            {
                "schema": "gcw.benchmark_matrix.v3",
                "config": {"modes": ["thinking_off", "thinking_on"]},
                "cells": {
                    "thinking_off": {"mode": "thinking_off", "accuracy": 0.5},
                    "thinking_on": {"mode": "thinking_on", "accuracy": 0.75},
                },
            }
        )
    )
    result = load_benchmark_result(path, enable_thinking=True)
    assert result["schema"] == "gcw.gsm8k_reproduction.v4"
    assert result["enable_thinking"] is True
    assert result["apply_chat_template"] is True
    assert result["accuracy"] == 0.75
    assert "mode" not in result and "modes" not in result["config"]


@pytest.fixture
def small_checkpoint(tmp_path, request):
    variant = request.param
    config = ModelConfig(
        variant=variant,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=16,
        cache_processor=(
            None
            if variant == "tf"
            else CacheProcessorConfig(
                process_every=2,
                generation_process_every=2,
                topk=2,
                topk_heads=1,
                hidden_size=8,
                intermediate_size=16,
                num_layers=1,
                num_heads=1,
                num_kv_heads=1,
            )
        ),
    )
    model = build_model(config)
    model.init_weights()
    save_checkpoint(
        tmp_path, model, config, {"eos_token_id": 63, "pad_token_id": 0}
    )
    return tmp_path


@pytest.mark.parametrize(
    "small_checkpoint", ["tf", "bt", "bgt", "icarus", "longshot"], indirect=True
)
@pytest.mark.parametrize("requested", [False, True])
def test_native_processor_policy_and_tf_off_regression(
    small_checkpoint, requested, monkeypatch
):
    loaded, config, state = load_checkpoint(small_checkpoint)
    monkeypatch.setattr(
        "gcw.checkpoint.load_checkpoint",
        lambda *args, **kwargs: (loaded, config, state),
    )
    present = config.variant != "tf"
    if not present:
        # This unconditional wrapper was the former TF-off failure.
        with pytest.raises(ValueError, match="composed GCW checkpoint"):
            ProcessorBypass(loaded)
    backend = GCWBenchmarkBackend(
        small_checkpoint, torch.device("cpu"), processor_enabled=requested
    )
    assert backend.identity["processor_requested"] is requested
    assert backend.identity["processor_present"] is present
    assert backend.identity["processor_enabled"] is (present and requested)
    assert backend.eos_ids == (63,)
    assert backend.pad_id == 0
    assert not backend.supports_left_padding
    if present and not requested:
        assert isinstance(backend.model, ProcessorBypass)
        assert backend.model.backbone is loaded.backbone
    else:
        assert backend.model is loaded
    assert backend.identity["generation_process_every"] == (
        2 if present else None
    )
    inputs = torch.tensor([[1, 2, 3, 4]])
    generated = backend.generate(inputs, torch.ones_like(inputs), 2)
    assert 4 < generated.shape[1] <= 6
    if present:
        assert (loaded.cache_processor.pop_call_count() > 0) is requested
    if not present or not requested:
        backbone = loaded.backbone if present else loaded
        expected = backbone.eval().generate_fast(
            inputs, max_new_tokens=2, eos_token_id=63, pad_token_id=0
        )
        torch.testing.assert_close(generated, expected)


@pytest.mark.parametrize("small_checkpoint", ["tf"], indirect=True)
@pytest.mark.parametrize("requested", [False, True])
def test_processor_presence_is_not_config_or_backbone_attribute(
    small_checkpoint, requested, monkeypatch
):
    loaded, config, state = load_checkpoint(small_checkpoint)
    config.cache_processor = CacheProcessorConfig()
    loaded.cache_processor = config.cache_processor
    loaded.backbone = torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        "gcw.checkpoint.load_checkpoint",
        lambda *args, **kwargs: (loaded, config, state),
    )
    backend = GCWBenchmarkBackend(
        small_checkpoint, torch.device("cpu"), processor_enabled=requested
    )
    assert backend.model is loaded
    assert backend.identity["processor_requested"] is requested
    assert backend.identity["processor_present"] is False
    assert backend.identity["processor_enabled"] is False
    if requested:
        with pytest.raises(ValueError, match="loaded cache processor"):
            GCWBenchmarkBackend(
                small_checkpoint,
                torch.device("cpu"),
                processor_enabled=True,
                generation_process_every=1,
            )


@pytest.mark.parametrize("small_checkpoint", ["tf", "bgt"], indirect=True)
@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize(
    "override", ["generation_process_every", "diffusion_eval_solver_steps"]
)
def test_processor_overrides_require_applicable_active_processor(
    small_checkpoint, requested, override
):
    _, config, _ = load_checkpoint(small_checkpoint)
    kwargs = {override: 1}
    if (
        requested
        and config.variant == "bgt"
        and override == "generation_process_every"
    ):
        backend = GCWBenchmarkBackend(
            small_checkpoint,
            torch.device("cpu"),
            processor_enabled=True,
            **kwargs,
        )
        assert backend.identity["generation_process_every"] == 1
        assert backend.model.config.cache_processor.process_every == 2
        _, saved_config, _ = load_checkpoint(small_checkpoint)
        assert saved_config.cache_processor.generation_process_every == 2
    else:
        with pytest.raises(ValueError, match="requires|require"):
            GCWBenchmarkBackend(
                small_checkpoint,
                torch.device("cpu"),
                processor_enabled=requested,
                **kwargs,
            )


@pytest.mark.parametrize("requested", [False, True])
def test_hf_has_no_gcw_processor_and_matrix_records_request(
    tmp_path, monkeypatch, requested
):
    from transformers import GPT2Config, GPT2LMHeadModel
    import gcw.evaluation.benchmarks as benchmarks

    source = tmp_path / "hf"
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=64,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            eos_token_id=63,
            pad_token_id=0,
        )
    )
    model.save_pretrained(source)
    backend = HFBenchmarkBackend(
        source, torch.device("cpu"), processor_enabled=requested
    )
    assert isinstance(backend.model, GPT2LMHeadModel)
    assert backend.identity["processor_requested"] is requested
    assert backend.identity["processor_present"] is False
    assert backend.identity["processor_enabled"] is False

    def cpu_backend(path, device, *, processor_enabled):
        assert path == source
        assert processor_enabled is requested
        return backend

    monkeypatch.setattr(benchmarks, "HFBenchmarkBackend", cpu_backend)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test")
    monkeypatch.setattr(
        benchmarks.subprocess, "check_output", lambda *a, **k: ""
    )
    monkeypatch.setattr(
        benchmarks, "_source_commit", lambda path: "test-commit"
    )
    monkeypatch.setattr(
        benchmarks, "_load_tokenizer", lambda path: (EvaluationTokenizer(), {})
    )
    monkeypatch.setattr(benchmarks, "_load_test_rows", lambda limit: ([], {}))
    monkeypatch.setattr(benchmarks, "evaluate_benchmark", lambda *a, **k: {})
    result = run_gsm8k_reproduction(
        BenchmarkConfig(
            model_path=str(source),
            output_dir=str(tmp_path / "result"),
            processor_enabled=requested,
        )
    )
    assert result["schema"] == "gcw.gsm8k_reproduction.v4"
    assert result["config"]["apply_chat_template"] is True
    assert result["config"]["enable_thinking"] is False
    assert result["config"]["processor_requested"] is requested
    assert "processor_enabled" not in result["config"]
    assert result["target"] == backend.identity
    assert json.loads(
        (tmp_path / "result" / "result.json").read_text()
    ) == json.loads(json.dumps(result))
    digest = result.pop("payload_sha256")
    assert (
        digest
        == hashlib.sha256(
            json.dumps(
                result, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    )
