import pytest
import torch

from gcw.config import EvalConfig
from gcw.evaluation.core import (
    _normalize_until,
    _trim_until,
)
from gcw.evaluation.scoring import (
    classify_generated_tokens,
    tokenizer_for_checkpoint,
    trim_generated_token_ids,
    truncate_context_continuation,
)
from gcw.evaluation.lm_eval_adapter import GCWEvalLM


def test_auto_batching_retries_cuda_oom_and_retains_reduced_limit():
    lm = GCWEvalLM.__new__(GCWEvalLM)
    lm.batch_size = "auto"
    lm._auto_batch_sizes = {"generation": 8}
    attempts = []

    def run_batch(rows):
        attempts.append(len(rows))
        if len(rows) > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return [row * 2 for row in rows]

    assert lm._run_batches(list(range(5)), "generation", run_batch) == [
        0,
        2,
        4,
        6,
        8,
    ]
    assert attempts == [5, 2, 2, 1]
    assert lm._auto_batch_sizes["generation"] == 2

    attempts.clear()
    assert lm._run_batches(list(range(3)), "generation", run_batch) == [0, 2, 4]
    assert attempts == [2, 1]
    assert lm._metrics()["successful_batch_sizes"]["generation"] == [
        2,
        2,
        1,
        2,
        1,
    ]


class ThinkingTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"<think>": [10, 11], "</think>": [12, 13]}[text]


def test_generated_token_classification_handles_thinking_delimiters():
    tokenizer = ThinkingTokenizer()
    disabled = classify_generated_tokens(
        tokenizer,
        [10, 11, 20, 12, 13, 30],
        enable_thinking=False,
        termination="stop",
    )
    assert (disabled.answer_tokens, disabled.thinking_tokens) == (6, 0)

    closed = classify_generated_tokens(
        tokenizer,
        [10, 11, 20, 21, 12, 13, 30, 31],
        enable_thinking=True,
        termination="stop",
    )
    assert (closed.answer_tokens, closed.thinking_tokens) == (2, 2)
    assert closed.thinking_terminated is True
    assert closed.thinking_truncated is False

    without_opening = classify_generated_tokens(
        tokenizer,
        [20, 12, 13, 30],
        enable_thinking=True,
        termination="stop",
    )
    assert (without_opening.answer_tokens, without_opening.thinking_tokens) == (
        1,
        1,
    )


def test_unterminated_thinking_distinguishes_stop_from_token_cap():
    tokenizer = ThinkingTokenizer()
    stopped = classify_generated_tokens(
        tokenizer,
        [10, 11, 20],
        enable_thinking=True,
        termination="eos",
    )
    capped = classify_generated_tokens(
        tokenizer,
        [20, 21],
        enable_thinking=True,
        termination="max_new_tokens",
    )
    assert stopped.answer_tokens == capped.answer_tokens == 0
    assert stopped.thinking_tokens == 1
    assert stopped.thinking_terminated is False
    assert stopped.thinking_truncated is False
    assert capped.thinking_tokens == 2
    assert capped.thinking_truncated is True


def test_lm_runtime_metrics_aggregate_and_reset():
    lm = GCWEvalLM.__new__(GCWEvalLM)
    lm.batch_size = "auto"
    lm.eval_config = type(
        "Config", (), {"max_batch_size": 8, "enable_thinking": True}
    )()
    lm.tokenizer = ThinkingTokenizer()
    lm.model = type(
        "Model", (), {"get_device": lambda self: torch.device("cpu")}
    )()
    lm.reset_runtime_metrics()
    lm._record_generated_tokens([10, 11, 20, 12, 13, 30], "eos")
    lm._record_generated_tokens([20, 21], "max_new_tokens")
    lm._metrics()["successful_batch_sizes"]["generation"] = [2]

    runtime = lm.pop_runtime_metrics()

    assert runtime["generated_responses"] == 2
    assert runtime["answer_tokens_total"] == 1
    assert runtime["answer_tokens_mean"] == 0.5
    assert runtime["thinking_tokens_total"] == 3
    assert runtime["thinking_tokens_mean"] == 1.5
    assert runtime["unterminated_thinking_responses"] == 1
    assert runtime["thinking_truncated_responses"] == 1
    assert runtime["thinking_truncation_rate"] == 0.5
    assert runtime["successful_batch_sizes"] == {"generation": [2]}
    assert lm._metrics()["generated_responses"] == 0


def test_lm_runtime_metrics_aggregate_across_distributed_ranks(monkeypatch):
    import gcw.evaluation.lm_eval_adapter as adapter

    lm = GCWEvalLM.__new__(GCWEvalLM)
    lm.batch_size = 4
    lm.eval_config = type(
        "Config", (), {"max_batch_size": 8, "enable_thinking": False}
    )()
    lm.model = type(
        "Model", (), {"get_device": lambda self: torch.device("cpu")}
    )()
    lm.reset_runtime_metrics()
    lm._runtime_metrics.update({"generated_responses": 1, "answer_tokens": 3})
    lm._runtime_metrics["successful_batch_sizes"]["generation"] = [1]
    monkeypatch.setattr(adapter.dist, "is_available", lambda: True)
    monkeypatch.setattr(adapter.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(adapter.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        adapter.dist,
        "all_reduce",
        lambda counters: counters.add_(torch.tensor([2, 5, 0, 0, 0])),
    )

    def gather(output, local):
        output[0] = local
        output[1] = {"generation": [2], "loglikelihood": [4]}

    monkeypatch.setattr(adapter.dist, "all_gather_object", gather)

    runtime = lm.pop_runtime_metrics()

    assert runtime["generated_responses"] == 3
    assert runtime["answer_tokens_total"] == 8
    assert runtime["successful_batch_sizes"] == {
        "generation": [1, 2],
        "loglikelihood": [4],
    }


def test_fineweb_reports_execution_and_end_to_end_timing(monkeypatch):
    import gcw.evaluation.bpb as bpb

    model = object()
    checkpoint_config = type(
        "Config",
        (),
        {
            "family": "smollm",
            "vocab_size": bpb.COSMO2_VOCAB_SIZE,
            "sequence_len": 4,
        },
    )()
    monkeypatch.setattr(
        bpb,
        "load_checkpoint",
        lambda *args, **kwargs: (model, checkpoint_config, {"step": 3}),
    )
    monkeypatch.setattr(bpb, "data_loader", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        bpb, "checkpoint_loss_backend", lambda *args, **kwargs: "reference"
    )
    monkeypatch.setattr(bpb, "get_token_bytes", lambda device: object())
    monkeypatch.setattr(
        bpb, "evaluate_nll_bpb", lambda *args, **kwargs: (1.25, 2.5)
    )

    result = bpb.run_fineweb(
        "checkpoint",
        eval_tokens=8,
        config=EvalConfig(batch_size=2),
    )

    assert result["val_nll"] == 1.25
    assert result["elapsed_seconds"] >= 0.0
    assert result["end_to_end_elapsed_seconds"] >= result["elapsed_seconds"]


def test_auto_batching_does_not_hide_batch_one_oom():
    lm = GCWEvalLM.__new__(GCWEvalLM)
    lm.batch_size = "auto"
    lm._auto_batch_sizes = {"loglikelihood": 1}

    def run_batch(_rows):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        lm._run_batches([1], "loglikelihood", run_batch)


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
