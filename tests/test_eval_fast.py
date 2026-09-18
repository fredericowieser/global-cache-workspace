import sys
import types

import pytest
import torch

from gcw.config import EvalConfig
from gcw.evaluation.core import (
    _normalize_until,
    _trim_until,
)
from gcw.evaluation.scoring import (
    classify_generated_tokens,
    token_length_distribution,
    tokenizer_for_checkpoint,
    trim_generated_token_ids,
    truncate_context_continuation,
)
from gcw.evaluation.lm_eval_adapter import GCWEvalLM
from gcw.evaluation.update_metrics import CacheUpdateMetrics


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
    assert lm._metrics()["successful_batch_sizes"]["generation"] == {
        "1": 2,
        "2": 3,
    }


@pytest.mark.parametrize(
    "enable_thinking,expected", [(False, 2048), (True, 4096)]
)
@pytest.mark.parametrize("reference", [False, True])
def test_generation_default_depends_on_thinking_and_task_override_wins(
    monkeypatch, enable_thinking, expected, reference
):
    if reference:
        monkeypatch.setenv("GCW_EVAL_REFERENCE_GENERATE", "1")
    else:
        monkeypatch.delenv("GCW_EVAL_REFERENCE_GENERATE", raising=False)

    seen = []

    class Model:
        def get_device(self):
            return torch.device("cpu")

        def generate(self, ids, *, max_new_tokens, **kwargs):
            seen.append(max_new_tokens)
            return torch.cat((ids, torch.tensor([[9, 0]])), dim=1)

        generate_fast = generate

    class Tokenizer:
        eot_id = 0
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [4]

        def decode(self, tokens):
            return "".join(map(str, tokens))

    lm = GCWEvalLM.__new__(GCWEvalLM)
    lm.model = Model()
    lm.update_metrics = CacheUpdateMetrics(lm.model, None)
    lm._diagnostic_requests = None
    lm.tokenizer = Tokenizer()
    lm.eval_config = EvalConfig(
        apply_chat_template=enable_thinking,
        enable_thinking=enable_thinking,
    )
    lm.max_length = 16
    lm.add_special_tokens_for_generate = False
    lm.batch_size = 1
    lm.reset_runtime_metrics()
    requests = [
        types.SimpleNamespace(args=("question", {})),
        types.SimpleNamespace(args=("question", {"max_gen_toks": 777})),
    ]

    assert lm.generate_until(requests) == ["9", "9"]
    assert seen == [expected, 777]


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


def test_token_length_distribution_exposes_tail_count_and_exact_percentiles():
    summary = token_length_distribution({1: 4, 10: 1}, "answer_tokens")
    assert summary["answer_tokens_stddev"] == pytest.approx(3.6)
    assert summary["answer_tokens_max"] == 10
    assert summary["answer_tokens_p50"] == 1
    assert summary["answer_tokens_p95"] == 10
    assert summary["answer_tokens_p99"] == 10
    assert summary["answer_tokens_above_2x_mean_count"] == 1
    assert summary["answer_tokens_above_2x_mean_rate"] == 0.2


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
    lm._metrics()["successful_batch_sizes"]["generation"] = {"2": 1}

    runtime = lm.pop_runtime_metrics()

    assert runtime["generated_responses"] == 2
    assert runtime["answer_tokens_total"] == 1
    assert runtime["answer_tokens_mean"] == 0.5
    assert runtime["answer_tokens_stddev"] == 0.5
    assert runtime["answer_tokens_max"] == 1
    assert runtime["response_tokens_mean"] == 4.0
    assert runtime["response_tokens_stddev"] == 2.0
    assert runtime["response_tokens_p95"] == 6
    assert runtime["thinking_tokens_total"] == 3
    assert runtime["thinking_tokens_mean"] == 1.5
    assert runtime["thinking_tokens_stddev"] == 0.5
    assert runtime["thinking_tokens_max"] == 2
    assert runtime["generation_cap_hits"] == 1
    assert runtime["generation_cap_hit_rate"] == 0.5
    assert runtime["unterminated_thinking_responses"] == 1
    assert runtime["thinking_truncated_responses"] == 1
    assert runtime["thinking_truncation_rate"] == 0.5
    assert runtime["successful_batch_sizes"] == {"generation": {"2": 1}}
    assert lm._metrics()["generated_responses"] == 0


def test_lm_cap_hits_are_distinct_from_thinking_truncation():
    lm = GCWEvalLM.__new__(GCWEvalLM)
    lm.batch_size = 2
    lm.eval_config = type(
        "Config", (), {"max_batch_size": 2, "enable_thinking": True}
    )()
    lm.tokenizer = ThinkingTokenizer()
    lm.model = type(
        "Model", (), {"get_device": lambda self: torch.device("cpu")}
    )()
    lm.reset_runtime_metrics()
    lm._record_generated_tokens([20, 12, 13, 30], "max_new_tokens")
    lm._record_generated_tokens([20], "eos")

    runtime = lm.pop_runtime_metrics()
    assert runtime["generation_cap_hits"] == 1
    assert runtime["generation_cap_hit_rate"] == 0.5
    assert runtime["unterminated_thinking_responses"] == 1
    assert runtime["thinking_truncated_responses"] == 0


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
    lm._runtime_metrics["successful_batch_sizes"]["generation"] = {"1": 1}
    for category, length in (("response", 3), ("answer", 3), ("thinking", 0)):
        lm._runtime_metrics["token_lengths"][category] = {length: 1}
    monkeypatch.setattr(adapter.dist, "is_available", lambda: True)
    monkeypatch.setattr(adapter.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(adapter.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        adapter.dist,
        "all_reduce",
        lambda counters: counters.add_(torch.tensor([2, 5, 0, 0, 0, 2])),
    )

    def gather(output, local):
        output[0] = local
        output[1] = {
            "batches": {
                "generation": {"1": 2, "2": 1},
                "loglikelihood": {"4": 1},
            },
            "lengths": {
                "response": {2: 1, 3: 1},
                "answer": {2: 1, 3: 1},
                "thinking": {0: 2},
            },
        }

    monkeypatch.setattr(adapter.dist, "all_gather_object", gather)

    runtime = lm.pop_runtime_metrics()

    assert runtime["generated_responses"] == 3
    assert runtime["answer_tokens_total"] == 8
    assert runtime["answer_tokens_max"] == 3
    assert runtime["response_tokens_p95"] == 3
    assert runtime["generation_cap_hits"] == 2
    assert runtime["generation_cap_hit_rate"] == pytest.approx(2 / 3)
    assert runtime["successful_batch_sizes"] == {
        "generation": {"1": 3, "2": 1},
        "loglikelihood": {"4": 1},
    }


def test_stale_partial_is_ignored_and_does_not_resume_progress(
    tmp_path, monkeypatch
):
    import gcw.evaluation.progress as progress

    runs = [(["task"], 0)]
    stale = progress._empty_eval_results()
    stale["results"]["task"] = {"metric": 1.0}
    stale["gcw_runtime"]["task"] = {
        "elapsed_seconds": 1.0,
        "processor_calls": 0,
    }
    monkeypatch.setenv("GCW_EVAL_PARTIAL_DIR", str(tmp_path))
    progress._write_partial_results(stale)
    old_progress = progress._new_progress(runs, resume=False)
    old_progress["started_time"] = 1.0
    progress._write_progress(old_progress, stale)

    with pytest.warns(UserWarning, match="ignoring the stale partial"):
        assert progress._load_partial_results(runs) is None
    fresh_progress = progress._new_progress(runs, resume=False)

    assert fresh_progress["started_time"] != 1.0
    assert fresh_progress["runs"][0]["status"] == "pending"

    stale_distribution = progress._empty_eval_results()
    stale_distribution["results"]["task"] = {"metric": 1.0}
    stale_distribution["gcw_runtime"].update(
        {
            "task": {
                "requested_batch_size": 4,
                "configured_auto_max_batch_size": None,
                "successful_batch_sizes": {"generation": {"1": 1}},
                "generated_responses": 1,
                "generation_cap_hits": 0,
                "generation_cap_hit_rate": 0.0,
            },
            "evaluation": {
                "scope": "cumulative_active_end_to_end_including_loading_and_setup",
                "elapsed_seconds": 5.0,
            },
        }
    )
    progress._write_partial_results(stale_distribution)
    with pytest.warns(UserWarning, match="distribution metrics"):
        assert progress._load_partial_results(runs) is None
    del stale_distribution["gcw_runtime"]["task"]["generation_cap_hits"]
    progress._write_partial_results(stale_distribution)
    with pytest.warns(UserWarning, match="cap-hit metrics"):
        assert progress._load_partial_results(runs) is None

    current = progress._empty_eval_results()
    current["results"]["task"] = {"metric": 1.0}
    current["gcw_runtime"].update(
        {
            "task": {
                "requested_batch_size": 4,
                "configured_auto_max_batch_size": None,
                "successful_batch_sizes": {"loglikelihood": {"4": 2, "1": 1}},
            },
            "evaluation": {
                "scope": (
                    "cumulative_active_end_to_end_including_loading_and_setup"
                ),
                "elapsed_seconds": 5.0,
            },
        }
    )
    progress._write_partial_results(current)
    assert progress._load_partial_results(runs) == current


def test_evaluation_timer_accumulates_resume_time_and_preserves_noop(
    monkeypatch,
):
    import gcw.evaluation.progress as progress

    results = progress._empty_eval_results()
    results["gcw_runtime"]["evaluation"] = {
        "scope": "cumulative_active_end_to_end_including_loading_and_setup",
        "elapsed_seconds": 7.0,
    }
    timer = progress.EvaluationTimer.from_results(results, started=10.0)
    monkeypatch.setattr(progress.time, "monotonic", lambda: 20.0)
    timer.update_results(results)
    assert results["gcw_runtime"]["evaluation"]["elapsed_seconds"] == 7.0

    timer.mark_evaluated()
    monkeypatch.setattr(progress.time, "monotonic", lambda: 23.0)
    timer.update_results(results)
    assert results["gcw_runtime"]["evaluation"] == {
        "scope": "cumulative_active_end_to_end_including_loading_and_setup",
        "elapsed_seconds": 20.0,
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


def test_task_group_cuda_metrics_follow_the_selected_device(monkeypatch):
    import gcw.evaluation.core as core

    class TaskManager:
        def __init__(self, **kwargs):
            pass

    def simple_evaluate(**kwargs):
        return {
            "results": {"task": {"metric": 1.0}},
            "groups": {},
            "n-samples": {},
            "n-shot": {},
            "versions": {},
            "higher_is_better": {},
        }

    tasks_module = types.ModuleType("lm_eval.tasks")
    tasks_module.TaskManager = TaskManager
    tasks_module.get_task_dict = lambda tasks, manager: {}
    monkeypatch.setitem(
        sys.modules,
        "lm_eval",
        types.SimpleNamespace(simple_evaluate=simple_evaluate),
    )
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", tasks_module)
    monkeypatch.setattr(core, "HAS_LM_EVAL", True)
    monkeypatch.setattr(core, "_distributed_eval_state", lambda: (None, 0, 1))
    monkeypatch.setattr(core, "_eval_device", lambda rank: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: pytest.fail(
            "CUDA instrumentation received a CPU device"
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda device: pytest.fail(
            "CUDA instrumentation received a CPU device"
        ),
    )

    class Tokenizer:
        eot_id = 0

        @staticmethod
        def encode(text, **kwargs):
            return [1]

        @staticmethod
        def decode(tokens):
            return ""

    class Model(torch.nn.Module):
        config = type("Config", (), {"sequence_len": 8})()

        @staticmethod
        def get_device():
            return torch.device("cpu")

    result = core.run_task_groups(
        None,
        [(["task"], 0)],
        loaded_target=(Model(), Tokenizer()),
    )

    assert result["gcw_runtime"]["task"]["peak_memory_mib"] == 0.0


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


def test_native_hf_families_require_original_tokenizer(monkeypatch, tmp_path):
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
            {"source": str(tmp_path)},
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
