from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import gcw.eval as eval_module


def _result_for(task: str) -> dict:
    return {
        "results": {task: {"exact_match,none": 0.5}},
        "groups": {},
        "n-samples": {task: {"effective": 1, "original": 1}},
        "n-shot": {task: 0},
        "versions": {task: 1},
        "higher_is_better": {task: {"exact_match,none": True}},
    }


def test_math_runs_are_independently_persisted_units():
    assert eval_module.MATH_RUNS == [([task], 0) for task in eval_module.MATH_TASKS]


def test_hendrycks_resume_leaf_names_match_frozen_evaluation_schema():
    from gcw.flow_math_experiment import HENDRYCKS_MATH_SAMPLE_COUNTS

    assert eval_module.HENDRYCKS_MATH_LEAF_TASKS == set(HENDRYCKS_MATH_SAMPLE_COUNTS)


def test_group_completion_accepts_hendrycks_leaf_closure():
    merged = eval_module._empty_eval_results()
    merged["groups"]["hendrycks_math"] = {"exact_match,none": 0.5}
    for task in eval_module.HENDRYCKS_MATH_LEAF_TASKS:
        merged["results"][task] = {"exact_match,none": 0.5}
    assert eval_module._run_is_complete(["hendrycks_math"], merged)


def test_partial_results_are_authenticated_and_tampering_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("GCW_EVAL_PARTIAL_DIR", str(tmp_path))
    runs = [(["gsm8k_openmath"], 0)]
    payload = eval_module._empty_eval_results()
    payload["results"]["gsm8k_openmath"] = {"exact_match,none": 0.5}
    eval_module._write_partial_results(payload)
    assert eval_module._load_partial_results(runs) == payload

    path = tmp_path / "partial.json"
    tampered = json.loads(path.read_text())
    tampered["payload"]["results"]["svamp"] = {"exact_match,none": 1.0}
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="integrity"):
        eval_module._load_partial_results(runs)


def test_progress_is_authenticated_and_tampering_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("GCW_EVAL_PARTIAL_DIR", str(tmp_path))
    runs = [(["gsm8k_openmath"], 0)]
    merged = eval_module._empty_eval_results()
    progress = eval_module._new_progress(runs)
    eval_module._write_progress(progress, merged)
    assert eval_module._new_progress(runs)["total_runs"] == 1

    path = tmp_path / "progress.json"
    tampered = json.loads(path.read_text())
    tampered["payload"]["completed_runs"] = 1
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="integrity"):
        eval_module._new_progress(runs)


def test_atomic_partial_publication_preserves_previous_snapshot_on_replace_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("GCW_EVAL_PARTIAL_DIR", str(tmp_path))
    runs = [(["gsm8k_openmath"], 0)]
    first = eval_module._empty_eval_results()
    first["results"]["gsm8k_openmath"] = {"exact_match,none": 0.25}
    eval_module._write_partial_results(first)

    replacement = eval_module._empty_eval_results()
    replacement["results"]["gsm8k_openmath"] = {"exact_match,none": 0.75}
    monkeypatch.setattr(eval_module.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError, match="interrupted"):
        eval_module._write_partial_results(replacement)
    assert eval_module._load_partial_results(runs) == first


def test_interrupted_evaluation_resumes_after_completed_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("GCW_EVAL_PARTIAL_DIR", str(tmp_path))
    monkeypatch.delenv("GCW_EVAL_PROGRESS_PATH", raising=False)
    monkeypatch.setattr(eval_module, "HAS_LM_EVAL", True)
    monkeypatch.setattr(eval_module, "load_checkpoint", lambda *_args, **_kwargs: (object(), SimpleNamespace(), {}))
    monkeypatch.setattr(eval_module, "_tokenizer_for", lambda *_args: SimpleNamespace(chat_template=None, default_chat_template=None))
    monkeypatch.setattr(eval_module, "GCWEvalLM", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(eval_module, "_distributed_eval_state", lambda: (None, 0, 1))
    monkeypatch.setattr(eval_module, "_eval_device", lambda _rank: torch.device("cpu"))

    lm_eval = ModuleType("lm_eval")
    tasks_module = ModuleType("lm_eval.tasks")
    tasks_module.TaskManager = lambda **_kwargs: object()
    tasks_module.get_task_dict = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", tasks_module)

    first_calls: list[str] = []

    def first_evaluate(*, tasks, **_kwargs):
        task = tasks[0]
        first_calls.append(task)
        if task == "svamp":
            raise RuntimeError("simulated interruption")
        return _result_for(task)

    lm_eval.simple_evaluate = first_evaluate
    runs = [(["gsm8k_openmath"], 0), (["svamp"], 0)]
    with pytest.raises(RuntimeError, match="simulated interruption"):
        eval_module.run_task_groups("checkpoint", runs, batch_size=16)
    assert first_calls == ["gsm8k_openmath", "svamp"]

    resumed_calls: list[str] = []

    def resumed_evaluate(*, tasks, **_kwargs):
        resumed_calls.append(tasks[0])
        return _result_for(tasks[0])

    lm_eval.simple_evaluate = resumed_evaluate
    merged = eval_module.run_task_groups("checkpoint", runs, batch_size=16)
    assert resumed_calls == ["svamp"]
    assert set(merged["results"]) == {"gsm8k_openmath", "svamp"}
