from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

from gcw.evaluation import legacy
import gcw.evaluation.lm_eval_adapter as lm_eval_adapter


def test_paper7_legacy_task_order_and_skip(monkeypatch):
    assert legacy.PAPER7_TASKS == [
        "gsm8k_openmath",
        "svamp",
        "theoremqa",
        "logiqa",
        "agieval_gaokao_mathqa",
        "gsm8k_hard",
    ]

    monkeypatch.setenv("GCW_EVAL_SKIP_TASKS", "theoremqa")
    assert legacy.run_paper7_legacy_groups.__globals__["_filter_skipped_tasks"](legacy.PAPER7_RUNS) == [
        (["gsm8k_openmath"], 0),
        (["svamp"], 0),
        (["logiqa"], 0),
        (["agieval_gaokao_mathqa"], 0),
        (["gsm8k_hard"], 0),
    ]


def test_legacy_task_yaml_preserves_martin_generation_caps():
    root = Path("configs/tasks_legacy/paper7")
    assert "max_gen_toks: 512" in (root / "gsm8k_openmath.yaml").read_text()
    assert "revision:" not in (root / "gsm8k_openmath.yaml").read_text()
    assert "max_gen_toks: 256" in (root / "svamp.yaml").read_text()
    assert "max_gen_toks: 512" in (root / "gsm8k_hard.yaml").read_text()
    assert "max_gen_toks: 2048" in (root / "theoremqa.yaml").read_text()


def test_legacy_generate_add_special_tokens_only_for_old_hf_families():
    def model(family, legacy_generate=True, variant="longshot"):
        config = type(
            "Config",
            (),
            {
                "family": family,
                "legacy_bottleneck_generate": legacy_generate,
                "variant": variant,
            },
        )()
        return type("Model", (), {"config": config})()

    assert legacy._legacy_generate_add_special_tokens(model("qwen3_5"))
    assert legacy._legacy_generate_add_special_tokens(model("gemma4"))
    assert not legacy._legacy_generate_add_special_tokens(model("qwen3_5", False))
    assert not legacy._legacy_generate_add_special_tokens(model("smollm"))


def test_legacy_chat_template_does_not_pass_thinking_kwarg(monkeypatch):
    monkeypatch.setattr(lm_eval_adapter, "HAS_LM_EVAL", True)
    calls = []

    class Tokenizer:
        eot_id = 0
        chat_template = "template"
        default_chat_template = None

        def apply_chat_template(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "rendered"

        def encode(self, text, add_special_tokens=False):
            return [1, 2]

        def decode(self, ids):
            return ""

    class Model(torch.nn.Module):
        config = type(
            "Config",
            (),
            {
                "sequence_len": 8,
                "family": "qwen3_5",
                "legacy_bottleneck_generate": True,
                "variant": "longshot",
            },
        )()

        def get_device(self):
            return torch.device("cpu")

    lm = legacy.LegacyPaper7EvalLM(Model(), Tokenizer())
    assert lm.add_special_tokens_for_generate
    assert lm.apply_chat_template([{"role": "user", "content": "hi"}]) == "rendered"
    assert calls[0][1] == {"tokenize": False, "add_generation_prompt": True}


def test_run_paper7_legacy_calls_lm_eval_with_martin_protocol(monkeypatch):
    monkeypatch.setattr(legacy, "HAS_LM_EVAL", True)
    monkeypatch.setattr(lm_eval_adapter, "HAS_LM_EVAL", True)
    monkeypatch.setenv("GCW_EVAL_SKIP_TASKS", "theoremqa")

    calls = []
    task_manager_paths = []

    class TaskManager:
        def __init__(self, include_path, verbosity):
            task_manager_paths.append((include_path, verbosity))

    def simple_evaluate(**kwargs):
        calls.append(kwargs)
        task = kwargs["tasks"][0]
        return {
            "results": {task: {"metric": 1.0}},
            "groups": {},
            "n-samples": {task: {"effective": 1, "original": 1}},
            "n-shot": {task: 0},
            "versions": {task: 1},
            "higher_is_better": {task: {"metric": True}},
        }

    lm_eval_module = types.SimpleNamespace(simple_evaluate=simple_evaluate)
    tasks_module = types.ModuleType("lm_eval.tasks")
    tasks_module.TaskManager = TaskManager
    tasks_module.get_task_dict = lambda tasks, manager: {}
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_module)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", tasks_module)

    class Tokenizer:
        eot_id = 0
        chat_template = "template"
        default_chat_template = None

        def encode(self, text, add_special_tokens=False):
            return [1, 2]

        def decode(self, ids):
            return ""

        def apply_chat_template(self, *args, **kwargs):
            return ""

    class Model(torch.nn.Module):
        config = type(
            "Config",
            (),
            {
                "sequence_len": 8,
                "family": "qwen3_5",
                "legacy_bottleneck_generate": True,
                "variant": "longshot",
            },
        )()

        def get_device(self):
            return torch.device("cpu")

    monkeypatch.setattr(
        legacy,
        "_load_eval_target",
        lambda **kwargs: (Model(), Tokenizer()),
    )
    monkeypatch.setattr(legacy, "_distributed_eval_state", lambda: (None, 0, 1))
    monkeypatch.setattr(legacy, "_eval_device", lambda local_rank: torch.device("cpu"))
    monkeypatch.setattr(legacy, "_write_progress", lambda progress, merged: None)
    monkeypatch.setattr(legacy, "_write_partial_results", lambda merged: None)

    out = legacy.run_paper7_legacy("checkpoint", batch_size=16)

    assert [call["tasks"] for call in calls] == [
        ["gsm8k_openmath"],
        ["svamp"],
        ["logiqa"],
        ["agieval_gaokao_mathqa"],
        ["gsm8k_hard"],
    ]
    first = calls[0]
    assert first["num_fewshot"] == 0
    assert first["bootstrap_iters"] == 0
    assert first["log_samples"] is False
    assert first["confirm_run_unsafe_code"] is True
    assert first["apply_chat_template"] is True
    assert "configs/tasks_legacy/paper7" in task_manager_paths[0][0]
    assert set(out) == {"results", "groups", "n-samples", "n-shot", "versions", "higher_is_better"}
    assert "gcw_protocol" not in out
    assert "gcw_runtime" not in out
