from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

from gcw.config import EvalConfig, ModelConfig
from gcw.evaluation import legacy, paper7
import gcw.evaluation.lm_eval_adapter as lm_eval_adapter
from gcw.models.cache_processors.longshot_processor import CacheProcessor


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
    assert paper7._filter_skipped_tasks(legacy.PAPER7_RUNS) == [
        (["gsm8k_openmath"], 0),
        (["svamp"], 0),
        (["logiqa"], 0),
        (["agieval_gaokao_mathqa"], 0),
        (["gsm8k_hard"], 0),
    ]


def test_legacy_task_yaml_uses_thinking_generation_budget():
    root = Path("configs/tasks_legacy/paper7")
    gsm8k = (root / "gsm8k_openmath.yaml").read_text()
    svamp = (root / "svamp.yaml").read_text()
    hard = (root / "gsm8k_hard.yaml").read_text()
    theoremqa = (root / "theoremqa.yaml").read_text()
    logiqa = (root / "logiqa.yaml").read_text()
    assert "max_gen_toks: 2048" in gsm8k
    assert "version: 1.2" in gsm8k
    assert "revision:" not in gsm8k
    assert "max_gen_toks: 2048" in svamp
    assert "version: 1.1" in svamp
    assert "max_gen_toks: 2048" in hard
    assert "version: 1.1" in hard
    assert "max_gen_toks: 2048" in theoremqa
    assert "version: 2.1" in theoremqa
    assert "version: 1.1" in logiqa


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
    assert not legacy._legacy_generate_add_special_tokens(
        model("qwen3_5", False)
    )
    assert not legacy._legacy_generate_add_special_tokens(model("smollm"))


def test_legacy_chat_template_explicitly_disables_thinking_by_default(
    monkeypatch,
):
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
    assert (
        lm.apply_chat_template([{"role": "user", "content": "hi"}])
        == "rendered"
    )
    assert calls[0][1] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_legacy_chat_template_can_enable_thinking(monkeypatch):
    monkeypatch.setattr(lm_eval_adapter, "HAS_LM_EVAL", True)
    calls = []

    class Tokenizer:
        eot_id = 0

        def apply_chat_template(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "rendered"

    class Model(torch.nn.Module):
        config = type("Config", (), {"sequence_len": 8})()

    lm = legacy.LegacyPaper7EvalLM(
        Model(),
        Tokenizer(),
        EvalConfig(apply_chat_template=True, enable_thinking=True),
    )
    assert (
        lm.apply_chat_template([{"role": "user", "content": "hi"}])
        == "rendered"
    )
    assert calls[0][1]["enable_thinking"] is True


def test_eval_target_applies_generation_cadence_without_changing_checkpoint(
    monkeypatch,
):
    config = ModelConfig(variant="bgt")
    model = type("Model", (), {"config": config})()
    monkeypatch.setattr(
        lm_eval_adapter,
        "load_checkpoint",
        lambda *args, **kwargs: (model, config, {}),
    )
    monkeypatch.setattr(
        lm_eval_adapter,
        "tokenizer_for_checkpoint",
        lambda config, state: "tokenizer",
    )

    loaded, tokenizer = lm_eval_adapter._load_eval_target(
        checkpoint="checkpoint",
        hf_model=None,
        processor_enabled=True,
        device=torch.device("cpu"),
        generation_process_every=1,
    )

    assert loaded is model
    assert tokenizer == "tokenizer"
    assert config.cache_processor is not None
    assert config.cache_processor.generation_process_every == 1


def test_eval_target_applies_random_head_selection_after_checkpoint_load(
    monkeypatch,
):
    config = ModelConfig(
        variant="longshot",
        cache_processor={
            "head_selection_method": "learned_contextual",
            "internal_iteration_score": "learned_contextual",
            "iterations": 2,
            "encoder_seq_len": 3,
        },
    )
    assert config.cache_processor is not None
    processor = CacheProcessor(16, 4, config.cache_processor)
    assert processor.head_selector is not None
    model = type("Model", (), {"config": config})()
    monkeypatch.setattr(
        lm_eval_adapter,
        "load_checkpoint",
        lambda *args, **kwargs: (model, config, {}),
    )
    monkeypatch.setattr(
        lm_eval_adapter,
        "tokenizer_for_checkpoint",
        lambda config, state: "tokenizer",
    )

    loaded, tokenizer = lm_eval_adapter._load_eval_target(
        checkpoint="checkpoint",
        hf_model=None,
        processor_enabled=True,
        device=torch.device("cpu"),
        head_selection_method="random",
    )

    assert loaded is model
    assert tokenizer == "tokenizer"
    assert config.cache_processor.head_selection_method == "random"
    assert config.cache_processor.internal_iteration_score == "random"

    def reject_learned_routing(*_args, **_kwargs):
        raise AssertionError("runtime override invoked the learned selector")

    monkeypatch.setattr(
        processor.head_selector, "route", reject_learned_routing
    )
    with torch.no_grad():
        processor(torch.randn(2, 4, 3, 16), topk_heads=2)


def test_eval_target_rejects_head_selection_override_for_non_longshot():
    config = ModelConfig(variant="bgt")
    try:
        lm_eval_adapter._apply_head_selection_method(config, "random")
    except ValueError as exc:
        assert "Longshot checkpoint" in str(exc)
    else:
        raise AssertionError("accepted random head routing for non-Longshot")


def test_eval_target_rejects_generation_cadence_without_active_processor():
    common = {
        "processor_enabled": True,
        "device": torch.device("cpu"),
        "generation_process_every": 1,
    }
    try:
        lm_eval_adapter._load_eval_target(
            checkpoint=None, hf_model="hf-model", **common
        )
    except ValueError as exc:
        assert "only to GCW checkpoints" in str(exc)
    else:
        raise AssertionError("accepted generation cadence for an HF target")

    try:
        lm_eval_adapter._load_eval_target(
            checkpoint="checkpoint",
            hf_model=None,
            **{**common, "processor_enabled": False},
        )
    except ValueError as exc:
        assert "processor_enabled=True" in str(exc)
    else:
        raise AssertionError("accepted generation cadence with processor off")


@pytest.mark.parametrize(
    ("chat_template", "expected_apply_chat_template"),
    [("template", True), (None, False)],
)
def test_run_paper7_legacy_calls_lm_eval_with_martin_protocol(
    monkeypatch, chat_template, expected_apply_chat_template
):
    monkeypatch.setattr(paper7, "HAS_LM_EVAL", True)
    monkeypatch.setattr(lm_eval_adapter, "HAS_LM_EVAL", True)
    monkeypatch.setenv("GCW_EVAL_SKIP_TASKS", "theoremqa")

    calls = []
    load_calls = []
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
        chat_template = None
        default_chat_template = None

        def encode(self, text, add_special_tokens=False):
            return [1, 2]

        def decode(self, ids):
            return ""

        def apply_chat_template(self, *args, **kwargs):
            return ""

    Tokenizer.chat_template = chat_template

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
        paper7,
        "_load_eval_target",
        lambda **kwargs: (load_calls.append(kwargs) or Model(), Tokenizer()),
    )
    monkeypatch.setattr(paper7, "_distributed_eval_state", lambda: (None, 0, 1))
    monkeypatch.setattr(
        paper7, "_eval_device", lambda local_rank: torch.device("cpu")
    )
    monkeypatch.setattr(
        paper7, "_write_progress", lambda progress, merged: None
    )
    monkeypatch.setattr(paper7, "_write_partial_results", lambda merged: None)

    eval_config = EvalConfig(batch_size=16, apply_chat_template=True)
    out = legacy.run_paper7_legacy(
        "checkpoint",
        config=eval_config,
        generation_process_every=1,
        head_selection_method="random",
    )

    assert load_calls[0]["generation_process_every"] == 1
    assert load_calls[0]["head_selection_method"] == "random"

    assert [call["tasks"] for call in calls] == [
        ["gsm8k_openmath"],
        ["svamp"],
        ["logiqa"],
        ["agieval_gaokao_mathqa"],
        ["gsm8k_hard"],
    ]
    first = calls[0]
    assert isinstance(first["model"], legacy.LegacyPaper7EvalLM)
    assert first["num_fewshot"] == 0
    assert first["bootstrap_iters"] == 0
    assert first["log_samples"] is False
    assert first["confirm_run_unsafe_code"] is True
    assert first["apply_chat_template"] is expected_apply_chat_template
    assert (
        first["model"].eval_config.apply_chat_template
        is expected_apply_chat_template
    )
    assert eval_config.apply_chat_template is expected_apply_chat_template
    assert first["model"].eval_config.enable_thinking is False
    assert "configs/tasks_legacy/paper7" in task_manager_paths[0][0]
    assert set(out) == {
        "results",
        "groups",
        "n-samples",
        "n-shot",
        "versions",
        "higher_is_better",
        "gcw_runtime",
    }
    assert "gcw_protocol" not in out
    assert "evaluation" in out["gcw_runtime"]
    assert "gsm8k_openmath" in out["gcw_runtime"]


def test_run_paper7_uses_mixed_prompt_protocol(monkeypatch):
    monkeypatch.setattr(paper7, "HAS_LM_EVAL", True)
    monkeypatch.setattr(lm_eval_adapter, "HAS_LM_EVAL", True)

    calls = []

    class TaskManager:
        def __init__(self, include_path, verbosity):
            pass

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
        paper7,
        "_load_eval_target",
        lambda **kwargs: (Model(), Tokenizer()),
    )
    monkeypatch.setattr(paper7, "_distributed_eval_state", lambda: (None, 0, 1))
    monkeypatch.setattr(
        paper7, "_eval_device", lambda local_rank: torch.device("cpu")
    )
    monkeypatch.setattr(
        paper7, "_write_progress", lambda progress, merged: None
    )
    monkeypatch.setattr(paper7, "_write_partial_results", lambda merged: None)

    out = paper7.run_paper7(
        "checkpoint",
        config=EvalConfig(
            batch_size=16,
            apply_chat_template=True,
            enable_thinking=True,
        ),
    )

    formatting = {
        call["tasks"][0]: call["apply_chat_template"] for call in calls
    }
    assert len({id(call["model"]) for call in calls}) == 1
    assert type(calls[0]["model"]) is lm_eval_adapter.GCWEvalLM
    assert formatting == {
        "gsm8k_openmath": True,
        "svamp": True,
        "theoremqa": True,
        "logiqa": False,
        "agieval_gaokao_mathqa": False,
        "gsm8k_hard": True,
    }
    assert out["gcw_protocol"] == {
        "name": "paper7_mixed_prompt_v1",
        "version": 1,
        "tasks": {
            "gsm8k_openmath": {
                "apply_chat_template": True,
                "enable_thinking": True,
                "request_type": "answer_generation",
            },
            "svamp": {
                "apply_chat_template": True,
                "enable_thinking": True,
                "request_type": "answer_generation",
            },
            "theoremqa": {
                "apply_chat_template": True,
                "enable_thinking": True,
                "request_type": "answer_generation",
            },
            "logiqa": {
                "apply_chat_template": False,
                "enable_thinking": False,
                "request_type": "multiple_choice_likelihood",
            },
            "agieval_gaokao_mathqa": {
                "apply_chat_template": False,
                "enable_thinking": False,
                "request_type": "multiple_choice_likelihood",
            },
            "gsm8k_hard": {
                "apply_chat_template": True,
                "enable_thinking": True,
                "request_type": "answer_generation",
            },
        },
    }
