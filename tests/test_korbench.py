from __future__ import annotations

import hashlib
import json
from pathlib import Path
import types

import pytest
import torch
import yaml

from gcw.cli import build_parser
from gcw.config import CacheProcessorConfig, EvalConfig, ModelConfig
from gcw.evaluation import korbench
from gcw.evaluation.korbench_scoring import (
    arithmetic_value,
    extract_answer,
    is_correct,
)
from gcw.evaluation.lm_eval_adapter import GCWEvalLM, ProcessorBypass
from gcw.evaluation.scoring import encode_generate_prompt
from gcw.models import build_model

pytest.importorskip(
    "lm_eval", reason="KOR-Bench integration requires the eval extra"
)
pytest.importorskip(
    "antlr4", reason="KOR-Bench symbolic scoring requires the eval extra"
)

FIXTURES = Path(__file__).parent / "fixtures" / "korbench" / "scoring.json"


@pytest.mark.parametrize(
    "case",
    json.loads(FIXTURES.read_text())["cases"],
    ids=lambda c: f"{c['category']}-{c['rule_id']}",
)
def test_pinned_upstream_scoring_parity(case):
    assert (
        bool(
            is_correct(
                case["response"],
                case["answer"],
                case["category"],
                case["rule_id"],
                case["idx"],
            )
        )
        == case["expected"]
    )


@pytest.mark.parametrize(
    "text", ["", "No answer", "[[]]", "[[ ]]", "Reasoning without brackets: 24"]
)
@pytest.mark.parametrize("category", korbench.CATEGORIES)
def test_unparseable_answers_fail(text, category):
    assert not is_correct(text, "[[24]]", category, "1", "1")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').getcwd()",
        "(1).__class__",
        "[1][0]",
        "True",
        "2**1000000",
        "1/0",
        "1e999",
        "1+" * 300 + "1",
    ],
)
def test_arithmetic_does_not_execute_code(expression):
    with pytest.raises((ValueError, SyntaxError, ArithmeticError)):
        arithmetic_value(expression)


def test_rule_specific_grading_and_counterfactual_metric():
    assert arithmetic_value("(2 + 4) * -3 / 2") == -9
    assert is_correct("[[3*(4+4)=24]]", "[[3*(4+4)=24]]", "puzzle", "10", "91")
    assert is_correct(
        "[[x ≥ 2; x ≤ 7]]", "[[x ≤ 7; x ≥ 2]]", "operation", "18", "178"
    )
    assert not is_correct(
        "[[x ≥ 3; x ≤ 7]]", "[[x ≤ 7; x ≥ 2]]", "operation", "18", "178"
    )
    doc = {
        "category": "counterfactual",
        "rule_id": "1",
        "idx": "1",
        "answer": "[[C]]",
        "real_life_answer": "[[A]]",
    }
    assert korbench.process_results(doc, ["[[A]]"]) == {
        "accuracy": 0.0,
        "parse_failure": 0.0,
        "real_life_answer_rate": 1.0,
    }
    assert extract_answer("[[first]] then [[second]]") == "[[first]]"


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source = tmp_path / "source"
    files = {}
    templates = {}
    for category in korbench.CATEGORIES:
        templates[f"{category}_prompt_format"] = [
            "Rules:\n{}\nQuestion:\n{}\nAnswer:\n"
        ]
        rules = [
            {
                "idx": str(i),
                "rule_content": f"Rule {i}: apply the supplied operation.",
            }
            for i in range(1, 26)
        ]
        samples = [
            {
                "idx": str(i),
                "rule_id": str((i - 1) // 10 + 1),
                "question": f"Question {i}?",
                "answer": "[[24]]",
                "needle": ["Do not insert this hint"],
                "trick": "Do not insert this solution",
            }
            for i in range(1, 251)
        ]
        for kind, rows in (("rule", rules), ("sample", samples)):
            name = f"data/{category}/{kind}.jsonl"
            p = source / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("".join(json.dumps(row) + "\n" for row in rows))
            files[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    name = "config/prompt/zero-shot.yaml"
    p = source / name
    p.parent.mkdir(parents=True)
    p.write_text(yaml.safe_dump(templates))
    files[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    pin = {"revision": korbench.REVISION, "files": files}
    monkeypatch.setattr(korbench, "SOURCES", pin)
    output = tmp_path / "prepared"
    manifest = korbench.prepare_dataset(output, source)
    pin["prepared"] = {
        category: item["sha256"] for category, item in manifest["files"].items()
    }
    monkeypatch.setenv("GCW_KORBENCH_DATA", str(output))
    return source, output, manifest


def test_preparation_joins_rules_without_hints_and_verifies_data(prepared):
    _, output, manifest = prepared
    assert sum(item["count"] for item in manifest["files"].values()) == 1250
    dataset = korbench.load_dataset("logic")["test"]
    row = dataset[12]
    assert row["id"] == "logic:13"
    assert row["rule_id"] == "2"
    assert (
        row["prompt"]
        == "Rules:\nRule 2: apply the supplied operation.\nQuestion:\nQuestion 13?\nAnswer:\n"
    )
    assert "needle" not in row and "trick" not in row
    with (output / "logic.jsonl").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="data hash differs"):
        korbench.load_dataset("logic")


def test_preparation_refuses_overwrite_and_changed_upstream(prepared, tmp_path):
    source, output, _ = prepared
    with pytest.raises(FileExistsError):
        korbench.prepare_dataset(output, source)
    (source / "data/logic/sample.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="source hash differs"):
        korbench.prepare_dataset(tmp_path / "another", source)
    assert not (tmp_path / "another").exists()


def test_preparation_rejects_duplicate_ids(prepared, tmp_path):
    source, _, _ = prepared
    name = "data/logic/sample.jsonl"
    p = source / name
    rows = p.read_text().splitlines()
    rows[1] = rows[0]
    p.write_text("\n".join(rows) + "\n")
    korbench.SOURCES["files"][name] = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="question population differs"):
        korbench.prepare_dataset(tmp_path / "duplicate", source)


def test_loader_rejects_changed_protocol(prepared):
    _, output, manifest = prepared
    manifest["revision"] = "wrong"
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned protocol"):
        korbench.load_dataset("logic")


def test_prompt_length_guard_and_cli():
    tokenizer = types.SimpleNamespace(
        encode=lambda text, **kwargs: list(range(len(text)))
    )
    with pytest.raises(ValueError, match="complete prompt"):
        encode_generate_prompt(
            tokenizer, "too long", 3, require_full_prompt=True
        )
    assert encode_generate_prompt(tokenizer, "long", 3) == [1, 2, 3]
    args = build_parser().parse_args(
        [
            "eval",
            "tasks",
            "--checkpoint",
            "model",
            "--tasks",
            "korbench",
            "--log-samples",
        ]
    )
    assert args.log_samples
    with pytest.raises(ValueError, match="log_samples"):
        EvalConfig(log_samples="true")


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_sample_logging_survives_resume_and_rejects_protocol_changes(
    tmp_path, monkeypatch, enable_thinking
):
    import lm_eval
    import lm_eval.tasks
    from gcw.evaluation import core, progress

    monkeypatch.setenv("GCW_EVAL_PARTIAL_DIR", str(tmp_path))
    monkeypatch.delenv("GCW_EVAL_PROGRESS_PATH", raising=False)
    monkeypatch.setattr(lm_eval.tasks, "TaskManager", lambda **kwargs: object())
    monkeypatch.setattr(core, "_distributed_eval_state", lambda: (None, 0, 1))
    monkeypatch.setattr(core, "_eval_device", lambda rank: torch.device("cpu"))
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        assert kwargs["log_samples"] is True
        return {
            "results": {"korbench_operation": {"accuracy,none": 1.0}},
            "groups": {"korbench": {"accuracy,none": 1.0}},
            "n-samples": {
                "korbench_operation": {"original": 250, "effective": 1}
            },
            "configs": {"korbench_operation": {"metadata": {"version": 1}}},
            "samples": {
                "korbench_operation": [
                    {
                        "doc_id": 0,
                        "filter": "none",
                        "doc": {"category": "operation", "id": "operation:1"},
                        "filtered_resps": ["[[24]]"],
                        "resps": [["[[24]]"]],
                        "accuracy": 1.0,
                    }
                ]
            },
        }

    monkeypatch.setattr(lm_eval, "simple_evaluate", evaluate)

    class Model(torch.nn.Module):
        config = types.SimpleNamespace(sequence_len=16)

        def get_device(self):
            return torch.device("cpu")

    tokenizer = types.SimpleNamespace(eot_id=0)
    args = {
        "checkpoint": "checkpoint",
        "runs": [(["korbench"], 0)],
        "config": EvalConfig(
            log_samples=True,
            limit=1,
            apply_chat_template=enable_thinking,
            enable_thinking=enable_thinking,
        ),
        "loaded_target": (Model(), tokenizer),
    }
    result = core.run_task_groups(**args)
    sample = result["samples"]["korbench_operation"][0]
    assert sample["extracted_answer"] == "[[24]]"
    assert (
        result["gcw_runtime"]["korbench"]["benchmark"]["name"]
        == korbench.PROTOCOL
    )
    assert (
        result["gcw_runtime"]["korbench"]["benchmark"]["max_new_tokens"]
        == args["config"].default_max_new_tokens
    )
    resumed = core.run_task_groups(**args)
    assert resumed["samples"] == result["samples"]
    assert len(calls) == 1
    for config in (
        EvalConfig(log_samples=False),
        EvalConfig(log_samples=True, limit=2),
    ):
        with pytest.raises(ValueError, match="changed"):
            core.run_task_groups(**{**args, "config": config})
    partial = progress._load_partial_results(args["runs"])
    partial["samples"]["korbench_operation"] = []
    with pytest.raises(ValueError, match="population differs"):
        progress._validate_partial_results(partial, args["runs"])


@pytest.mark.parametrize("enabled", [False, True])
def test_native_longshot_generation_smoke(enabled):
    torch.manual_seed(7)
    config = ModelConfig(
        variant="longshot",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=16,
        cache_processor=CacheProcessorConfig(
            process_every=2,
            generation_process_every=1,
            topk=2,
            topk_heads=1,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
        ),
    )
    model = build_model(config)
    model.init_weights()
    target = model if enabled else ProcessorBypass(model)
    tokenizer = types.SimpleNamespace(
        eot_id=0,
        eos_token_id=0,
        encode=lambda text, **kwargs: [3, 4, 5, 6],
        decode=lambda ids: " ".join(map(str, ids)),
    )
    evaluator = GCWEvalLM(target, tokenizer, EvalConfig(batch_size=1))
    request = types.SimpleNamespace(
        args=(
            "Rule: A implies B. A holds. Does B hold?",
            {"max_gen_toks": 2, "until": [], "gcw_require_full_prompt": True},
        )
    )
    responses = evaluator.generate_until([request])
    assert len(responses) == 1
    assert evaluator.pop_runtime_metrics()["generated_responses"] == 1
    calls = model.cache_processor.pop_call_count()
    assert (calls > 0) is enabled


def test_harness_group_generation_scoring_and_samples(prepared, tmp_path):
    import shutil
    from lm_eval import evaluator
    from lm_eval.api.model import LM
    from lm_eval.tasks import TaskManager

    _, _, manifest = prepared
    task_root = tmp_path / "tasks"
    shutil.copytree(Path("configs/tasks/korbench"), task_root)
    for category in korbench.CATEGORIES:
        path = task_root / f"{category}.yaml"
        config = yaml.safe_load(path.read_text())
        config["metadata"]["data_sha256"] = manifest["files"][category][
            "sha256"
        ]
        path.write_text(yaml.safe_dump(config))

    class Oracle(LM):
        def loglikelihood(self, requests):
            pytest.fail("KOR-Bench must use generation")

        def loglikelihood_rolling(self, requests):
            pytest.fail("KOR-Bench must use generation")

        def generate_until(self, requests):
            for request in requests:
                assert request.args[0] == request.doc["prompt"]
                assert "max_gen_toks" not in request.args[1]
                assert request.args[1]["gcw_require_full_prompt"] is True
            return [request.doc["answer"] for request in requests]

    manager = TaskManager(include_path=str(task_root), include_defaults=False)
    result = evaluator.evaluate(
        Oracle(),
        manager.load(["korbench"]),
        limit=2,
        bootstrap_iters=0,
        log_samples=True,
    )
    korbench.annotate_results(result)
    assert result["results"]["korbench"]["accuracy,none"] == 1.0
    for category in korbench.CATEGORIES:
        task = f"korbench_{category}"
        assert result["n-samples"][task] == {"original": 250, "effective": 2}
        assert result["results"][task]["accuracy,none"] == 1.0
        assert len(result["samples"][task]) == 2
        assert result["samples"][task][0]["extracted_answer"] is not None
