from __future__ import annotations

import copy

import pytest

from configs.tasks.utils_opsd_mc import process_gaokao, process_logiqa
from gcw.cli import _atomic_write_new
from gcw.eval import (
    OPSD_FULL_TASK_SAMPLE_COUNTS,
    OPSD_SCREEN_TASKS,
    _opsd_dataset_file_identity,
    _opsd_sample_manifest,
    _validate_opsd_full_sample_counts,
)
from gcw.opsd_eval import (
    OPSD_SCREEN_CONDITIONS,
    _condition_metrics,
    aggregate_opsd_screen,
    aggregate_opsd_three_way,
    audit_opsd_prompt_overlap,
)


from scripts import log_opsd_matrix_wandb as wandb_publisher
from scripts.log_opsd_matrix_wandb import flatten_finite_metrics


def test_wandb_eval_flattening_keeps_only_finite_numeric_metrics():
    metrics = flatten_finite_metrics(
        {
            "score": 0.5,
            "nested": {"count": 3, "ok": True, "bad": float("nan")},
            "text": "ignore",
        }
    )

    assert metrics == {
        "eval/score": 0.5,
        "eval/nested/count": 3.0,
        "eval/nested/ok": 1.0,
    }



def test_wandb_eval_publisher_is_online_and_checkpoint_step_bound(
    tmp_path, monkeypatch
):
    tasks = list(OPSD_SCREEN_TASKS)
    out = tmp_path / "matrix"
    checkpoint = tmp_path / "checkpoint"
    out.mkdir()
    checkpoint.mkdir()
    (checkpoint / "train_state.json").write_text(
        __import__("json").dumps({"optimizer_steps": 1000})
    )
    aggregate = {"tasks": tasks, "metrics": {"score": 0.5}}
    (out / "aggregate.json").write_text(__import__("json").dumps(aggregate))
    for name in (
        "cot-off__cp-off.json",
        "cot-off__cp-on.json",
        "cot-on__cp-off.json",
        "cot-on__cp-on.json",
    ):
        (out / name).write_text("{}")

    init_calls = []
    logged = []

    class FakeArtifact:
        def __init__(self, name, **kwargs):
            self.name = name
            self.files = []

        def add_file(self, path, name):
            assert __import__("pathlib").Path(path).is_file()
            self.files.append(name)

    class FakeRun:
        id = "eval-run-id"
        url = "https://wandb.example/eval-run-id"

        def log(self, metrics, step):
            logged.append((metrics, step))

        def log_artifact(self, artifact):
            assert len(artifact.files) == 5

        def finish(self):
            return None

    fake_wandb = type(
        "FakeWandb",
        (),
        {
            "init": staticmethod(
                lambda **kwargs: (init_calls.append(kwargs) or FakeRun())
            ),
            "Artifact": FakeArtifact,
        },
    )()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    monkeypatch.setenv("OUT_DIR", str(out))
    monkeypatch.setenv("CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)

    wandb_publisher.main()

    assert init_calls[0]["mode"] == "online"
    assert init_calls[0]["config"]["training_step"] == 1000
    assert logged[0][1] == 1000
    evidence = __import__("json").loads((out / "wandb-publication.json").read_text())
    assert evidence["wandb_run_id"] == "eval-run-id"
    assert evidence["tasks"] == tasks



def test_generative_multiple_choice_scoring_requires_explicit_terminal_answer():
    logiqa = {"options": ["one", "two", "three", "four"], "label": "b"}
    gaokao = {"choices": ["one", "two", "three", "four"], "gold": [2]}

    assert process_logiqa(logiqa, ["A is tempting.\nFinal answer: B"]) == {"acc": 1.0}
    assert process_logiqa(logiqa, ["Reasoning ends here. Answer: B"]) == {"acc": 1.0}
    assert process_logiqa(logiqa, ["Therefore, Final answer is B."]) == {"acc": 1.0}
    assert process_logiqa(logiqa, ["Conclusion: Answer: B"]) == {"acc": 1.0}
    assert process_logiqa(logiqa, ["推理。答案：B"]) == {"acc": 1.0}
    assert process_logiqa(logiqa, ["推理，答案：B"]) == {"acc": 1.0}
    assert process_logiqa(logiqa, ["Reasoning — Answer: B"]) == {"acc": 1.0}
    assert process_logiqa(logiqa, ["Reasoning… Answer: B"]) == {"acc": 1.0}
    assert process_logiqa(logiqa, ["Reasoning. **Answer:** **B**"]) == {"acc": 1.0}
    assert process_logiqa(logiqa, ["Reasoning. **Answer**: B"]) == {"acc": 1.0}
    assert process_logiqa(logiqa, ["Reasoning.\n**Final answer is B.**"]) == {"acc": 1.0}
    assert process_logiqa(logiqa, ["Reasoning.\nAnswer: D"]) == {"acc": 0.0}
    assert process_gaokao(gaokao, ["Choice A fails; therefore \\boxed{C}"]) == {"acc": 1.0}
    assert process_gaokao(gaokao, ["推理。\n答案：C"]) == {"acc": 1.0}

    ambiguous_or_nonfinal = (
        ({**logiqa, "label": "a"}, "A reasonable derivation gives the result."),
        ({**logiqa, "label": "d"}, "A is tempting. I conclude C because D is wrong."),
        ({**logiqa, "label": "a"}, "Final answer: A and D"),
        ({**logiqa, "label": "d"}, "the final answer is definitely C"),
        ({**logiqa, "label": "c"}, "answeris C"),
        ({**logiqa, "label": "b"}, "Answer: (B"),
        ({**logiqa, "label": "b"}, "Answer: B)"),
        ({**logiqa, "label": "b"}, "Answer: A. Answer: B"),
        ({**logiqa, "label": "b"}, "The incorrect answer: B"),
        (logiqa, "no final option"),
    )
    for document, response in ambiguous_or_nonfinal:
        assert process_logiqa(document, [response]) == {"acc": 0.0}


def test_condition_metrics_retains_task_specific_accuracy_metrics():
    tasks = list(OPSD_SCREEN_TASKS)
    results = {
        task: (
            {"acc,none": 0.75, "alias": task}
            if task in {"logiqa_opsd_generate", "agieval_gaokao_mathqa_opsd_generate"}
            else {"exact_match,strict-match": 0.5, "alias": task}
        )
        for task in tasks
    }
    document = {
        "tasks": tasks,
        "lm_eval": {"results": results},
        "generation_records": [
            {"response_tokens": 10, "max_response_tokens": 20}
        ],
    }

    metrics = _condition_metrics(document)

    assert metrics["logiqa_opsd_generate"] == {"acc,none": 0.75}
    assert metrics["agieval_gaokao_mathqa_opsd_generate"] == {"acc,none": 0.75}
    assert metrics["gsm8k_openmath"] == {"exact_match,strict-match": 0.5}

    broken = copy.deepcopy(document)
    broken["lm_eval"]["results"]["logiqa_opsd_generate"] = {"alias": "logiqa"}
    with pytest.raises(ValueError, match="no supported finite accuracy metric"):
        _condition_metrics(broken)



def test_opsd_screen_supports_full_five_task_matrix_and_authenticates_files(
    tmp_path, monkeypatch
):
    expected = (
        "gsm8k_openmath",
        "svamp",
        "gsm8k_hard",
        "logiqa_opsd_generate",
        "agieval_gaokao_mathqa_opsd_generate",
    )
    assert OPSD_SCREEN_TASKS == expected
    filenames = (
        "gsm8k-test.arrow",
        "svamp-test.arrow",
        "gsm-hard-train.arrow",
        "logiqa-test.arrow",
        "agieval-gaokao-mathqa-test.arrow",
    )
    for filename in filenames:
        (tmp_path / filename).write_bytes(filename.encode())
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))

    identity = _opsd_dataset_file_identity(expected)

    assert tuple(identity) == expected
    assert {value["path"].rsplit("/", 1)[-1] for value in identity.values()} == set(
        filenames
    )



def test_opsd_full_sample_counts_fail_closed_on_incomplete_split():
    manifest = [
        {"task": task}
        for task in OPSD_SCREEN_TASKS
        for _ in range(OPSD_FULL_TASK_SAMPLE_COUNTS[task])
    ]
    _validate_opsd_full_sample_counts(manifest, tasks=OPSD_SCREEN_TASKS, limit=None)

    with pytest.raises(ValueError, match="sample counts"):
        _validate_opsd_full_sample_counts(
            manifest[:-1], tasks=OPSD_SCREEN_TASKS, limit=None
        )
    _validate_opsd_full_sample_counts(
        manifest[:-1], tasks=OPSD_SCREEN_TASKS, limit=64
    )



def _document(name: str, score: float) -> dict:
    processor_enabled = name.endswith("cp-on")
    return {
        "source_revision": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "checkpoint_manifest": {
            "files": {
                "model.safetensors": {"size_bytes": 1, "sha256": "b" * 64},
                "config.json": {"size_bytes": 2, "sha256": "2" * 64},
                "train_state.json": {"size_bytes": 3, "sha256": "3" * 64},
            },
            "manifest_sha256": "4" * 64,
        },
        "model_config": {"variant": "bgt"},
        "tokenizer_source": "/models/qwen3-sft",
        "tokenizer_identity_sha256": "f" * 64,
        "datasets_cache": "/cache",
        "tasks": ["gsm8k_openmath"],
        "task_config_sha256": {"gsm8k_openmath": "c" * 64},
        "dataset_files": {
            "gsm8k_openmath": {
                "path": "/cache/gsm8k-test.arrow",
                "size_bytes": 123,
                "sha256": "1" * 64,
            }
        },
        "limit_per_task": 1,
        "generation_policy": {
            "do_sample": False,
            "temperature": 0.0,
            "top_k": 1,
            "fewshot": 0,
            "apply_chat_template_in_lm_eval": False,
        },
        "condition": {
            "name": name,
            "enable_thinking": name.startswith("cot-on"),
            "processor_enabled": processor_enabled,
        },
        "sample_manifest": [
            {
                "task": "gsm8k_openmath",
                "doc_id": 0,
                "doc_hash": "5" * 64,
                "prompt_hash": "6" * 64,
                "target_hash": "7" * 64,
            }
        ],
        "generation_records": [
            {
                "request_index": 0,
                "prompt_sha256": "d" * 64,
                "prompt_text_sha256": "d" * 64,
                "normalized_prompt_text_sha256": "e" * 64,
                "base_token_ids_sha256": "8" * 64,
                "full_prompt_token_ids_sha256": (
                    "9" * 64 if name.startswith("cot-off") else "a" * 64
                ),
                "base_tokens": 5,
                "trigger_tokens": 2,
                "response_tokens": 5,
                "max_response_tokens": 10,
                "raw_response_tokens": 5,
                "cache_event_positions": [0] if processor_enabled else [],
                "cache_event_lengths": [5] if processor_enabled else [],
            }
        ],
        "lm_eval": {
            "results": {
                "gsm8k_openmath": {
                    "exact_match,flexible": score,
                    "exact_match_stderr,flexible": 0.01,
                    "sample_len": 1,
                    "alias": "gsm8k_openmath",
                }
            }
        },
    }


def test_opsd_screen_aggregate_computes_causal_contrasts():
    documents = [
        _document("cot-off__cp-off", 0.1),
        _document("cot-off__cp-on", 0.2),
        _document("cot-on__cp-off", 0.3),
        _document("cot-on__cp-on", 0.5),
    ]

    result = aggregate_opsd_screen(documents)

    assert result["conditions"] == list(OPSD_SCREEN_CONDITIONS)
    assert result["prompt_count"] == 1
    assert result["metrics"]["cot-on__cp-on"]["gsm8k_openmath"] == {
        "exact_match,flexible": 0.5
    }
    assert result["contrasts"]["cp_effect_cot_off"]["gsm8k_openmath"]["exact_match,flexible"] == pytest.approx(0.1)
    assert result["contrasts"]["cp_effect_cot_on"]["gsm8k_openmath"]["exact_match,flexible"] == pytest.approx(0.2)
    assert result["contrasts"]["interaction"]["gsm8k_openmath"]["exact_match,flexible"] == pytest.approx(0.1)


def test_opsd_screen_aggregate_rejects_prompt_manifest_drift():
    documents = [_document(name, 0.1) for name in OPSD_SCREEN_CONDITIONS]
    documents = copy.deepcopy(documents)
    documents[-1]["generation_records"][0]["prompt_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="prompt manifest"):
        aggregate_opsd_screen(documents)



def test_opsd_screen_publication_is_atomic_and_refuses_overwrite(tmp_path):
    output = tmp_path / "condition.json"
    _atomic_write_new(output, '{"ok": true}')

    assert output.read_text() == '{"ok": true}'
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _atomic_write_new(output, '{"ok": false}')



def test_opsd_sample_manifest_deduplicates_lm_eval_filter_rows():
    identity = {
        "doc_id": 7,
        "doc_hash": "a",
        "prompt_hash": "b",
        "target_hash": "c",
    }
    evaluated = {"samples": {"gsm8k_openmath": [identity, dict(identity), dict(identity)]}}

    assert _opsd_sample_manifest(evaluated, ("gsm8k_openmath",)) == [
        {"task": "gsm8k_openmath", **identity}
    ]



def test_opsd_screen_aggregate_rejects_impossible_cache_lengths():
    documents = [
        _document(name, score)
        for name, score in zip(OPSD_SCREEN_CONDITIONS, (0.1, 0.2, 0.3, 0.5), strict=True)
    ]
    documents[1]["generation_records"][0]["cache_event_lengths"] = [999999]

    with pytest.raises(ValueError, match="cache-event evidence"):
        aggregate_opsd_screen(documents)


def test_opsd_screen_aggregate_rejects_condition_flag_mismatch():
    documents = [
        _document(name, score)
        for name, score in zip(OPSD_SCREEN_CONDITIONS, (0.1, 0.2, 0.3, 0.5), strict=True)
    ]
    documents[1]["condition"]["processor_enabled"] = False
    documents[1]["generation_records"][0]["cache_event_positions"] = []
    documents[1]["generation_records"][0]["cache_event_lengths"] = []

    with pytest.raises(ValueError, match="condition name disagrees"):
        aggregate_opsd_screen(documents)


def test_atomic_write_new_refuses_broken_symlink(tmp_path):
    destination = tmp_path / "broken.json"
    destination.symlink_to(tmp_path / "missing-target")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _atomic_write_new(destination, "new")

    assert destination.is_symlink()
    assert not destination.resolve().exists()


def _three_way_documents() -> dict[str, dict]:
    off = _document("cot-off__cp-off", 0.0)
    untrained = _document("cot-off__cp-on", 0.0)
    trained = _document("cot-off__cp-on", 1.0)
    for document, score, response in (
        (off, 0, "off"),
        (untrained, 0, "untrained"),
        (trained, 1, "trained"),
    ):
        document["lm_eval"]["samples"] = {
            "gsm8k_openmath": [
                {
                    "doc_id": 0,
                    "filter": "flexible",
                    "exact_match": score,
                    "resps": [[response]],
                }
            ]
        }
        document["train_state"] = {
            "dtype": "mixed_bfloat16_backbone_float32_processor",
            "training_seed": 24,
        }
    trained["train_state"]["post_load_processor_sha256"] = "9" * 64
    off["train_state"]["post_load_processor_sha256"] = "9" * 64
    untrained["checkpoint_sha256"] = "e" * 64
    untrained["checkpoint_manifest"] = copy.deepcopy(untrained["checkpoint_manifest"])
    untrained["checkpoint_manifest"]["files"]["model.safetensors"]["sha256"] = "e" * 64
    untrained["checkpoint_manifest"]["files"]["train_state.json"]["sha256"] = "5" * 64
    untrained["checkpoint_manifest"]["manifest_sha256"] = "f" * 64
    untrained["train_state"] = {
        "kind": "derived_untrained_fp32_cache_processor_control",
        "dtype": "mixed_bfloat16_backbone_float32_processor",
        "source_revision": "a" * 40,
        "source_model_sha256": "b" * 64,
        "tokenizer_identity_sha256": "f" * 64,
        "initial_processor_sha256": "8" * 64,
        "trained_processor_sha256": "9" * 64,
        "processor_initialization_seed": 24,
        "processor_storage_dtype": "float32",
        "source_backbone_sha256": "1" * 64,
        "control_backbone_sha256": "1" * 64,
        "checkpoint_dtype_counts": {"torch.bfloat16": 2, "torch.float32": 1},
    }
    return {"off": off, "untrained": untrained, "trained": trained}


def test_opsd_three_way_aggregate_computes_paired_control_contrasts():
    result = aggregate_opsd_three_way(
        _three_way_documents(),
        bootstrap_seed=24,
        bootstrap_samples=100,
        expected_tasks=("gsm8k_openmath",),
        expected_limit=1,
    )

    assert result["conditions"] == ["off", "untrained", "trained"]
    assert result["prompt_count"] == 1
    assert result["accuracy"]["trained"]["gsm8k_openmath"] == 1.0
    assert (
        result["paired_contrasts"]["trained_minus_untrained"]["tasks"]
        ["gsm8k_openmath"]["delta"]
        == 1.0
    )
    assert result["response_identity_counts"]["off_equals_untrained"] == {
        "gsm8k_openmath": 0
    }


def test_opsd_three_way_aggregate_rejects_noncanonical_protocols():
    documents = _three_way_documents()
    with pytest.raises(ValueError, match="tasks differ from the canonical"):
        aggregate_opsd_three_way(documents, bootstrap_samples=10)

    documents = _three_way_documents()
    documents["trained"]["generation_policy"]["do_sample"] = True
    with pytest.raises(ValueError, match="generation policy"):
        aggregate_opsd_three_way(
            documents,
            bootstrap_samples=10,
            expected_tasks=("gsm8k_openmath",),
            expected_limit=1,
        )

    documents = _three_way_documents()
    documents["untrained"]["train_state"]["processor_initialization_seed"] = 25
    with pytest.raises(ValueError, match="FP32 untrained processor provenance"):
        aggregate_opsd_three_way(
            documents,
            bootstrap_samples=10,
            expected_tasks=("gsm8k_openmath",),
            expected_limit=1,
        )


def test_opsd_prompt_overlap_audit_reports_exact_and_normalized_collisions():
    documents = _three_way_documents()
    held_out = audit_opsd_prompt_overlap(
        [
            {
                "source_index": 0,
                "prompt_text_sha256": "a" * 64,
                "normalized_prompt_text_sha256": "b" * 64,
            }
        ],
        documents,
        expected_training_rows=1,
        expected_evaluation_prompts=1,
    )
    assert held_out["held_out_prompt_text_verified"]

    collision = audit_opsd_prompt_overlap(
        [
            {
                "source_index": 0,
                "prompt_text_sha256": "d" * 64,
                "normalized_prompt_text_sha256": "e" * 64,
            }
        ],
        documents,
        expected_training_rows=1,
        expected_evaluation_prompts=1,
    )
    assert collision["exact_text_overlap_count"] == 1
    assert collision["normalized_text_overlap_count"] == 1
    assert not collision["held_out_prompt_text_verified"]


def test_opsd_three_way_aggregate_rejects_untrained_provenance_drift():
    documents = _three_way_documents()
    documents["untrained"]["train_state"]["processor_storage_dtype"] = "bfloat16"

    with pytest.raises(ValueError, match="FP32 untrained processor provenance"):
        aggregate_opsd_three_way(
            documents,
            bootstrap_seed=24,
            bootstrap_samples=10,
            expected_tasks=("gsm8k_openmath",),
            expected_limit=1,
        )
