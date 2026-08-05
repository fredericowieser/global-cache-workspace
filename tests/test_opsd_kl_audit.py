from __future__ import annotations

from copy import deepcopy
import json

import pytest
import torch
import torch.nn.functional as F

from gcw.config import DistillationConfig

import gcw.distill.audit as audit_module
from gcw.distill.audit import (
    OpsdMatchedAuditConfig,
    aggregate_matched_opsd_audit,
    classify_teacher_completion,
    dataset_solution_token_nll,
    iter_heldout_source_rows,
    load_opsd_matched_audit_config,
    run_matched_opsd_audit,
)


def _rows(*, trained: bool) -> list[dict]:
    kls = (0.1, 0.5) if trained else (0.2, 0.4)
    return [
        {
            "source_index": source_index,
            "prompt_text_sha256": ("a" if source_index == 128 else "0") * 64,
            "normalized_prompt_text_sha256": ("9" if source_index == 128 else "7") * 64,
            "dataset_answer_text_sha256": "b" * 64,
            "dataset_solution_ids_sha256": "8" * 64,
            "student_response_ids_sha256": ("c" if source_index == 128 else "d") * 64,
            "teacher_reasoning_ids_sha256": ("e" if source_index == 128 else "f") * 64,
            "raw_teacher_completion_ids_sha256": ("5" if source_index == 128 else "6") * 64,
            "teacher_trace_termination": "native_close",
            "raw_teacher_completion_tokens": 3,
            "teacher_reasoning_tokens": 3,
            "response_anchor": "seed_identical_untrained_cp_on",
            "student_prompt_ids_sha256": ("1" if source_index == 128 else "2") * 64,
            "teacher_prompt_ids_sha256": ("3" if source_index == 128 else "4") * 64,
            "teacher_logits_sha256": ("d" if source_index == 128 else "c") * 64,
            "event_cache_lengths": [10, 74],
            "answer_tokens": answer_tokens,
            "dataset_solution_tokens": dataset_tokens,
            "exact_forward_kl": kl,
            "no_privilege_dataset_solution_nll": no_privilege,
            "self_reasoning_dataset_solution_nll": self_reasoning,
        }
        for source_index, answer_tokens, dataset_tokens, kl, no_privilege, self_reasoning in (
            (128, 2, 1, kls[0], 0.8, 0.6),
            (129, 1, 3, kls[1], 0.4, 0.5),
        )
    ]


def _config(**overrides) -> OpsdMatchedAuditConfig:
    values = {
        "train_config": "pilot.yaml",
        "trained_checkpoint": "trained",
        "untrained_checkpoint": "untrained",
        "training_metrics": "metrics.jsonl",
        "evaluation_inputs": ["off.json", "untrained.json", "trained.json"],
        "output": "audit.json",
        "expected_source_revision": "1" * 40,
    }
    values.update(overrides)
    return OpsdMatchedAuditConfig(**values)


def test_matched_opsd_audit_config_locks_production_protocol():
    config = _config()

    assert config.source_skip_rows == 128
    assert config.valid_examples == 32
    assert config.max_attempts == 64
    assert config.student_seed == 24
    assert config.bootstrap_samples == 10_000


def test_matched_opsd_audit_rejects_native_full_objective():
    distill = DistillationConfig(
        teacher_trajectory_mode="native_full",
        visible_trajectory_mode="both",
        close_reasoning_at_eos=False,
    )

    with pytest.raises(ValueError, match="legacy reasoning_only/student"):
        audit_module._validate_matched_audit_distillation_protocol(distill)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_skip_rows", 129),
        ("valid_examples", 31),
        ("max_attempts", 63),
        ("student_seed", 25),
        ("bootstrap_samples", 9999),
    ],
)
def test_matched_opsd_audit_config_rejects_protocol_drift(field, value):
    with pytest.raises(ValueError, match="production protocol"):
        _config(**{field: value})


def test_matched_opsd_audit_publishes_fresh_result_atomically(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"
    config = _config(output=str(output))
    expected = {"kind": "test-audit", "schema_version": 1}
    monkeypatch.setattr(audit_module, "_execute_matched_opsd_audit", lambda _: expected)

    actual = run_matched_opsd_audit(config)

    assert actual == expected
    assert json.loads(output.read_text()) == expected
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_matched_opsd_audit(config)


def test_matched_opsd_audit_refuses_existing_output_before_loading_artifacts(tmp_path):
    output = tmp_path / "audit.json"
    output.write_text("existing")
    config_path = tmp_path / "audit.yaml"
    values = {
        "train_config": "missing-pilot.yaml",
        "trained_checkpoint": "missing-trained",
        "untrained_checkpoint": "missing-untrained",
        "training_metrics": "missing-metrics.jsonl",
        "evaluation_inputs": ["missing-off", "missing-u", "missing-t"],
        "output": str(output),
        "expected_source_revision": "1" * 40,
    }
    config_path.write_text(json.dumps(values))

    config = load_opsd_matched_audit_config(config_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_matched_opsd_audit(config)


def test_heldout_source_rows_preserve_raw_source_indices(tmp_path):
    source = tmp_path / "rows.jsonl"
    rows = [
        {"prompt": "training", "answer": "zero"},
        {"prompt": "held one", "answer": "one"},
        {"prompt": "held two", "answer": "two"},
        {"prompt": "outside", "answer": "three"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    selected = list(iter_heldout_source_rows(source, skip_rows=1, max_attempts=2))

    assert [(row.source_index, row.prompt, row.dataset_answer) for row in selected] == [
        (1, "held one", "one"),
        (2, "held two", "two"),
    ]


def test_live_mixed_dtype_validation_rejects_wrong_processor_dtype():
    class MixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2).to(torch.bfloat16)
            self.cache_processor = torch.nn.Linear(2, 2).to(torch.float32)

    model = MixedModel()
    audit_module._validate_live_mixed_dtype(model, label="test")
    model.cache_processor.to(torch.bfloat16)
    with pytest.raises(ValueError, match="FP32 processor"):
        audit_module._validate_live_mixed_dtype(model, label="test")


def test_teacher_completion_records_native_close():
    reasoning, mode = classify_teacher_completion(
        (1, 2, 8, 9), close_ids=(8, 9), eos_token_id=7, close_at_eos=True
    )
    assert reasoning == (1, 2, 8, 9)
    assert mode == "native_close"


def test_teacher_completion_records_eot_rewrite():
    reasoning, mode = classify_teacher_completion(
        (1, 2, 7), close_ids=(8, 9), eos_token_id=7, close_at_eos=True
    )
    assert reasoning == (1, 2, 8, 9)
    assert mode == "eot_rewritten"


def test_teacher_completion_rejects_budget_exhaustion():
    with pytest.raises(ValueError, match="budget_exhausted"):
        classify_teacher_completion(
            (1, 2, 3), close_ids=(8, 9), eos_token_id=7, close_at_eos=True
        )


def test_dataset_solution_nll_scores_each_target_token_in_fp32():
    logits = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.0, 3.0, -2.0]]], dtype=torch.bfloat16
    )
    targets = (0, 1)

    actual = dataset_solution_token_nll(logits, targets)
    expected = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), torch.tensor(targets)
    )

    assert actual == pytest.approx(float(expected.item()))


def test_matched_opsd_audit_aggregates_paired_kl_and_teacher_context_consistency():
    result = aggregate_matched_opsd_audit(
        _rows(trained=False),
        _rows(trained=True),
        bootstrap_seed=24,
        bootstrap_samples=100,
        expected_examples=2,
    )

    assert result["examples"] == 2
    assert result["student_response_tokens"] == 3
    assert result["exact_forward_kl"]["untrained_token_weighted"] == pytest.approx(
        (0.2 * 2 + 0.4) / 3
    )
    assert result["exact_forward_kl"]["trained_token_weighted"] == pytest.approx(
        (0.1 * 2 + 0.5) / 3
    )
    assert result["exact_forward_kl"]["trained_relative_reduction"] == pytest.approx(
        1.0 - ((0.1 * 2 + 0.5) / (0.2 * 2 + 0.4))
    )
    assert result["teacher_context_consistency"]["no_privilege_dataset_solution_nll"] == pytest.approx(
        (0.8 + 0.4 * 3) / 4
    )
    assert result["teacher_context_consistency"]["self_reasoning_dataset_solution_nll"] == pytest.approx(
        (0.6 + 0.5 * 3) / 4
    )
    assert len(result["bootstrap"]["index_matrix_sha256"]) == 64
    assert result["bootstrap"]["samples"] == 100


def test_matched_opsd_audit_rejects_duplicate_heldout_prompts():
    untrained = _rows(trained=False)
    trained = _rows(trained=True)
    for rows in (untrained, trained):
        rows[1]["prompt_text_sha256"] = rows[0]["prompt_text_sha256"]

    with pytest.raises(ValueError, match="duplicate prompt_text_sha256"):
        aggregate_matched_opsd_audit(
            untrained,
            trained,
            bootstrap_seed=24,
            bootstrap_samples=10,
            expected_examples=2,
        )


def test_matched_opsd_audit_rejects_teacher_logit_drift():
    trained = deepcopy(_rows(trained=True))
    trained[0]["teacher_logits_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="differ in teacher_logits_sha256"):
        aggregate_matched_opsd_audit(
            _rows(trained=False),
            trained,
            bootstrap_seed=24,
            bootstrap_samples=10,
            expected_examples=2,
        )


def test_matched_opsd_audit_rejects_student_prefix_identity_drift():
    trained = deepcopy(_rows(trained=True))
    trained[1]["student_response_ids_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="differ in student_response_ids_sha256"):
        aggregate_matched_opsd_audit(
            _rows(trained=False),
            trained,
            bootstrap_seed=24,
            bootstrap_samples=10,
            expected_examples=2,
        )
