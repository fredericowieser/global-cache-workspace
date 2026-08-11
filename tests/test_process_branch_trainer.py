from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import replace

import gcw.process_branch_trainer as process_branch_trainer
import pytest
import torch
from gcw.checkpoint import load_checkpoint, save_checkpoint
from gcw.config import (
    CacheProcessorConfig,
    DataConfig,
    ModelConfig,
    ProcessBranchConfig,
    ResumeConfig,
    TrainConfig,
)
from gcw.models import build_model
from gcw.process_data import AnswerOnlyRecord, ProcessBranchRecord
from gcw.process_branch_trainer import _sample_cycle_count
from gcw.train import train
from gcw.state_hash import tensor_state_sha256
from gcw.training_checkpoint import (
    CheckpointCompatibilityError,
    TrainingCheckpoint,
    save_training_checkpoint,
)


class _TinyTokenizer:
    eot_id = 0
    vocab_size = 128

    def encode(self, text: str) -> list[int]:
        return [ord(character) % 127 + 1 for character in text]

    def get_vocab_size(self) -> int:
        return self.vocab_size


def _model_config() -> ModelConfig:
    return ModelConfig(
        variant="longshot",
        family="smollm",
        sequence_len=32,
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=0,
            topk_heads=4,
            score="kv_norm",
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            encoder_method="query_attention",
            encoder_query_source="final_token",
            update_stream="v",
            gate_init=0.0,
        ),
    )


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def test_cycle_sampling_is_seeded_and_not_round_robin() -> None:
    process = ProcessBranchConfig(cycle_choices=(1, 2, 4))
    first = random.Random(24)
    second = random.Random(24)

    first_sequence = [_sample_cycle_count(process, first) for _ in range(32)]
    second_sequence = [_sample_cycle_count(process, second) for _ in range(32)]

    assert first_sequence == second_sequence
    assert set(first_sequence) == {1, 2, 4}
    assert first_sequence != [1, 2, 4] * 10 + [1, 2]


def test_process_schedule_separates_record_order_from_depth_choices() -> None:
    fixed = process_branch_trainer.ProcessSchedule(
        record_count=5,
        cycle_choices=(1,),
        seed=24,
    )
    random_depth = process_branch_trainer.ProcessSchedule(
        record_count=5,
        cycle_choices=(1, 2, 4),
        seed=24,
    )

    fixed_sequence = [fixed.next_example() for _ in range(8)]
    random_sequence = [random_depth.next_example() for _ in range(8)]

    assert fixed_sequence == [
        (4, 1),
        (2, 1),
        (0, 1),
        (1, 1),
        (3, 1),
        (3, 1),
        (0, 1),
        (2, 1),
    ]
    assert random_sequence == [
        (4, 4),
        (2, 2),
        (0, 4),
        (1, 1),
        (3, 1),
        (3, 1),
        (0, 1),
        (2, 1),
    ]
    assert [index for index, _ in fixed_sequence] == [
        index for index, _ in random_sequence
    ]


def test_answer_only_training_fingerprint_binds_loss_contract() -> None:
    base = TrainConfig(
        model=_model_config(),
        data=DataConfig(
            kind="answer_only_jsonl",
            data_dir="records.jsonl",
            tokenizer="hf",
            tokenizer_source="fake-tokenizer",
            response_prefix="P",
        ),
        mode="process_branch",
        distributed="none",
        freeze_backbone=True,
        backbone_checkpoint="backbone",
        batch_size=1,
        max_steps=1,
        process_branch=ProcessBranchConfig(
            objective="answer_only",
            process_weight=0.0,
            answer_weight=1.0,
            heldout_take_rows=2,
        ),
    )
    assert base.process_branch is not None
    changed = replace(
        base,
        process_branch=replace(base.process_branch, answer_weight=0.5),
    )

    base_fingerprint = process_branch_trainer._process_training_fingerprint(
        base,
        data_identity={"kind": "test"},
        initial_processor_sha256="0" * 64,
    )
    changed_fingerprint = process_branch_trainer._process_training_fingerprint(
        changed,
        data_identity={"kind": "test"},
        initial_processor_sha256="0" * 64,
    )

    assert base_fingerprint != changed_fingerprint



@pytest.mark.parametrize(
    ("epoch", "order", "offset"),
    [
        (0, (), 0),
        (-1, (0,), 0),
        (0, (0, 2), 0),
        (0, (0, 1), 3),
    ],
)
def test_process_cursor_rejects_malformed_positions_direct_and_from_dict(
    epoch: int,
    order: tuple[int, ...],
    offset: int,
) -> None:
    state = random.Random(24).getstate()
    with pytest.raises(ValueError):
        process_branch_trainer.ProcessCursor(epoch, order, offset, state, state)

    payload = {
        "epoch": epoch,
        "order": list(order),
        "offset": offset,
        "order_rng_state": process_branch_trainer._random_state_to_dict(state),
        "depth_rng_state": process_branch_trainer._random_state_to_dict(state),
    }
    with pytest.raises(ValueError):
        process_branch_trainer.ProcessCursor.from_dict(payload)


@pytest.mark.parametrize(
    "state",
    [
        (1, (1,), None),
        (3, random.Random(24).getstate()[1], float("nan")),
        (3, random.Random(24).getstate()[1], float("inf")),
        (3, random.Random(24).getstate()[1], True),
        (3, random.Random(24).getstate()[1], "bad"),
    ],
)
def test_process_cursor_rejects_rng_states_random_cannot_restore(
    state,
) -> None:
    valid = random.Random(24).getstate()
    with pytest.raises(ValueError):
        process_branch_trainer.ProcessCursor(-1, (), 0, state, valid)

    payload = process_branch_trainer.ProcessCursor(-1, (), 0, valid, valid).to_dict()
    payload["depth_rng_state"] = process_branch_trainer._random_state_to_dict(state)
    with pytest.raises(ValueError):
        process_branch_trainer.ProcessCursor.from_dict(payload)


def test_process_training_fingerprint_binds_bf16_policy() -> None:
    base = TrainConfig(
        model=_model_config(),
        data=DataConfig(kind="answer_only_jsonl", data_dir="records.jsonl", tokenizer="hf", tokenizer_source="fake", response_prefix="P"),
        mode="process_branch",
        distributed="none",
        freeze_backbone=True,
        backbone_checkpoint="backbone",
        process_branch=ProcessBranchConfig(objective="answer_only", process_weight=0.0, heldout_take_rows=2),
        max_steps=1,
        bf16=False,
    )
    fp32 = process_branch_trainer._process_training_fingerprint(
        base, data_identity={"kind": "test"}, initial_processor_sha256="0" * 64
    )
    bf16 = process_branch_trainer._process_training_fingerprint(
        replace(base, bf16=True),
        data_identity={"kind": "test"},
        initial_processor_sha256="0" * 64,
    )

    assert fp32 != bf16
    assert fp32["bf16"] is False
    assert bf16["bf16"] is True


def test_selected_record_identity_binds_reasoning_only_for_process_objective() -> None:
    answer = AnswerOnlyRecord(source_id="id", prompt="Q", final_answer="A")
    process = ProcessBranchRecord(
        source_id="id", prompt="Q", final_answer="A", reasoning_steps=("R1", "R2")
    )

    assert process_branch_trainer._selected_record_payload(answer) == {
        "source_id": "id", "prompt": "Q", "final_answer": "A"
    }
    assert process_branch_trainer._selected_record_payload(process) == {
        "source_id": "id",
        "prompt": "Q",
        "final_answer": "A",
        "reasoning_steps": ["R1", "R2"],
    }


def test_training_data_audit_does_not_disclose_absolute_source_path(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    source = tmp_path / "private" / "secret-records.jsonl"
    source.parent.mkdir()
    source.write_text("{}\n")

    process_branch_trainer._write_training_data_audit(
        output, source, {"kind": "identity"}
    )

    encoded = (output / "training-data-audit.json").read_text()
    payload = json.loads(encoded)
    assert str(source.resolve()) not in encoded
    assert payload["source"] == {
        "basename": "secret-records.jsonl",
        "role": "informational",
    }
    assert "source_path" not in payload

def test_process_schedule_cursor_roundtrip_preserves_exact_continuation() -> None:
    schedule = process_branch_trainer.ProcessSchedule(
        record_count=5,
        cycle_choices=(1, 2, 4),
        seed=24,
    )
    for _ in range(8):
        schedule.next_example()

    payload = schedule.cursor.to_dict()

    assert payload["epoch"] == 1
    assert payload["order"] == [3, 0, 2, 4, 1]
    assert payload["offset"] == 3
    assert set(payload) == {
        "epoch",
        "order",
        "offset",
        "order_rng_state",
        "depth_rng_state",
    }
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    restored_cursor = process_branch_trainer.ProcessCursor.from_dict(json.loads(encoded))
    resumed = process_branch_trainer.ProcessSchedule(
        record_count=5,
        cycle_choices=(1, 2, 4),
        seed=999,
        cursor=restored_cursor,
    )

    expected_continuation = [(4, 4), (1, 4), (4, 1), (0, 4)]
    assert [schedule.next_example() for _ in range(4)] == expected_continuation
    assert [resumed.next_example() for _ in range(4)] == expected_continuation


def test_process_branch_trainer_updates_processor_and_writes_reloadable_checkpoint(
    monkeypatch,
    tmp_path,
) -> None:
    model_config = _model_config()
    backbone = tmp_path / "backbone"
    initial_model = build_model(model_config)
    initial_model.init_weights()
    save_checkpoint(backbone, initial_model, model_config)

    records = tmp_path / "records.jsonl"
    records.write_text(
        "".join(
            json.dumps(
                {
                    "source_id": f"gsm8k-train:{index}",
                    "prompt": f"Q{index}",
                    "final_answer": f"A{index}",
                    "reasoning_steps": [f"R{index}"],
                }
            )
            + "\n"
            for index in range(3)
        )
    )
    output = tmp_path / "run"
    config = TrainConfig(
        model=model_config,
        data=DataConfig(
            kind="process_branch_jsonl",
            data_dir=str(records),
            tokenizer="hf",
            tokenizer_source="fake-tokenizer",
            sft_take_rows=1,
            response_prefix="P",
        ),
        mode="process_branch",
        distributed="none",
        freeze_backbone=True,
        backbone_checkpoint=str(backbone),
        output_dir=str(output),
        batch_size=1,
        max_steps=1,
        learning_rate=1e-3,
        warmup_steps=0,
        weight_decay=0.0,
        process_branch=ProcessBranchConfig(
            cycle_choices=(1,),
            gradient_accumulation_steps=1,
            max_optimizer_steps=1,
            max_prompt_tokens=8,
            max_step_tokens=8,
            max_answer_tokens=8,
            heldout_skip_rows=1,
            heldout_take_rows=2,
        ),
    )
    tokenizer_names: list[str] = []

    def get_tiny_tokenizer(name: str) -> _TinyTokenizer:
        tokenizer_names.append(name)
        return _TinyTokenizer()

    monkeypatch.setattr(
        "gcw.process_branch_trainer.get_tokenizer",
        get_tiny_tokenizer,
    )

    train(config)

    assert tokenizer_names == ["hf:fake-tokenizer"]

    summary = json.loads((output / "summary.json").read_text())
    assert summary["optimizer_steps"] == 1
    assert summary["valid_records"] == 1
    assert summary["process_cursor"]["epoch"] == 0
    assert summary["process_cursor"]["order"] == [0]
    assert summary["process_cursor"]["offset"] == 1
    assert set(summary["process_cursor"]) == {
        "epoch",
        "order",
        "offset",
        "order_rng_state",
        "depth_rng_state",
    }
    assert summary["initial_processor_sha256"] != summary["final_processor_sha256"]
    assert summary["checkpoint_reload_verified"] is True
    assert summary["heldout_controls"]["examples"] == 2
    assert set(summary["heldout_controls"]) >= {
        "matched_answer_loss",
        "processor_off_answer_loss",
        "shuffled_answer_loss",
    }
    audit_path = output / "training-data-audit.json"
    audit_sha256 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    assert summary["training_data_audit"] == {
        "path": "training-data-audit.json",
        "sha256": audit_sha256,
    }
    audit = json.loads(audit_path.read_text())
    assert audit["data_identity"] == summary["data_identity"]
    assert audit["data_identity"]["source"]["sha256"] == hashlib.sha256(
        records.read_bytes()
    ).hexdigest()
    assert audit["source"] == {
        "basename": records.name,
        "role": "informational",
    }
    _, _, checkpoint_state = load_checkpoint(output / "final")
    assert checkpoint_state["data_identity"] == summary["data_identity"]
    assert checkpoint_state["training_data_audit"]["sha256"] == audit_sha256
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 1
    assert metrics[0]["train/cycles"] == 1
    assert (output / "final" / "model.safetensors").is_file()
    final_manifest = output / "final" / "manifest.json"
    assert final_manifest.is_file()
    assert summary["final_checkpoint_manifest_sha256"] == hashlib.sha256(
        final_manifest.read_bytes()
    ).hexdigest()


def test_answer_only_fixed_and_random_depth_runs_share_population_and_skip_process_loss(
    monkeypatch,
    tmp_path,
) -> None:
    model_config = _model_config()
    backbone = tmp_path / "backbone"
    initial_model = build_model(model_config)
    initial_model.init_weights()
    save_checkpoint(backbone, initial_model, model_config)
    records = tmp_path / "answer-only.jsonl"
    records.write_text(
        "".join(
            json.dumps(
                {
                    "source_id": f"gsm8k-train:{index}",
                    "prompt": f"Q{index}",
                    "final_answer": f"A{index}",
                }
            )
            + "\n"
            for index in range(3)
        )
    )
    monkeypatch.setattr(
        "gcw.process_branch_trainer.get_tokenizer",
        lambda _name: _TinyTokenizer(),
    )

    def run(name: str, cycle_choices: tuple[int, ...]) -> tuple[dict, dict, list[dict]]:
        output = tmp_path / name
        config = TrainConfig(
            model=model_config,
            data=DataConfig(
                kind="answer_only_jsonl",
                data_dir=str(records),
                tokenizer="hf",
                tokenizer_source="fake-tokenizer",
                sft_take_rows=1,
                response_prefix="P",
            ),
            mode="process_branch",
            distributed="none",
            freeze_backbone=True,
            backbone_checkpoint=str(backbone),
            output_dir=str(output),
            batch_size=1,
            max_steps=1,
            learning_rate=1e-3,
            warmup_steps=0,
            weight_decay=0.0,
            process_branch=ProcessBranchConfig(
                objective="answer_only",
                cycle_choices=cycle_choices,
                process_weight=0.0,
                gradient_accumulation_steps=1,
                max_optimizer_steps=4,
                max_prompt_tokens=8,
                max_answer_tokens=8,
                heldout_skip_rows=1,
                heldout_take_rows=2,
            ),
        )
        train(config)
        summary = json.loads((output / "summary.json").read_text())
        audit = json.loads((output / "training-data-audit.json").read_text())
        metrics = [
            json.loads(line)
            for line in (output / "metrics.jsonl").read_text().splitlines()
        ]
        return summary, audit, metrics

    fixed_summary, fixed_audit, fixed_metric = run("p0", (1,))
    random_summary, random_audit, random_metric = run("p1", (1, 2, 4))

    source_bytes = records.read_bytes()
    train_payloads = [
        {"source_id": "gsm8k-train:0", "prompt": "Q0", "final_answer": "A0"}
    ]
    heldout_payloads = [
        {"source_id": "gsm8k-train:1", "prompt": "Q1", "final_answer": "A1"},
        {"source_id": "gsm8k-train:2", "prompt": "Q2", "final_answer": "A2"},
    ]
    expected_identity = {
        "kind": "gcw.process_branch.training_data_identity.v1",
        "objective": "answer_only",
        "source": {
            "size_bytes": len(source_bytes),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "splits": {
            "train": {
                "bounds": {"skip_rows": 0, "take_rows": 1},
                "source_records": 1,
                "eligible_records": 1,
                "rejected_records": 0,
                "rejection_reasons": {},
                "ordered_source_ids": ["gsm8k-train:0"],
                "ordered_source_ids_sha256": _canonical_sha256(["gsm8k-train:0"]),
                "selected_records_sha256": _canonical_sha256(train_payloads),
            },
            "heldout": {
                "bounds": {"skip_rows": 1, "take_rows": 2},
                "source_records": 2,
                "eligible_records": 2,
                "rejected_records": 0,
                "rejection_reasons": {},
                "ordered_source_ids": ["gsm8k-train:1", "gsm8k-train:2"],
                "ordered_source_ids_sha256": _canonical_sha256(
                    ["gsm8k-train:1", "gsm8k-train:2"]
                ),
                "selected_records_sha256": _canonical_sha256(heldout_payloads),
            },
        },
    }

    assert fixed_summary["objective"] == random_summary["objective"] == "answer_only"
    assert fixed_summary["data_identity"] == expected_identity
    assert random_summary["data_identity"] == expected_identity
    expected_audit = {
        "kind": "gcw.process_branch.training_data_audit.v3",
        "data_identity": expected_identity,
        "source": {
            "basename": records.name,
            "role": "informational",
        },
    }
    assert fixed_audit == expected_audit
    assert random_audit == expected_audit
    for output_name, summary in (("p0", fixed_summary), ("p1", random_summary)):
        audit_path = tmp_path / output_name / "training-data-audit.json"
        assert summary["training_data_audit"] == {
            "path": "training-data-audit.json",
            "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        }
        _, _, checkpoint_state = load_checkpoint(tmp_path / output_name / "final")
        assert checkpoint_state["data_identity"] == expected_identity
    assert all("train/process_loss" not in metric for metric in fixed_metric)
    assert all("train/process_loss" not in metric for metric in random_metric)
    assert [metric["train/cycles"] for metric in fixed_metric] == [1, 1, 1, 1]
    sampled_depths = [metric["train/cycles"] for metric in random_metric]
    assert sampled_depths == [4, 2, 4, 1]
    assert len(set(sampled_depths)) > 1
    assert sampled_depths != [1, 2, 4, 1]


def test_answer_only_training_and_heldout_controls_never_use_process_rationales(
    monkeypatch,
    tmp_path,
) -> None:
    model_config = _model_config()
    backbone = tmp_path / "backbone"
    initial_model = build_model(model_config)
    initial_model.init_weights()
    expected_backbone_sha256 = tensor_state_sha256(initial_model.backbone.state_dict())
    save_checkpoint(backbone, initial_model, model_config)
    records = tmp_path / "answer-only-heldout.jsonl"
    records.write_text(
        "".join(
            json.dumps(
                {
                    "source_id": f"gsm8k-train:{index}",
                    "prompt": f"Q{index}",
                    "final_answer": f"A{index}",
                }
            )
            + "\n"
            for index in range(3)
        )
    )
    output = tmp_path / "answer-only-heldout-run"
    config = TrainConfig(
        model=model_config,
        data=DataConfig(
            kind="answer_only_jsonl",
            data_dir=str(records),
            tokenizer="hf",
            tokenizer_source="fake-tokenizer",
            sft_take_rows=1,
            response_prefix="P",
        ),
        mode="process_branch",
        distributed="none",
        freeze_backbone=True,
        backbone_checkpoint=str(backbone),
        output_dir=str(output),
        batch_size=1,
        max_steps=1,
        learning_rate=1e-3,
        warmup_steps=0,
        weight_decay=0.0,
        process_branch=ProcessBranchConfig(
            objective="answer_only",
            cycle_choices=(1, 4),
            process_weight=0.0,
            gradient_accumulation_steps=1,
            max_optimizer_steps=1,
            max_prompt_tokens=8,
            max_answer_tokens=8,
            heldout_skip_rows=1,
            heldout_take_rows=2,
        ),
    )
    monkeypatch.setattr(
        process_branch_trainer,
        "get_tokenizer",
        lambda _name: _TinyTokenizer(),
    )

    def reject_process_rationale_path(*_args, **_kwargs):
        raise AssertionError("answer-only execution entered a process-rationale path")

    monkeypatch.setattr(
        "gcw.process_data.read_process_branch_records",
        reject_process_rationale_path,
    )
    monkeypatch.setattr(
        process_branch_trainer,
        "tokenize_process_record",
        reject_process_rationale_path,
    )
    monkeypatch.setattr(
        process_branch_trainer,
        "process_branch_example_loss",
        reject_process_rationale_path,
    )

    train(config)

    summary = json.loads((output / "summary.json").read_text())
    assert summary["heldout_controls"]["examples"] == 2
    assert summary["heldout_controls"]["cycles"] == 4
    assert summary["initial_backbone_sha256"] == expected_backbone_sha256
    assert summary["final_backbone_sha256"] == expected_backbone_sha256
    reloaded, _, checkpoint_state = load_checkpoint(output / "final")
    assert tensor_state_sha256(reloaded.backbone.state_dict()) == expected_backbone_sha256
    assert checkpoint_state["initial_backbone_sha256"] == expected_backbone_sha256
    assert checkpoint_state["final_backbone_sha256"] == expected_backbone_sha256


def test_answer_only_identity_never_reopens_path_during_record_loading(
    monkeypatch,
    tmp_path,
) -> None:
    model_config = _model_config()
    backbone = tmp_path / "backbone"
    initial_model = build_model(model_config)
    initial_model.init_weights()
    save_checkpoint(backbone, initial_model, model_config)
    records = tmp_path / "answer-only.jsonl"
    records.write_text(
        "".join(
            json.dumps(
                {
                    "source_id": f"gsm8k-train:{index}",
                    "prompt": f"Q{index}",
                    "final_answer": f"A{index}",
                }
            )
            + "\n"
            for index in range(3)
        )
    )
    config = TrainConfig(
        model=model_config,
        data=DataConfig(
            kind="answer_only_jsonl",
            data_dir=str(records),
            tokenizer="hf",
            tokenizer_source="fake-tokenizer",
            sft_take_rows=1,
            response_prefix="P",
        ),
        mode="process_branch",
        distributed="none",
        freeze_backbone=True,
        backbone_checkpoint=str(backbone),
        output_dir=str(tmp_path / "run"),
        batch_size=1,
        max_steps=1,
        learning_rate=1e-3,
        warmup_steps=0,
        weight_decay=0.0,
        process_branch=ProcessBranchConfig(
            objective="answer_only",
            cycle_choices=(1,),
            process_weight=0.0,
            gradient_accumulation_steps=1,
            max_optimizer_steps=1,
            max_prompt_tokens=8,
            max_answer_tokens=8,
            heldout_skip_rows=1,
            heldout_take_rows=2,
        ),
    )
    monkeypatch.setattr(
        "gcw.process_branch_trainer.get_tokenizer",
        lambda _name: _TinyTokenizer(),
    )
    def mutable_path_reader_must_not_run(*_args, **_kwargs):
        raise AssertionError("training reopened mutable answer-only path")

    monkeypatch.setattr(
        "gcw.process_data.read_answer_only_records",
        mutable_path_reader_must_not_run,
    )

    train(config)
    assert (tmp_path / "run" / "summary.json").is_file()





def test_authenticated_training_tokenizer_snapshot_survives_source_mutation(tmp_path) -> None:
    source = tmp_path / "tokenizer"
    source.mkdir()
    (source / "tokenizer.json").write_bytes(b"authenticated")

    with process_branch_trainer._authenticated_tokenizer_snapshot(source) as (snapshot, digest):
        (source / "tokenizer.json").write_bytes(b"replaced")
        assert (snapshot / "tokenizer.json").read_bytes() == b"authenticated"
        assert digest == process_branch_trainer._path_tree_sha256(snapshot)


def test_canonical_source_identity_fails_closed_when_git_fails(monkeypatch) -> None:
    def fail_git(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(process_branch_trainer.subprocess, "check_output", fail_git)
    with pytest.raises(RuntimeError, match="requires Git provenance"):
        process_branch_trainer._source_code_identity(require_git=True)


def test_resume_schedule_consistency_requires_exact_deterministic_replay() -> None:
    schedule = process_branch_trainer.ProcessSchedule(
        record_count=4,
        cycle_choices=(1, 2, 4),
        seed=24,
    )
    schedule.next_example()
    schedule.next_example()
    valid_cursor = schedule.cursor

    process_branch_trainer._validate_resume_schedule_consistency(
        step=1,
        seen_examples=2,
        cursor=valid_cursor,
        record_count=4,
        cycle_choices=(1, 2, 4),
        seed=24,
        gradient_accumulation_steps=2,
        max_optimizer_steps=3,
    )
    with pytest.raises(ValueError, match="seen_examples.*optimizer step"):
        process_branch_trainer._validate_resume_schedule_consistency(
            step=1,
            seen_examples=1,
            cursor=valid_cursor,
            record_count=4,
            cycle_choices=(1, 2, 4),
            seed=24,
            gradient_accumulation_steps=2,
            max_optimizer_steps=3,
        )
    with pytest.raises(ValueError, match="step exceeds"):
        process_branch_trainer._validate_resume_schedule_consistency(
            step=4,
            seen_examples=8,
            cursor=valid_cursor,
            record_count=4,
            cycle_choices=(1, 2, 4),
            seed=24,
            gradient_accumulation_steps=2,
            max_optimizer_steps=3,
        )
    wrong_schedule = process_branch_trainer.ProcessSchedule(
        record_count=4,
        cycle_choices=(1, 2, 4),
        seed=24,
    )
    wrong_schedule.next_example()
    with pytest.raises(ValueError, match="cursor/RNG.*replay"):
        process_branch_trainer._validate_resume_schedule_consistency(
            step=1,
            seen_examples=2,
            cursor=wrong_schedule.cursor,
            record_count=4,
            cycle_choices=(1, 2, 4),
            seed=24,
            gradient_accumulation_steps=2,
            max_optimizer_steps=3,
        )


def test_answer_only_interrupted_resume_matches_uninterrupted_training(
    monkeypatch,
    tmp_path,
) -> None:
    model_config = _model_config()
    backbone = tmp_path / "backbone"
    initial_model = build_model(model_config)
    initial_model.init_weights()
    save_checkpoint(backbone, initial_model, model_config)
    records = tmp_path / "answer-only-resume.jsonl"
    records.write_text(
        "".join(
            json.dumps(
                {
                    "source_id": f"gsm8k-train:{index}",
                    "prompt": f"Q{index}",
                    "final_answer": f"A{index}",
                }
            )
            + "\n"
            for index in range(6)
        )
    )
    monkeypatch.setattr(
        "gcw.process_branch_trainer.get_tokenizer",
        lambda _name: _TinyTokenizer(),
    )

    def new_config(output_name: str) -> TrainConfig:
        return TrainConfig(
            model=model_config,
            data=DataConfig(
                kind="answer_only_jsonl",
                data_dir=str(records),
                tokenizer="hf",
                tokenizer_source=str(tmp_path / "fake-tokenizer"),
                sft_take_rows=4,
                response_prefix="P",
            ),
            mode="process_branch",
            distributed="none",
            freeze_backbone=True,
            backbone_checkpoint=str(backbone),
            output_dir=str(tmp_path / output_name),
            batch_size=1,
            max_steps=1,
            learning_rate=1e-3,
            warmup_steps=0,
            weight_decay=0.0,
            save_every=1,
            process_branch=ProcessBranchConfig(
                objective="answer_only",
                cycle_choices=(1, 2, 4),
                process_weight=0.0,
                gradient_accumulation_steps=1,
                max_optimizer_steps=2,
                max_prompt_tokens=8,
                max_answer_tokens=8,
                heldout_skip_rows=4,
                heldout_take_rows=2,
            ),
        )

    tokenizer_artifact = tmp_path / "fake-tokenizer"
    tokenizer_artifact.mkdir()
    (tokenizer_artifact / "tokenizer.json").write_text("{}")
    reference_config = new_config("reference")
    train(reference_config)

    class _IntentionalInterruption(RuntimeError):
        pass

    def interrupt_after_step_one(*args, **kwargs) -> None:
        save_training_checkpoint(*args, **kwargs)
        resume_state = args[4]
        if resume_state.step == 1:
            raise _IntentionalInterruption("stop after authenticated step one")

    monkeypatch.setattr(
        process_branch_trainer,
        "save_training_checkpoint",
        interrupt_after_step_one,
        raising=False,
    )
    interrupted_config = new_config("interrupted")
    with pytest.raises(_IntentionalInterruption, match="authenticated step one"):
        train(interrupted_config)

    step_one_checkpoint = tmp_path / "interrupted" / "training-checkpoint"
    assert (step_one_checkpoint / "manifest.json").is_file()
    bf16_drift = replace(
        new_config("bf16-drift"),
        bf16=True,
        backbone_checkpoint=None,
        resume=ResumeConfig(checkpoint=str(step_one_checkpoint)),
    )
    with pytest.raises(CheckpointCompatibilityError, match="bf16"):
        train(bf16_drift)
    assert not (tmp_path / "bf16-drift").exists()
    monkeypatch.setattr(
        process_branch_trainer,
        "save_training_checkpoint",
        save_training_checkpoint,
    )
    resumed_config = replace(
        new_config("resumed"),
        backbone_checkpoint=None,
        resume=ResumeConfig(checkpoint=str(step_one_checkpoint)),
    )
    assert resumed_config.process_branch is not None
    assert reference_config.process_branch is not None
    assert (
        resumed_config.process_branch.max_optimizer_steps
        == reference_config.process_branch.max_optimizer_steps
        == 2
    )
    original_validate_frozen = process_branch_trainer._validate_frozen_training_identities
    resume_backbone_hashes: list[str] = []

    def validate_frozen_after_resume(*args, **kwargs):
        resume_backbone_hashes.append(kwargs["initial_backbone_sha256"])
        return original_validate_frozen(*args, **kwargs)

    monkeypatch.setattr(
        process_branch_trainer,
        "_validate_frozen_training_identities",
        validate_frozen_after_resume,
    )
    train(resumed_config)

    reference_metrics = [
        json.loads(line)
        for line in (tmp_path / "reference" / "metrics.jsonl").read_text().splitlines()
    ]
    resumed_metrics = [
        json.loads(line)
        for line in (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(reference_metrics) == 2
    assert len(resumed_metrics) == 1
    assert resumed_metrics[0]["train/step"] == 2
    assert resumed_metrics[0]["train/source_ids"] == reference_metrics[1]["train/source_ids"]
    assert resumed_metrics[0]["train/cycle_counts"] == reference_metrics[1]["train/cycle_counts"]
    assert resumed_metrics[0]["train/loss"] == reference_metrics[1]["train/loss"]

    reference_summary = json.loads((tmp_path / "reference" / "summary.json").read_text())
    resumed_summary = json.loads((tmp_path / "resumed" / "summary.json").read_text())
    assert resume_backbone_hashes == [reference_summary["initial_backbone_sha256"]]
    assert resumed_summary["seen_examples"] == reference_summary["seen_examples"] == 2
    assert resumed_summary["data_identity"] == reference_summary["data_identity"]
    assert (
        resumed_summary["initial_processor_sha256"]
        == reference_summary["initial_processor_sha256"]
    )
    assert resumed_summary["final_processor_sha256"] == reference_summary["final_processor_sha256"]

    reference_model, _, _ = load_checkpoint(tmp_path / "reference" / "final")
    resumed_model, _, _ = load_checkpoint(tmp_path / "resumed" / "final")
    for name, reference_tensor in reference_model.cache_processor.state_dict().items():
        assert torch.equal(reference_tensor, resumed_model.cache_processor.state_dict()[name])

    def restored_optimizer_state(checkpoint_path):
        manifest = json.loads((checkpoint_path / "manifest.json").read_text())
        checkpoint = TrainingCheckpoint.open(
            checkpoint_path,
            expected_fingerprint=manifest["training_fingerprint"],
            topology="same",
            current_world_size=1,
        )
        model = build_model(model_config)
        model.init_weights()
        optimizer = torch.optim.AdamW(
            model.cache_processor.parameters(),
            lr=reference_config.learning_rate,
            betas=(reference_config.adam_beta1, reference_config.adam_beta2),
            eps=reference_config.adam_eps,
            weight_decay=reference_config.weight_decay,
        )
        checkpoint.restore_model(model)
        checkpoint.restore_optimizer(optimizer, device="cpu")
        return optimizer.state_dict()

    _assert_nested_equal(
        restored_optimizer_state(tmp_path / "reference" / "training-checkpoint"),
        restored_optimizer_state(tmp_path / "resumed" / "training-checkpoint"),
    )


def test_answer_only_resume_rejects_changed_data_identity_before_consuming_example(
    monkeypatch,
    tmp_path,
) -> None:
    model_config = _model_config()
    backbone = tmp_path / "backbone"
    initial_model = build_model(model_config)
    initial_model.init_weights()
    save_checkpoint(backbone, initial_model, model_config)
    records = tmp_path / "answer-only-resume-identity.jsonl"
    rows = [
        {
            "source_id": f"gsm8k-train:{index}",
            "prompt": f"Q{index}",
            "final_answer": f"A{index}",
        }
        for index in range(4)
    ]

    def write_rows() -> None:
        records.write_text("".join(json.dumps(row) + "\n" for row in rows))

    write_rows()
    monkeypatch.setattr(
        "gcw.process_branch_trainer.get_tokenizer",
        lambda _name: _TinyTokenizer(),
    )

    def new_config(output_name: str) -> TrainConfig:
        return TrainConfig(
            model=model_config,
            data=DataConfig(
                kind="answer_only_jsonl",
                data_dir=str(records),
                tokenizer="hf",
                tokenizer_source=str(tmp_path / "fake-tokenizer"),
                sft_take_rows=2,
                response_prefix="P",
            ),
            mode="process_branch",
            distributed="none",
            freeze_backbone=True,
            backbone_checkpoint=str(backbone),
            output_dir=str(tmp_path / output_name),
            batch_size=1,
            max_steps=1,
            learning_rate=1e-3,
            warmup_steps=0,
            weight_decay=0.0,
            save_every=1,
            process_branch=ProcessBranchConfig(
                objective="answer_only",
                cycle_choices=(1, 2, 4),
                process_weight=0.0,
                gradient_accumulation_steps=1,
                max_optimizer_steps=2,
                max_prompt_tokens=8,
                max_answer_tokens=8,
                heldout_skip_rows=2,
                heldout_take_rows=2,
            ),
        )

    tokenizer_artifact = tmp_path / "fake-tokenizer"
    tokenizer_artifact.mkdir()
    (tokenizer_artifact / "tokenizer.json").write_text("{}")

    class _IntentionalInterruption(RuntimeError):
        pass

    def interrupt_after_step_one(*args, **kwargs) -> None:
        save_training_checkpoint(*args, **kwargs)
        if args[4].step == 1:
            raise _IntentionalInterruption

    monkeypatch.setattr(
        process_branch_trainer,
        "save_training_checkpoint",
        interrupt_after_step_one,
    )
    with pytest.raises(_IntentionalInterruption):
        train(new_config("interrupted"))

    rows[0]["prompt"] = "CHANGED"
    write_rows()
    monkeypatch.setattr(
        process_branch_trainer,
        "save_training_checkpoint",
        save_training_checkpoint,
    )
    monkeypatch.setattr(
        process_branch_trainer,
        "answer_only_example_loss",
        lambda *_args, **_kwargs: pytest.fail("resume consumed an example before identity check"),
    )
    resumed_config = replace(
        new_config("changed-data-resume"),
        backbone_checkpoint=None,
        resume=ResumeConfig(
            checkpoint=str(tmp_path / "interrupted" / "training-checkpoint")
        ),
    )

    with pytest.raises(CheckpointCompatibilityError, match="data_identity"):
        train(resumed_config)


def _assert_nested_equal(expected, actual) -> None:
    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            _assert_nested_equal(left, right)
    else:
        assert expected == actual
