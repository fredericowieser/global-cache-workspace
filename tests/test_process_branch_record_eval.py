from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import hashlib
from typing import cast
import json

import pytest
import torch

import scripts.evaluate_process_branch_records as evaluator
from gcw import process_branch_trainer
from gcw.config import CacheProcessorConfig, ModelConfig, load_train_config
from gcw.models import build_model
from gcw.models.longshot import GlobalCacheProcessor, Longshot
from gcw.process_data import AnswerOnlyRecord


def _tiny_longshot() -> Longshot:
    config = ModelConfig(
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
    model = build_model(config)
    assert isinstance(model, Longshot)
    model.init_weights()
    return model


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("OFF", Counter()),
        ("INIT", Counter({"encode_source": 1, "initialise_workspace": 1, "write_for_readout": 1})),
        ("K1", Counter({"encode_source": 1, "initialise_workspace": 1, "deliberate": 1, "write_for_readout": 1})),
        ("K2", Counter({"encode_source": 1, "initialise_workspace": 1, "deliberate": 2, "write_for_readout": 1})),
        ("K4", Counter({"encode_source": 1, "initialise_workspace": 1, "deliberate": 4, "write_for_readout": 1})),
        ("K8", Counter({"encode_source": 1, "initialise_workspace": 1, "deliberate": 8, "write_for_readout": 1})),
    ],
)
def test_canonical_conditions_have_disjoint_processor_call_graphs(
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
    expected: Counter[str],
) -> None:
    torch.manual_seed(31)
    model = _tiny_longshot()
    processor = cast(GlobalCacheProcessor, model.cache_processor)
    calls: Counter[str] = Counter()
    for method_name in ("encode_source", "initialise_workspace", "deliberate", "write_for_readout"):
        original = getattr(GlobalCacheProcessor, method_name)

        def spy(self, *args, _name=method_name, _original=original, **kwargs):
            if self is processor:
                calls[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(GlobalCacheProcessor, method_name, spy)

    result = evaluator.generate_answer_only_condition(
        model,
        prompt_ids=torch.tensor([4, 5, 6, 7]),
        answer_prefix_ids=torch.tensor([8, 9]),
        condition=condition,
        max_new_tokens=3,
        eos_token_id=None,
        valid_vocab_size=64,
    )

    assert result.generated_ids.shape == (3,)
    assert result.condition == condition
    assert result.cycles == {"OFF": None, "INIT": 0, "K1": 1, "K2": 2, "K4": 4, "K8": 8}[condition]
    assert calls == expected



class _TinyTokenizer:
    eot_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(character) % 63 + 1 for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "\\boxed{42}" if token_ids else ""

    def get_vocab_size(self) -> int:
        return 64


def test_canonical_rows_share_decoder_contract_and_ordered_sources() -> None:
    torch.manual_seed(32)
    model = _tiny_longshot()
    records = [
        AnswerOnlyRecord(source_id="gsm8k-train:10", prompt="Q1", final_answer="\\boxed{42}"),
        AnswerOnlyRecord(source_id="gsm8k-train:11", prompt="Q2", final_answer="\\boxed{42}"),
    ]
    identities = evaluator.EvaluationIdentities(
        run_id="run-1",
        run_sha256="a" * 64,
        arm="p0",
        arm_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        config_sha256="d" * 64,
        data_sha256="e" * 64,
    )
    decode = evaluator.DeterministicDecode(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0)

    rows = evaluator.evaluate_canonical_records(
        model,
        _TinyTokenizer(),
        records,
        conditions=("OFF", "INIT", "K1", "K2", "K4", "K8"),
        identities=identities,
        decode=decode,
        prompt_suffix="S",
        answer_prefix="P",
        initial_processor_state={
            key: value.detach().clone()
            for key, value in model.cache_processor.state_dict().items()
        },
    )

    expected_order = [record.source_id for record in records]
    assert len(rows) == 12
    for condition_index, condition in enumerate(("OFF", "INIT", "K1", "K2", "K4", "K8")):
        condition_rows = rows[condition_index * 2 : (condition_index + 1) * 2]
        assert [row["source_id"] for row in condition_rows] == expected_order
        assert {row["condition"] for row in condition_rows} == {condition}
    assert {json.dumps(row["decode"], sort_keys=True) for row in rows} == {
        json.dumps({"max_new_tokens": 2, "temperature": 0.0, "top_k": 0, "top_p": 1.0}, sort_keys=True)
    }
    assert [row["source_index"] for row in rows[:2]] == [10, 11]
    assert all(row["raw_generated_text"] == "\\boxed{42}" for row in rows)
    assert all(row["parsed_prediction"] == "42" for row in rows)
    assert all(row["correctness"] is True for row in rows)
    assert all(row["extraction_status"] == "ok" for row in rows)
    assert all(row["prompt_token_count"] > 0 and row["answer_token_count"] == 2 for row in rows)
    assert all(set(row["diagnostics"]) == {
        "delta_value_norm", "cache_head_coverage", "update_concentration", "update_entropy"
    } for row in rows)
    assert all(row["schema"] == evaluator.CANONICAL_ROW_SCHEMA for row in rows)


def test_canonical_rows_include_gold_nll_and_termination_contract() -> None:
    torch.manual_seed(33)
    rows = evaluator.evaluate_canonical_records(
        _tiny_longshot(),
        _TinyTokenizer(),
        [AnswerOnlyRecord(source_id="gsm8k-train:12", prompt="Q", final_answer="\\boxed{42}")],
        conditions=("K8",),
        identities=evaluator.EvaluationIdentities(
            run_id="run-k8",
            run_sha256="a" * 64,
            arm="p1",
            arm_sha256="b" * 64,
            checkpoint_sha256="c" * 64,
            config_sha256="d" * 64,
            data_sha256="e" * 64,
        ),
        decode=evaluator.DeterministicDecode(
            max_new_tokens=2,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        ),
        prompt_suffix="S",
        answer_prefix="P",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["condition"] == "K8"
    assert row["K"] == 8
    assert row["target_token_count"] > 0
    assert row["answer_nll_sum"] >= 0.0
    assert row["answer_nll_per_token"] >= 0.0
    assert row["answer_nll_sum"] == pytest.approx(
        row["answer_nll_per_token"] * row["target_token_count"]
    )
    assert row["termination_reason"] in {"eos", "max_tokens"}
    assert row["cap_hit"] is (row["termination_reason"] == "max_tokens")


def _valid_ledger() -> tuple[list[dict], dict]:
    conditions = ("OFF", "INIT", "K1", "K2", "K4", "K8")
    source_ids = tuple(f"gsm8k-train:{index}" for index in range(6000, 6256))
    decode = {"max_new_tokens": 2, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
    protocol_sha256 = "f" * 64
    identities = {
        "run_id": "run-1",
        "arm": "p0",
        "arm_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "data_sha256": "e" * 64,
    }
    identities["run_sha256"] = evaluator._canonical_json_sha256({
        "run_id": identities["run_id"],
        "arm": identities["arm"],
        "arm_contract_sha256": identities["arm_sha256"],
        "checkpoint_sha256": identities["checkpoint_sha256"],
        "training_contract_sha256": identities["config_sha256"],
        "data_sha256": identities["data_sha256"],
        "evaluator_source_sha256": evaluator.CANONICAL_EVALUATOR_SOURCE_SHA256,
        "evaluation_precision": "float32",
        "split": "heldout",
        "conditions": conditions,
        "decode": decode,
        "eos_token_id": 0,
    })
    rows = []
    for condition in conditions:
        for evaluation_index, source_id in enumerate(source_ids):
            rows.append(
                {
                    "schema": evaluator.CANONICAL_ROW_SCHEMA,
                    "evaluator_version": evaluator.CANONICAL_EVALUATOR_VERSION,
                    "evaluator_source_sha256": evaluator.CANONICAL_EVALUATOR_SOURCE_SHA256,
                    **identities,
                    "source_id": source_id,
                    "source_index": int(source_id.rsplit(":", 1)[1]),
                    "evaluation_index": evaluation_index,
                    "condition": condition,
                    "K": {"OFF": None, "INIT": 0, "K1": 1, "K2": 2, "K4": 4, "K8": 8}[condition],
                    "decode": decode,
                    "prompt_token_count": 3,
                    "answer_token_count": 2,
                    "generated_token_ids": [1, 2],
                    "target_token_count": 3,
                    "answer_nll_sum": 3.0,
                    "answer_nll_per_token": 1.0,
                    "termination_reason": "max_tokens",
                    "cap_hit": True,
                    "raw_generated_text": "\\boxed{42}",
                    "parsed_prediction": "42",
                    "target": "\\boxed{42}",
                    "correctness": True,
                    "extraction_status": "ok",
                    "latency_ms": 1.0,
                    "diagnostics": {
                        "delta_value_norm": None if condition == "OFF" else 0.0,
                        "cache_head_coverage": None if condition == "OFF" else 1.0,
                        "update_concentration": None if condition == "OFF" else 0.0,
                        "update_entropy": None if condition == "OFF" else 0.0,
                    },
                }
            )
    manifest = {
        "schema": evaluator.CANONICAL_MANIFEST_SCHEMA,
        "evaluator_version": evaluator.CANONICAL_EVALUATOR_VERSION,
        "evaluator_source_sha256": evaluator.CANONICAL_EVALUATOR_SOURCE_SHA256,
        "row_schema": evaluator.CANONICAL_ROW_SCHEMA,
        "identities": identities,
        "conditions": list(conditions),
        "ordered_source_ids": list(source_ids),
        "generation_settings": decode,
        "eos_token_id": 0,
        "row_count": len(rows),
        "selection": {"split": "heldout", "skip_rows": 6000, "take_rows": 256},
        "checkpoint_path": "checkpoint",
        "config_path": "train-config",
        "data_path": "answer-only-data",
        "protocol_sha256": protocol_sha256,
        "runtime": {
            "device": "cpu",
            "torch_version": torch.__version__,
            "cuda_version": None,
            "gpu_name": None,
            "evaluation_precision": "float32",
            "elapsed_seconds": 1.0,
        },
    }
    return rows, manifest



@pytest.mark.parametrize(
    "conditions",
    [
        ["OFF", "INIT", "K1"],
        ["INIT", "OFF", "K1", "K2", "K4", "K8"],
    ],
)
def test_canonical_manifest_requires_exact_ordered_condition_matrix(conditions) -> None:
    rows, manifest = _valid_ledger()
    manifest["conditions"] = conditions
    with pytest.raises(ValueError, match="exact OFF/INIT/K1/K2/K4/K8"):
        evaluator.validate_canonical_ledger(rows, manifest)






def test_evaluator_source_tree_digest_binds_dependency_bytes(tmp_path) -> None:
    evaluator_source = tmp_path / "evaluate.py"
    dependency = tmp_path / "checkpoint.py"
    evaluator_source.write_bytes(b"evaluator-v1")
    dependency.write_bytes(b"dependency-v1")

    first = evaluator._evaluator_source_tree_sha256((evaluator_source, dependency))
    dependency.write_bytes(b"dependency-v2")
    second = evaluator._evaluator_source_tree_sha256((evaluator_source, dependency))

    assert first != second


def test_canonical_ledger_rejects_undeclared_row_fields() -> None:
    rows, manifest = _valid_ledger()
    rows[0]["unreviewed_extra"] = "must-not-publish"

    with pytest.raises(ValueError, match="row fields differ"):
        evaluator.validate_canonical_ledger(rows, manifest)


def test_canonical_ledger_requires_exact_heldout_interval() -> None:
    rows, manifest = _valid_ledger()
    manifest["selection"] = {"split": "heldout", "skip_rows": 0, "take_rows": 2}

    with pytest.raises(ValueError, match="heldout interval"):
        evaluator.validate_canonical_ledger(rows, manifest)


def test_realized_evaluation_precision_reports_mixed_cpu_parameters() -> None:
    model = torch.nn.Module()
    model.register_parameter(
        "backbone_weight",
        torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16)),
    )
    model.register_parameter(
        "processor_weight",
        torch.nn.Parameter(torch.ones(1, dtype=torch.float32)),
    )

    assert evaluator._realized_evaluation_precision(
        model,
        device=torch.device("cpu"),
        autocast_bf16=False,
    ) == "mixed_bfloat16_float32"


@pytest.mark.parametrize("mutation", ["missing", "extra", "bad_sha", "bad_run_digest"])
def test_canonical_manifest_enforces_exact_authenticated_schema(mutation: str) -> None:
    rows, manifest = _valid_ledger()
    if mutation == "missing":
        del manifest["runtime"]
    elif mutation == "extra":
        manifest["unreviewed"] = True
    elif mutation == "bad_sha":
        manifest["protocol_sha256"] = "A" * 64
    else:
        manifest["identities"]["run_sha256"] = "0" * 64
        for row in rows:
            row["run_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="manifest|required schema|protocol_sha256|run_sha256"):
        evaluator.validate_canonical_ledger(rows, manifest)


def test_published_canonical_manifest_requires_complete_row_source_hash_binding(tmp_path) -> None:
    rows, manifest = _valid_ledger()
    manifest["rows_path"] = str(tmp_path / "rows.jsonl")
    with pytest.raises(ValueError, match="published.*hash|schema"):
        evaluator.validate_canonical_ledger(rows, manifest)

def test_canonical_ledger_rejects_malformed_rows() -> None:
    rows, manifest = _valid_ledger()
    del rows[0]["raw_generated_text"]

    with pytest.raises(ValueError, match="row fields differ"):
        evaluator.validate_canonical_ledger(rows, manifest)


def test_canonical_ledger_rejects_duplicate_condition_source_rows() -> None:
    rows, manifest = _valid_ledger()
    rows[-1] = dict(rows[0])

    with pytest.raises(ValueError, match="duplicate condition/source row"):
        evaluator.validate_canonical_ledger(rows, manifest)


def test_canonical_ledger_rejects_missing_condition_coverage() -> None:
    rows, manifest = _valid_ledger()
    rows.pop()

    with pytest.raises(ValueError, match="row count|missing condition coverage"):
        evaluator.validate_canonical_ledger(rows, manifest)


@pytest.mark.parametrize("identity", ["checkpoint_sha256", "config_sha256", "data_sha256"])
def test_canonical_ledger_rejects_identity_hash_mismatch(identity: str) -> None:
    rows, manifest = _valid_ledger()
    rows[-1][identity] = "f" * 64

    with pytest.raises(ValueError, match=f"{identity} mismatch"):
        evaluator.validate_canonical_ledger(rows, manifest)


def test_canonical_ledger_rejects_different_source_order_between_conditions() -> None:
    rows, manifest = _valid_ledger()
    rows[2], rows[3] = rows[3], rows[2]

    with pytest.raises(ValueError, match="ordered source coverage mismatch"):
        evaluator.validate_canonical_ledger(rows, manifest)




def test_canonical_ledger_bundle_is_invisible_before_atomic_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    rows, manifest = _valid_ledger()
    bundle = tmp_path / "evaluation.bundle"
    real_replace = evaluator.os.replace

    def interrupt_bundle_commit(source, destination):
        if destination == bundle:
            raise KeyboardInterrupt("simulated hard interruption")
        return real_replace(source, destination)

    monkeypatch.setattr(evaluator.os, "replace", interrupt_bundle_commit)
    with pytest.raises(KeyboardInterrupt, match="hard interruption"):
        evaluator.publish_canonical_ledger(bundle, rows, manifest)

    assert not bundle.exists()


def test_canonical_ledger_publication_binds_jsonl_and_manifest(tmp_path) -> None:
    rows, manifest = _valid_ledger()
    bundle = tmp_path / "evaluation.bundle"

    published = evaluator.publish_canonical_ledger(bundle, rows, manifest)

    assert published == bundle
    loaded_rows, loaded_manifest = evaluator.load_canonical_ledger(
        bundle,
        expected_manifest_sha256=evaluator._sha256_file(bundle / "manifest.json"),
        expected_protocol_sha256=manifest["protocol_sha256"],
    )
    assert loaded_rows == rows
    assert loaded_manifest["rows_path"] == "rows.jsonl"
    assert loaded_manifest["rows_sha256"] == evaluator._sha256_file(
        bundle / "rows.jsonl"
    )
    with pytest.raises(ValueError, match="manifest sha256 mismatch"):
        evaluator.load_canonical_ledger(
            bundle,
            expected_manifest_sha256="0" * 64,
            expected_protocol_sha256=manifest["protocol_sha256"],
        )
    assert loaded_manifest["ordered_source_ids_sha256"] == evaluator._canonical_json_sha256(
        manifest["ordered_source_ids"]
    )
    evaluator.validate_canonical_ledger(loaded_rows, loaded_manifest)


def test_canonical_ledger_publication_preserves_unicode_line_separator(tmp_path) -> None:
    rows, manifest = _valid_ledger()
    rows[0]["raw_generated_text"] += "\u2028"
    bundle = tmp_path / "unicode-line-separator.bundle"

    evaluator.publish_canonical_ledger(bundle, rows, manifest)
    loaded_rows, _ = evaluator.load_canonical_ledger(
        bundle,
        expected_manifest_sha256=evaluator._sha256_file(bundle / "manifest.json"),
        expected_protocol_sha256=manifest["protocol_sha256"],
    )

    assert loaded_rows[0]["raw_generated_text"].endswith("\u2028")


def test_canonical_ledger_publication_rejects_existing_bundle(tmp_path) -> None:
    rows, manifest = _valid_ledger()
    bundle = tmp_path / "existing.bundle"
    bundle.mkdir()
    with pytest.raises(FileExistsError):
        evaluator.publish_canonical_ledger(bundle, rows, manifest)


def test_canonical_ledger_publication_is_invisible_when_staging_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    rows, manifest = _valid_ledger()
    bundle = tmp_path / "evaluation.bundle"
    real_write = evaluator._write_fsynced_bytes_new

    def fail_manifest(path, payload):
        if path.name == "manifest.json":
            raise RuntimeError("forced manifest failure")
        return real_write(path, payload)

    monkeypatch.setattr(evaluator, "_write_fsynced_bytes_new", fail_manifest)
    with pytest.raises(RuntimeError, match="forced manifest failure"):
        evaluator.publish_canonical_ledger(bundle, rows, manifest)
    assert not bundle.exists()

def test_canonical_manifest_binds_schema_settings_and_ordered_examples() -> None:
    rows, base = _valid_ledger()
    identities = evaluator.EvaluationIdentities(**base["identities"])
    decode = evaluator.DeterministicDecode(**base["generation_settings"])

    manifest = evaluator.build_canonical_manifest(
        rows,
        identities=identities,
        conditions=tuple(base["conditions"]),
        decode=decode,
        eos_token_id=0,
        checkpoint_path="checkpoint",
        config_path="train-config",
        data_path="answer-only-data",
        protocol_sha256="f" * 64,
        split="heldout",
        skip_rows=6000,
        take_rows=256,
        runtime={
            "device": "cpu",
            "torch_version": torch.__version__,
            "cuda_version": None,
            "gpu_name": None,
            "evaluation_precision": "float32",
            "elapsed_seconds": 1.0,
        },
    )

    assert manifest["schema"] == evaluator.CANONICAL_MANIFEST_SCHEMA
    assert manifest["row_schema"] == evaluator.CANONICAL_ROW_SCHEMA
    assert manifest["generation_settings"] == decode.as_dict()
    assert len(manifest["ordered_source_ids"]) == 256
    assert manifest["ordered_source_ids"][0] == "gsm8k-train:6000"
    assert manifest["ordered_source_ids"][-1] == "gsm8k-train:6255"
    assert manifest["conditions"] == ["OFF", "INIT", "K1", "K2", "K4", "K8"]
    assert manifest["selection"] == {
        "split": "heldout",
        "skip_rows": 6000,
        "take_rows": 256,
    }
    assert manifest["checkpoint_path"] == "checkpoint"
    assert manifest["config_path"] == "train-config"
    assert manifest["data_path"] == "answer-only-data"
    evaluator.validate_canonical_ledger(rows, manifest)



def test_completed_training_provenance_rejects_mutated_process_cursor() -> None:
    config = load_train_config(
        "configs/train/process_branch_qwen3_06b/qwen3_06b_answer_only_k1_6000_seed24.yaml"
    )
    process = config.process_branch
    assert process is not None
    process = replace(
        process,
        expected_train_eligible_records=2,
        expected_train_eligible_source_ids_sha256="2" * 64,
        expected_heldout_eligible_records=2,
        expected_heldout_eligible_source_ids_sha256="3" * 64,
        expected_tokenizer_sha256="1" * 64,
    )
    config.process_branch = process
    data_identity = {
        "splits": {
            "train": {
                "eligible_records": 2,
                "ordered_source_ids_sha256": "2" * 64,
            },
            "heldout": {
                "eligible_records": 2,
                "ordered_source_ids_sha256": "3" * 64,
            },
        }
    }
    fingerprint = process_branch_trainer._process_training_fingerprint(
        config,
        data_identity=data_identity,
        initial_processor_sha256="0" * 64,
        observed_tokenizer_sha256=cast(str, process.expected_tokenizer_sha256),
    )
    seen_examples = process.max_optimizer_steps * process.gradient_accumulation_steps
    schedule = evaluator.ProcessSchedule(
        record_count=2,
        cycle_choices=process.cycle_choices,
        seed=config.seed,
    )
    for _ in range(seen_examples):
        schedule.next_example()
    train_state = {
        "data_identity": data_identity,
        "code_identity": fingerprint["code_identity"],
        "training_fingerprint": fingerprint,
        "optimizer_steps": process.max_optimizer_steps,
        "seen_examples": seen_examples,
        "valid_records": 2,
        "heldout_valid_records": 2,
        "process_cursor": schedule.cursor.to_dict(),
    }
    manifest = {
        "initial_processor_sha256": "0" * 64,
    }
    evaluator._validate_completed_training_provenance(
        train_state=train_state,
        train_config=config,
        checkpoint_manifest=manifest,
        observed_data_identity=data_identity,
    )
    wrong_data_identity = {
        "splits": {
            "train": {
                "eligible_records": 3,
                "ordered_source_ids_sha256": "4" * 64,
            },
            "heldout": data_identity["splits"]["heldout"],
        }
    }
    wrong_population_state = {
        **train_state,
        "data_identity": wrong_data_identity,
        "training_fingerprint": {
            **fingerprint,
            "data_identity": wrong_data_identity,
        },
        "valid_records": 3,
    }
    with pytest.raises(ValueError, match="train population differs from frozen identity"):
        evaluator._validate_completed_training_provenance(
            train_state=wrong_population_state,
            train_config=config,
            checkpoint_manifest=manifest,
            observed_data_identity=wrong_data_identity,
        )

    train_state["process_cursor"] = {
        **cast(dict, train_state["process_cursor"]),
        "offset": 0,
    }
    with pytest.raises(ValueError, match="exact schedule replay"):
        evaluator._validate_completed_training_provenance(
            train_state=train_state,
            train_config=config,
            checkpoint_manifest=manifest,
            observed_data_identity=data_identity,
        )


def test_canonical_ledger_rejects_false_published_rows_digest() -> None:
    rows, manifest = _valid_ledger()
    manifest.update(
        {
            "rows_path": "rows.jsonl",
            "rows_sha256": "4" * 64,
            "ordered_source_ids_sha256": evaluator._canonical_json_sha256(
                manifest["ordered_source_ids"]
            ),
        }
    )
    with pytest.raises(ValueError, match="published rows_sha256 mismatch"):
        evaluator.validate_canonical_ledger(rows, manifest)


def test_canonical_ledger_rejects_arbitrary_rows_path() -> None:
    rows, manifest = _valid_ledger()
    manifest.update(
        {
            "rows_path": "arbitrary/renamed.jsonl",
            "rows_sha256": "4" * 64,
            "ordered_source_ids_sha256": "5" * 64,
        }
    )
    with pytest.raises(ValueError, match="rows_path must equal fixed artifact role"):
        evaluator.validate_canonical_ledger(rows, manifest)


def test_canonical_ledger_rejects_arbitrary_artifact_labels() -> None:
    rows, manifest = _valid_ledger()
    manifest["checkpoint_path"] = "renamed-checkpoint"
    with pytest.raises(ValueError, match="fixed artifact role"):
        evaluator.validate_canonical_ledger(rows, manifest)


def test_canonical_checkpoint_authentication_rejects_train_state_tamper(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    files = {
        "model.safetensors": b"model",
        "initial_processor.safetensors": b"initial",
        "config.json": b'{"model":{}}\n',
        "run_config.json": b'{"mode":"process_branch"}\n',
        "train_state.json": b'{"objective":"answer_only"}\n',
    }
    for name, content in files.items():
        (checkpoint / name).write_bytes(content)
    manifest = {
        "schema": "gcw.final_checkpoint.manifest.v2",
        "files": {name: evaluator._sha256_file(checkpoint / name) for name in sorted(files)},
        "initial_processor_sha256": "a" * 64,
        "final_processor_sha256": "b" * 64,
    }
    manifest_path = checkpoint / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    manifest_sha256 = evaluator._sha256_file(manifest_path)

    assert evaluator.authenticate_canonical_checkpoint(
        checkpoint,
        expected_manifest_sha256=manifest_sha256,
        expected_model_sha256=manifest["files"]["model.safetensors"],
    ) == manifest

    (checkpoint / "train_state.json").write_text('{"objective":"tampered"}\n')
    with pytest.raises(ValueError, match="train_state.json.*sha256 mismatch"):
        evaluator.authenticate_canonical_checkpoint(
            checkpoint,
            expected_manifest_sha256=manifest_sha256,
            expected_model_sha256=manifest["files"]["model.safetensors"],
        )



def test_authenticated_checkpoint_snapshot_is_the_only_loader_input(tmp_path) -> None:
    from safetensors.torch import save_file

    from gcw.checkpoint import publish_final_checkpoint_manifest, save_checkpoint
    from gcw.state_hash import tensor_state_sha256

    model = _tiny_longshot()
    checkpoint = tmp_path / "checkpoint-snapshot"
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in model.cache_processor.state_dict().items()
    }
    save_checkpoint(
        checkpoint,
        model,
        model.config,
        {"objective": "answer_only", "marker": "authenticated"},
    )
    save_file(initial_state, checkpoint / "initial_processor.safetensors")
    (checkpoint / "run_config.json").write_text("{}\n")
    final_hash = tensor_state_sha256(model.cache_processor.state_dict())
    initial_hash = tensor_state_sha256(initial_state)
    manifest_sha256 = publish_final_checkpoint_manifest(
        checkpoint,
        final_hash,
        initial_hash,
    )
    model_file_sha256 = evaluator._sha256_file(checkpoint / "model.safetensors")

    snapshot = evaluator.authenticate_canonical_checkpoint_snapshot(
        checkpoint,
        expected_manifest_sha256=manifest_sha256,
        expected_model_sha256=model_file_sha256,
    )
    for name in (
        "config.json",
        "initial_processor.safetensors",
        "model.safetensors",
        "run_config.json",
        "train_state.json",
    ):
        (checkpoint / name).write_bytes(b"replaced-after-authentication")

    loaded, loaded_config, train_state = evaluator.load_checkpoint_bytes(
        config_bytes=snapshot.files["config.json"],
        train_state_bytes=snapshot.files["train_state.json"],
        model_bytes=snapshot.files["model.safetensors"],
        device="cpu",
    )
    assert loaded_config == model.config
    assert train_state["marker"] == "authenticated"
    assert tensor_state_sha256(loaded.state_dict()) == tensor_state_sha256(model.state_dict())






@pytest.mark.parametrize(
    ("tokenizer_name", "tokenizer_source", "expected"),
    [
        ("hf", "artifacts/models/qwen", Path("artifacts/models/qwen")),
        ("hf:artifacts/models/qwen", None, Path("artifacts/models/qwen")),
    ],
)
def test_canonical_tokenizer_source_accepts_local_hf_forms(
    tokenizer_name: str,
    tokenizer_source: str | None,
    expected: Path,
) -> None:
    assert evaluator._canonical_tokenizer_source(
        tokenizer_name,
        tokenizer_source,
    ) == expected


def test_canonical_tokenizer_source_rejects_non_hf_tokenizer() -> None:
    with pytest.raises(ValueError, match="local HF tokenizer"):
        evaluator._canonical_tokenizer_source("cosmo2", None)


def test_authenticated_tokenizer_snapshot_is_private_and_immutable(tmp_path) -> None:
    from gcw.process_branch_trainer import _path_tree_sha256

    source = tmp_path / "tokenizer"
    source.mkdir()
    (source / "tokenizer.json").write_bytes(b"original-tokenizer")
    nested = source / "nested"
    nested.mkdir()
    (nested / "config.json").write_bytes(b"original-config")
    expected_sha256 = _path_tree_sha256(source)

    with evaluator.authenticated_tokenizer_snapshot(
        source,
        expected_sha256=expected_sha256,
    ) as snapshot:
        assert snapshot != source
        (source / "tokenizer.json").write_bytes(b"mutated")
        assert (snapshot / "tokenizer.json").read_bytes() == b"original-tokenizer"
        assert (snapshot / "nested" / "config.json").read_bytes() == b"original-config"

    assert not snapshot.exists()


def test_canonical_execution_plan_refuses_protocol_blocked_final_split(tmp_path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema": "gcw.answer_only_eval.protocol.v1",
                "evaluator_version": evaluator.CANONICAL_EVALUATOR_VERSION,
                "conditions": ["OFF", "INIT", "K1", "K2", "K4", "K8"],
                "generation": {"answer_prefix": "P", "prompt_suffix": "S", "max_new_tokens": 2, "temperature": 0.0, "top_p": 1.0, "top_k": 0},
                "dataset": {"path": "/data.jsonl", "sha256": "a" * 64, "ordered_source_ids_sha256": "b" * 64},
                "splits": {"final": {"skip_rows": 6256, "take_rows": 256, "execution_allowed": False}},
                "arms": {"p0": {"train_config": "/p0.yaml", "checkpoint": "/p0/final", "checkpoint_sha256": None}},
            }
        )
    )

    with pytest.raises(ValueError, match="final.*forbidden"):
        evaluator.resolve_canonical_execution_plan(protocol, arm="p0", split="final")



def test_canonical_execution_plan_rejects_final_even_when_protocol_enables_it(tmp_path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "schema": "gcw.answer_only_eval.protocol.v1",
        "evaluator_version": evaluator.CANONICAL_EVALUATOR_VERSION,
        "conditions": ["OFF", "INIT", "K1", "K2", "K4", "K8"],
        "generation": {
            "answer_prefix": "P", "prompt_suffix": "S", "max_new_tokens": 2,
            "temperature": 0.0, "top_p": 1.0, "top_k": 0,
        },
        "dataset": {
            "path": "/data.jsonl", "sha256": "a" * 64,
            "ordered_source_ids_sha256": "b" * 64,
        },
        "splits": {
            "final": {"skip_rows": 6256, "take_rows": 256, "execution_allowed": True}
        },
        "arms": {
            "p0": {
                "train_config": "/p0.yaml", "checkpoint": "/p0/final",
                "checkpoint_sha256": "c" * 64, "config_sha256": "d" * 64,
            }
        },
    }))

    with pytest.raises(ValueError, match="final.*forbidden"):
        evaluator.resolve_canonical_execution_plan(protocol, arm="p0", split="final")
    with pytest.raises(ValueError, match="final.*forbidden"):
        evaluator.main([
            "canonical", "--protocol", str(protocol), "--arm", "p0", "--split", "final",
            "--bundle-out", str(tmp_path / "ledger.bundle"),
        ])

def test_evaluator_cli_rejects_legacy_arbitrary_slices() -> None:
    with pytest.raises(ValueError, match="canonical-only"):
        evaluator.main(["--records", "copied-final.jsonl"])


def test_record_evaluator_cli_routes_canonical_protocol_without_legacy_cycles(tmp_path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema": "gcw.answer_only_eval.protocol.v1",
                "evaluator_version": evaluator.CANONICAL_EVALUATOR_VERSION,
                "conditions": ["OFF", "INIT", "K1", "K2", "K4", "K8"],
                "generation": {"answer_prefix": "P", "prompt_suffix": "S", "max_new_tokens": 2, "temperature": 0.0, "top_p": 1.0, "top_k": 0},
                "dataset": {"path": "/data.jsonl", "sha256": "a" * 64, "ordered_source_ids_sha256": "b" * 64},
                "splits": {"final": {"skip_rows": 6256, "take_rows": 256, "execution_allowed": False}},
                "arms": {"p0": {"train_config": "/p0.yaml", "checkpoint": "/p0/final", "checkpoint_sha256": None}},
            }
        )
    )

    with pytest.raises(ValueError, match="final.*forbidden"):
        evaluator.main(
            [
                "canonical", "--protocol", str(protocol), "--arm", "p0", "--split", "final",
                "--bundle-out", str(tmp_path / "ledger.bundle"),
            ]
        )


def test_canonical_ledger_rejects_boolean_decode_numbers() -> None:
    rows, manifest = _valid_ledger()
    manifest["generation_settings"]["max_new_tokens"] = True
    for row in rows:
        row["decode"]["max_new_tokens"] = True

    with pytest.raises(ValueError, match="max_new_tokens.*integer"):
        evaluator.validate_canonical_ledger(rows, manifest)


def test_canonical_ledger_rejects_short_max_token_claim() -> None:
    rows, manifest = _valid_ledger()
    rows[0]["answer_token_count"] = 1
    rows[0]["generated_token_ids"] = [1]
    with pytest.raises(ValueError, match="max-token termination evidence mismatch"):
        evaluator.validate_canonical_ledger(rows, manifest)


def test_canonical_ledger_rejects_eos_claim_without_terminal_eos() -> None:
    rows, manifest = _valid_ledger()
    rows[0]["termination_reason"] = "eos"
    rows[0]["cap_hit"] = False
    with pytest.raises(ValueError, match="terminal EOS evidence"):
        evaluator.validate_canonical_ledger(rows, manifest)


def test_canonical_ledger_rejects_interior_eos_for_max_tokens() -> None:
    rows, manifest = _valid_ledger()
    rows[0]["generated_token_ids"] = [0, 2]
    with pytest.raises(ValueError, match="max-token termination evidence mismatch"):
        evaluator.validate_canonical_ledger(rows, manifest)


def test_canonical_ledger_rejects_eos_sequence_beyond_decode_cap() -> None:
    rows, manifest = _valid_ledger()
    rows[0]["termination_reason"] = "eos"
    rows[0]["cap_hit"] = False
    rows[0]["answer_token_count"] = 3
    rows[0]["generated_token_ids"] = [1, 2, 0]
    with pytest.raises(ValueError, match="exceeds decode bounds"):
        evaluator.validate_canonical_ledger(rows, manifest)


def test_canonical_ledger_recomputes_extraction_and_correctness() -> None:
    rows, manifest = _valid_ledger()
    rows[0]["correctness"] = False

    with pytest.raises(ValueError, match="correctness mismatch"):
        evaluator.validate_canonical_ledger(rows, manifest)

def test_atomic_write_json_new_publishes_complete_payload(tmp_path) -> None:
    output = tmp_path / "nested" / "result.json"

    evaluator._atomic_write_json_new(output, {"accuracy": 0.5, "rows": [1, 2]})

    assert json.loads(output.read_text()) == {"accuracy": 0.5, "rows": [1, 2]}
    assert list(output.parent.iterdir()) == [output]


def test_atomic_write_json_new_does_not_overwrite(tmp_path) -> None:
    output = tmp_path / "result.json"
    output.write_text("original\n")

    with pytest.raises(FileExistsError):
        evaluator._atomic_write_json_new(output, {"replacement": True})

    assert output.read_text() == "original\n"


def test_atomic_write_json_new_does_not_publish_partial_json(monkeypatch, tmp_path) -> None:
    output = tmp_path / "result.json"

    def fail_after_partial_write(payload, handle, **kwargs) -> None:
        del payload, kwargs
        handle.write('{"partial":')
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(evaluator.json, "dump", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        evaluator._atomic_write_json_new(output, {"complete": True})

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_canonical_plan_binds_the_same_protocol_snapshot_it_parses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "schema": "gcw.answer_only_eval.protocol.v1",
        "evaluator_version": evaluator.CANONICAL_EVALUATOR_VERSION,
        "conditions": ["OFF", "INIT", "K1", "K2", "K4", "K8"],
        "dataset": {
            "path": "/data.jsonl", "sha256": "a" * 64,
            "ordered_source_ids_sha256": "b" * 64,
        },
        "generation": {
            "answer_prefix": "P", "prompt_suffix": "S", "max_new_tokens": 2,
            "temperature": 0.0, "top_p": 1.0, "top_k": 0,
        },
        "splits": {
            "heldout": {"skip_rows": 6000, "take_rows": 256, "execution_allowed": True},
            "final": {"skip_rows": 6256, "take_rows": 256, "execution_allowed": False},
        },
        "arms": {
            name: {
                "run_id": name, "train_config": f"/{name}.yaml",
                "checkpoint": f"/{name}/final", "checkpoint_sha256": "c" * 64,
                "checkpoint_manifest_sha256": "d" * 64, "config_sha256": "e" * 64,
            }
            for name in ("p0", "p1")
        },
    }))
    authenticated_bytes = protocol.read_bytes()
    expected_sha256 = hashlib.sha256(authenticated_bytes).hexdigest()
    expected_arm_sha256 = evaluator._canonical_json_sha256(
        json.loads(authenticated_bytes)["arms"]["p0"]
    )
    original_read_bytes = Path.read_bytes

    def mutate_after_read(path: Path) -> bytes:
        payload = original_read_bytes(path)
        if path == protocol:
            protocol.write_text('{"tampered":true}')
        return payload

    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)
    plan = evaluator.resolve_canonical_execution_plan(protocol, arm="p0", split="heldout")

    assert plan.protocol_sha256 == expected_sha256
    assert plan.arm_sha256 == expected_arm_sha256
