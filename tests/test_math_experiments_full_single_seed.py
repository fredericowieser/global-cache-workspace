from argparse import Namespace
import hashlib
import json

from scripts import run_math_experiments_full_single_seed as series


def test_plan_declares_full_order_scheduler_and_drop_last_epoch(tmp_path, monkeypatch):
    dataset = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    dataset.write_text("{}\n")
    validation.write_text("{}\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "gcw.math_experiment_data.v1",
                "rows": 697_744,
                "validation_rows": 4096,
                "seed": 24,
                "sequence_length": 4096,
                "source_ids_sha256": "12" * 32,
                "tokenizer_identity_sha256": "23" * 32,
                "dataset_path": str(dataset),
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "validation_path": str(validation),
                "validation_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
            }
        )
    )
    monkeypatch.setattr(
        series,
        "_capture_jsonl_identity",
        lambda path: (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            4096 if path == validation else 697_744,
        ),
    )
    args = Namespace(
        data_manifest=str(manifest),
        dataset=None,
        validation_dataset=None,
        experiment="experiment",
        model="model",
        backbone_checkpoint="checkpoint",
    )

    data = series._load_data_manifest(args)
    plan = series._plan(args, data, "a" * 40)

    assert plan["training_order"] == list(series.TRAINABLE_ARMS)
    assert plan["imported_arms"] == ["full-sft"]
    assert len(plan["training_order"]) == 10
    assert len(plan["evaluation_order"]) == 12
    assert "icarus" in plan["training_order"]
    assert "diffusion-bgt-t4-sigma0" in plan["training_order"]
    assert series.PROCESSOR_ARMS == frozenset(
        {
            "bt",
            "bgt",
            "icarus",
            "diffusion-bgt-t4-sigma015",
            "diffusion-bgt-t4-sigma0",
            "longshot",
            "longshot-internal-r4",
            "longshot-external-r4",
            "flow-bgt-t4",
        }
    )
    assert all(series.PEAK_LEARNING_RATES[arm] == 1e-3 for arm in series.PROCESSOR_ARMS)
    assert series.PEAK_LEARNING_RATES["full-sft"] == 5e-6
    assert series.PEAK_LEARNING_RATES["lora-r32"] == 1e-4
    assert all(series.MICRO_BATCH_ROWS[arm] > 0 for arm in series.TRAIN_ORDER)
    assert plan["max_steps"] == 10_902
    assert plan["effective_epoch_rows"] == 697_728
    assert plan["drop_last_rows"] == 16
    assert plan["scheduler"] == {
        "name": "warmup_stable_decay",
        "warmup_fraction": 0.03,
        "stable_fraction": 0.87,
        "warmdown_fraction": 0.10,
        "final_learning_rate_fraction": 0.0,
    }
    assert series._evaluation_cells("bgt") == (
        ("off", "thinking_off"),
        ("off", "thinking_on"),
        ("on", "thinking_off"),
        ("on", "thinking_on"),
    )
    assert series._evaluation_cells("full-sft") == (
        ("on", "thinking_off"),
        ("on", "thinking_on"),
    )
    assert series._processor_states("bgt") == ("off", "on")
    assert series._processor_states("icarus") == ("off", "on")
    assert series._processor_states("diffusion-bgt-t4-sigma0") == ("off", "on")
    assert series._processor_states("full-sft") == ("on",)
    assert plan["schema"] == "gcw.math_experiment_series.v2"
    assert "lambada_openai" not in plan["tasks"]["canonical_multiple_choice"]
    assert plan["tasks"]["canonical_multiple_choice"] == list(series.CANONICAL_TASKS)
    assert plan["tasks"]["numeric_math"] == list(series.NUMERIC_MATH_TASKS)
    assert plan["evaluation_protocols"]["canonical_likelihood"]["thinking_dimension"] is False
    assert plan["evaluation_protocols"]["numeric_generation"]["max_new_tokens"] == 2048
    assert plan["evaluation_protocols"]["numeric_generation"]["scoring"] == "numeric_last_answer"


def test_reuse_result_preserves_bytes_and_records_source_hash(tmp_path):
    source = tmp_path / "v2" / "result.json"
    target = tmp_path / "v3" / "result.json"
    source.parent.mkdir()
    source.write_bytes(b'{"evidence":true}\n')

    assert series._reuse_result(source, target, protocol="test-v1") is True
    assert target.read_bytes() == source.read_bytes()
    marker = json.loads((target.parent / "reused-evidence.json").read_text())
    assert marker["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert marker["bytes_preserved"] is True
    assert series._reuse_result(source, target, protocol="test-v1") is False


def test_numeric_result_validation_rejects_choice_rows(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "mode": "thinking_off",
                "max_new_tokens": 2048,
                "tasks": {"gsm8k_openmath": {}},
                "datasets": {"gsm8k_openmath": {}},
                "rows": [{"target_kind": "choice"}],
            }
        )
    )
    try:
        series._validate_numeric_result(result, "gsm8k_openmath", "thinking_off")
    except ValueError as error:
        assert "non-numeric" in str(error)
    else:
        raise AssertionError("choice rows must not pass numeric protocol validation")
