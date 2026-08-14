from argparse import Namespace
import hashlib
import json

from scripts import run_math_experiments_full_single_seed as series


def test_plan_declares_full_order_scheduler_and_drop_last_epoch(tmp_path):
    dataset = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    dataset.write_text("{}\n")
    validation.write_text("{}\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": 697_744,
                "dataset_path": str(dataset),
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "validation_path": str(validation),
                "validation_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
            }
        )
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

    assert plan["training_order"] == list(series.TRAIN_ORDER)
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
