from __future__ import annotations

import hashlib
import json

import pytest

from gcw.experiment import validate_experiment
from gcw.config import DataConfig, TrainConfig


def test_experiment_contract_accepts_only_matched_execution(
    tmp_path, monkeypatch
):
    class Tokenizer:
        @staticmethod
        def identity_sha256():
            return "1" * 64

    monkeypatch.setattr("gcw.experiment.get_tokenizer", lambda _: Tokenizer())
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{}\n")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"weights")
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = tmp_path / "experiment.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "gcw.math_experiment.v1",
                "dataset_sha256": digest,
                "tokenizer_identity_sha256": "1" * 64,
                "backbone_sha256": checkpoint_digest,
                "sequence_length": 4096,
                "global_batch_rows": 64,
                "max_steps": 10,
                "dataset_path": str(dataset.resolve()),
                "tokenizer_source": "Qwen/Qwen3-0.6B",
                "backbone_checkpoint": str(tmp_path),
            }
        )
    )
    config = TrainConfig(
        data=DataConfig(
            kind="sft",
            dataset=str(dataset),
            tokenizer="hf",
            tokenizer_source="Qwen/Qwen3-0.6B",
            experiment_manifest=str(manifest),
        ),
        mode="sft",
        backbone_checkpoint=str(tmp_path),
        sequence_len=4096,
        global_batch_rows=64,
        max_steps=10,
    )

    assert validate_experiment(config) is not None
    config.max_steps = 11
    with pytest.raises(ValueError, match="max_steps"):
        validate_experiment(config)
