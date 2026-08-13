from __future__ import annotations

import hashlib
import json

import pytest

from gcw.campaign import validate_campaign
from gcw.config import DataConfig, TrainConfig


def test_campaign_contract_accepts_only_matched_execution(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{}\n')
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "gcw.math_campaign.v1",
                "dataset_sha256": digest,
                "tokenizer_identity_sha256": "1" * 64,
                "backbone_sha256": "2" * 64,
                "sequence_length": 4096,
                "global_batch_rows": 64,
                "max_steps": 10,
                "dataset_path": str(dataset.resolve()),
                "tokenizer_source": "Qwen/Qwen3-0.6B",
                "backbone_checkpoint": "/models/qwen",
            }
        )
    )
    config = TrainConfig(
        data=DataConfig(
            kind="sft",
            dataset=str(dataset),
            tokenizer="hf",
            tokenizer_source="Qwen/Qwen3-0.6B",
            campaign_manifest=str(manifest),
        ),
        mode="sft",
        backbone_checkpoint="/models/qwen",
        sequence_len=4096,
        total_batch_size=64,
        max_steps=10,
    )

    assert validate_campaign(config) is not None
    config.max_steps = 11
    with pytest.raises(ValueError, match="max_steps"):
        validate_campaign(config)
