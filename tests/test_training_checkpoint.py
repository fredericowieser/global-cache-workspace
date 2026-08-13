import copy
from dataclasses import replace
import json
import shutil

import pytest
import torch

from gcw.checkpoint import save_checkpoint
from gcw.config import ModelConfig
from gcw.data.training import FineWebCursor, RandomCursor
from gcw.models import build_model
from gcw.training.checkpoint import (
    CheckpointCompatibilityError,
    LegacyCheckpointError,
    ResumeState,
    TrainingCheckpoint,
    TrainingCheckpointError,
    UnsupportedTopologyError,
    save_training_checkpoint,
    verify_training_checkpoint_files,
)


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        variant="tf",
        family="smollm",
        sequence_len=8,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        tie_embeddings=True,
    )


def test_training_checkpoint_restores_model_optimizer_and_resume_state(tmp_path):
    config = tiny_model_config()
    model = build_model(config)
    model.init_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, config.vocab_size, (1, 4))
    y = torch.roll(x, -1, 1)
    model(x, y).backward()
    optimizer.step()
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    fingerprint = {"model": "tiny", "distributed": "none", "data_kind": "random", "seed": 24}
    resume = ResumeState(
        step=1,
        smooth_loss=1.25,
        loader_cursor=RandomCursor(rng_state=(1, 2, 3)),
        saved_world_size=1,
        exact_topology=True,
    )

    save_training_checkpoint(tmp_path / "checkpoint", model, config, optimizer, resume, fingerprint)

    restored_model = build_model(config)
    restored_model.init_weights()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    checkpoint = TrainingCheckpoint.open(
        tmp_path / "checkpoint",
        expected_fingerprint=fingerprint,
        topology="same",
        current_world_size=1,
    )
    for payload_name in (
        "model.safetensors",
        "optimizer.pt",
        "rng.pt",
        "config.json",
        "train_state.json",
    ):
        (tmp_path / "checkpoint" / payload_name).write_bytes(b"replaced-after-open")
    checkpoint.restore_model(restored_model)
    checkpoint.restore_optimizer(restored_optimizer, device="cpu")
    checkpoint.restore_rng(rank=0)

    assert checkpoint.resume_state == resume
    for expected, actual in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(expected, actual)
    _assert_nested_equal(expected_optimizer, restored_optimizer.state_dict())



def test_verify_training_checkpoint_files_returns_authenticated_state(tmp_path):
    config, model, optimizer, resume, fingerprint = _checkpoint_fixture()
    path = tmp_path / "checkpoint"
    save_training_checkpoint(path, model, config, optimizer, resume, fingerprint)

    verified = verify_training_checkpoint_files(path)

    assert verified["path"] == str(path)
    assert verified["manifest"]["step"] == 1
    assert verified["train_state"]["step"] == 1
    assert verified["manifest"]["training_fingerprint"] == fingerprint


def test_exact_resume_rejects_legacy_model_only_checkpoint(tmp_path):
    config = tiny_model_config()
    model = build_model(config)
    model.init_weights()
    save_checkpoint(tmp_path / "legacy", model, config, {"step": 3})

    with pytest.raises(LegacyCheckpointError, match="version-2"):
        TrainingCheckpoint.open(
            tmp_path / "legacy",
            expected_fingerprint={},
            topology="same",
            current_world_size=1,
        )


def test_fsdp_resume_fails_with_specific_topology_error(tmp_path):
    with pytest.raises(UnsupportedTopologyError, match="FSDP"):
        TrainingCheckpoint.open(
            tmp_path / "legacy-or-missing",
            expected_fingerprint={"distributed": "fsdp"},
            topology="same",
            current_world_size=2,
        )


def test_training_checkpoint_rejects_corrupt_payload(tmp_path):
    config, model, optimizer, resume, fingerprint = _checkpoint_fixture()
    path = tmp_path / "checkpoint"
    save_training_checkpoint(path, model, config, optimizer, resume, fingerprint)
    with open(path / "optimizer.pt", "ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(TrainingCheckpointError, match="hash mismatch"):
        TrainingCheckpoint.open(path, expected_fingerprint=fingerprint, topology="same", current_world_size=1)


def test_incomplete_destination_falls_back_to_previous_complete_checkpoint(tmp_path):
    config, model, optimizer, resume, fingerprint = _checkpoint_fixture()
    path = tmp_path / "checkpoint"
    save_training_checkpoint(path, model, config, optimizer, resume, fingerprint)
    path.rename(tmp_path / "checkpoint.previous")
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"partial")

    checkpoint = TrainingCheckpoint.open(
        path,
        expected_fingerprint=fingerprint,
        topology="same",
        current_world_size=1,
    )

    assert checkpoint.path.name == "checkpoint.previous"
    assert checkpoint.resume_state.step == 1


def test_corrupt_destination_falls_back_to_previous_complete_checkpoint(tmp_path):
    config, model, optimizer, resume, fingerprint = _checkpoint_fixture()
    path = tmp_path / "checkpoint"
    save_training_checkpoint(path, model, config, optimizer, resume, fingerprint)
    previous = tmp_path / "checkpoint.previous"
    path.rename(previous)
    shutil.copytree(previous, path)
    with (path / "optimizer.pt").open("ab") as handle:
        handle.write(b"corrupt")

    checkpoint = TrainingCheckpoint.open(
        path,
        expected_fingerprint=fingerprint,
        topology="same",
        current_world_size=1,
    )

    assert checkpoint.path == previous


def test_manifest_must_hash_every_required_payload(tmp_path):
    config, model, optimizer, resume, fingerprint = _checkpoint_fixture()
    path = tmp_path / "checkpoint"
    save_training_checkpoint(path, model, config, optimizer, resume, fingerprint)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["files"]["optimizer.pt"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TrainingCheckpointError, match="required file hashes"):
        TrainingCheckpoint.open(path, expected_fingerprint=fingerprint, topology="same", current_world_size=1)


def test_elastic_resume_requires_fineweb_ddp_and_marks_changed_topology(tmp_path):
    config, model, optimizer, resume, _ = _checkpoint_fixture()
    resume = replace(resume, loader_cursor=FineWebCursor())
    fingerprint = {"distributed": "ddp", "data_kind": "fineweb", "seed": 24}
    path = tmp_path / "checkpoint"
    save_training_checkpoint(path, model, config, optimizer, resume, fingerprint)

    checkpoint = TrainingCheckpoint.open(
        path,
        expected_fingerprint=fingerprint,
        topology="elastic_ddp",
        current_world_size=2,
    )

    assert checkpoint.resume_state.exact_topology is False
    with pytest.raises(CheckpointCompatibilityError, match="world size"):
        TrainingCheckpoint.open(path, expected_fingerprint=fingerprint, topology="same", current_world_size=2)


def test_elastic_resume_rejects_non_fineweb_data(tmp_path):
    config, model, optimizer, resume, _ = _checkpoint_fixture()
    fingerprint = {"distributed": "ddp", "data_kind": "random", "seed": 24}
    path = tmp_path / "checkpoint"
    save_training_checkpoint(path, model, config, optimizer, resume, fingerprint)

    with pytest.raises(UnsupportedTopologyError, match="FineWeb"):
        TrainingCheckpoint.open(
            path,
            expected_fingerprint=fingerprint,
            topology="elastic_ddp",
            current_world_size=2,
        )


def _checkpoint_fixture():
    config = tiny_model_config()
    model = build_model(config)
    model.init_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    resume = ResumeState(
        step=1,
        smooth_loss=1.25,
        loader_cursor=RandomCursor(rng_state=(1, 2, 3)),
        saved_world_size=1,
        exact_topology=True,
    )
    fingerprint = {"model": "tiny", "distributed": "none", "data_kind": "random", "seed": 24}
    return config, model, optimizer, resume, fingerprint


def _assert_nested_equal(expected, actual):
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
