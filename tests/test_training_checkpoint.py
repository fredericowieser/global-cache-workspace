import copy
from dataclasses import replace
import json
import shutil
import warnings

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from gcw.checkpoint import save_checkpoint
from gcw.config import (
    CacheProcessorConfig,
    DataConfig,
    ModelConfig,
    ResumeConfig,
    TrainConfig,
    to_dict,
)
from gcw.data.training import FineWebCursor, RandomCursor
from gcw.models import build_model
from gcw.training.audit import _training_fingerprint
from gcw.training.core import train
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


def test_training_checkpoint_restores_model_optimizer_and_resume_state(
    tmp_path,
):
    config = tiny_model_config()
    model = build_model(config)
    model.init_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, config.vocab_size, (1, 4))
    y = torch.roll(x, -1, 1)
    model(x, y).backward()
    optimizer.step()
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    fingerprint = {
        "model": "tiny",
        "distributed": "none",
        "data_kind": "random",
        "seed": 24,
    }
    resume = ResumeState(
        step=1,
        smooth_loss=1.25,
        loader_cursor=RandomCursor(rng_state=(1, 2, 3)),
        saved_world_size=1,
        exact_topology=True,
    )

    save_training_checkpoint(
        tmp_path / "checkpoint", model, config, optimizer, resume, fingerprint
    )

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
        (tmp_path / "checkpoint" / payload_name).write_bytes(
            b"replaced-after-open"
        )
    checkpoint.restore_model(restored_model)
    checkpoint.restore_optimizer(restored_optimizer, device="cpu")
    checkpoint.restore_rng(rank=0)

    assert checkpoint.resume_state == resume
    for expected, actual in zip(
        model.parameters(), restored_model.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)
    _assert_nested_equal(expected_optimizer, restored_optimizer.state_dict())


def test_verify_training_checkpoint_files_returns_authenticated_state(tmp_path):
    config, model, optimizer, resume, fingerprint = _checkpoint_fixture()
    path = tmp_path / "checkpoint"
    save_training_checkpoint(
        path, model, config, optimizer, resume, fingerprint
    )

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
    save_training_checkpoint(
        path, model, config, optimizer, resume, fingerprint
    )
    with open(path / "optimizer.pt", "ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(TrainingCheckpointError, match="hash mismatch"):
        TrainingCheckpoint.open(
            path,
            expected_fingerprint=fingerprint,
            topology="same",
            current_world_size=1,
        )


def test_incomplete_destination_falls_back_to_previous_complete_checkpoint(
    tmp_path,
):
    config, model, optimizer, resume, fingerprint = _checkpoint_fixture()
    path = tmp_path / "checkpoint"
    save_training_checkpoint(
        path, model, config, optimizer, resume, fingerprint
    )
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


def test_corrupt_destination_falls_back_to_previous_complete_checkpoint(
    tmp_path,
):
    config, model, optimizer, resume, fingerprint = _checkpoint_fixture()
    path = tmp_path / "checkpoint"
    save_training_checkpoint(
        path, model, config, optimizer, resume, fingerprint
    )
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
    save_training_checkpoint(
        path, model, config, optimizer, resume, fingerprint
    )
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["files"]["optimizer.pt"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TrainingCheckpointError, match="required file hashes"):
        TrainingCheckpoint.open(
            path,
            expected_fingerprint=fingerprint,
            topology="same",
            current_world_size=1,
        )


def test_elastic_resume_requires_fineweb_ddp_and_marks_changed_topology(
    tmp_path,
):
    config, model, optimizer, resume, _ = _checkpoint_fixture()
    resume = replace(resume, loader_cursor=FineWebCursor())
    fingerprint = {"distributed": "ddp", "data_kind": "fineweb", "seed": 24}
    path = tmp_path / "checkpoint"
    save_training_checkpoint(
        path, model, config, optimizer, resume, fingerprint
    )

    checkpoint = TrainingCheckpoint.open(
        path,
        expected_fingerprint=fingerprint,
        topology="elastic_ddp",
        current_world_size=2,
    )

    assert checkpoint.resume_state.exact_topology is False
    with pytest.raises(CheckpointCompatibilityError, match="world size"):
        TrainingCheckpoint.open(
            path,
            expected_fingerprint=fingerprint,
            topology="same",
            current_world_size=2,
        )


def test_elastic_resume_rejects_non_fineweb_data(tmp_path):
    config, model, optimizer, resume, _ = _checkpoint_fixture()
    fingerprint = {"distributed": "ddp", "data_kind": "random", "seed": 24}
    path = tmp_path / "checkpoint"
    save_training_checkpoint(
        path, model, config, optimizer, resume, fingerprint
    )

    with pytest.raises(UnsupportedTopologyError, match="FineWeb"):
        TrainingCheckpoint.open(
            path,
            expected_fingerprint=fingerprint,
            topology="elastic_ddp",
            current_world_size=2,
        )


def _legacy_precision_checkpoint(tmp_path, legacy_bf16: bool):
    config, model, optimizer, resume, fingerprint = _checkpoint_fixture()
    path = tmp_path / "checkpoint"
    save_training_checkpoint(
        path,
        model,
        config,
        optimizer,
        resume,
        {**fingerprint, "bf16": legacy_bf16},
    )
    return path, fingerprint


@pytest.mark.parametrize("precision", ["bf16_mixed", "bf16"])
def test_legacy_bf16_fingerprint_resumes_into_either_bf16_precision(
    tmp_path, precision
):
    path, fingerprint = _legacy_precision_checkpoint(tmp_path, legacy_bf16=True)

    with pytest.warns(
        RuntimeWarning, match="predates precision fingerprinting"
    ):
        checkpoint = TrainingCheckpoint.open(
            path,
            expected_fingerprint={**fingerprint, "precision": precision},
            topology="same",
            current_world_size=1,
        )

    assert checkpoint.resume_state.step == 1
    assert checkpoint.manifest["training_fingerprint"]["bf16"] is True


def test_legacy_fp32_fingerprint_resumes_without_warning(tmp_path):
    path, fingerprint = _legacy_precision_checkpoint(
        tmp_path, legacy_bf16=False
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        checkpoint = TrainingCheckpoint.open(
            path,
            expected_fingerprint={**fingerprint, "precision": "fp32"},
            topology="same",
            current_world_size=1,
        )

    assert checkpoint.resume_state.step == 1


@pytest.mark.parametrize(
    ("legacy_bf16", "precision"),
    [(True, "fp32"), (False, "bf16_mixed"), (False, "bf16")],
)
def test_legacy_fingerprint_still_rejects_a_disagreeing_precision(
    tmp_path, legacy_bf16, precision
):
    path, fingerprint = _legacy_precision_checkpoint(
        tmp_path, legacy_bf16=legacy_bf16
    )

    with pytest.raises(CheckpointCompatibilityError) as failure:
        TrainingCheckpoint.open(
            path,
            expected_fingerprint={**fingerprint, "precision": precision},
            topology="same",
            current_world_size=1,
        )

    assert "precision" in str(failure.value)
    assert "bf16," not in str(failure.value)


def test_current_fingerprint_separates_the_two_bf16_precisions(tmp_path):
    config, model, optimizer, resume, fingerprint = _checkpoint_fixture()
    path = tmp_path / "checkpoint"
    save_training_checkpoint(
        path,
        model,
        config,
        optimizer,
        resume,
        {**fingerprint, "precision": "bf16_mixed"},
    )

    with pytest.raises(CheckpointCompatibilityError, match="precision"):
        TrainingCheckpoint.open(
            path,
            expected_fingerprint={**fingerprint, "precision": "bf16"},
            topology="same",
            current_world_size=1,
        )


def test_legacy_model_fingerprint_is_upgraded_to_current_defaults(tmp_path):
    config, model, optimizer, resume, fingerprint = _checkpoint_fixture()
    current_model = to_dict(
        replace(
            tiny_model_config(),
            variant="longshot",
            cache_processor=CacheProcessorConfig(),
        )
    )
    legacy_model = copy.deepcopy(current_model)
    legacy_model["workspace_flow_refinement"] = None
    for field in (
        "generation_process_every",
        "head_selection_method",
        "head_selector_hidden_size",
        "head_selector_use_identity_bias",
    ):
        legacy_model["cache_processor"].pop(field)
    path = tmp_path / "checkpoint"
    save_training_checkpoint(
        path,
        model,
        config,
        optimizer,
        resume,
        {**fingerprint, "model": legacy_model},
    )

    checkpoint = TrainingCheckpoint.open(
        path,
        expected_fingerprint={**fingerprint, "model": current_model},
        topology="same",
        current_world_size=1,
    )

    assert checkpoint.resume_state.step == 1
    assert checkpoint.manifest["training_fingerprint"]["model"] == legacy_model


def test_legacy_non_null_workspace_fingerprint_is_rejected(tmp_path):
    config, model, optimizer, resume, fingerprint = _checkpoint_fixture()
    current_model = to_dict(tiny_model_config())
    legacy_model = {
        **current_model,
        "workspace_flow_refinement": {"solver": "euler"},
    }
    path = tmp_path / "checkpoint"
    save_training_checkpoint(
        path,
        model,
        config,
        optimizer,
        resume,
        {**fingerprint, "model": legacy_model},
    )

    with pytest.raises(
        CheckpointCompatibilityError, match="workspace_flow_refinement"
    ):
        TrainingCheckpoint.open(
            path,
            expected_fingerprint={**fingerprint, "model": current_model},
            topology="same",
            current_world_size=1,
        )


def _legacy_generate_model_config() -> ModelConfig:
    return replace(
        tiny_model_config(),
        variant="longshot",
        cache_processor=CacheProcessorConfig(),
        legacy_bottleneck_generate=True,
    )


def test_model_fingerprint_upgrade_keeps_legacy_bottleneck_generate(tmp_path):
    config, model, optimizer, resume, fingerprint = _checkpoint_fixture()
    current_model = to_dict(_legacy_generate_model_config())
    legacy_model = copy.deepcopy(current_model)
    legacy_model["workspace_flow_refinement"] = None
    for field in (
        "generation_process_every",
        "head_selection_method",
        "head_selector_hidden_size",
        "head_selector_use_identity_bias",
    ):
        legacy_model["cache_processor"].pop(field)
    path = tmp_path / "checkpoint"
    save_training_checkpoint(
        path,
        model,
        config,
        optimizer,
        resume,
        {**fingerprint, "model": legacy_model},
    )

    checkpoint = TrainingCheckpoint.open(
        path,
        expected_fingerprint={**fingerprint, "model": current_model},
        topology="same",
        current_world_size=1,
    )

    assert checkpoint.resume_state.step == 1


def test_current_model_fingerprint_resumes_unchanged(tmp_path):
    config, model, optimizer, resume, fingerprint = _checkpoint_fixture()
    current_model = to_dict(_legacy_generate_model_config())
    path = tmp_path / "checkpoint"
    save_training_checkpoint(
        path,
        model,
        config,
        optimizer,
        resume,
        {**fingerprint, "model": current_model},
    )

    checkpoint = TrainingCheckpoint.open(
        path,
        expected_fingerprint={**fingerprint, "model": current_model},
        topology="same",
        current_world_size=1,
    )

    assert checkpoint.resume_state.step == 1


def test_legacy_upgrade_does_not_touch_a_manifest_that_already_records_precision(
    tmp_path,
):
    config, model, optimizer, resume, fingerprint = _checkpoint_fixture()
    path = tmp_path / "checkpoint"
    saved = {**fingerprint, "precision": "bf16", "bf16": True}
    save_training_checkpoint(path, model, config, optimizer, resume, saved)

    with pytest.raises(CheckpointCompatibilityError, match="bf16"):
        TrainingCheckpoint.open(
            path,
            expected_fingerprint={**fingerprint, "precision": "bf16"},
            topology="same",
            current_world_size=1,
        )


def _truncated_train_config(tmp_path):
    return TrainConfig(
        model=replace(
            tiny_model_config(),
            variant="bgt",
            family="qwen3",
            sequence_len=16,
            cache_processor=CacheProcessorConfig(
                process_every=4,
                topk=2,
                hidden_size=8,
                intermediate_size=16,
                num_layers=1,
                num_heads=1,
                num_kv_heads=1,
                attention_dropout=0.2,
            ),
        ),
        data=DataConfig(kind="random"),
        output_dir=str(tmp_path / "baseline"),
        trainable="all",
        batch_size=1,
        sequence_len=14,
        max_steps=3,
        learning_rate=1e-3,
        warmup_steps=1,
        truncated_bptt_windows=2,
        save_cache_processor_only=False,
        save_final=True,
        eval_every=0,
        eval_on_start=False,
    )


def test_truncated_bptt_fingerprint_binds_horizon_and_rejects_changed_resume(
    tmp_path,
):
    enabled = _truncated_train_config(tmp_path)
    disabled = replace(enabled, truncated_bptt_windows=0)
    changed = replace(enabled, truncated_bptt_windows=3)
    fingerprints = [
        _training_fingerprint(config, 1, 14, 14, 1)
        for config in (disabled, enabled, changed)
    ]
    absent, bound, different = fingerprints
    assert "truncated_bptt_windows" not in absent
    assert bound == absent | {"truncated_bptt_windows": 2}
    assert different == absent | {"truncated_bptt_windows": 3}
    assert different != bound
    config, model, optimizer, resume, _ = _checkpoint_fixture(enabled.model)
    path = tmp_path / "checkpoint"
    save_training_checkpoint(path, model, config, optimizer, resume, bound)
    checkpoint = TrainingCheckpoint.open(
        path, expected_fingerprint=bound, topology="same", current_world_size=1
    )
    assert checkpoint.resume_state.step == 1
    for fingerprint in (different, absent):
        with pytest.raises(
            CheckpointCompatibilityError, match="truncated_bptt_windows"
        ):
            TrainingCheckpoint.open(
                path,
                expected_fingerprint=fingerprint,
                topology="same",
                current_world_size=1,
            )


@pytest.mark.parametrize(
    "checkpoint_windows", [0, 3], ids=["eager", "misaligned-checkpoint"]
)
def test_truncated_bptt_complete_checkpoint_exact_resume_matches_uninterrupted(
    monkeypatch, tmp_path, checkpoint_windows
):
    baseline = replace(
        _truncated_train_config(tmp_path),
        activation_checkpoint_windows=checkpoint_windows,
    )
    # Keep the standard trainer/data/optimizer/resume path, forcing only device
    # selection so this remains a tiny CPU regression on GPU hosts as well.
    monkeypatch.setattr(
        "gcw.training.assembly.init_distributed",
        lambda _mode: torch.device("cpu"),
    )
    train(baseline)
    interrupted = replace(
        baseline, output_dir=str(tmp_path / "resumed"), save_every=1
    )
    path = tmp_path / "resumed" / "checkpoint"

    def save_then_interrupt(*args, **kwargs):
        assert not kwargs.get("cache_processor_only", False)
        save_training_checkpoint(*args, **kwargs)
        if args[4].step == 1:
            raise RuntimeError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(
            "gcw.training.state.save_training_checkpoint", save_then_interrupt
        )
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train(interrupted)

    verified = verify_training_checkpoint_files(path)
    assert verified["manifest"]["step"] == 1
    assert (
        verified["manifest"]["training_fingerprint"]["truncated_bptt_windows"]
        == 2
    )
    saved_weights = load_file(str(path / "model.safetensors"))
    assert (
        saved_weights.keys()
        == build_model(interrupted.model).state_dict().keys()
    )
    assert any(name.startswith("backbone.") for name in saved_weights)
    assert any(name.startswith("cache_processor.") for name in saved_weights)
    # A changed horizon must fail in the production trainer, not merely when
    # manually opening a manifest. The original checkpoint remains resumable.
    with pytest.raises(
        CheckpointCompatibilityError, match="truncated_bptt_windows"
    ):
        train(
            replace(
                interrupted,
                truncated_bptt_windows=3,
                resume=ResumeConfig(checkpoint=str(path)),
            )
        )
    train(replace(interrupted, resume=ResumeConfig(checkpoint=str(path))))

    baseline_path = tmp_path / "baseline" / "checkpoint"
    fingerprint = _training_fingerprint(baseline, 1, 14, 14, 1)
    checkpoints = [
        TrainingCheckpoint.open(
            checkpoint_path,
            expected_fingerprint=fingerprint,
            topology="same",
            current_world_size=1,
        )
        for checkpoint_path in (baseline_path, path)
    ]
    expected, actual = [checkpoint.resume_state for checkpoint in checkpoints]
    assert expected.step == actual.step == 3
    assert expected.smooth_loss == actual.smooth_loss
    assert expected.loader_cursor == actual.loader_cursor
    assert expected.exact_topology is actual.exact_topology is True
    _assert_nested_equal(
        load_file(str(baseline_path / "model.safetensors")),
        load_file(str(path / "model.safetensors")),
    )
    for payload in ("optimizer.pt", "rng.pt"):
        _assert_nested_equal(
            torch.load(
                baseline_path / payload, map_location="cpu", weights_only=False
            ),
            torch.load(path / payload, map_location="cpu", weights_only=False),
        )


def _checkpoint_fixture(config=None):
    config = config or tiny_model_config()
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
    fingerprint = {
        "model": "tiny",
        "distributed": "none",
        "data_kind": "random",
        "seed": 24,
    }
    return config, model, optimizer, resume, fingerprint


def _assert_nested_equal(expected, actual):
    if isinstance(expected, torch.Tensor):
        assert expected.dtype == actual.dtype
        assert torch.equal(expected, actual)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(expected, actual, strict=True)
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
