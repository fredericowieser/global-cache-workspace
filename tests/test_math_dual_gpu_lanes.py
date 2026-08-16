from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_math_experiments_full_single_seed as series
from scripts.math_evaluation_bundle import create_bundle, verify_bundle
from scripts.run_math_evaluation_lane import build_parser as build_eval_parser
from scripts.run_math_training_lane import (
    _reject_performance_overrides,
    build_parser as build_train_parser,
)


def _write_checkpoint(path: Path, *, step: int = 10_902) -> None:
    path.mkdir()
    files = {
        "config.json": b'{"model":{}}\n',
        "model.safetensors": b"fake-weights",
        "train_state.json": json.dumps({"step": step, "dtype": "bfloat16"}).encode(),
    }
    for name, content in files.items():
        (path / name).write_bytes(content)
    manifest = {
        "complete": True,
        "step": step,
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_evaluation_bundle_is_minimal_authenticated_and_atomic(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    bundle = tmp_path / "outbox" / "bt"
    _write_checkpoint(checkpoint)

    manifest = create_bundle(
        checkpoint,
        bundle,
        arm="bt",
        expected_step=10_902,
        repository=Path(__file__).resolve().parents[1],
        training_config=None,
    )

    assert set(path.name for path in bundle.iterdir()) == {
        "READY.json",
        "bundle-manifest.json",
        "config.json",
        "model.safetensors",
        "train_state.json",
    }
    assert manifest == verify_bundle(bundle, expected_arm="bt", expected_step=10_902)
    assert manifest["files"]["model.safetensors"]["sha256"] == hashlib.sha256(b"fake-weights").hexdigest()


def test_completed_full_sft_import_is_digest_bound(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    bundle = tmp_path / "bundle"
    training_config = tmp_path / "full-sft.yaml"
    training_config.write_text("mode: sft\n")
    _write_checkpoint(checkpoint)
    create_bundle(
        checkpoint,
        bundle,
        arm="full-sft",
        expected_step=10_902,
        repository=Path(__file__).resolve().parents[1],
        training_config=training_config,
    )
    generated = verify_bundle(bundle)
    monkeypatch.setattr(
        series,
        "FULL_SFT_IMPORT_IDENTITIES",
        {
            "source_commit": generated["source_commit"],
            "training_config_sha256": generated["training_config_sha256"],
            "source_checkpoint_manifest_sha256": generated["source_checkpoint_manifest_sha256"],
            **{name: value["sha256"] for name, value in generated["files"].items()},
        },
    )

    manifest = series._verify_full_sft_import(bundle)

    assert manifest["step"] == 10_902
    assert manifest["training_config_sha256"] == hashlib.sha256(training_config.read_bytes()).hexdigest()


def test_evaluation_bundle_rejects_tampering(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    bundle = tmp_path / "bundle"
    _write_checkpoint(checkpoint)
    create_bundle(
        checkpoint,
        bundle,
        arm="full-sft",
        expected_step=10_902,
        repository=Path(__file__).resolve().parents[1],
        training_config=None,
    )
    (bundle / "model.safetensors").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="mismatch"):
        verify_bundle(bundle, expected_arm="full-sft", expected_step=10_902)


def test_training_lane_rejects_unqualified_performance_overrides():
    with pytest.raises(RuntimeError, match="GCW_COMPILE"):
        _reject_performance_overrides({"GCW_COMPILE": "1"})
    _reject_performance_overrides({"GCW_COMPILE": ""})


def test_lane_parsers_are_dry_run_by_default():
    eval_args = build_eval_parser().parse_args(
        [
            "--arm",
            "instruct-baseline",
            "--hf-model",
            "model",
            "--tokenizer-model",
            "model",
            "--experiment-root",
            "root",
        ]
    )
    train_args = build_train_parser().parse_args(
        [
            "--arm",
            "bt",
            "--config",
            "bt.yaml",
            "--experiment-root",
            "root",
            "--bundle-outbox",
            "outbox",
            "--stop-after-checkpoint-step",
            "256",
            "--stop-after-checkpoint-utc",
            "2026-08-17T00:00:00Z",
        ]
    )
    assert eval_args.execute is False
    assert eval_args.evaluation_lineage == "lrc-h100-v1"
    assert train_args.execute is False
    assert train_args.stop_after_checkpoint_step == 256
    assert train_args.stop_after_checkpoint_utc == "2026-08-17T00:00:00Z"
