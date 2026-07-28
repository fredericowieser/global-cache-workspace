from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import gcw.flow_math_experiment as flow_math_experiment
import torch
from safetensors.torch import save_file

from gcw.config import load_train_config, to_dict
from gcw.flow_math_experiment import (
    EXPECTED_EVAL_SAMPLE_COUNTS,
    EXPECTED_EVAL_VERSIONS,
    EVALUATION_ARTIFACT_ROLES,
    HENDRYCKS_MATH_SAMPLE_COUNTS,
    REQUIRED_TRAINING_MANIFEST_SHA256,
    REQUIRED_TRAINING_SOURCE_COMMIT,
    build_evaluation_ledger,
    build_evaluation_manifest,
    build_experiment_manifest,
    build_directory_manifest,
    checkpoint_metadata_sha256,
    extract_primary_metrics,
    load_matrix_file,
    materialize_internal_symlinks,
    probe_flow_dynamics,
    summarise_metric_markers,
    verify_completed_flow_math_run,
    verify_cache_payload_copy,
    verify_directory_manifest,
    verify_evaluation_context,
    verify_execution_context,
    verify_required_training_identity,
    verify_run_marker,
    verify_training_audit,
)
from gcw.models import build_model
from scripts.flow_math_experiment import _parser, _validate_dynamics_payload
from scripts.flow_math_eval_runner import build_eval_command, build_runtime_evidence
from scripts.flow_smoke import smoke_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_eval_payload(metrics: dict[str, float]) -> dict:
    results = {
        "gsm8k_openmath": {"exact_match,flexible": metrics["gsm8k_openmath"]},
        "svamp": {"exact_match,flexible": metrics["svamp"]},
        "logiqa": {"acc_norm,none": metrics["logiqa"]},
        "agieval_gaokao_mathqa": {"acc_norm,none": metrics["agieval_gaokao_mathqa"]},
        "gsm8k_hard": {"exact_match,flexible": metrics["gsm8k_hard"]},
    }
    for task in HENDRYCKS_MATH_SAMPLE_COUNTS:
        results[task] = {"exact_match,none": metrics["hendrycks_math"]}
    return {
        "results": results,
        "groups": {"hendrycks_math": {"exact_match,none": metrics["hendrycks_math"]}},
        "n-samples": {
            task: {
                "effective": HENDRYCKS_MATH_SAMPLE_COUNTS[task]
                if task in HENDRYCKS_MATH_SAMPLE_COUNTS
                else EXPECTED_EVAL_SAMPLE_COUNTS[task]
            }
            for task in EXPECTED_EVAL_VERSIONS
        },
        "versions": EXPECTED_EVAL_VERSIONS,
        "n-shot": {task: 0 for task in EXPECTED_EVAL_VERSIONS},
    }


def test_build_experiment_manifest_authenticates_configs_and_external_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    backbone = tmp_path / "model.safetensors"
    dataset = tmp_path / "train.jsonl"
    backbone.write_bytes(b"backbone")
    dataset.write_bytes(b"dataset")

    manifest = build_experiment_manifest(
        repo_root=repo_root,
        source_commit="3ea69336dcce92b2b7db7dd9953a0c60fea25c35",
        artifacts={
            "backbone": (backbone, _sha256(backbone)),
            "dataset": (dataset, _sha256(dataset)),
        },
    )

    assert manifest["experiment_id"] == "flow_math_llama32_1b_v1"
    assert manifest["source_commit"] == "3ea69336dcce92b2b7db7dd9953a0c60fea25c35"
    assert len(manifest["runs"]) == 9
    assert len(manifest["pilots"]) == 4
    assert {(run["variant"], run["seed"]) for run in manifest["runs"]} == {
        (variant, seed)
        for variant in ("bgt", "icarus", "longshot")
        for seed in (24, 25, 26)
    }
    assert all(len(run["config_sha256"]) == 64 for run in manifest["runs"])
    assert manifest["artifacts"]["backbone"]["sha256"] == _sha256(backbone)
    assert manifest["artifacts"]["dataset"]["sha256"] == _sha256(dataset)
    assert manifest["evaluation"]["excluded_tasks"] == ["theoremqa"]
    assert manifest["evaluation"]["sample_counts"] == EXPECTED_EVAL_SAMPLE_COUNTS


def test_evaluation_ledger_binds_cache_pilot_and_exact_ten_markers(tmp_path):
    training_source = REQUIRED_TRAINING_SOURCE_COMMIT
    training_manifest = REQUIRED_TRAINING_MANIFEST_SHA256
    evaluation_source = "c" * 40
    evaluation_manifest = "d" * 64
    cache_sha = "e" * 64

    def write_marker(name, kind, target=None):
        path = tmp_path / name
        payload = {
            "kind": kind,
            "training_source_commit": training_source,
            "training_experiment_manifest_sha256": training_manifest,
            "evaluation_source_commit": evaluation_source,
            "evaluation_manifest_sha256": evaluation_manifest,
            "eval_cache_manifest_sha256": cache_sha,
        }
        if target is not None:
            payload["target"] = target
        path.write_text(json.dumps(payload))
        return path

    cache = write_marker("cache.json", "flow_math_eval_cache")
    pilot = write_marker("pilot.json", "flow_math_eval_pilot")
    targets = ["tf-backbone"] + [
        f"{variant}-seed{seed}"
        for variant in ("bgt", "icarus", "longshot")
        for seed in (24, 25, 26)
    ]
    markers = [write_marker(f"{target}.json", "flow_math_eval_result", target) for target in targets]
    ledger = build_evaluation_ledger(
        cache_marker=cache,
        pilot_marker=pilot,
        metric_markers=markers,
        training_source_commit=training_source,
        training_experiment_manifest_sha256=training_manifest,
        evaluation_source_commit=evaluation_source,
        evaluation_manifest_sha256=evaluation_manifest,
        eval_cache_manifest_sha256=cache_sha,
    )
    assert set(ledger["metric_marker_sha256"]) == set(targets)
    assert ledger["cache_marker_sha256"] == _sha256(cache)
    markers[0].write_text("{}")
    with pytest.raises(ValueError, match="wrong kind"):
        build_evaluation_ledger(
            cache_marker=cache,
            pilot_marker=pilot,
            metric_markers=markers,
            training_source_commit=training_source,
            training_experiment_manifest_sha256=training_manifest,
            evaluation_source_commit=evaluation_source,
            evaluation_manifest_sha256=evaluation_manifest,
            eval_cache_manifest_sha256=cache_sha,
        )


def test_required_training_identity_is_fixed_to_completed_matrix():
    verify_required_training_identity(
        REQUIRED_TRAINING_SOURCE_COMMIT,
        REQUIRED_TRAINING_MANIFEST_SHA256,
    )
    with pytest.raises(ValueError, match="required completed training identity"):
        verify_required_training_identity("f" * 40, REQUIRED_TRAINING_MANIFEST_SHA256)
    with pytest.raises(ValueError, match="required completed training identity"):
        verify_required_training_identity(REQUIRED_TRAINING_SOURCE_COMMIT, "f" * 64)


def test_flow_math_cli_exposes_split_evaluation_manifest_commands(tmp_path):
    parser = _parser()
    build_args = parser.parse_args(
        [
            "build-evaluation-manifest",
            "--evaluation-source-commit",
            "b" * 40,
            "--training-source-commit",
            "a" * 40,
            "--training-experiment-manifest",
            str(tmp_path / "training.json"),
            "--training-experiment-manifest-sha256",
            "c" * 64,
            "--out",
            str(tmp_path / "evaluation.json"),
        ]
    )
    verify_args = parser.parse_args(
        [
            "verify-evaluation-context",
            "--evaluation-manifest",
            str(tmp_path / "evaluation.json"),
            "--evaluation-manifest-sha256",
            "d" * 64,
            "--evaluation-source-commit",
            "b" * 40,
            "--training-experiment-manifest",
            str(tmp_path / "training.json"),
            "--training-experiment-manifest-sha256",
            "c" * 64,
            "--training-source-commit",
            "a" * 40,
        ]
    )

    assert build_args.command == "build-evaluation-manifest"
    assert verify_args.command == "verify-evaluation-context"


def test_build_evaluation_manifest_binds_new_evaluator_to_immutable_training_identity(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    training_source_commit = "a" * 40
    evaluation_source_commit = "b" * 40
    training_manifest = tmp_path / "training-manifest.json"
    training_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "flow_math_llama32_1b_v1",
                "source_commit": training_source_commit,
            },
            sort_keys=True,
        )
    )
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"model")
    monkeypatch.setattr(
        flow_math_experiment,
        "CANONICAL_ARTIFACT_SPECS",
        {
            role: {"path": str(artifact), "sha256": _sha256(artifact), "size_bytes": artifact.stat().st_size}
            for role in EVALUATION_ARTIFACT_ROLES
        },
    )

    manifest = build_evaluation_manifest(
        repo_root=repo_root,
        evaluation_source_commit=evaluation_source_commit,
        training_manifest_path=training_manifest,
        approved_training_manifest_sha256=_sha256(training_manifest),
        training_source_commit=training_source_commit,
        artifacts={role: (artifact, _sha256(artifact)) for role in EVALUATION_ARTIFACT_ROLES},
    )

    assert manifest["manifest_kind"] == "flow_math_evaluation"
    assert manifest["evaluation_source_commit"] == evaluation_source_commit
    assert manifest["training_provenance"] == {
        "source_commit": training_source_commit,
        "experiment_manifest_sha256": _sha256(training_manifest),
    }
    assert manifest["evaluation"]["protocol"]["world_size"] == 1
    assert "scripts/slurm/flow_math_eval_array.sbatch" in manifest["evaluation"]["evaluation_code_sha256"]
    assert "scripts/slurm/flow_math_eval_cache.sbatch" in manifest["evaluation"]["evaluation_code_sha256"]
    assert "scripts/flow_math_eval_runner.py" in manifest["evaluation"]["evaluation_code_sha256"]


def test_build_evaluation_manifest_rejects_wrong_training_manifest_digest(tmp_path):
    training_manifest = tmp_path / "training-manifest.json"
    training_manifest.write_text(json.dumps({"experiment_id": "flow_math_llama32_1b_v1", "source_commit": "a" * 40}))
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"model")

    with pytest.raises(ValueError, match="training manifest differs"):
        build_evaluation_manifest(
            repo_root=Path(__file__).resolve().parents[1],
            evaluation_source_commit="b" * 40,
            training_manifest_path=training_manifest,
            approved_training_manifest_sha256="0" * 64,
            training_source_commit="a" * 40,
            artifacts={role: (artifact, _sha256(artifact)) for role in EVALUATION_ARTIFACT_ROLES},
        )


def test_build_experiment_manifest_rejects_artifact_hash_mismatch(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"actual")
    with pytest.raises(ValueError, match="artifact sha256 mismatch"):
        build_experiment_manifest(
            repo_root=repo_root,
            source_commit="a" * 40,
            artifacts={"backbone": (artifact, "0" * 64)},
        )


def test_verify_evaluation_context_accepts_new_head_with_old_training_manifest(tmp_path, monkeypatch):
    git = shutil.which("git")
    assert git is not None
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([git, "init", "-q"], cwd=repo, check=True)
    subprocess.run([git, "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run([git, "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "uv.lock").write_text("locked\n")
    artifact = repo / "model.safetensors"
    artifact.write_bytes(b"model")
    evaluation_code = repo / "eval.py"
    evaluation_code.write_text("VERSION = 1\n")
    subprocess.run([git, "add", "."], cwd=repo, check=True)
    subprocess.run([git, "commit", "-qm", "training"], cwd=repo, check=True)
    training_source_commit = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    training_manifest_path = tmp_path / "training-manifest.json"
    training_manifest_path.write_text(
        json.dumps({"experiment_id": "flow_math_llama32_1b_v1", "source_commit": training_source_commit})
    )
    training_manifest_sha = _sha256(training_manifest_path)

    evaluation_code.write_text("VERSION = 2\n")
    subprocess.run([git, "add", "eval.py"], cwd=repo, check=True)
    subprocess.run([git, "commit", "-qm", "evaluation"], cwd=repo, check=True)
    evaluation_source_commit = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    evaluation_manifest_path = tmp_path / "evaluation-manifest.json"
    evaluation_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_kind": "flow_math_evaluation",
                "experiment_id": "flow_math_llama32_1b_v1",
                "evaluation_source_commit": evaluation_source_commit,
                "training_provenance": {
                    "source_commit": training_source_commit,
                    "experiment_manifest_sha256": training_manifest_sha,
                },
                "uv_lock_sha256": _sha256(repo / "uv.lock"),
                "artifacts": {
                    "backbone": {
                        "path": str(artifact),
                        "size_bytes": artifact.stat().st_size,
                        "sha256": _sha256(artifact),
                    }
                },
                "evaluation": {
                    "evaluation_code_sha256": {"eval.py": _sha256(evaluation_code)},
                    "protocol": {
                        "num_fewshot": 0,
                        "batch_size_per_rank": 16,
                        "world_size": 1,
                        "limit": None,
                        "offline_datasets": True,
                    },
                    "excluded_tasks": ["theoremqa"],
                },
            },
            sort_keys=True,
        )
    )

    monkeypatch.setattr(flow_math_experiment, "REQUIRED_TRAINING_SOURCE_COMMIT", training_source_commit)
    monkeypatch.setattr(flow_math_experiment, "REQUIRED_TRAINING_MANIFEST_SHA256", training_manifest_sha)
    canonical = json.loads(evaluation_manifest_path.read_text())
    monkeypatch.setattr(flow_math_experiment, "build_evaluation_manifest", lambda **_: canonical)

    verified = verify_evaluation_context(
        repo_root=repo,
        evaluation_manifest_path=evaluation_manifest_path,
        approved_evaluation_manifest_sha256=_sha256(evaluation_manifest_path),
        training_manifest_path=training_manifest_path,
        approved_training_manifest_sha256=training_manifest_sha,
        training_source_commit=training_source_commit,
    )

    assert verified["evaluation_source_commit"] == evaluation_source_commit
    assert verified["training_provenance"]["source_commit"] == training_source_commit

    (repo / "torch.py").write_text("raise RuntimeError('shadow')\n")
    with pytest.raises(ValueError, match="tracked or untracked"):
        verify_evaluation_context(
            repo_root=repo,
            evaluation_manifest_path=evaluation_manifest_path,
            approved_evaluation_manifest_sha256=_sha256(evaluation_manifest_path),
            training_manifest_path=training_manifest_path,
            approved_training_manifest_sha256=training_manifest_sha,
            training_source_commit=training_source_commit,
        )


def test_verify_evaluation_context_rejects_two_process_protocol(tmp_path):
    manifest = tmp_path / "evaluation.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError):
        verify_evaluation_context(
            repo_root=tmp_path,
            evaluation_manifest_path=manifest,
            approved_evaluation_manifest_sha256=_sha256(manifest),
            training_manifest_path=manifest,
            approved_training_manifest_sha256=_sha256(manifest),
            training_source_commit="a" * 40,
        )


def test_verify_execution_context_binds_manifest_git_lock_and_artifacts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    lock = repo / "uv.lock"
    lock.write_text("locked\n")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"approved")
    git = shutil.which("git")
    assert git is not None
    subprocess.run([git, "init", "-q"], cwd=repo, check=True)
    subprocess.run([git, "add", "uv.lock"], cwd=repo, check=True)
    subprocess.run(
        [git, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "init"],
        cwd=repo,
        check=True,
    )
    source_commit = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    manifest = {
        "source_commit": source_commit,
        "uv_lock_sha256": _sha256(lock),
        "artifacts": {
            "artifact": {
                "path": str(artifact),
                "size_bytes": artifact.stat().st_size,
                "sha256": _sha256(artifact),
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    approved_sha = _sha256(manifest_path)

    verified = verify_execution_context(
        repo_root=repo,
        manifest_path=manifest_path,
        approved_manifest_sha256=approved_sha,
    )
    assert verified["source_commit"] == source_commit

    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact"):
        verify_execution_context(
            repo_root=repo,
            manifest_path=manifest_path,
            approved_manifest_sha256=approved_sha,
        )


def test_materialize_internal_cache_symlinks_rejects_external_targets(tmp_path):
    cache = tmp_path / "cache"
    blob = cache / "blobs" / "payload"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"cached")
    snapshot = cache / "snapshots" / "payload"
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)

    records = materialize_internal_symlinks(cache)
    assert not snapshot.is_symlink()
    assert snapshot.read_bytes() == b"cached"
    assert records == [{
        "path": "snapshots/payload",
        "link_target": str(blob),
        "resolved_path": "blobs/payload",
        "size": 6,
        "sha256": hashlib.sha256(b"cached").hexdigest(),
    }]

    outside = tmp_path / "outside"
    outside.write_bytes(b"forbidden")
    bad = cache / "bad-link"
    bad.symlink_to(outside)
    with pytest.raises(ValueError, match="outside cache"):
        materialize_internal_symlinks(cache)
    assert bad.is_symlink()


def test_cache_payload_copy_allows_transient_locks_but_rejects_persistent_changes(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    payload_file = frozen / "dataset.arrow"
    payload_file.write_bytes(b"payload")
    payload_file.chmod(0o444)
    frozen.chmod(0o555)
    manifest = tmp_path / "cache-manifest.json"
    manifest.write_text(json.dumps(build_directory_manifest(frozen, strict=True)))

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    copied = scratch / "dataset.arrow"
    copied.write_bytes(b"payload")
    (scratch / "dataset_builder.lock").write_text("transient")
    assert verify_cache_payload_copy(cache_dir=scratch, manifest_path=manifest)["files"] == 1

    (scratch / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="payload differs"):
        verify_cache_payload_copy(cache_dir=scratch, manifest_path=manifest)


def test_build_directory_manifest_strict_rejects_symlinks_locks_and_mode_changes(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    target = cache / "target.bin"
    target.write_bytes(b"x")
    (cache / "alias.bin").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        build_directory_manifest(cache, strict=True)
    (cache / "alias.bin").unlink()
    lock = cache / "download.lock"
    lock.write_text("lock evidence")
    with pytest.raises(ValueError, match="lock"):
        build_directory_manifest(cache, strict=True)
    lock.unlink()
    target.chmod(0o444)
    cache.chmod(0o555)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(build_directory_manifest(cache, strict=True)))
    digest = _sha256(manifest)
    verify_directory_manifest(
        root=cache, manifest_path=manifest, approved_manifest_sha256=digest,
        require_read_only=True, strict=True,
    )
    target.chmod(0o400)
    with pytest.raises(ValueError, match="contents differ"):
        verify_directory_manifest(
            root=cache, manifest_path=manifest, approved_manifest_sha256=digest,
            require_read_only=True, strict=True,
        )

def test_directory_manifest_rejects_mutated_eval_cache(tmp_path):
    cache = tmp_path / "cache"
    (cache / "dataset").mkdir(parents=True)
    (cache / "dataset/data.arrow").write_bytes(b"rows")
    (cache / "dataset/dataset_info.json").write_text("{}")
    (cache / "transient.lock").write_text("ignored")
    payload = build_directory_manifest(cache)
    manifest = tmp_path / "eval-cache-manifest.json"
    manifest.write_text(json.dumps(payload, sort_keys=True))
    approved_sha = _sha256(manifest)

    verified = verify_directory_manifest(
        root=cache,
        manifest_path=manifest,
        approved_manifest_sha256=approved_sha,
    )
    assert set(verified["files"]) == {"dataset/data.arrow", "dataset/dataset_info.json"}

    (cache / "dataset/data.arrow").write_bytes(b"changed")
    with pytest.raises(ValueError, match="directory contents differ"):
        verify_directory_manifest(
            root=cache,
            manifest_path=manifest,
            approved_manifest_sha256=approved_sha,
        )


def test_flow_math_matrix_tsv_maps_exact_array_indices():
    repo_root = Path(__file__).resolve().parents[1]
    rows = load_matrix_file(repo_root / "configs/train/flow_math_llama32_1b/matrix.tsv")

    assert len(rows) == 9
    assert rows[0] == {
        "index": 0,
        "variant": "bgt",
        "seed": 24,
        "config": "configs/train/flow_math_llama32_1b/llama32_1b_flow_bgt_seed24.yaml",
    }
    assert rows[-1] == {
        "index": 8,
        "variant": "longshot",
        "seed": 26,
        "config": "configs/train/flow_math_llama32_1b/llama32_1b_flow_longshot_seed26.yaml",
    }
    assert [row["index"] for row in rows] == list(range(9))
    assert len({row["config"] for row in rows}) == 9


def test_flow_math_cli_returns_exact_matrix_row():
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/flow_math_experiment.py",
            "matrix-row",
            "--matrix",
            "configs/train/flow_math_llama32_1b/matrix.tsv",
            "--index",
            "4",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == (
        "icarus\t25\tconfigs/train/flow_math_llama32_1b/llama32_1b_flow_icarus_seed25.yaml\n"
    )


def test_extract_primary_metrics_requires_exact_math_suite_without_theoremqa():
    results = {
        "gsm8k_openmath": {"exact_match,flexible": 0.41},
        "svamp": {"exact_match,flexible": 0.42},
        "logiqa": {"acc_norm,none": 0.43},
        "agieval_gaokao_mathqa": {"acc_norm,none": 0.44},
        "gsm8k_hard": {"exact_match,flexible": 0.45},
    }
    math_counts = {
        "hendrycks_math_algebra": 1187,
        "hendrycks_math_counting_and_prob": 474,
        "hendrycks_math_geometry": 479,
        "hendrycks_math_intermediate_algebra": 903,
        "hendrycks_math_num_theory": 540,
        "hendrycks_math_prealgebra": 871,
        "hendrycks_math_precalc": 546,
    }
    for task in math_counts:
        results[task] = {"exact_match,none": 0.4}
    sample_counts = {
        task: {"effective": count}
        for task, count in {
            "gsm8k_openmath": 1319,
            **math_counts,
            "svamp": 300,
            "logiqa": 651,
            "agieval_gaokao_mathqa": 351,
            "gsm8k_hard": 1319,
        }.items()
    }
    payload = {
        "results": results,
        "groups": {"hendrycks_math": {"exact_match,none": 0.40}},
        "n-samples": sample_counts,
        "versions": {
            task: 1.1 if task == "gsm8k_openmath" else 1.0
            for task in results
        },
        "n-shot": {task: 0 for task in results},
    }

    assert extract_primary_metrics(payload) == {
        "gsm8k_openmath": 0.41,
        "hendrycks_math": 0.40,
        "svamp": 0.42,
        "logiqa": 0.43,
        "agieval_gaokao_mathqa": 0.44,
        "gsm8k_hard": 0.45,
    }

    payload["results"]["theoremqa"] = {"exact_match,none": 1.0}
    with pytest.raises(ValueError, match="theoremqa must be absent"):
        extract_primary_metrics(payload)
    del payload["results"]["theoremqa"]

    payload["results"]["unexpected_task"] = {"acc,none": 1.0}
    payload["n-samples"]["unexpected_task"] = {"effective": 1}
    payload["versions"]["unexpected_task"] = 1.0
    payload["n-shot"]["unexpected_task"] = 0
    with pytest.raises(ValueError, match="exactly the approved leaf tasks"):
        extract_primary_metrics(payload)


def test_summarise_metric_markers_reports_seed_mean_sd_and_secondary_macros(tmp_path):
    training_source_commit = "a" * 40
    training_manifest_sha = "b" * 64
    evaluation_source_commit = "c" * 40
    evaluation_manifest_sha = "d" * 64
    cache_sha = "e" * 64

    def marker(target, metrics):
        result = tmp_path / f"{target}-result.json"
        result.write_text(json.dumps(_raw_eval_payload(metrics)))
        checkpoint = tmp_path / target / "model.safetensors"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(target.encode())
        (checkpoint.parent / "config.json").write_text("{}")
        (checkpoint.parent / "train_state.json").write_text("{}")
        return {
            "schema_version": 2,
            "kind": "flow_math_eval_result",
            "target": target,
            "training_source_commit": training_source_commit,
            "training_experiment_manifest_sha256": training_manifest_sha,
            "evaluation_source_commit": evaluation_source_commit,
            "evaluation_manifest_sha256": evaluation_manifest_sha,
            "eval_cache_manifest_sha256": cache_sha,
            "checkpoint": str(checkpoint.parent),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_metadata_sha256": checkpoint_metadata_sha256(checkpoint.parent),
            "result": str(result),
            "result_sha256": _sha256(result),
            "metrics": metrics,
        }

    markers = [marker("tf-backbone", {task: 0.25 for task in EXPECTED_EVAL_SAMPLE_COUNTS})]
    for variant_index, variant in enumerate(("bgt", "icarus", "longshot")):
        for seed in (24, 25, 26):
            score = 0.1 * variant_index + 0.01 * (seed - 24)
            target = f"{variant}-seed{seed}"
            markers.append(marker(target, {task: score for task in EXPECTED_EVAL_SAMPLE_COUNTS}))

    summary = summarise_metric_markers(
        markers,
        training_source_commit=training_source_commit,
        training_experiment_manifest_sha256=training_manifest_sha,
        evaluation_source_commit=evaluation_source_commit,
        evaluation_manifest_sha256=evaluation_manifest_sha,
        eval_cache_manifest_sha256=cache_sha,
    )

    assert summary["schema_version"] == 2
    assert summary["training_provenance"] == {
        "source_commit": training_source_commit,
        "experiment_manifest_sha256": training_manifest_sha,
    }
    assert summary["evaluation_provenance"] == {
        "source_commit": evaluation_source_commit,
        "manifest_sha256": evaluation_manifest_sha,
    }
    assert len(summary["runs"]) == 9
    assert set(summary["runs"][0]["evidence"]) == {
        "result", "result_sha256", "runtime_evidence", "runtime_evidence_sha256",
        "cache_copy_evidence", "cache_copy_evidence_sha256",
        "partial_identity", "partial_identity_sha256", "partial_manifest", "partial_manifest_sha256",
        "checkpoint", "checkpoint_sha256",
        "checkpoint_metadata_sha256", "training_marker", "training_marker_sha256",
    }
    assert summary["baseline"]["target"] == "tf-backbone"
    assert summary["architectures"]["bgt"]["gsm8k_openmath"] == {
        "mean": pytest.approx(0.01),
        "sample_sd": pytest.approx(0.01),
        "n": 3,
    }
    assert summary["architectures"]["icarus"]["historical_macro"]["mean"] == pytest.approx(0.11)
    assert summary["architectures"]["longshot"]["math_only_macro"]["mean"] == pytest.approx(0.21)

    tampered = [dict(item) for item in markers]
    tampered[0]["evaluation_source_commit"] = "f" * 40
    with pytest.raises(ValueError, match="evaluation source commit"):
        summarise_metric_markers(
            tampered,
            training_source_commit=training_source_commit,
            training_experiment_manifest_sha256=training_manifest_sha,
            evaluation_source_commit=evaluation_source_commit,
            evaluation_manifest_sha256=evaluation_manifest_sha,
            eval_cache_manifest_sha256=cache_sha,
        )


def test_extract_metrics_writes_dual_provenance_runtime_bound_marker(tmp_path):
    metrics = {task: 0.25 for task in EXPECTED_EVAL_SAMPLE_COUNTS}
    result = tmp_path / "result.json"
    result.write_text(json.dumps(_raw_eval_payload(metrics)))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"model")
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "train_state.json").write_text("{}")
    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    partial_identity = partial_dir / "identity.json"
    partial_identity.write_text('{"identity": 1}\n')
    partial_identity.chmod(0o444)
    partial_identity.parent.chmod(0o555)
    partial_manifest = tmp_path / "partial-manifest.json"
    partial_manifest.write_text(json.dumps(build_directory_manifest(partial_identity.parent, strict=True)))
    temporary_output = tmp_path / ".result.json.test.tmp.json"
    command = build_eval_command(
        python=Path(sys.executable), checkpoint=checkpoint, batch_size=16, limit=None, out=temporary_output
    )
    runtime_payload = build_runtime_evidence(
        target="bgt-seed24",
        command=command,
        checkpoint=checkpoint,
        output=result,
        temporary_output=temporary_output,
        partial_identity=partial_identity,
        eval_cache_dir=tmp_path / "cache",
        expected_array_task_id=1,
        batch_size=16,
        limit=None,
        cuda_device_count=1,
        environ={
            "CUDA_VISIBLE_DEVICES": "0",
            "WORLD_SIZE": "1", "LOCAL_RANK": "0", "RANK": "0",
            "SLURM_JOB_ID": "123", "SLURM_ARRAY_JOB_ID": "120", "SLURM_ARRAY_TASK_ID": "1",
            "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "16",
            "GCW_STRICT_FLASH": "1", "WANDB_MODE": "offline",
            "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "GCW_EVAL_PARTIAL_DIR": str(partial_dir.resolve()), "GCW_EVAL_SKIP_TASKS": "theoremqa",
            "GCW_EVAL_REFERENCE_GENERATE": "0", "PYTHONPATH": "",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HOME": str((tmp_path / "cache/huggingface").resolve()),
            "HF_DATASETS_CACHE": str((tmp_path / "cache/huggingface/datasets").resolve()),
            "HF_HUB_CACHE": str((tmp_path / "cache/huggingface/hub").resolve()),
        },
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(runtime_payload))
    training_marker = tmp_path / "train.json"
    training_marker.write_text("{}")
    cache_copy_evidence = tmp_path / "cache-copy.json"
    cache_copy_evidence.write_text(json.dumps({
        "schema_version": 1,
        "kind": "flow_math_eval_cache_copy",
        "target": "bgt-seed24",
        "eval_cache_manifest_sha256": "e" * 64,
        "files": 1,
        "content_sha256": "f" * 64,
        "post_evaluation_verified": True,
    }))
    out = tmp_path / "marker.json"
    args = _parser().parse_args(
        [
            "extract-metrics",
            "--result", str(result),
            "--target", "bgt-seed24",
            "--training-source-commit", "a" * 40,
            "--training-experiment-manifest-sha256", "b" * 64,
            "--evaluation-source-commit", "c" * 40,
            "--evaluation-manifest-sha256", "d" * 64,
            "--eval-cache-manifest-sha256", "e" * 64,
            "--eval-cache-dir", str(tmp_path / "cache"),
            "--array-task-id", "1",
            "--cache-copy-evidence", str(cache_copy_evidence),
            "--checkpoint", str(checkpoint),
            "--checkpoint-sha256", _sha256(model),
            "--checkpoint-metadata-sha256", checkpoint_metadata_sha256(checkpoint),
            "--runtime-evidence", str(runtime),
            "--partial-identity", str(partial_identity),
            "--partial-manifest", str(partial_manifest),
            "--training-marker", str(training_marker),
            "--out", str(out),
        ]
    )
    args.func(args)
    marker = json.loads(out.read_text())

    assert marker["schema_version"] == 2
    assert marker["kind"] == "flow_math_eval_result"
    assert marker["training_source_commit"] == "a" * 40
    assert marker["training_experiment_manifest_sha256"] == "b" * 64
    assert marker["evaluation_source_commit"] == "c" * 40
    assert marker["evaluation_manifest_sha256"] == "d" * 64
    assert marker["runtime_evidence_sha256"] == _sha256(runtime)
    assert marker["partial_identity_sha256"] == _sha256(partial_identity)
    assert marker["training_marker_sha256"] == _sha256(training_marker)


def test_probe_flow_dynamics_detects_nontrivial_solver_behavior():
    torch.manual_seed(1234)
    config = smoke_config("bgt", flow_steps=2)
    model = build_model(config)
    model.init_weights()
    input_ids = torch.randint(0, config.vocab_size, (1, config.sequence_len))

    result = probe_flow_dynamics(model, input_ids)

    assert result["flow_modules"] == 1
    assert result["solver_steps"] == [1, 2]
    assert result["max_abs_logit_diff"] > 0.0
    assert result["max_flow_displacement_l2"] > 0.0
    assert len(result["flow_gate_values"]) == 1
    assert all(torch.isfinite(torch.tensor(value)) for value in result["flow_gate_values"])


def test_verify_completed_flow_math_run_authenticates_semantics_and_frozen_backbone(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs/train/flow_math_llama32_1b/llama32_1b_flow_bgt_seed24.yaml"
    source_backbone = tmp_path / "source.safetensors"
    source_tensor = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    save_file({"backbone.weight": source_tensor}, source_backbone)
    checkpoint = _write_completed_checkpoint(tmp_path, config_path, source_tensor)

    verified = verify_completed_flow_math_run(
        config_path=config_path,
        checkpoint_path=checkpoint,
        backbone_model_path=source_backbone,
    )

    assert verified["variant"] == "bgt"
    assert verified["seed"] == 24
    assert verified["step"] == 1000
    assert verified["saved_world_size"] == 1
    assert verified["loader_cursor"] == {"epoch": 1, "row": 0, "kind": "sft"}
    assert verified["backbone_tensors"] == 1


def test_verify_completed_flow_math_run_rejects_changed_backbone_tensor(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs/train/flow_math_llama32_1b/llama32_1b_flow_bgt_seed24.yaml"
    source_backbone = tmp_path / "source.safetensors"
    source_tensor = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    save_file({"backbone.weight": source_tensor}, source_backbone)
    checkpoint = _write_completed_checkpoint(tmp_path, config_path, torch.tensor([[1.0, -1.0]]))

    with pytest.raises(ValueError, match="frozen backbone tensor differs"):
        verify_completed_flow_math_run(
            config_path=config_path,
            checkpoint_path=checkpoint,
            backbone_model_path=source_backbone,
        )


def test_verify_completed_flow_math_run_rejects_altered_optimizer_fingerprint(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs/train/flow_math_llama32_1b/llama32_1b_flow_bgt_seed24.yaml"
    source_backbone = tmp_path / "source.safetensors"
    source_tensor = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    save_file({"backbone.weight": source_tensor}, source_backbone)
    checkpoint = _write_completed_checkpoint(tmp_path, config_path, source_tensor)
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training_fingerprint"]["optimizer"]["learning_rate"] = 99.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="training fingerprint"):
        verify_completed_flow_math_run(
            config_path=config_path,
            checkpoint_path=checkpoint,
            backbone_model_path=source_backbone,
        )


def test_verify_run_marker_reauthenticates_live_checkpoint(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs/train/flow_math_llama32_1b/llama32_1b_flow_bgt_seed24.yaml"
    source_backbone = tmp_path / "source.safetensors"
    source_tensor = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    save_file({"backbone.weight": source_tensor}, source_backbone)
    checkpoint = _write_completed_checkpoint(tmp_path, config_path, source_tensor)
    source_commit = "a" * 40
    manifest_sha = "b" * 64
    payload = verify_completed_flow_math_run(
        config_path=config_path,
        checkpoint_path=checkpoint,
        backbone_model_path=source_backbone,
    )
    payload.update({"source_commit": source_commit, "experiment_manifest_sha256": manifest_sha})
    marker = tmp_path / "marker.json"
    marker.write_text(json.dumps(payload))
    protected = (checkpoint.parent, checkpoint, *checkpoint.rglob("*"))
    for path in protected:
        path.chmod(0o555 if path.is_dir() else 0o444)

    verified = verify_run_marker(
        marker_path=marker,
        config_path=config_path,
        checkpoint_path=checkpoint,
        backbone_model_path=source_backbone,
        source_commit=source_commit,
        experiment_manifest_sha256=manifest_sha,
    )
    assert verified["checkpoint_model_sha256"] == payload["checkpoint_model_sha256"]

    payload["source_commit"] = "c" * 40
    marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="marker"):
        verify_run_marker(
            marker_path=marker,
            config_path=config_path,
            checkpoint_path=checkpoint,
            backbone_model_path=source_backbone,
            source_commit=source_commit,
            experiment_manifest_sha256=manifest_sha,
        )
    for path in protected:
        path.chmod(0o755 if path.is_dir() else 0o644)


def test_dynamics_validator_rejects_nonfinite_values(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    identity = {"checkpoint_model_sha256": "a" * 64, "checkpoint_manifest_sha256": "b" * 64}
    payload = {
        "source_commit": "c" * 40,
        "experiment_manifest_sha256": "d" * 64,
        "checkpoint": str(checkpoint.resolve()),
        **identity,
        "max_abs_logit_diff": float("nan"),
        "max_flow_displacement_l2": 1.0,
        "flow_gate_delta_from_init": [1.0],
    }
    with pytest.raises(ValueError, match="finite"):
        _validate_dynamics_payload(
            payload,
            checkpoint=checkpoint,
            checkpoint_payload=identity,
            source_commit="c" * 40,
            experiment_manifest_sha256="d" * 64,
        )


def test_verify_training_audit_rejects_backbone_gradients(tmp_path):
    source_commit = "a" * 40
    manifest_sha = "b" * 64
    audit_path = tmp_path / "audit.json"
    audit = {
        "architecture": "flow_bgt",
        "step": 2,
        "processor_grad_nonzero": True,
        "flow_gate_grad_nonzero": True,
        "backbone_grad_tensors": 0,
        "processor_grad_tensors": 4,
        "flow_gate_grad_tensors": 1,
        "nonfinite_grad_tensors": 0,
        "tokens_per_second": 100.0,
        "peak_memory_mib": 42.0,
        "source_commit": source_commit,
        "experiment_manifest_sha256": manifest_sha,
    }
    audit_path.write_text(json.dumps(audit))
    verified = verify_training_audit(
        audit_path,
        architecture="flow_bgt",
        max_steps=2,
        source_commit=source_commit,
        experiment_manifest_sha256=manifest_sha,
    )
    assert verified["sha256"] == _sha256(audit_path)

    audit["backbone_grad_tensors"] = 1
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="backbone gradients"):
        verify_training_audit(
            audit_path,
            architecture="flow_bgt",
            max_steps=2,
            source_commit=source_commit,
            experiment_manifest_sha256=manifest_sha,
        )

    audit["backbone_grad_tensors"] = 0
    audit["processor_grad_tensors"] = True
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="counts"):
        verify_training_audit(
            audit_path,
            architecture="flow_bgt",
            max_steps=2,
            source_commit=source_commit,
            experiment_manifest_sha256=manifest_sha,
        )


def _write_completed_checkpoint(tmp_path: Path, config_path: Path, backbone_tensor: torch.Tensor) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = load_train_config(config_path)
    save_file(
        {
            "backbone.weight": backbone_tensor,
            "cache_processor.weight": torch.tensor([3.0]),
        },
        checkpoint / "model.safetensors",
    )
    torch.save(
        {"state": {0: {"exp_avg": torch.tensor([0.5]), "exp_avg_sq": torch.tensor([0.25])}}},
        checkpoint / "optimizer.pt",
    )
    torch.save({"cpu": torch.get_rng_state()}, checkpoint / "rng.pt")
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    train_state = {
        "step": 1000,
        "smooth_loss": 1.0,
        "loader_cursor": {"epoch": 1, "row": 0, "kind": "sft"},
        "rank_loader_cursors": [{"epoch": 1, "row": 0, "kind": "sft"}],
        "saved_world_size": 1,
        "exact_topology": True,
    }
    (checkpoint / "train_state.json").write_text(json.dumps(train_state), encoding="utf-8")
    assert config.data.dataset is not None
    fingerprint = {
        "model": to_dict(config.model),
        "data": to_dict(config.data),
        "data_identity": {
            "kind": "sft",
            "path": str(Path(config.data.dataset).resolve()),
            "size": 1_392_578_835,
        },
        "data_kind": config.data.kind,
        "mode": config.mode,
        "distributed": config.distributed,
        "freeze_backbone": config.freeze_backbone,
        "loss": config.loss,
        "checkpoint_format": config.checkpoint_format,
        "bf16": config.bf16,
        "seed": config.seed,
        "optimizer": {
            "learning_rate": config.learning_rate,
            "adam_beta1": config.adam_beta1,
            "adam_beta2": config.adam_beta2,
            "adam_eps": config.adam_eps,
            "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip,
        },
        "schedule": {
            "max_steps": config.max_steps,
            "warmup_steps": config.warmup_steps,
            "warmdown_ratio": config.warmdown_ratio,
            "final_lr_frac": config.final_lr_frac,
        },
        "batch": {
            "sequence_len": config.sequence_len,
            "micro_tokens": config.batch_size * config.sequence_len,
            "tokens_per_step": config.total_batch_size,
            "grad_accum_steps": config.total_batch_size // (config.batch_size * config.sequence_len),
        },
    }
    files = {
        name: _sha256(checkpoint / name)
        for name in ("model.safetensors", "optimizer.pt", "rng.pt", "config.json", "train_state.json")
    }
    manifest = {
        "format_version": 2,
        "complete": True,
        "step": 1000,
        "saved_world_size": 1,
        "topology": "ddp",
        "training_fingerprint": fingerprint,
        "files": files,
    }
    (checkpoint / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint
