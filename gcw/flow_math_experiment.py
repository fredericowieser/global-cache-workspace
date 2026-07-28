from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
from typing import Any

from safetensors import safe_open
import torch

from gcw.config import load_train_config, to_dict
from gcw.models import LatentFlow, set_flow_solver_steps
from gcw.training_checkpoint import verify_training_checkpoint_files


EXPERIMENT_ID = "flow_math_llama32_1b_v1"
REQUIRED_TRAINING_SOURCE_COMMIT = "a623bca1fb5c0cedfb5b8b945b5c0b6ccde7efb5"
REQUIRED_TRAINING_MANIFEST_SHA256 = "c86000c60ed445fffd5d05a19445371957389fe93a344a3da00589dd4596b6c8"
EVALUATION_ARTIFACT_ROLES = (
    "backbone_model",
    "baseline_config",
    "baseline_train_state",
    "tokenizer_json",
    "tokenizer_config",
    "tokenizer_model_config",
)
CONFIG_DIR = Path("configs/train/flow_math_llama32_1b")
VARIANTS = ("bgt", "icarus", "longshot")
SEEDS = (24, 25, 26)
EXPECTED_EVAL_SAMPLE_COUNTS = {
    "gsm8k_openmath": 1319,
    "hendrycks_math": 5000,
    "svamp": 300,
    "logiqa": 651,
    "agieval_gaokao_mathqa": 351,
    "gsm8k_hard": 1319,
}
HENDRYCKS_MATH_SAMPLE_COUNTS = {
    "hendrycks_math_algebra": 1187,
    "hendrycks_math_counting_and_prob": 474,
    "hendrycks_math_geometry": 479,
    "hendrycks_math_intermediate_algebra": 903,
    "hendrycks_math_num_theory": 540,
    "hendrycks_math_prealgebra": 871,
    "hendrycks_math_precalc": 546,
}
PRIMARY_METRICS = {
    "gsm8k_openmath": "exact_match,flexible",
    "hendrycks_math": "exact_match,none",
    "svamp": "exact_match,flexible",
    "logiqa": "acc_norm,none",
    "agieval_gaokao_mathqa": "acc_norm,none",
    "gsm8k_hard": "exact_match,flexible",
}
EXPECTED_EVAL_LEAF_TASKS = {
    "gsm8k_openmath",
    "svamp",
    "logiqa",
    "agieval_gaokao_mathqa",
    "gsm8k_hard",
    *HENDRYCKS_MATH_SAMPLE_COUNTS,
}
EXPECTED_EVAL_VERSIONS = {
    task: 1.1 if task == "gsm8k_openmath" else 1.0 for task in EXPECTED_EVAL_LEAF_TASKS
}
DATASET_SIZE_BYTES = 1_392_578_835
BASELINE_DIR = Path("/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf")
BACKBONE_PATH = BASELINE_DIR / "model.safetensors"
DATASET_PATH = Path("/mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl")
TOKENIZER_DIR = Path("/mnt/jfs-01/shared/bottleneck_llm/sft/llama-3.2-1b/final_model")
CANONICAL_ARTIFACT_SPECS = {
    "backbone_model": {
        "path": str(BACKBONE_PATH),
        "size": 2_471_644_704,
        "sha256": "6944fef6b2a7e0a5279f6a0147d49152388239406d05b3902d220bd5d714cb26",
    },
    "baseline_config": {
        "path": str(BASELINE_DIR / "config.json"),
        "size": 459,
        "sha256": "bc78736bf8ea4b3b3ff13e128110fc2d3af87f96bd5a957df9e445b99e699aae",
    },
    "baseline_train_state": {
        "path": str(BASELINE_DIR / "train_state.json"),
        "size": 149,
        "sha256": "81da63d661db6c599d3ac101939f4fba0e5d2f75352d94f8edbc3f4c78c686dc",
    },
    "dataset": {
        "path": str(DATASET_PATH),
        "size": DATASET_SIZE_BYTES,
        "sha256": "576f26f134bb20d326c7812a44347a81699c4698dfaeacca1bea3a352abc696a",
    },
    "tokenizer_json": {
        "path": str(TOKENIZER_DIR / "tokenizer.json"),
        "size": 17_209_920,
        "sha256": "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b",
    },
    "tokenizer_config": {
        "path": str(TOKENIZER_DIR / "tokenizer_config.json"),
        "size": 335,
        "sha256": "e7ed4d8b6cd6bb8d53a6c89d174991b5eee950a749f8dff956e5bf673163dcd3",
    },
    "tokenizer_model_config": {
        "path": str(TOKENIZER_DIR / "config.json"),
        "size": 861,
        "sha256": "47dacf3f6fdc73e383fcfd8adbea6a36be0b5f8e06126061cc58da15ddf55889",
    },
}


def load_json_strict(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def checkpoint_metadata_sha256(path: Path) -> str:
    manifest = path / "manifest.json"
    if manifest.is_file():
        from gcw.training_checkpoint import verify_training_checkpoint_files

        verify_training_checkpoint_files(path)
        return sha256_file(manifest)
    metadata_files = ("config.json", "train_state.json")
    payload = {}
    for name in metadata_files:
        candidate = path / name
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        payload[name] = sha256_file(candidate)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_execution_context(
    *,
    repo_root: Path,
    manifest_path: Path,
    approved_manifest_sha256: str,
    required_artifacts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", approved_manifest_sha256):
        raise ValueError("approved manifest sha256 must be 64 lowercase hexadecimal characters")
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != approved_manifest_sha256:
        raise ValueError("experiment manifest differs from the independently approved digest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("experiment manifest must be a JSON object")

    git_binary = shutil.which("git")
    if git_binary is None and Path("/run/current-system/sw/bin/git").is_file():
        git_binary = "/run/current-system/sw/bin/git"
    if git_binary is None:
        raise ValueError("git executable is unavailable")

    def git(*args: str) -> str:
        completed = subprocess.run(
            [git_binary, *args],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
        return completed.stdout.strip()

    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or git("rev-parse", "HEAD") != source_commit:
        raise ValueError("checked-out Git HEAD differs from the experiment manifest")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("tracked Git worktree or index is dirty")
    expected_lock_sha = manifest.get("uv_lock_sha256")
    if not isinstance(expected_lock_sha, str) or sha256_file(repo_root / "uv.lock") != expected_lock_sha:
        raise ValueError("uv.lock differs from the experiment manifest")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("experiment manifest has no authenticated artifacts")
    for name, entry in artifacts.items():
        if not isinstance(entry, dict):
            raise ValueError(f"artifact manifest entry is invalid: {name}")
        path = Path(entry.get("path", ""))
        if not path.is_file():
            raise ValueError(f"artifact is missing: {name}: {path}")
        if path.stat().st_size != entry.get("size_bytes") or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"artifact differs from manifest: {name}: {path}")
    if required_artifacts is not None:
        if set(artifacts) != set(required_artifacts):
            raise ValueError("manifest artifact roles differ from the canonical runtime artifact set")
        for name, spec in required_artifacts.items():
            entry = artifacts[name]
            if (
                Path(entry["path"]).resolve() != Path(spec["path"]).resolve()
                or entry.get("size_bytes") != spec["size"]
                or entry.get("sha256") != spec["sha256"]
            ):
                raise ValueError(f"manifest artifact role is not canonical: {name}")
        train_state = json.loads(Path(required_artifacts["baseline_train_state"]["path"]).read_text())
        if (
            train_state.get("source") != str(TOKENIZER_DIR)
            or train_state.get("dtype") != "bfloat16"
            or train_state.get("family") != "llama"
        ):
            raise ValueError("baseline train_state runtime policy is not canonical")
        for section in ("runs", "pilots"):
            entries = manifest.get(section)
            if not isinstance(entries, list):
                raise ValueError(f"manifest {section} section is invalid")
            for run in entries:
                config_path = (repo_root / run["config"]).resolve()
                config = load_train_config(config_path)
                if (
                    Path(config.data.dataset or "").resolve() != DATASET_PATH.resolve()
                    or Path(config.backbone_checkpoint or "").resolve() != BASELINE_DIR.resolve()
                    or Path(config.data.tokenizer_source or "").resolve() != TOKENIZER_DIR.resolve()
                ):
                    raise ValueError(f"config runtime artifacts are not canonical: {config_path}")
    return manifest


def materialize_internal_symlinks(cache_dir: Path) -> list[dict[str, Any]]:
    root = cache_dir.resolve(strict=True)
    planned: list[tuple[Path, Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_symlink():
            continue
        try:
            resolved = path.resolve(strict=True)
            resolved_relative = resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"evaluation cache symlink resolves outside cache: {path}") from exc
        if not resolved.is_file() or resolved.is_symlink():
            raise ValueError(f"evaluation cache symlink target is not a regular file: {path}")
        record = {
            "path": path.relative_to(root).as_posix(),
            "link_target": os.readlink(path),
            "resolved_path": resolved_relative.as_posix(),
            "size": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
        planned.append((path, resolved, record))

    for path, resolved, _record in planned:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".materializing",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as destination, resolved.open("rb") as source:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            os.chmod(temporary, resolved.stat().st_mode & 0o7777)
            os.replace(temporary, path)
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)
    return [record for _path, _resolved, record in planned]


def build_directory_manifest(root: Path, *, strict: bool = False) -> dict[str, Any]:
    if strict and root.is_symlink():
        raise ValueError(f"strict directory manifest rejects symlink root: {root}")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"manifest root is not a directory: {root}")
    files: dict[str, dict[str, Any]] = {}
    directories: dict[str, str] = {".": f"{root.stat().st_mode & 0o777:04o}"}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        is_lock = path.suffix == ".lock" or ".locks" in relative.parts
        if strict and path.is_symlink():
            raise ValueError(f"strict directory manifest rejects symlink: {relative}")
        if strict and is_lock:
            raise ValueError(f"strict directory manifest rejects lock artefact: {relative}")
        if path.is_dir():
            if strict:
                directories[relative.as_posix()] = f"{path.stat().st_mode & 0o777:04o}"
            continue
        if not path.is_file():
            if strict:
                raise ValueError(f"strict directory manifest rejects non-regular entry: {relative}")
            continue
        if is_lock and not strict:
            continue
        entry: dict[str, Any] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if strict:
            entry["mode"] = f"{path.stat().st_mode & 0o777:04o}"
        files[relative.as_posix()] = entry
    if not files:
        raise ValueError(f"manifest root contains no persistent files: {root}")
    payload: dict[str, Any] = {"schema_version": 1, "root": str(root), "files": files}
    if strict:
        payload["directories"] = directories
    return payload

def verify_cache_payload_copy(*, cache_dir: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = load_json_strict(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError("frozen cache manifest is invalid")
    expected = {
        str(relative): {
            "size_bytes": entry["size_bytes"],
            "sha256": entry["sha256"],
        }
        for relative, entry in manifest["files"].items()
        if isinstance(entry, dict)
    }
    actual: dict[str, dict[str, Any]] = {}
    root = cache_dir.resolve()
    if not root.is_dir() or cache_dir.is_symlink():
        raise ValueError("cache payload copy root is invalid")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        is_lock = path.suffix == ".lock" or ".locks" in relative.parts
        if path.is_symlink():
            raise ValueError(f"cache payload copy contains symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"cache payload copy contains special entry: {relative}")
        if is_lock:
            continue
        actual[relative.as_posix()] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if actual != expected:
        raise ValueError("cache payload differs from frozen authenticated contents")
    content_sha256 = hashlib.sha256(
        json.dumps(actual, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"files": len(actual), "content_sha256": content_sha256}


def verify_directory_manifest(
    *,
    root: Path,
    manifest_path: Path,
    approved_manifest_sha256: str,
    require_read_only: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    if sha256_file(manifest_path) != approved_manifest_sha256:
        raise ValueError("directory manifest differs from the approved digest")
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = build_directory_manifest(root, strict=strict)
    if expected != actual:
        raise ValueError("directory contents differ from the authenticated manifest")
    if require_read_only:
        writable = [path for path in (root, *root.rglob("*")) if path.stat().st_mode & 0o222]
        if writable:
            raise ValueError(f"authenticated directory is writable: {writable[0]}")
    return expected


def matrix_config_paths(repo_root: Path) -> list[Path]:
    return [
        repo_root / CONFIG_DIR / f"llama32_1b_flow_{variant}_seed{seed}.yaml"
        for variant in VARIANTS
        for seed in SEEDS
    ]


def load_matrix_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"matrix line {line_number} must have four tab-separated fields")
        index_text, variant, seed_text, config_path = fields
        try:
            index = int(index_text)
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError(f"matrix line {line_number} has a non-integer index or seed") from exc
        rows.append({"index": index, "variant": variant, "seed": seed, "config": config_path})

    expected_pairs = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    if [row["index"] for row in rows] != list(range(len(expected_pairs))):
        raise ValueError("matrix indices must be contiguous from zero")
    if {(row["variant"], row["seed"]) for row in rows} != expected_pairs:
        raise ValueError("matrix must contain each approved variant/seed pair exactly once")
    if len({row["config"] for row in rows}) != len(rows):
        raise ValueError("matrix config paths must be unique")
    for row in rows:
        expected_name = f"llama32_1b_flow_{row['variant']}_seed{row['seed']}.yaml"
        if Path(row["config"]).name != expected_name:
            raise ValueError(f"matrix config does not match its variant/seed identity: {row['config']}")
    return rows


def verify_required_training_identity(source_commit: str, manifest_sha256: str) -> None:
    if (
        source_commit != REQUIRED_TRAINING_SOURCE_COMMIT
        or manifest_sha256 != REQUIRED_TRAINING_MANIFEST_SHA256
    ):
        raise ValueError("training provenance differs from the required completed training identity")


def verify_evaluation_context(
    *,
    repo_root: Path,
    evaluation_manifest_path: Path,
    approved_evaluation_manifest_sha256: str,
    training_manifest_path: Path,
    approved_training_manifest_sha256: str,
    training_source_commit: str,
) -> dict[str, Any]:
    verify_required_training_identity(training_source_commit, approved_training_manifest_sha256)
    for label, digest in (
        ("evaluation manifest", approved_evaluation_manifest_sha256),
        ("training manifest", approved_training_manifest_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"approved {label} sha256 must be lowercase hexadecimal")
    if re.fullmatch(r"[0-9a-f]{40}", training_source_commit) is None:
        raise ValueError("training_source_commit must be a full lowercase Git SHA")
    if sha256_file(evaluation_manifest_path) != approved_evaluation_manifest_sha256:
        raise ValueError("evaluation manifest differs from the independently approved digest")
    if sha256_file(training_manifest_path) != approved_training_manifest_sha256:
        raise ValueError("training manifest differs from the independently approved digest")
    evaluation_manifest = load_json_strict(evaluation_manifest_path)
    training_manifest = load_json_strict(training_manifest_path)
    if not isinstance(evaluation_manifest, dict) or not isinstance(training_manifest, dict):
        raise ValueError("evaluation and training manifests must be JSON objects")
    if training_manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("training manifest experiment identity differs")
    if training_manifest.get("source_commit") != training_source_commit:
        raise ValueError("training manifest source commit differs")

    git_binary = shutil.which("git")
    if git_binary is None:
        fallback = Path("/run/current-system/sw/bin/git")
        if fallback.is_file():
            git_binary = str(fallback)
        else:
            raise ValueError("git executable is unavailable")

    def git(*args: str) -> str:
        completed = subprocess.run(
            [git_binary, *args], cwd=repo_root, check=True, text=True, capture_output=True
        )
        return completed.stdout.strip()

    evaluation_source_commit = evaluation_manifest.get("evaluation_source_commit")
    if not isinstance(evaluation_source_commit, str) or git("rev-parse", "HEAD") != evaluation_source_commit:
        raise ValueError("checked-out Git HEAD differs from the evaluation manifest")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Git worktree contains tracked or untracked changes")
    subprocess.run(
        [git_binary, "cat-file", "-e", f"{training_source_commit}^{{commit}}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    artifacts_payload = evaluation_manifest.get("artifacts")
    if not isinstance(artifacts_payload, dict):
        raise ValueError("evaluation manifest artifacts must be an object")
    artifacts: dict[str, tuple[Path, str]] = {}
    for name, entry in artifacts_payload.items():
        if not isinstance(entry, dict):
            raise ValueError(f"evaluation artifact entry is invalid: {name}")
        artifacts[str(name)] = (Path(str(entry.get("path", ""))), str(entry.get("sha256", "")))
    expected = build_evaluation_manifest(
        repo_root=repo_root,
        evaluation_source_commit=evaluation_source_commit,
        training_manifest_path=training_manifest_path,
        approved_training_manifest_sha256=approved_training_manifest_sha256,
        training_source_commit=training_source_commit,
        artifacts=artifacts,
    )
    if evaluation_manifest != expected:
        raise ValueError("evaluation manifest differs from the exact canonical evaluation manifest")
    return evaluation_manifest


def build_evaluation_manifest(
    *,
    repo_root: Path,
    evaluation_source_commit: str,
    training_manifest_path: Path,
    approved_training_manifest_sha256: str,
    training_source_commit: str,
    artifacts: dict[str, tuple[Path, str]],
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", evaluation_source_commit) is None:
        raise ValueError("evaluation_source_commit must be a full lowercase Git SHA")
    if re.fullmatch(r"[0-9a-f]{40}", training_source_commit) is None:
        raise ValueError("training_source_commit must be a full lowercase Git SHA")
    if re.fullmatch(r"[0-9a-f]{64}", approved_training_manifest_sha256) is None:
        raise ValueError("approved training manifest sha256 must be lowercase hexadecimal")
    if sha256_file(training_manifest_path) != approved_training_manifest_sha256:
        raise ValueError("training manifest differs from the independently approved digest")
    training_manifest = load_json_strict(training_manifest_path)
    if not isinstance(training_manifest, dict):
        raise ValueError("training manifest must be a JSON object")
    if training_manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("training manifest experiment identity differs")
    if training_manifest.get("source_commit") != training_source_commit:
        raise ValueError("training manifest source commit differs")

    if set(artifacts) != set(EVALUATION_ARTIFACT_ROLES):
        raise ValueError("evaluation artifact roles differ from the canonical role set")
    for name in EVALUATION_ARTIFACT_ROLES:
        path, expected_sha256 = artifacts[name]
        specification = CANONICAL_ARTIFACT_SPECS[name]
        if path.resolve() != Path(str(specification["path"])).resolve() or expected_sha256 != specification["sha256"]:
            raise ValueError(f"evaluation artifact differs from canonical specification: {name}")
    authenticated_artifacts: dict[str, dict[str, Any]] = {}
    for name, (path, expected_sha256) in sorted(artifacts.items()):
        if not path.is_file():
            raise ValueError(f"artifact is not a file: {name}: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"artifact sha256 mismatch for {name}: expected {expected_sha256}, got {actual_sha256}")
        authenticated_artifacts[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": actual_sha256,
        }

    evaluation_files = sorted(
        {
            repo_root / "configs/eval/benchmarks.yaml",
            *(repo_root / "configs/tasks" / f"{task}.yaml" for task in ("gsm8k_openmath", "svamp", "logiqa", "gsm8k_hard")),
            *(repo_root / "configs/tasks").glob("*.py"),
            *(repo_root / "gcw").rglob("*.py"),
            repo_root / "scripts/__init__.py",
            repo_root / "scripts/flow_math_eval_runner.py",
            repo_root / "scripts/flow_math_experiment.py",
            repo_root / "scripts/submit_flow_math.sh",
            repo_root / "scripts/slurm/flow_math_eval_array.sbatch",
            repo_root / "scripts/slurm/flow_math_eval_cache.sbatch",
            repo_root / "scripts/slurm/flow_math_eval_pilot.sbatch",
        }
    )
    missing = [path for path in evaluation_files if not path.is_file()]
    if missing:
        raise ValueError(f"missing evaluation code file: {missing[0]}")
    evaluation_code_sha256 = {
        str(path.relative_to(repo_root)): sha256_file(path) for path in evaluation_files
    }
    return {
        "schema_version": 1,
        "manifest_kind": "flow_math_evaluation",
        "experiment_id": EXPERIMENT_ID,
        "evaluation_source_commit": evaluation_source_commit,
        "training_provenance": {
            "source_commit": training_source_commit,
            "experiment_manifest_sha256": approved_training_manifest_sha256,
        },
        "uv_lock_sha256": sha256_file(repo_root / "uv.lock"),
        "artifacts": authenticated_artifacts,
        "evaluation": {
            "suite": "math",
            "excluded_tasks": ["theoremqa"],
            "top_level_tasks": list(EXPECTED_EVAL_SAMPLE_COUNTS),
            "sample_counts": EXPECTED_EVAL_SAMPLE_COUNTS,
            "hendrycks_math_leaf_counts": HENDRYCKS_MATH_SAMPLE_COUNTS,
            "primary_metrics": PRIMARY_METRICS,
            "leaf_task_versions": EXPECTED_EVAL_VERSIONS,
            "evaluation_code_sha256": evaluation_code_sha256,
            "environment": {"lm_eval_version": "0.4.12", "lockfile": "uv.lock"},
            "protocol": {
                "num_fewshot": 0,
                "apply_chat_template": "tokenizer_auto",
                "fewshot_as_multiturn": False,
                "batch_size_per_rank": 16,
                "world_size": 1,
                "limit": None,
                "offline_datasets": True,
                "legacy_bottleneck_generate": False,
                "resume": {
                    "unit": "one_top_level_task",
                    "tasks": list(EXPECTED_EVAL_SAMPLE_COUNTS),
                    "group_completion": "results_or_groups",
                    "publication": "same_directory_fsync_os_replace",
                    "integrity": "embedded_sha256_strict_schema",
                },
            },
        },
    }


def build_experiment_manifest(
    *,
    repo_root: Path,
    source_commit: str,
    artifacts: dict[str, tuple[Path, str]],
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source_commit must be a full lowercase Git SHA")

    runs: list[dict[str, Any]] = []
    for path in matrix_config_paths(repo_root):
        if not path.is_file():
            raise ValueError(f"missing matrix config: {path}")
        config = load_train_config(path)
        variant = config.model.variant
        if variant not in VARIANTS or config.seed not in SEEDS:
            raise ValueError(f"unexpected matrix identity in {path}")
        runs.append(
            {
                "variant": variant,
                "architecture": config.model.architecture_name,
                "seed": config.seed,
                "config": str(path.relative_to(repo_root)),
                "config_sha256": sha256_file(path),
                "output_dir": config.output_dir,
                "wandb_id": config.wandb.id,
            }
        )

    authenticated_artifacts: dict[str, dict[str, Any]] = {}
    for name, (path, expected_sha256) in sorted(artifacts.items()):
        if not path.is_file():
            raise ValueError(f"artifact is not a file: {name}: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"artifact sha256 mismatch for {name}: expected {expected_sha256}, got {actual_sha256}"
            )
        authenticated_artifacts[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": actual_sha256,
        }

    pilots: list[dict[str, Any]] = []
    pilot_dir = repo_root / CONFIG_DIR / "pilots"
    for path in sorted(pilot_dir.glob("*.yaml")):
        config = load_train_config(path)
        pilots.append(
            {
                "variant": config.model.variant,
                "architecture": config.model.architecture_name,
                "seed": config.seed,
                "max_steps": config.max_steps,
                "config": str(path.relative_to(repo_root)),
                "config_sha256": sha256_file(path),
                "output_dir": config.output_dir,
            }
        )
    if len(pilots) != 4:
        raise ValueError(f"expected four pilot configs, found {len(pilots)}")

    evaluation_files = [
        repo_root / "configs/eval/benchmarks.yaml",
        repo_root / "gcw/eval.py",
        *(
            repo_root / "configs/tasks" / f"{task}.yaml"
            for task in ("gsm8k_openmath", "svamp", "logiqa", "gsm8k_hard")
        ),
    ]
    evaluation_code_sha256 = {
        str(path.relative_to(repo_root)): sha256_file(path) for path in evaluation_files
    }
    lockfile = repo_root / "uv.lock"
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_commit": source_commit,
        "uv_lock_sha256": sha256_file(lockfile),
        "runs": runs,
        "pilots": pilots,
        "artifacts": authenticated_artifacts,
        "evaluation": {
            "suite": "math",
            "excluded_tasks": ["theoremqa"],
            "top_level_tasks": list(EXPECTED_EVAL_SAMPLE_COUNTS),
            "sample_counts": EXPECTED_EVAL_SAMPLE_COUNTS,
            "hendrycks_math_leaf_counts": HENDRYCKS_MATH_SAMPLE_COUNTS,
            "primary_metrics": PRIMARY_METRICS,
            "leaf_task_versions": EXPECTED_EVAL_VERSIONS,
            "evaluation_code_sha256": evaluation_code_sha256,
            "environment": {"lm_eval_version": "0.4.12", "lockfile": "uv.lock"},
            "protocol": {
                "num_fewshot": 0,
                "apply_chat_template": "tokenizer_auto",
                "fewshot_as_multiturn": False,
                "batch_size_per_rank": 16,
                "world_size": 1,
                "limit": None,
                "offline_datasets": True,
                "legacy_bottleneck_generate": False,
            },
        },
    }


def extract_pilot_metrics(payload: dict[str, Any], *, limit: int) -> dict[str, float]:
    results = payload.get("results")
    expected_result_tasks = {*EXPECTED_EVAL_LEAF_TASKS, "hendrycks_math"}
    if not isinstance(results, dict) or set(results) != expected_result_tasks:
        raise ValueError("evaluation pilot must contain the exact approved leaf tasks and MATH group row")
    groups = payload.get("groups")
    if not isinstance(groups, dict) or set(groups) != {"hendrycks_math"}:
        raise ValueError("evaluation pilot must contain exactly the approved MATH group")
    if "theoremqa" in results:
        raise ValueError("TheoremQA is excluded from the evaluation pilot")
    versions = payload.get("versions")
    expected_versions = {**EXPECTED_EVAL_VERSIONS, "hendrycks_math": "1.0"}
    if versions != expected_versions:
        raise ValueError("evaluation pilot task versions differ from the approved versions")
    n_shot = payload.get("n-shot")
    if n_shot != {task: 0 for task in expected_result_tasks}:
        raise ValueError("evaluation pilot must be zero-shot")
    sample_counts = payload.get("n-samples")
    if not isinstance(sample_counts, dict) or set(sample_counts) != set(EXPECTED_EVAL_LEAF_TASKS):
        raise ValueError("evaluation pilot sample counts differ from the approved leaf tasks")
    for task, entry in sample_counts.items():
        if not isinstance(entry, dict) or type(entry.get("effective")) is not int:
            raise ValueError(f"evaluation pilot sample count is invalid for {task}")
        approved_count = (
            HENDRYCKS_MATH_SAMPLE_COUNTS[task]
            if task in HENDRYCKS_MATH_SAMPLE_COUNTS
            else EXPECTED_EVAL_SAMPLE_COUNTS[task]
        )
        if entry["effective"] != min(limit, approved_count):
            raise ValueError(f"evaluation pilot sample count is not exactly the approved limit for {task}")
    metrics: dict[str, float] = {}
    for task in ("gsm8k_openmath", "svamp", "logiqa", "agieval_gaokao_mathqa", "gsm8k_hard"):
        metric_key = PRIMARY_METRICS[task]
        row = results.get(task)
        if not isinstance(row, dict) or metric_key not in row:
            raise ValueError(f"evaluation pilot missing primary metric {task}/{metric_key}")
        value = row[metric_key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"evaluation pilot metric is non-finite for {task}")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"evaluation pilot metric is outside [0, 1] for {task}")
        metrics[task] = float(value)
    math_weighted_total = 0.0
    math_sample_total = 0
    for task in HENDRYCKS_MATH_SAMPLE_COUNTS:
        row = results[task]
        value = row.get("exact_match,none") if isinstance(row, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"evaluation pilot MATH metric is non-finite for {task}")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"evaluation pilot MATH metric is outside [0, 1] for {task}")
        effective_count = sample_counts[task]["effective"]
        math_weighted_total += float(value) * effective_count
        math_sample_total += effective_count
    leaf_aggregate = math_weighted_total / math_sample_total
    group_row = groups["hendrycks_math"]
    result_group_row = results["hendrycks_math"]
    group_value = group_row.get("exact_match,none") if isinstance(group_row, dict) else None
    result_group_value = result_group_row.get("exact_match,none") if isinstance(result_group_row, dict) else None
    if (
        isinstance(group_value, bool)
        or not isinstance(group_value, (int, float))
        or not math.isfinite(float(group_value))
    ):
        raise ValueError("evaluation pilot MATH group metric is non-finite")
    if (
        isinstance(result_group_value, bool)
        or not isinstance(result_group_value, (int, float))
        or not math.isfinite(float(result_group_value))
    ):
        raise ValueError("evaluation pilot MATH group metric is non-finite")
    group_score = float(group_value)
    result_group_score = float(result_group_value)
    if not 0.0 <= group_score <= 1.0 or not 0.0 <= result_group_score <= 1.0:
        raise ValueError("evaluation pilot MATH group metric is outside [0, 1]")
    if not math.isclose(group_score, result_group_score, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("evaluation pilot MATH group rows disagree")
    if not math.isclose(group_score, leaf_aggregate, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("evaluation pilot MATH group score differs from the weighted leaf aggregate")
    metrics["hendrycks_math"] = group_score
    return metrics


def extract_cache_warmup_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Validate exact all-task limit-1 cache coverage without passing the pilot gate."""
    return extract_pilot_metrics(payload, limit=1)


def build_evaluation_ledger(
    *,
    cache_marker: Path,
    pilot_marker: Path,
    metric_markers: list[Path],
    training_source_commit: str,
    training_experiment_manifest_sha256: str,
    evaluation_source_commit: str,
    evaluation_manifest_sha256: str,
    eval_cache_manifest_sha256: str,
) -> dict[str, Any]:
    verify_required_training_identity(training_source_commit, training_experiment_manifest_sha256)
    expected_identity = {
        "training_source_commit": training_source_commit,
        "training_experiment_manifest_sha256": training_experiment_manifest_sha256,
        "evaluation_source_commit": evaluation_source_commit,
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "eval_cache_manifest_sha256": eval_cache_manifest_sha256,
    }

    def checked(path: Path, kind: str) -> dict[str, Any]:
        payload = load_json_strict(path)
        if not isinstance(payload, dict) or payload.get("kind") != kind:
            raise ValueError(f"ledger input has wrong kind: {path.name}")
        for key, value in expected_identity.items():
            if payload.get(key) != value:
                raise ValueError(f"ledger input provenance mismatch for {path.name}: {key}")
        return payload

    checked(cache_marker, "flow_math_eval_cache")
    checked(pilot_marker, "flow_math_eval_pilot")
    expected_targets = {"tf-backbone"} | {
        f"{variant}-seed{seed}" for variant in VARIANTS for seed in SEEDS
    }
    hashes: dict[str, str] = {}
    for marker_path in metric_markers:
        payload = checked(marker_path, "flow_math_eval_result")
        target = payload.get("target")
        if not isinstance(target, str) or target in hashes:
            raise ValueError("ledger has invalid or duplicate evaluation target")
        hashes[target] = sha256_file(marker_path)
    if set(hashes) != expected_targets:
        raise ValueError("ledger must contain exactly the TF baseline and nine Flow targets")
    return {
        "schema_version": 1,
        "kind": "flow_math_eval_ledger",
        **expected_identity,
        "cache_marker_name": cache_marker.name,
        "cache_marker_sha256": sha256_file(cache_marker),
        "pilot_marker_name": pilot_marker.name,
        "pilot_marker_sha256": sha256_file(pilot_marker),
        "metric_marker_sha256": dict(sorted(hashes.items())),
    }


def verify_evaluation_ledger(
    *,
    ledger_path: Path,
    approved_ledger_sha256: str,
    cache_marker: Path,
    pilot_marker: Path,
    metric_markers: list[Path],
    training_source_commit: str,
    training_experiment_manifest_sha256: str,
    evaluation_source_commit: str,
    evaluation_manifest_sha256: str,
    eval_cache_manifest_sha256: str,
) -> dict[str, Any]:
    if sha256_file(ledger_path) != approved_ledger_sha256:
        raise ValueError("evaluation ledger differs from the independently retained digest")
    expected = build_evaluation_ledger(
        cache_marker=cache_marker,
        pilot_marker=pilot_marker,
        metric_markers=metric_markers,
        training_source_commit=training_source_commit,
        training_experiment_manifest_sha256=training_experiment_manifest_sha256,
        evaluation_source_commit=evaluation_source_commit,
        evaluation_manifest_sha256=evaluation_manifest_sha256,
        eval_cache_manifest_sha256=eval_cache_manifest_sha256,
    )
    actual = load_json_strict(ledger_path)
    if actual != expected:
        raise ValueError("evaluation ledger differs from live authenticated marker hashes")
    return expected


def summarise_metric_markers(
    markers: list[dict[str, Any]],
    *,
    training_source_commit: str,
    training_experiment_manifest_sha256: str,
    evaluation_source_commit: str,
    evaluation_manifest_sha256: str,
    eval_cache_manifest_sha256: str,
) -> dict[str, Any]:
    expected_targets = {"tf-backbone"} | {
        f"{variant}-seed{seed}" for variant in VARIANTS for seed in SEEDS
    }
    targets = [marker.get("target") for marker in markers]
    if set(targets) != expected_targets or len(targets) != len(expected_targets):
        raise ValueError("metric markers must contain exactly the TF baseline and nine flow targets")

    by_target: dict[str, dict[str, Any]] = {}
    for marker in markers:
        target = marker.get("target")
        if not isinstance(target, str):
            raise ValueError("metric marker target must be a string")
        if marker.get("schema_version") != 2 or marker.get("kind") != "flow_math_eval_result":
            raise ValueError(f"metric marker schema differs for {target}")
        if marker.get("training_source_commit") != training_source_commit:
            raise ValueError(f"metric marker training source commit mismatch for {target}")
        if marker.get("training_experiment_manifest_sha256") != training_experiment_manifest_sha256:
            raise ValueError(f"metric marker training manifest hash mismatch for {target}")
        if marker.get("evaluation_source_commit") != evaluation_source_commit:
            raise ValueError(f"metric marker evaluation source commit mismatch for {target}")
        if marker.get("evaluation_manifest_sha256") != evaluation_manifest_sha256:
            raise ValueError(f"metric marker evaluation manifest hash mismatch for {target}")
        if marker.get("eval_cache_manifest_sha256") != eval_cache_manifest_sha256:
            raise ValueError(f"metric marker evaluation-cache hash mismatch for {target}")
        result_path = Path(marker.get("result", ""))
        checkpoint_path = Path(marker.get("checkpoint", ""))
        checkpoint_model_path = checkpoint_path / "model.safetensors"
        if not result_path.is_file() or sha256_file(result_path) != marker.get("result_sha256"):
            raise ValueError(f"metric marker raw result changed for {target}")
        if not checkpoint_model_path.is_file() or sha256_file(checkpoint_model_path) != marker.get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"metric marker checkpoint changed for {target}")
        if checkpoint_metadata_sha256(checkpoint_path) != marker.get("checkpoint_metadata_sha256"):
            raise ValueError(f"metric marker checkpoint metadata changed for {target}")
        metrics = marker.get("metrics")
        raw_payload = load_json_strict(result_path)
        if not isinstance(raw_payload, dict):
            raise ValueError(f"raw evaluation result is not an object for {target}")
        derived_metrics = extract_primary_metrics(raw_payload)
        if metrics != derived_metrics:
            raise ValueError(f"metric marker scores differ from authenticated raw results for {target}")
        if set(derived_metrics) != set(EXPECTED_EVAL_SAMPLE_COUNTS):
            raise ValueError(f"metric marker has an unexpected task set for {target}")
        clean_metrics: dict[str, float] = {}
        for task, value in derived_metrics.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"metric marker has a non-finite value for {target}/{task}")
            score = float(value)
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"metric marker value is outside [0, 1] for {target}/{task}")
            clean_metrics[task] = score
        by_target[target] = {**marker, "metrics": clean_metrics}

    runs: list[dict[str, Any]] = []
    math_only_tasks = [task for task in EXPECTED_EVAL_SAMPLE_COUNTS if task != "logiqa"]
    for variant in VARIANTS:
        for seed in SEEDS:
            marker = by_target[f"{variant}-seed{seed}"]
            metrics = marker["metrics"]
            runs.append(
                {
                    "target": marker["target"],
                    "variant": variant,
                    "seed": seed,
                    "metrics": metrics,
                    "evidence": {
                        key: marker.get(key)
                        for key in (
                            "result", "result_sha256", "runtime_evidence", "runtime_evidence_sha256",
                            "cache_copy_evidence", "cache_copy_evidence_sha256",
                            "partial_identity", "partial_identity_sha256", "partial_manifest", "partial_manifest_sha256",
                            "checkpoint", "checkpoint_sha256",
                            "checkpoint_metadata_sha256", "training_marker", "training_marker_sha256",
                        )
                    },
                    "historical_macro": statistics.fmean(metrics.values()),
                    "math_only_macro": statistics.fmean(metrics[task] for task in math_only_tasks),
                }
            )

    architectures: dict[str, dict[str, Any]] = {}
    summary_keys = [*EXPECTED_EVAL_SAMPLE_COUNTS, "historical_macro", "math_only_macro"]
    for variant in VARIANTS:
        variant_runs = [run for run in runs if run["variant"] == variant]
        variant_summary: dict[str, Any] = {}
        for key in summary_keys:
            values = [
                run[key] if key in ("historical_macro", "math_only_macro") else run["metrics"][key]
                for run in variant_runs
            ]
            variant_summary[key] = {
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values),
                "n": len(values),
            }
        architectures[variant] = variant_summary

    paired_deltas: list[dict[str, Any]] = []
    for left, right in (("bgt", "icarus"), ("bgt", "longshot"), ("icarus", "longshot")):
        for seed in SEEDS:
            left_run = next(run for run in runs if run["variant"] == left and run["seed"] == seed)
            right_run = next(run for run in runs if run["variant"] == right and run["seed"] == seed)
            paired_deltas.append(
                {
                    "left": left,
                    "right": right,
                    "seed": seed,
                    "metrics": {
                        task: left_run["metrics"][task] - right_run["metrics"][task]
                        for task in EXPECTED_EVAL_SAMPLE_COUNTS
                    },
                    "historical_macro": left_run["historical_macro"] - right_run["historical_macro"],
                    "math_only_macro": left_run["math_only_macro"] - right_run["math_only_macro"],
                }
            )

    return {
        "schema_version": 2,
        "experiment_id": EXPERIMENT_ID,
        "training_provenance": {
            "source_commit": training_source_commit,
            "experiment_manifest_sha256": training_experiment_manifest_sha256,
        },
        "evaluation_provenance": {
            "source_commit": evaluation_source_commit,
            "manifest_sha256": evaluation_manifest_sha256,
        },
        "eval_cache_manifest_sha256": eval_cache_manifest_sha256,
        "baseline": by_target["tf-backbone"],
        "runs": runs,
        "architectures": architectures,
        "paired_deltas": paired_deltas,
        "aggregate_caveats": [
            "historical_macro double-counts GSM8K-derived problem structures through GSM8K-Hard",
            "math_only_macro excludes LogiQA but still includes both GSM8K and GSM8K-Hard",
        ],
    }


def probe_flow_dynamics(model: torch.nn.Module, input_ids: torch.Tensor) -> dict[str, Any]:
    flows = [module for module in model.modules() if isinstance(module, LatentFlow)]
    if not flows:
        raise ValueError("dynamics probe requires a flow-enabled model")
    original_steps = [flow.config.num_steps for flow in flows]
    displacements: list[float] = []

    def capture_displacement(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all().item():
            raise ValueError("flow displacement is non-finite or not a tensor")
        displacements.append(float(torch.linalg.vector_norm(output.float()).detach().cpu()))

    hooks = [flow.register_forward_hook(capture_displacement) for flow in flows]
    logits: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for steps in (1, 2):
                set_flow_solver_steps(model, steps)
                output = model(input_ids)
                if not isinstance(output, torch.Tensor):
                    output = getattr(output, "logits", None)
                if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all().item():
                    raise ValueError(f"solver-{steps} logits are non-finite or unavailable")
                logits.append(output.detach().float().cpu())
    finally:
        for flow, steps in zip(flows, original_steps, strict=True):
            flow.set_num_steps(steps)
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    max_abs_logit_diff = float((logits[0] - logits[1]).abs().max())
    max_flow_displacement_l2 = max(displacements, default=0.0)
    if max_abs_logit_diff <= 0.0:
        raise ValueError("flow solver steps 1 and 2 produced identical logits")
    if max_flow_displacement_l2 <= 0.0:
        raise ValueError("flow displacement is zero")
    gate_values = [float(flow.flow_gate.detach().float().cpu()) for flow in flows]
    if not all(math.isfinite(value) for value in gate_values):
        raise ValueError("flow gate is non-finite")
    return {
        "flow_modules": len(flows),
        "solver_steps": [1, 2],
        "max_abs_logit_diff": max_abs_logit_diff,
        "max_flow_displacement_l2": max_flow_displacement_l2,
        "flow_gate_values": gate_values,
        "flow_gate_delta_from_init": [
            value - float(flow.config.flow_gate_init) for value, flow in zip(gate_values, flows, strict=True)
        ],
    }


def _expected_training_fingerprint(config: Any) -> dict[str, Any]:
    if config.data.dataset is None or config.total_batch_size is None:
        raise ValueError("flow math fingerprint requires a fixed SFT dataset and token batch")
    world_size = 1
    micro_tokens = config.batch_size * config.sequence_len * world_size
    if config.total_batch_size % micro_tokens != 0:
        raise ValueError("approved total batch must be divisible by one-GPU micro batch tokens")
    return {
        "model": to_dict(config.model),
        "data": to_dict(config.data),
        "data_identity": {
            "kind": "sft",
            "path": str(Path(config.data.dataset).resolve()),
            "size": DATASET_SIZE_BYTES,
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
            "micro_tokens": micro_tokens,
            "tokens_per_step": config.total_batch_size,
            "grad_accum_steps": config.total_batch_size // micro_tokens,
        },
    }


def verify_completed_flow_math_run(
    *,
    config_path: Path,
    checkpoint_path: Path,
    backbone_model_path: Path,
) -> dict[str, Any]:
    config = load_train_config(config_path)
    verified = verify_training_checkpoint_files(checkpoint_path)
    manifest = verified["manifest"]
    train_state = verified["train_state"]
    fingerprint = manifest.get("training_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("checkpoint has no training fingerprint")

    expected_fingerprint = _expected_training_fingerprint(config)
    if fingerprint != expected_fingerprint:
        raise ValueError("checkpoint training fingerprint differs from the approved full fingerprint")
    if manifest.get("step") != config.max_steps or train_state.get("step") != config.max_steps:
        raise ValueError(f"checkpoint is not complete at step {config.max_steps}")
    if not config.freeze_backbone:
        raise ValueError("flow math run must freeze the backbone")
    if (
        manifest.get("saved_world_size") != 1
        or manifest.get("topology") != config.distributed
        or train_state.get("saved_world_size") != 1
        or train_state.get("exact_topology") is not True
    ):
        raise ValueError("flow math run must use the approved exact one-GPU topology")
    smooth_loss = train_state.get("smooth_loss")
    if not isinstance(smooth_loss, (int, float)) or not math.isfinite(float(smooth_loss)):
        raise ValueError("checkpoint smooth loss is not finite")

    rows_per_step = config.total_batch_size // config.sequence_len
    rows_consumed = rows_per_step * config.max_steps
    selected_rows = config.data.sft_take_rows
    if selected_rows is None or selected_rows <= 0:
        raise ValueError("flow math run must have a bounded positive SFT row selection")
    expected_cursor = {
        "epoch": rows_consumed // selected_rows,
        "row": rows_consumed % selected_rows,
        "kind": "sft",
    }
    if train_state.get("loader_cursor") != expected_cursor:
        raise ValueError(f"unexpected final SFT cursor; expected {expected_cursor}")
    if train_state.get("rank_loader_cursors") != [expected_cursor]:
        raise ValueError("rank SFT cursor does not match the approved one-GPU topology")

    checkpoint_dir = Path(verified["path"])
    model_path = checkpoint_dir / "model.safetensors"
    backbone_tensors, processor_tensors = _verify_model_tensors(model_path, backbone_model_path)
    optimizer_tensors = _verify_optimizer_tensors(checkpoint_dir / "optimizer.pt")
    return {
        "checkpoint": verified["path"],
        "variant": config.model.variant,
        "architecture": config.model.architecture_name,
        "seed": config.seed,
        "step": config.max_steps,
        "saved_world_size": train_state["saved_world_size"],
        "loader_cursor": train_state["loader_cursor"],
        "smooth_loss": float(smooth_loss),
        "backbone_tensors": backbone_tensors,
        "processor_tensors": processor_tensors,
        "optimizer_tensors": optimizer_tensors,
        "checkpoint_manifest_sha256": sha256_file(checkpoint_dir / "manifest.json"),
        "checkpoint_model_sha256": sha256_file(model_path),
        "config_sha256": sha256_file(config_path),
    }


def verify_run_marker(
    *,
    marker_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    backbone_model_path: Path,
    source_commit: str,
    experiment_manifest_sha256: str,
) -> dict[str, Any]:
    expected = verify_completed_flow_math_run(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        backbone_model_path=backbone_model_path,
    )
    expected.update(
        {
            "source_commit": source_commit,
            "experiment_manifest_sha256": experiment_manifest_sha256,
        }
    )
    protected_paths = (checkpoint_path.parent, checkpoint_path, *checkpoint_path.rglob("*"))
    writable = [path for path in protected_paths if path.stat().st_mode & 0o222]
    if writable:
        raise ValueError(f"completed checkpoint or containing run directory is writable: {writable[0]}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker != expected:
        raise ValueError("training marker differs from the re-authenticated live checkpoint")
    return expected


def verify_training_audit(
    path: Path,
    *,
    architecture: str,
    max_steps: int,
    source_commit: str,
    experiment_manifest_sha256: str,
) -> dict[str, Any]:
    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise ValueError("training audit must be a JSON object")
    count_fields = (
        "backbone_grad_tensors",
        "nonfinite_grad_tensors",
        "processor_grad_tensors",
        "flow_gate_grad_tensors",
    )
    if any(type(payload.get(field)) is not int for field in count_fields):
        raise ValueError("training audit gradient counts must be integers")
    if (
        type(payload.get("step")) is not int
        or payload.get("architecture") != architecture
        or payload.get("step") != max_steps
        or payload.get("source_commit") != source_commit
        or payload.get("experiment_manifest_sha256") != experiment_manifest_sha256
    ):
        raise ValueError("training audit identity differs from the approved pilot")
    if payload.get("backbone_grad_tensors") != 0:
        raise ValueError("training audit detected backbone gradients")
    if payload.get("nonfinite_grad_tensors") != 0:
        raise ValueError("training audit detected non-finite gradients")
    if payload.get("processor_grad_nonzero") is not True or payload.get("processor_grad_tensors", 0) <= 0:
        raise ValueError("training audit has no non-zero processor gradients")
    if payload.get("flow_gate_grad_nonzero") is not True or payload.get("flow_gate_grad_tensors", 0) <= 0:
        raise ValueError("training audit has no non-zero flow-gate gradients")
    throughput = payload.get("tokens_per_second")
    peak_memory = payload.get("peak_memory_mib")
    if isinstance(throughput, bool) or not isinstance(throughput, (int, float)):
        raise ValueError("training audit throughput is invalid")
    if not math.isfinite(float(throughput)) or throughput <= 0:
        raise ValueError("training audit throughput is invalid")
    if isinstance(peak_memory, bool) or not isinstance(peak_memory, (int, float)):
        raise ValueError("training audit peak memory is invalid")
    if not math.isfinite(float(peak_memory)) or peak_memory < 0:
        raise ValueError("training audit peak memory is invalid")
    return {**payload, "sha256": sha256_file(path)}


def _verify_model_tensors(model_path: Path, backbone_model_path: Path) -> tuple[int, int]:
    with safe_open(backbone_model_path, framework="pt", device="cpu") as source, safe_open(
        model_path, framework="pt", device="cpu"
    ) as trained:
        source_keys = set(source.keys())
        trained_keys = set(trained.keys())
        expected_backbone_keys = source_keys
        if not all(key.startswith("backbone.") for key in source_keys):
            expected_backbone_keys = {f"backbone.{key}" for key in source_keys}
        trained_backbone_keys = {key for key in trained_keys if key.startswith("backbone.")}
        if trained_backbone_keys != expected_backbone_keys:
            raise ValueError("checkpoint backbone tensor keys differ from the approved source")

        for source_key in sorted(source_keys):
            trained_key = source_key if source_key.startswith("backbone.") else f"backbone.{source_key}"
            source_tensor = source.get_tensor(source_key)
            trained_tensor = trained.get_tensor(trained_key)
            if (
                source_tensor.shape != trained_tensor.shape
                or source_tensor.dtype != trained_tensor.dtype
                or not torch.equal(source_tensor, trained_tensor)
            ):
                raise ValueError(f"frozen backbone tensor differs: {source_key}")

        processor_keys = sorted(trained_keys - trained_backbone_keys)
        if not processor_keys:
            raise ValueError("checkpoint contains no trainable processor tensors")
        for key in processor_keys:
            tensor = trained.get_tensor(key)
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"processor tensor is non-finite: {key}")
    return len(source_keys), len(processor_keys)


def _verify_optimizer_tensors(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    count = 0

    def visit(value: Any) -> None:
        nonlocal count
        if isinstance(value, torch.Tensor):
            count += 1
            if value.is_floating_point() and not torch.isfinite(value).all().item():
                raise ValueError("optimizer state contains a non-finite tensor")
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(payload)
    if count == 0:
        raise ValueError("optimizer state contains no tensors")
    return count


def _effective_count(sample_counts: dict[str, Any], task: str) -> int:
    entry = sample_counts.get(task)
    if not isinstance(entry, dict) or not isinstance(entry.get("effective"), int):
        raise ValueError(f"missing effective sample count for {task}")
    return entry["effective"]


def _metric(task_payload: Any, metric: str, task: str) -> float:
    if not isinstance(task_payload, dict) or metric not in task_payload:
        raise ValueError(f"missing primary metric {metric} for {task}")
    value = task_payload[metric]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"non-finite primary metric for {task}")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"primary metric outside [0, 1] for {task}")
    return score


def extract_primary_metrics(payload: dict[str, Any]) -> dict[str, float]:
    results = payload.get("results")
    groups = payload.get("groups")
    sample_counts = payload.get("n-samples")
    versions = payload.get("versions")
    n_shot = payload.get("n-shot")
    if (
        not isinstance(results, dict)
        or not isinstance(groups, dict)
        or not isinstance(sample_counts, dict)
        or not isinstance(versions, dict)
        or not isinstance(n_shot, dict)
    ):
        raise ValueError("evaluation payload must contain results, groups, n-samples, versions, and n-shot mappings")

    all_names = {str(name).lower() for mapping in (results, groups, sample_counts) for name in mapping}
    if any("theoremqa" in name for name in all_names):
        raise ValueError("theoremqa must be absent from evaluation outputs")
    expected_result_tasks = {*EXPECTED_EVAL_LEAF_TASKS, "hendrycks_math"}
    if set(results) != expected_result_tasks or set(sample_counts) != EXPECTED_EVAL_LEAF_TASKS:
        raise ValueError("evaluation output must contain exactly the approved leaves and MATH group row")
    if set(groups) != {"hendrycks_math"}:
        raise ValueError("evaluation output must contain exactly the approved MATH group")
    expected_versions = {**EXPECTED_EVAL_VERSIONS, "hendrycks_math": "1.0"}
    if versions != expected_versions:
        raise ValueError("evaluation task versions differ from the approved versions")
    if n_shot != {task: 0 for task in expected_result_tasks}:
        raise ValueError("evaluation tasks must all use the approved zero-shot setting")

    for task, expected in HENDRYCKS_MATH_SAMPLE_COUNTS.items():
        if task not in results:
            raise ValueError(f"missing Hendrycks MATH leaf task: {task}")
        if _effective_count(sample_counts, task) != expected:
            raise ValueError(f"unexpected sample count for {task}")
    if sum(_effective_count(sample_counts, task) for task in HENDRYCKS_MATH_SAMPLE_COUNTS) != 5000:
        raise ValueError("Hendrycks MATH leaves must total 5000 examples")

    metrics: dict[str, float] = {}
    for task in ("gsm8k_openmath", "svamp", "logiqa", "agieval_gaokao_mathqa", "gsm8k_hard"):
        if task not in results:
            raise ValueError(f"missing evaluation task: {task}")
        if _effective_count(sample_counts, task) != EXPECTED_EVAL_SAMPLE_COUNTS[task]:
            raise ValueError(f"unexpected sample count for {task}")
        metrics[task] = _metric(results[task], PRIMARY_METRICS[task], task)

    hendrycks = groups["hendrycks_math"]
    hendrycks_result = results["hendrycks_math"]
    group_metric = _metric(hendrycks, PRIMARY_METRICS["hendrycks_math"], "hendrycks_math")
    result_metric = _metric(hendrycks_result, PRIMARY_METRICS["hendrycks_math"], "hendrycks_math result row")
    if not math.isclose(group_metric, result_metric, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Hendrycks MATH result and group metrics disagree")
    metrics["hendrycks_math"] = group_metric

    return {task: metrics[task] for task in EXPECTED_EVAL_SAMPLE_COUNTS}
