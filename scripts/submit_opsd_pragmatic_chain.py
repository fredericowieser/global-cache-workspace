from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from scripts.validate_opsd_pragmatic_waiver import WAIVER_RELATIVE, validate_all

CANARY_CONFIG = Path(
    "configs/train/opsd/qwen3_06b_opsd_native_base_pragmatic_canary_v1.yaml"
)
TRAIN_CONFIG = Path(
    "configs/train/opsd/qwen3_06b_opsd_native_base_pragmatic_steps1000_bs16_v1.yaml"
)
TRAIN_WRAPPER = Path("scripts/slurm/opsd_native_train.sbatch")
MATRIX_WRAPPER = Path("scripts/slurm/opsd_full_matrix.sbatch")
TOKENIZER = Path("/mnt/jfs-01/shared/gcw/hf/Qwen3-0.6B-base-v1")
DATASETS_CACHE = Path("/mnt/jfs-01/shared/gcw/hf_datasets_eval_v3")
EXPERIMENT_ROOT = Path(
    "/mnt/jfs-01/shared/gcw/experiments/qwen3_06b_opsd_native_base_pragmatic_v1"
)
CANARY_OUTPUT = EXPERIMENT_ROOT / "canary-steps1-bs1"
TRAIN_OUTPUT = EXPERIMENT_ROOT / "steps1000-bs16"
MATRIX_OUTPUT = Path(
    "/mnt/jfs-01/shared/gcw/evaluations/"
    "qwen3_06b_opsd_native_base_pragmatic_v1/steps1000-full-matrix"
)
BACKBONE_LABEL = "converted_GCW_Qwen3_0.6B_base_not_numerically_identical_to_HF"


def _run(command: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


def _git(repo: Path, *args: str) -> str:
    return _run(["git", *args], cwd=repo).stdout.strip()


def _job_id(result: subprocess.CompletedProcess[str]) -> str:
    value = result.stdout.strip().split(";", 1)[0]
    if not re.fullmatch(r"[0-9]+", value):
        raise RuntimeError(f"sbatch returned an invalid job id: {result.stdout!r}")
    return value


def _export(values: dict[str, str]) -> str:
    for key, value in values.items():
        if "," in value or "\n" in value:
            raise ValueError(f"unsafe Slurm export value for {key}")
    return "ALL," + ",".join(f"{key}={value}" for key, value in values.items())


def _write_new(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def submit_chain(repo: Path) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    source_revision = _git(repo, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise RuntimeError("source revision is not a full commit hash")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("pragmatic chain submission requires a clean checkout")
    for relative in (CANARY_CONFIG, TRAIN_CONFIG, TRAIN_WRAPPER, MATRIX_WRAPPER, WAIVER_RELATIVE):
        tracked = _git(repo, "ls-files", "--error-unmatch", "--", str(relative))
        if tracked != str(relative):
            raise RuntimeError(f"required source file is not tracked: {relative}")
    for output in (CANARY_OUTPUT, TRAIN_OUTPUT, MATRIX_OUTPUT):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing existing campaign output: {output}")
    if not TOKENIZER.resolve(strict=True).is_dir() or not DATASETS_CACHE.resolve(strict=True).is_dir():
        raise FileNotFoundError("tokenizer or evaluation cache is unavailable")
    waiver = repo / WAIVER_RELATIVE
    canary_config = repo / CANARY_CONFIG
    train_config = repo / TRAIN_CONFIG
    canary_evidence = validate_all(
        repo=repo,
        config_path=canary_config,
        waiver_path=waiver,
        source_revision=source_revision,
    )
    train_evidence = validate_all(
        repo=repo,
        config_path=train_config,
        waiver_path=waiver,
        source_revision=source_revision,
    )
    logs = Path.home() / "slurm_out/global-cache-workspace"
    logs.mkdir(parents=True, exist_ok=True)
    receipt = EXPERIMENT_ROOT / f"submission-receipt-{source_revision}.json"
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError(f"submission receipt already exists: {receipt}")

    submitted: list[str] = []
    try:
        canary_command = [
            "sbatch",
            "--parsable",
            "--partition=agent-dev-15",
            "--time=04:00:00",
            "--job-name=gcw-opsd-pragmatic-canary",
            f"--output={logs}/gcw-opsd-pragmatic-canary-%j.out",
            "--export="
            + _export(
                {
                    "REPO": str(repo),
                    "CONFIG": str(canary_config),
                    "SOURCE_REVISION": source_revision,
                }
            ),
            str(repo / TRAIN_WRAPPER),
        ]
        canary_job_id = _job_id(_run(canary_command, cwd=repo))
        submitted.append(canary_job_id)

        train_command = [
            "sbatch",
            "--parsable",
            "--partition=agent-xlong-15",
            "--time=5-00:00:00",
            f"--dependency=afterok:{canary_job_id}",
            "--kill-on-invalid-dep=yes",
            "--job-name=gcw-opsd-pragmatic-1k",
            f"--output={logs}/gcw-opsd-pragmatic-1k-%j.out",
            "--export="
            + _export(
                {
                    "REPO": str(repo),
                    "CONFIG": str(train_config),
                    "SOURCE_REVISION": source_revision,
                    "PROMOTION_EVIDENCE": str(CANARY_OUTPUT / "SUCCESS.json"),
                }
            ),
            str(repo / TRAIN_WRAPPER),
        ]
        train_job_id = _job_id(_run(train_command, cwd=repo))
        submitted.append(train_job_id)

        matrix_command = [
            "sbatch",
            "--parsable",
            "--partition=agent-xlong-15",
            "--time=2-00:00:00",
            "--nodes=1",
            "--ntasks=4",
            "--gres=gpu:4",
            f"--dependency=afterok:{train_job_id}",
            "--kill-on-invalid-dep=yes",
            "--job-name=gcw-opsd-pragmatic-matrix",
            f"--output={logs}/gcw-opsd-pragmatic-matrix-%j.out",
            "--export="
            + _export(
                {
                    "REPO": str(repo),
                    "CHECKPOINT": str(TRAIN_OUTPUT / "final"),
                    "SOURCE_REVISION": source_revision,
                    "TOKENIZER_SOURCE": str(TOKENIZER),
                    "DATASETS_CACHE": str(DATASETS_CACHE),
                    "OUT_DIR": str(MATRIX_OUTPUT),
                    "EXPECTED_PRAGMATIC_WAIVER": "true",
                    "WANDB_EVAL_NAME": "opsd-native-base-pragmatic-v1-steps1000-full-matrix",
                }
            ),
            str(repo / MATRIX_WRAPPER),
        ]
        matrix_job_id = _job_id(_run(matrix_command, cwd=repo))
        submitted.append(matrix_job_id)

        document = {
            "schema_version": 1,
            "kind": "opsd_pragmatic_slurm_chain_submission",
            "source_revision": source_revision,
            "backbone_label": BACKBONE_LABEL,
            "parity_gate_remains_failed": True,
            "pragmatic_parity_waiver": True,
            "canary": {
                "job_id": canary_job_id,
                "dependency": None,
                "config_sha256": canary_evidence["config_sha256"],
                "output": str(CANARY_OUTPUT),
                "command": canary_command,
            },
            "train_1k": {
                "job_id": train_job_id,
                "dependency": f"afterok:{canary_job_id}",
                "config_sha256": train_evidence["config_sha256"],
                "promotion_evidence": str(CANARY_OUTPUT / "SUCCESS.json"),
                "output": str(TRAIN_OUTPUT),
                "command": train_command,
            },
            "full_matrix": {
                "job_id": matrix_job_id,
                "dependency": f"afterok:{train_job_id}",
                "checkpoint": str(TRAIN_OUTPUT / "final"),
                "output": str(MATRIX_OUTPUT),
                "conditions": [
                    "cot-off__cp-off",
                    "cot-off__cp-on",
                    "cot-on__cp-off",
                    "cot-on__cp-on",
                ],
                "command": matrix_command,
            },
        }
        _write_new(receipt, document)
        return {**document, "receipt": str(receipt)}
    except BaseException:
        if submitted:
            subprocess.run(["scancel", *submitted], cwd=repo, check=False)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(submit_chain(args.repo), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
