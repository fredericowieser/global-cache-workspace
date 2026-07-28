from __future__ import annotations

from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts/slurm/flow_math_train_array.sbatch"
PILOT = ROOT / "scripts/slurm/flow_math_pilot.sbatch"
EVAL = ROOT / "scripts/slurm/flow_math_eval_array.sbatch"
EVAL_PILOT = ROOT / "scripts/slurm/flow_math_eval_pilot.sbatch"
SUBMIT = ROOT / "scripts/submit_flow_math.sh"


@pytest.mark.parametrize("path", [TRAIN, PILOT])
def test_flow_math_slurm_scripts_are_bounded_bash(path):
    assert path.is_file()
    subprocess.run(["/run/current-system/sw/bin/bash", "-n", str(path)], check=True)
    text = path.read_text()

    assert "#SBATCH --partition=agent-long-15,agent-long" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --mem=96G" in text
    assert "#SBATCH --time=24:00:00" in text
    assert 'SOURCE_COMMIT="${SOURCE_COMMIT:?' in text
    assert 'EXPERIMENT_MANIFEST="${EXPERIMENT_MANIFEST:?' in text
    assert 'APPROVED_MANIFEST_SHA256="${APPROVED_MANIFEST_SHA256:?' in text
    assert '[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]' in text
    assert "verify-context" in text
    assert "run --locked --extra gpu" in text
    assert "scripts/check_cuda_env.py" in text
    assert "verify-run" in text
    assert '--source-commit "$SOURCE_COMMIT"' in text
    assert '--experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256"' in text
    assert 'test ! -e "$OUTPUT_DIR"' in text
    assert "theoremqa" not in text.lower()


def test_flow_math_train_array_uses_exact_matrix_row_and_no_inline_eval():
    text = TRAIN.read_text()

    assert "SLURM_ARRAY_TASK_ID" in text
    assert "matrix-row" in text
    assert "configs/train/flow_math_llama32_1b/matrix.tsv" in text
    assert "gcw.cli train" in text
    assert "gcw.cli eval" not in text
    assert "check-pilot-gates" in text
    assert "train-${VARIANT}-seed${SEED}-${SOURCE_COMMIT:0:12}.json" in text


def test_flow_math_pilot_requires_an_explicit_approved_config():
    text = PILOT.read_text()

    assert 'CONFIG="${CONFIG:?' in text
    assert "configs/train/flow_math_llama32_1b/pilots/llama32_1b_flow_bgt_steps100.yaml" in text
    assert "DYNAMICS_REQUIRED=1" in text
    assert "flow_math_experiment.py dynamics" in text
    assert "GCW_TRAIN_AUDIT_OUT" in text
    assert "verify-pilot-audit" in text
    assert "finalize-pilot" in text
    assert "pilot-${PILOT_NAME}-${SOURCE_COMMIT:0:12}.json" in text


def test_flow_math_eval_array_is_full_math_without_theoremqa():
    assert EVAL.is_file()
    subprocess.run(["/run/current-system/sw/bin/bash", "-n", str(EVAL)], check=True)
    text = EVAL.read_text()

    assert "#SBATCH --partition=agent-long-15,agent-long" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --cpus-per-task=16" in text
    assert "#SBATCH --mem=120G" in text
    assert "#SBATCH --time=12:00:00" in text
    assert 'GCW_EVAL_SKIP_TASKS="theoremqa"' in text
    assert "-m gcw.cli eval math" in text
    assert "paper7" not in text
    assert 'test ! -e "$OUT"' in text
    assert "verify-cache-manifest" in text
    assert "check-run-marker" in text
    assert "HF_DATASETS_OFFLINE=1" in text
    assert "eval/partials/${TARGET}/${SOURCE_COMMIT:0:12}" in text
    assert "identity.json" in text
    assert "extract-metrics" in text
    assert "matrix-row" in text
    assert "train-${VARIANT}-seed${SEED}-${SOURCE_COMMIT:0:12}.json" in text
    assert "eval-${TARGET}-${SOURCE_COMMIT:0:12}.json" in text
    assert "check-eval-pilot" in text
    assert "checkpoint-metadata-sha256" in text


def test_flow_math_eval_pilot_is_two_target_all_task_gate():
    assert EVAL_PILOT.is_file()
    subprocess.run(["/run/current-system/sw/bin/bash", "-n", str(EVAL_PILOT)], check=True)
    text = EVAL_PILOT.read_text()
    assert "tf-backbone" in text
    assert "bgt-seed24" in text
    assert text.count("-m gcw.cli eval math") == 2
    assert text.count("--limit 16") == 2
    assert "theoremqa" in text.lower()
    assert "finalize-eval-pilot" in text
    assert "verify-cache-manifest" in text
    assert "check-run-marker" in text


def test_flow_math_submission_wrapper_precreates_logs_and_bounds_arrays():
    assert SUBMIT.is_file()
    subprocess.run(["/run/current-system/sw/bin/bash", "-n", str(SUBMIT)], check=True)
    text = SUBMIT.read_text()

    assert "mkdir -p slurm_out" in text
    assert "Refusing duplicate active job name" in text
    assert "--array=0-8%3" in text
    assert "--array=0-9%2" in text
    assert "--dry-run" in text
    assert "APPROVED_MANIFEST_SHA256" in text
    assert "verify-context" in text
    assert "eval-pilot" in text
    assert "check-eval-pilot" in text
