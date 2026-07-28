from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash") or "/bin/bash"
TRAIN = ROOT / "scripts/slurm/flow_math_train_array.sbatch"
PILOT = ROOT / "scripts/slurm/flow_math_pilot.sbatch"
EVAL = ROOT / "scripts/slurm/flow_math_eval_array.sbatch"
EVAL_PILOT = ROOT / "scripts/slurm/flow_math_eval_pilot.sbatch"
EVAL_RUNNER = ROOT / "scripts/flow_math_eval_runner.py"
EVAL_CACHE = ROOT / "scripts/slurm/flow_math_eval_cache.sbatch"
SUBMIT = ROOT / "scripts/submit_flow_math.sh"


@pytest.mark.parametrize("path", [TRAIN, PILOT])
def test_flow_math_slurm_scripts_are_bounded_bash(path):
    assert path.is_file()
    subprocess.run([BASH, "-n", str(path)], check=True)
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


def test_flow_math_eval_cache_job_warms_validates_and_freezes_strictly():
    assert EVAL_CACHE.is_file()
    subprocess.run([BASH, "-n", str(EVAL_CACHE)], check=True)
    text = EVAL_CACHE.read_text()

    assert "#SBATCH --gres=gpu:1" in text
    assert "--limit 1" in text
    assert 'GCW_EVAL_SKIP_TASKS="theoremqa"' in text
    assert "finalize-eval-cache" in text
    assert "build-cache-manifest" in text
    assert "verify-cache-manifest" in text
    assert "path.chmod(0o444)" in text
    assert "path.chmod(0o555)" in text
    assert "LOCK_INVENTORY" in text
    assert "cache-lock-inventory" in text
    assert "HF_HUB_OFFLINE=1" not in text
    assert "EVALUATION_SOURCE_COMMIT" in text
    assert "TRAINING_SOURCE_COMMIT" in text


def test_flow_math_eval_array_is_full_math_without_theoremqa():
    assert EVAL.is_file()
    subprocess.run([BASH, "-n", str(EVAL)], check=True)
    text = EVAL.read_text()

    assert "#SBATCH --partition=agent-long-15,agent-long" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --cpus-per-task=16" in text
    assert "#SBATCH --mem=120G" in text
    assert "#SBATCH --time=12:00:00" in text
    assert 'GCW_EVAL_SKIP_TASKS="theoremqa"' in text
    assert "scripts/flow_math_eval_runner.py" in text
    assert '--target "$TARGET"' in text
    assert '--runtime-evidence "$RUNTIME_EVIDENCE"' in text
    assert '--partial-identity "$IDENTITY"' in text
    assert 'export GCW_STRICT_FLASH=1' in text
    assert 'export WANDB_MODE=offline' in text
    assert "torch.distributed.run" not in text
    assert "--nproc-per-node" not in text
    assert "paper7" not in text
    assert 'test ! -e "$OUT"' in text
    assert "verify-cache-manifest" in text
    assert "check-run-marker" in text
    assert "HF_DATASETS_OFFLINE=1" in text
    assert "TRAINING_SOURCE_COMMIT" in text
    assert "APPROVED_TRAINING_MANIFEST_SHA256" in text
    assert "EVALUATION_SOURCE_COMMIT" in text
    assert "APPROVED_EVALUATION_MANIFEST_SHA256" in text
    assert "verify-evaluation-context" in text
    assert "identity.json" in text
    assert "extract-metrics" in text
    assert "matrix-row" in text
    assert "train-${VARIANT}-seed${SEED}-${TRAINING_SOURCE_COMMIT:0:12}.json" in text
    assert "check-eval-pilot" in text
    assert "checkpoint-metadata-sha256" in text
    assert 'TRAIN_MARKER_ARGS=()' in text
    assert 'if [[ -d "$PARTIAL_DIR" ]]' in text
    assert "existing partial identity differs from this exact evaluation" in text
    assert 'PARTIAL_MANIFEST=' in text
    assert '--partial-manifest "$PARTIAL_MANIFEST"' in text
    assert 'export GCW_EVAL_REFERENCE_GENERATE=0' in text
    assert 'export PYTHONPATH=' in text
    assert '"${TRAIN_MARKER_ARGS[@]}"' in text


def test_flow_math_eval_runner_is_shared_by_pilot_and_full_evaluation():
    assert EVAL_RUNNER.is_file()
    pilot = EVAL_PILOT.read_text()
    full = EVAL.read_text()

    assert "scripts/flow_math_eval_runner.py" in pilot
    assert "scripts/flow_math_eval_runner.py" in full
    assert "-m gcw.cli eval math" not in pilot
    assert "-m gcw.cli eval math" not in full


def test_flow_math_eval_pilot_is_two_target_all_task_gate():
    assert EVAL_PILOT.is_file()
    subprocess.run([BASH, "-n", str(EVAL_PILOT)], check=True)
    text = EVAL_PILOT.read_text()
    assert "tf-backbone" in text
    assert "bgt-seed24" in text
    assert text.count("scripts/flow_math_eval_runner.py") == 2
    assert text.count("--runtime-evidence") == 2
    pre_assignment = text[: text.index("JOB_CACHE_DIR=")]
    assert 'check-eval-cache' in pre_assignment
    assert '--eval-cache-dir "$JOB_CACHE_DIR"' not in pre_assignment
    assert text.count('--eval-cache-dir "$JOB_CACHE_DIR"') == 2
    assert 'export GCW_STRICT_FLASH=1' in text
    assert 'export WANDB_MODE=offline' in text
    assert 'unset GCW_EVAL_PARTIAL_DIR' in text
    assert 'export GCW_EVAL_REFERENCE_GENERATE=0' in text
    assert 'export PYTHONPATH=' in text
    assert 'tf-backbone-train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json' in text
    assert 'bgt-seed24-train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json' in text
    assert '--target "tf-backbone"' in text
    assert '--target "bgt-seed24"' in text
    assert text.count("--limit 16") == 2
    assert "theoremqa" in text.lower()
    assert "finalize-eval-pilot" in text
    assert "check-eval-cache" in text
    assert "verify-cache-manifest" in text
    assert "check-run-marker" in text
    assert "TRAINING_SOURCE_COMMIT" in text
    assert "EVALUATION_SOURCE_COMMIT" in text
    assert "verify-evaluation-context" in text


def test_flow_math_submission_wrapper_precreates_logs_and_bounds_arrays():
    assert SUBMIT.is_file()
    subprocess.run([BASH, "-n", str(SUBMIT)], check=True)
    text = SUBMIT.read_text()

    assert 'SLURM_LOG_DIR="${SLURM_LOG_DIR:-$HOME/slurm_out/global-cache-workspace}"' in text
    assert 'mkdir -p "$SLURM_LOG_DIR"' in text
    assert '--output="$SLURM_LOG_DIR/$output_pattern"' in text
    assert '--error="$SLURM_LOG_DIR/$error_pattern"' in text
    assert "mkdir -p slurm_out" not in text
    assert "Refusing duplicate active job name" in text
    assert "--array=0-8%3" in text
    assert '0-3%2|4-7%2|8-9%2|[0-9]' in text
    assert '--array="$EVAL_ARRAY_SPEC"' in text
    assert "--dry-run" in text
    assert "APPROVED_TRAINING_MANIFEST_SHA256" in text
    assert "APPROVED_EVALUATION_MANIFEST_SHA256" in text
    assert "verify-evaluation-context" in text
    assert "eval-cache" in text
    assert "flow_math_eval_cache.sbatch" in text
    assert "check-eval-cache" in text
    assert "eval-pilot" in text
    assert "check-eval-pilot" in text
    assert "build-eval-ledger" in text
    assert "approved-ledger-sha256" in text
    assert "summarise" in text
