#!/usr/bin/env bash
#SBATCH --job-name=flow-math-eval-pilot
#SBATCH --partition=agent-long-15,agent-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/%x-%j.out
#SBATCH --error=slurm_out/%x-%j.err

set -euo pipefail
umask 027

TRAINING_SOURCE_COMMIT="${TRAINING_SOURCE_COMMIT:?TRAINING_SOURCE_COMMIT is required}"
TRAINING_EXPERIMENT_MANIFEST="${TRAINING_EXPERIMENT_MANIFEST:?TRAINING_EXPERIMENT_MANIFEST is required}"
APPROVED_TRAINING_MANIFEST_SHA256="${APPROVED_TRAINING_MANIFEST_SHA256:?APPROVED_TRAINING_MANIFEST_SHA256 is required}"
EVALUATION_SOURCE_COMMIT="${EVALUATION_SOURCE_COMMIT:?EVALUATION_SOURCE_COMMIT is required}"
EVALUATION_MANIFEST="${EVALUATION_MANIFEST:?EVALUATION_MANIFEST is required}"
APPROVED_EVALUATION_MANIFEST_SHA256="${APPROVED_EVALUATION_MANIFEST_SHA256:?APPROVED_EVALUATION_MANIFEST_SHA256 is required}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:?EVAL_CACHE_DIR is required}"
EVAL_CACHE_MANIFEST="${EVAL_CACHE_MANIFEST:?EVAL_CACHE_MANIFEST is required}"
APPROVED_EVAL_CACHE_SHA256="${APPROVED_EVAL_CACHE_SHA256:?APPROVED_EVAL_CACHE_SHA256 is required}"
EXPERIMENT_ROOT="/mnt/jfs-01/shared/gcw/experiments/flow_math_llama32_1b_v1"
UV="$HOME/.local/bin/uv"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
test -x "$UV"
test -f "$TRAINING_EXPERIMENT_MANIFEST"
test -f "$EVALUATION_MANIFEST"
test -f "$EVAL_CACHE_MANIFEST"
[[ "$(git rev-parse HEAD)" == "$EVALUATION_SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
UV_RUN=("$UV" run --locked --extra gpu --extra eval)
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-evaluation-context \
  --repo-root "$PWD" --evaluation-manifest "$EVALUATION_MANIFEST" \
  --evaluation-manifest-sha256 "$APPROVED_EVALUATION_MANIFEST_SHA256" \
  --evaluation-source-commit "$EVALUATION_SOURCE_COMMIT" \
  --training-experiment-manifest "$TRAINING_EXPERIMENT_MANIFEST" \
  --training-experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256" \
  --training-source-commit "$TRAINING_SOURCE_COMMIT"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-cache-manifest \
  --cache-dir "$EVAL_CACHE_DIR" --manifest "$EVAL_CACHE_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" --require-read-only

IDENTITY="train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py check-eval-cache \
  --repo-root "$PWD" --evaluation-manifest "$EVALUATION_MANIFEST" \
  --evaluation-manifest-sha256 "$APPROVED_EVALUATION_MANIFEST_SHA256" \
  --evaluation-source-commit "$EVALUATION_SOURCE_COMMIT" \
  --training-experiment-manifest "$TRAINING_EXPERIMENT_MANIFEST" \
  --training-experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256" \
  --training-source-commit "$TRAINING_SOURCE_COMMIT" \
  --eval-cache-dir "$EVAL_CACHE_DIR" --eval-cache-manifest "$EVAL_CACHE_MANIFEST" \
  --eval-cache-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" \
  --warmup-result "$EXPERIMENT_ROOT/eval/cache-warmup/tf-backbone-$IDENTITY.json" \
  --lock-inventory "$EXPERIMENT_ROOT/eval/cache-lock-inventory-$IDENTITY.json" \
  --marker "$EXPERIMENT_ROOT/markers/eval-cache-$IDENTITY.json"

FLOW_CONFIG="configs/train/flow_math_llama32_1b/llama32_1b_flow_bgt_seed24.yaml"
FLOW_CHECKPOINT="$EXPERIMENT_ROOT/runs/bgt/seed-24/checkpoint"
TRAIN_MARKER="$EXPERIMENT_ROOT/markers/train-bgt-seed24-${TRAINING_SOURCE_COMMIT:0:12}.json"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py check-run-marker \
  --marker "$TRAIN_MARKER" --config "$FLOW_CONFIG" --checkpoint "$FLOW_CHECKPOINT" \
  --source-commit "$TRAINING_SOURCE_COMMIT" --experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256"

TF_RESULT="$EXPERIMENT_ROOT/eval/pilot/tf-backbone-train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json"
FLOW_RESULT="$EXPERIMENT_ROOT/eval/pilot/bgt-seed24-train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json"
TF_RUNTIME_EVIDENCE="$EXPERIMENT_ROOT/eval/pilot/tf-backbone-runtime-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json"
FLOW_RUNTIME_EVIDENCE="$EXPERIMENT_ROOT/eval/pilot/bgt-seed24-runtime-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json"
FINAL_MARKER="$EXPERIMENT_ROOT/markers/eval-pilot-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json"
CACHE_COPY_EVIDENCE="$EXPERIMENT_ROOT/eval/cache-copy-evidence/pilot-train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json"
JOB_CACHE_DIR="${SLURM_TMPDIR:-/tmp/${USER:?}}/flow-math-eval-cache-${SLURM_JOB_ID}-pilot"
test ! -e "$TF_RESULT"
test ! -e "$FLOW_RESULT"
test ! -e "$TF_RUNTIME_EVIDENCE"
test ! -e "$FLOW_RUNTIME_EVIDENCE"
test ! -e "$FINAL_MARKER"
test ! -e "$CACHE_COPY_EVIDENCE"
test ! -e "$JOB_CACHE_DIR"
mkdir -p "$EXPERIMENT_ROOT/eval/pilot" "$EXPERIMENT_ROOT/eval/cache-copy-evidence" "$EXPERIMENT_ROOT/markers"
mkdir -p "$(dirname "$JOB_CACHE_DIR")"
cp -a "$EVAL_CACHE_DIR" "$JOB_CACHE_DIR"
"${UV_RUN[@]}" python - "$JOB_CACHE_DIR" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
for path in root.rglob("*"):
    if path.is_dir():
        path.chmod(0o755)
root.chmod(0o755)
PY
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-cache-copy \
  --cache-dir "$JOB_CACHE_DIR" --manifest "$EVAL_CACHE_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" --target pilot

export GCW_STRICT_FLASH=1
export GCW_EVAL_SKIP_TASKS="theoremqa"
export GCW_EVAL_REFERENCE_GENERATE=0
export PYTHONPATH=
unset GCW_EVAL_PARTIAL_DIR
unset GCW_EVAL_PROGRESS_PATH
unset SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID
export HF_HOME="$JOB_CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$JOB_CACHE_DIR/huggingface/datasets"
export HF_HUB_CACHE="$JOB_CACHE_DIR/huggingface/hub"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

"${UV_RUN[@]}" python scripts/flow_math_eval_runner.py \
  --target "tf-backbone" \
  --checkpoint /mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf \
  --batch-size 16 --limit 16 --out "$TF_RESULT" \
  --runtime-evidence "$TF_RUNTIME_EVIDENCE" \
  --eval-cache-dir "$JOB_CACHE_DIR"
"${UV_RUN[@]}" python scripts/flow_math_eval_runner.py \
  --target "bgt-seed24" \
  --checkpoint "$FLOW_CHECKPOINT" --batch-size 16 --limit 16 --out "$FLOW_RESULT" \
  --runtime-evidence "$FLOW_RUNTIME_EVIDENCE" \
  --eval-cache-dir "$JOB_CACHE_DIR"

"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-cache-copy \
  --cache-dir "$JOB_CACHE_DIR" --manifest "$EVAL_CACHE_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" --target pilot \
  --out "$CACHE_COPY_EVIDENCE"

"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-evaluation-context \
  --repo-root "$PWD" --evaluation-manifest "$EVALUATION_MANIFEST" \
  --evaluation-manifest-sha256 "$APPROVED_EVALUATION_MANIFEST_SHA256" \
  --evaluation-source-commit "$EVALUATION_SOURCE_COMMIT" \
  --training-experiment-manifest "$TRAINING_EXPERIMENT_MANIFEST" \
  --training-experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256" \
  --training-source-commit "$TRAINING_SOURCE_COMMIT"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-cache-manifest \
  --cache-dir "$EVAL_CACHE_DIR" --manifest "$EVAL_CACHE_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" --require-read-only
"${UV_RUN[@]}" python scripts/flow_math_experiment.py check-run-marker \
  --marker "$TRAIN_MARKER" --config "$FLOW_CONFIG" --checkpoint "$FLOW_CHECKPOINT" \
  --source-commit "$TRAINING_SOURCE_COMMIT" --experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py finalize-eval-pilot \
  --repo-root "$PWD" --evaluation-manifest "$EVALUATION_MANIFEST" \
  --evaluation-manifest-sha256 "$APPROVED_EVALUATION_MANIFEST_SHA256" \
  --evaluation-source-commit "$EVALUATION_SOURCE_COMMIT" \
  --training-experiment-manifest "$TRAINING_EXPERIMENT_MANIFEST" \
  --training-experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256" \
  --training-source-commit "$TRAINING_SOURCE_COMMIT" \
  --eval-cache-dir "$EVAL_CACHE_DIR" --eval-cache-manifest "$EVAL_CACHE_MANIFEST" \
  --eval-cache-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" \
  --cache-copy-evidence "$CACHE_COPY_EVIDENCE" --out "$FINAL_MARKER"

printf 'FLOW_MATH_EVAL_PILOT_OK training=%s evaluation=%s marker=%s\n' "$TRAINING_SOURCE_COMMIT" "$EVALUATION_SOURCE_COMMIT" "$FINAL_MARKER"
