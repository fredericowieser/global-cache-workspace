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

SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
EXPERIMENT_MANIFEST="${EXPERIMENT_MANIFEST:?EXPERIMENT_MANIFEST is required}"
APPROVED_MANIFEST_SHA256="${APPROVED_MANIFEST_SHA256:?APPROVED_MANIFEST_SHA256 is required}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:?EVAL_CACHE_DIR is required}"
EVAL_CACHE_MANIFEST="${EVAL_CACHE_MANIFEST:?EVAL_CACHE_MANIFEST is required}"
APPROVED_EVAL_CACHE_SHA256="${APPROVED_EVAL_CACHE_SHA256:?APPROVED_EVAL_CACHE_SHA256 is required}"
EXPERIMENT_ROOT="/mnt/jfs-01/shared/gcw/experiments/flow_math_llama32_1b_v1"
UV="$HOME/.local/bin/uv"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
test -x "$UV"
test -f "$EXPERIMENT_MANIFEST"
test -f "$EVAL_CACHE_MANIFEST"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
UV_RUN=("$UV" run --locked --extra gpu --extra eval)
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-context \
  --repo-root "$PWD" --manifest "$EXPERIMENT_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_MANIFEST_SHA256" --source-commit "$SOURCE_COMMIT"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-cache-manifest \
  --cache-dir "$EVAL_CACHE_DIR" --manifest "$EVAL_CACHE_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" --require-read-only

FLOW_CONFIG="configs/train/flow_math_llama32_1b/llama32_1b_flow_bgt_seed24.yaml"
FLOW_CHECKPOINT="$EXPERIMENT_ROOT/runs/bgt/seed-24/checkpoint"
TRAIN_MARKER="$EXPERIMENT_ROOT/markers/train-bgt-seed24-${SOURCE_COMMIT:0:12}.json"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py check-run-marker \
  --marker "$TRAIN_MARKER" --config "$FLOW_CONFIG" --checkpoint "$FLOW_CHECKPOINT" \
  --source-commit "$SOURCE_COMMIT" --experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256"

TF_RESULT="$EXPERIMENT_ROOT/eval/pilot/tf-backbone-${SOURCE_COMMIT:0:12}.json"
FLOW_RESULT="$EXPERIMENT_ROOT/eval/pilot/bgt-seed24-${SOURCE_COMMIT:0:12}.json"
FINAL_MARKER="$EXPERIMENT_ROOT/markers/eval-pilot-${SOURCE_COMMIT:0:12}.json"
test ! -e "$TF_RESULT"
test ! -e "$FLOW_RESULT"
test ! -e "$FINAL_MARKER"
mkdir -p "$EXPERIMENT_ROOT/eval/pilot" "$EXPERIMENT_ROOT/markers"

export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"
export GCW_EVAL_SKIP_TASKS="theoremqa"
export HF_HOME="$EVAL_CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$EVAL_CACHE_DIR/huggingface/datasets"
export HF_HUB_CACHE="$EVAL_CACHE_DIR/huggingface/hub"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${UV_RUN[@]}" python scripts/check_cuda_env.py
"${UV_RUN[@]}" python -m gcw.cli eval math \
  --checkpoint /mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf \
  --batch-size 16 --limit 16 --out "$TF_RESULT"
"${UV_RUN[@]}" python -m gcw.cli eval math \
  --checkpoint "$FLOW_CHECKPOINT" --batch-size 16 --limit 16 --out "$FLOW_RESULT"

"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-context \
  --repo-root "$PWD" --manifest "$EXPERIMENT_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_MANIFEST_SHA256" --source-commit "$SOURCE_COMMIT"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-cache-manifest \
  --cache-dir "$EVAL_CACHE_DIR" --manifest "$EVAL_CACHE_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" --require-read-only
"${UV_RUN[@]}" python scripts/flow_math_experiment.py check-run-marker \
  --marker "$TRAIN_MARKER" --config "$FLOW_CONFIG" --checkpoint "$FLOW_CHECKPOINT" \
  --source-commit "$SOURCE_COMMIT" --experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py finalize-eval-pilot \
  --repo-root "$PWD" --experiment-manifest "$EXPERIMENT_MANIFEST" \
  --experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256" --source-commit "$SOURCE_COMMIT" \
  --eval-cache-dir "$EVAL_CACHE_DIR" --eval-cache-manifest "$EVAL_CACHE_MANIFEST" \
  --eval-cache-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" --out "$FINAL_MARKER"

printf 'FLOW_MATH_EVAL_PILOT_OK source=%s marker=%s\n' "$SOURCE_COMMIT" "$FINAL_MARKER"
