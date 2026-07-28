#!/usr/bin/env bash
#SBATCH --job-name=flow-math-train
#SBATCH --partition=agent-long-15,agent-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/flow-math-train-%A_%a.out

set -euo pipefail

SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
EXPERIMENT_MANIFEST="${EXPERIMENT_MANIFEST:?EXPERIMENT_MANIFEST is required}"
APPROVED_MANIFEST_SHA256="${APPROVED_MANIFEST_SHA256:?APPROVED_MANIFEST_SHA256 is required}"
MATRIX="configs/train/flow_math_llama32_1b/matrix.tsv"
EXPERIMENT_ROOT="/mnt/jfs-01/shared/gcw/experiments/flow_math_llama32_1b_v1"
BACKBONE_MODEL="/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf/model.safetensors"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
UV="$HOME/.local/bin/uv"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
test -x "$UV"
test -f "$EXPERIMENT_MANIFEST"
test -f "$MATRIX"
test -f "$BACKBONE_MODEL"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
read -r ACTUAL_MANIFEST_SHA _ < <(sha256sum "$EXPERIMENT_MANIFEST")
[[ "$ACTUAL_MANIFEST_SHA" == "$APPROVED_MANIFEST_SHA256" ]]
UV_RUN=("$UV" run --locked --extra gpu)
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-context \
  --repo-root "$PWD" \
  --manifest "$EXPERIMENT_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_MANIFEST_SHA256" \
  --source-commit "$SOURCE_COMMIT"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py check-pilot-gates \
  --markers-dir "$EXPERIMENT_ROOT/markers" \
  --source-commit "$SOURCE_COMMIT" \
  --experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256"

IFS=$'\t' read -r VARIANT SEED CONFIG < <(
  "${UV_RUN[@]}" python scripts/flow_math_experiment.py matrix-row --matrix "$MATRIX" --index "$TASK_ID"
)
test -f "$CONFIG"
OUTPUT_DIR="$("${UV_RUN[@]}" python -c 'import sys; from gcw.config import load_train_config; print(load_train_config(sys.argv[1]).output_dir)' "$CONFIG")"
test ! -e "$OUTPUT_DIR"

"${UV_RUN[@]}" python - "$EXPERIMENT_MANIFEST" "$CONFIG" <<'PY'
import json
from pathlib import Path
import sys
from gcw.flow_math_experiment import sha256_file
manifest = json.loads(Path(sys.argv[1]).read_text())
config_path = Path(sys.argv[2])
config_sha = sha256_file(config_path)
if not any(
    entry.get("config") == str(config_path) and entry.get("config_sha256") == config_sha
    for entry in manifest.get("runs", [])
):
    raise SystemExit("config is absent from or differs from experiment manifest")
PY

export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"
export WANDB_DIR="${WANDB_DIR:-$EXPERIMENT_ROOT/wandb}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$WANDB_DIR" "$EXPERIMENT_ROOT/markers"

"${UV_RUN[@]}" python scripts/check_cuda_env.py
"${UV_RUN[@]}" python -m gcw.cli train --config "$CONFIG"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-context \
  --repo-root "$PWD" \
  --manifest "$EXPERIMENT_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_MANIFEST_SHA256" \
  --source-commit "$SOURCE_COMMIT"

MARKER="$EXPERIMENT_ROOT/markers/train-${VARIANT}-seed${SEED}-${SOURCE_COMMIT:0:12}.json"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-run \
  --config "$CONFIG" \
  --checkpoint "$OUTPUT_DIR/checkpoint" \
  --source-commit "$SOURCE_COMMIT" \
  --experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256" \
  --out "$MARKER"
chmod -R a-w "$OUTPUT_DIR"

printf 'FLOW_MATH_TRAIN_OK variant=%s seed=%s source=%s marker=%s\n' \
  "$VARIANT" "$SEED" "$SOURCE_COMMIT" "$MARKER"
