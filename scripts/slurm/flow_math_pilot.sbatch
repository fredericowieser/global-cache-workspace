#!/usr/bin/env bash
#SBATCH --job-name=flow-math-pilot
#SBATCH --partition=agent-long-15,agent-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/flow-math-pilot-%j.out

set -euo pipefail

SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
EXPERIMENT_MANIFEST="${EXPERIMENT_MANIFEST:?EXPERIMENT_MANIFEST is required}"
APPROVED_MANIFEST_SHA256="${APPROVED_MANIFEST_SHA256:?APPROVED_MANIFEST_SHA256 is required}"
CONFIG="${CONFIG:?CONFIG is required}"
EXPERIMENT_ROOT="/mnt/jfs-01/shared/gcw/experiments/flow_math_llama32_1b_v1"
BACKBONE_MODEL="/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf/model.safetensors"
UV="$HOME/.local/bin/uv"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
test -x "$UV"
test -f "$EXPERIMENT_MANIFEST"
test -f "$BACKBONE_MODEL"
test -f "$CONFIG"
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

case "$CONFIG" in
  configs/train/flow_math_llama32_1b/pilots/llama32_1b_flow_bgt_steps2.yaml)
    PILOT_NAME="bgt-steps2"; DYNAMICS_REQUIRED=0 ;;
  configs/train/flow_math_llama32_1b/pilots/llama32_1b_flow_icarus_steps2.yaml)
    PILOT_NAME="icarus-steps2"; DYNAMICS_REQUIRED=0 ;;
  configs/train/flow_math_llama32_1b/pilots/llama32_1b_flow_longshot_steps2.yaml)
    PILOT_NAME="longshot-steps2"; DYNAMICS_REQUIRED=0 ;;
  configs/train/flow_math_llama32_1b/pilots/llama32_1b_flow_bgt_steps100.yaml)
    PILOT_NAME="bgt-steps100"; DYNAMICS_REQUIRED=1 ;;
  *) printf 'Refusing non-approved pilot config: %s\n' "$CONFIG" >&2; exit 2 ;;
esac
if [[ "$DYNAMICS_REQUIRED" == "1" ]]; then
  "${UV_RUN[@]}" python scripts/flow_math_experiment.py check-pilot-gates \
    --markers-dir "$EXPERIMENT_ROOT/markers" \
    --source-commit "$SOURCE_COMMIT" \
    --experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256" \
    --short-only
fi

OUTPUT_DIR="$("${UV_RUN[@]}" python -c 'import sys; from gcw.config import load_train_config; print(load_train_config(sys.argv[1]).output_dir)' "$CONFIG")"
MARKERS="$EXPERIMENT_ROOT/markers"
AUDIT="$EXPERIMENT_ROOT/pilots/audits/${PILOT_NAME}-${SOURCE_COMMIT:0:12}.json"
CHECKPOINT_MARKER="$MARKERS/pilot-${PILOT_NAME}-${SOURCE_COMMIT:0:12}-checkpoint.json"
DYNAMICS="$EXPERIMENT_ROOT/pilots/dynamics/${PILOT_NAME}-${SOURCE_COMMIT:0:12}.json"
FINAL_MARKER="$MARKERS/pilot-${PILOT_NAME}-${SOURCE_COMMIT:0:12}.json"
test ! -e "$OUTPUT_DIR"
test ! -e "$AUDIT"
test ! -e "$CHECKPOINT_MARKER"
test ! -e "$FINAL_MARKER"
if [[ "$DYNAMICS_REQUIRED" == "1" ]]; then test ! -e "$DYNAMICS"; fi
mkdir -p "$EXPERIMENT_ROOT/wandb" "$MARKERS" "$(dirname "$AUDIT")" "$(dirname "$DYNAMICS")"

export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"
export WANDB_MODE=disabled
export WANDB_DIR="${WANDB_DIR:-$EXPERIMENT_ROOT/wandb}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GCW_TRAIN_AUDIT_OUT="$AUDIT"
export GCW_AUDIT_SOURCE_COMMIT="$SOURCE_COMMIT"
export GCW_AUDIT_MANIFEST_SHA256="$APPROVED_MANIFEST_SHA256"

"${UV_RUN[@]}" python scripts/check_cuda_env.py
"${UV_RUN[@]}" python -m gcw.cli train --config "$CONFIG"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-context \
  --repo-root "$PWD" \
  --manifest "$EXPERIMENT_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_MANIFEST_SHA256" \
  --source-commit "$SOURCE_COMMIT"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-run \
  --config "$CONFIG" \
  --checkpoint "$OUTPUT_DIR/checkpoint" \
  --source-commit "$SOURCE_COMMIT" \
  --experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256" \
  --out "$CHECKPOINT_MARKER"
chmod -R a-w "$OUTPUT_DIR"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-pilot-audit \
  --audit "$AUDIT" \
  --config "$CONFIG" \
  --source-commit "$SOURCE_COMMIT" \
  --experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256"

DYNAMICS_ARGS=()
if [[ "$DYNAMICS_REQUIRED" == "1" ]]; then
  "${UV_RUN[@]}" python scripts/flow_math_experiment.py dynamics \
    --checkpoint "$OUTPUT_DIR/checkpoint" \
    --device cuda \
    --input-length 128 \
    --source-commit "$SOURCE_COMMIT" \
    --experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256" \
    --out "$DYNAMICS"
  DYNAMICS_ARGS=(--dynamics "$DYNAMICS")
fi
"${UV_RUN[@]}" python scripts/flow_math_experiment.py finalize-pilot \
  --config "$CONFIG" \
  --checkpoint-marker "$CHECKPOINT_MARKER" \
  --audit "$AUDIT" \
  "${DYNAMICS_ARGS[@]}" \
  --source-commit "$SOURCE_COMMIT" \
  --experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256" \
  --out "$FINAL_MARKER"

printf 'FLOW_MATH_PILOT_OK name=%s source=%s marker=%s\n' "$PILOT_NAME" "$SOURCE_COMMIT" "$FINAL_MARKER"
