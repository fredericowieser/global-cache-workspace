#!/usr/bin/env bash
#SBATCH --job-name=flow-math-eval
#SBATCH --partition=agent-long-15,agent-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_out/flow-math-eval-%A_%a.out

set -euo pipefail

TRAINING_SOURCE_COMMIT="${TRAINING_SOURCE_COMMIT:?TRAINING_SOURCE_COMMIT is required}"
TRAINING_EXPERIMENT_MANIFEST="${TRAINING_EXPERIMENT_MANIFEST:?TRAINING_EXPERIMENT_MANIFEST is required}"
APPROVED_TRAINING_MANIFEST_SHA256="${APPROVED_TRAINING_MANIFEST_SHA256:?APPROVED_TRAINING_MANIFEST_SHA256 is required}"
EVALUATION_SOURCE_COMMIT="${EVALUATION_SOURCE_COMMIT:?EVALUATION_SOURCE_COMMIT is required}"
EVALUATION_MANIFEST="${EVALUATION_MANIFEST:?EVALUATION_MANIFEST is required}"
APPROVED_EVALUATION_MANIFEST_SHA256="${APPROVED_EVALUATION_MANIFEST_SHA256:?APPROVED_EVALUATION_MANIFEST_SHA256 is required}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:?EVAL_CACHE_DIR is required}"
EVAL_CACHE_MANIFEST="${EVAL_CACHE_MANIFEST:?EVAL_CACHE_MANIFEST is required}"
APPROVED_EVAL_CACHE_SHA256="${APPROVED_EVAL_CACHE_SHA256:?APPROVED_EVAL_CACHE_SHA256 is required}"
MATRIX="configs/train/flow_math_llama32_1b/matrix.tsv"
EXPERIMENT_ROOT="/mnt/jfs-01/shared/gcw/experiments/flow_math_llama32_1b_v1"
BACKBONE_MODEL="/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf/model.safetensors"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
UV="$HOME/.local/bin/uv"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
test -x "$UV"
test -f "$TRAINING_EXPERIMENT_MANIFEST"
test -f "$EVALUATION_MANIFEST"
test -f "$EVAL_CACHE_MANIFEST"
test -f "$MATRIX"
[[ "$(git rev-parse HEAD)" == "$EVALUATION_SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
read -r ACTUAL_MANIFEST_SHA _ < <(sha256sum "$EVALUATION_MANIFEST")
[[ "$ACTUAL_MANIFEST_SHA" == "$APPROVED_EVALUATION_MANIFEST_SHA256" ]]
UV_RUN=("$UV" run --locked --extra gpu --extra eval)
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-evaluation-context \
  --repo-root "$PWD" \
  --evaluation-manifest "$EVALUATION_MANIFEST" \
  --evaluation-manifest-sha256 "$APPROVED_EVALUATION_MANIFEST_SHA256" \
  --evaluation-source-commit "$EVALUATION_SOURCE_COMMIT" \
  --training-experiment-manifest "$TRAINING_EXPERIMENT_MANIFEST" \
  --training-experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256" \
  --training-source-commit "$TRAINING_SOURCE_COMMIT"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-cache-manifest \
  --cache-dir "$EVAL_CACHE_DIR" \
  --manifest "$EVAL_CACHE_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" \
  --require-read-only
"${UV_RUN[@]}" python scripts/flow_math_experiment.py check-eval-pilot \
  --repo-root "$PWD" \
  --evaluation-manifest "$EVALUATION_MANIFEST" \
  --evaluation-manifest-sha256 "$APPROVED_EVALUATION_MANIFEST_SHA256" \
  --evaluation-source-commit "$EVALUATION_SOURCE_COMMIT" \
  --training-experiment-manifest "$TRAINING_EXPERIMENT_MANIFEST" \
  --training-experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256" \
  --training-source-commit "$TRAINING_SOURCE_COMMIT" \
  --eval-cache-dir "$EVAL_CACHE_DIR" \
  --eval-cache-manifest "$EVAL_CACHE_MANIFEST" \
  --eval-cache-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" \
  --cache-copy-evidence "$EXPERIMENT_ROOT/eval/cache-copy-evidence/pilot-train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json" \
  --marker "$EXPERIMENT_ROOT/markers/eval-pilot-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json"

TRAIN_MARKER_ARGS=()
if [[ "$TASK_ID" == "0" ]]; then
  TARGET="tf-backbone"
  CHECKPOINT="/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf"
  CHECKPOINT_SHA="$("${UV_RUN[@]}" python -c 'import sys; from pathlib import Path; from gcw.flow_math_experiment import sha256_file; print(sha256_file(Path(sys.argv[1])))' "$CHECKPOINT/model.safetensors")"
else
  MATRIX_INDEX=$((TASK_ID - 1))
  IFS=$'\t' read -r VARIANT SEED CONFIG < <(
    "${UV_RUN[@]}" python scripts/flow_math_experiment.py matrix-row --matrix "$MATRIX" --index "$MATRIX_INDEX"
  )
  OUTPUT_DIR="$("${UV_RUN[@]}" python -c 'import sys; from gcw.config import load_train_config; print(load_train_config(sys.argv[1]).output_dir)' "$CONFIG")"
  CHECKPOINT="$OUTPUT_DIR/checkpoint"
  TARGET="${VARIANT}-seed${SEED}"
  TRAIN_MARKER="$EXPERIMENT_ROOT/markers/train-${VARIANT}-seed${SEED}-${TRAINING_SOURCE_COMMIT:0:12}.json"
  TRAIN_MARKER_ARGS=(--training-marker "$TRAIN_MARKER")
  "${UV_RUN[@]}" python scripts/flow_math_experiment.py check-run-marker \
    --marker "$TRAIN_MARKER" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --source-commit "$TRAINING_SOURCE_COMMIT" \
    --experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256"
  CHECKPOINT_SHA="$("${UV_RUN[@]}" python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_model_sha256"])' "$TRAIN_MARKER")"
fi
CHECKPOINT_METADATA_SHA="$("${UV_RUN[@]}" python -c 'import sys; from pathlib import Path; from gcw.flow_math_experiment import checkpoint_metadata_sha256; print(checkpoint_metadata_sha256(Path(sys.argv[1])))' "$CHECKPOINT")"

test -d "$CHECKPOINT"
OUT="$EXPERIMENT_ROOT/eval/raw/${TARGET}-train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json"
MARKER="$EXPERIMENT_ROOT/markers/eval-${TARGET}-train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json"
RUNTIME_EVIDENCE="$EXPERIMENT_ROOT/eval/runtime/${TARGET}-train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json"
PARTIAL_DIR="$EXPERIMENT_ROOT/eval/partials/${TARGET}/train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}-${APPROVED_EVALUATION_MANIFEST_SHA256:0:12}-${CHECKPOINT_SHA:0:12}-${CHECKPOINT_METADATA_SHA:0:12}"
IDENTITY="$PARTIAL_DIR/identity.json"
PARTIAL_MANIFEST="$EXPERIMENT_ROOT/eval/partial-manifests/${TARGET}-train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json"
CACHE_COPY_EVIDENCE="$EXPERIMENT_ROOT/eval/cache-copy-evidence/${TARGET}-train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json"
JOB_CACHE_DIR="${SLURM_TMPDIR:-/tmp/${USER:?}}/flow-math-eval-cache-${SLURM_JOB_ID}-${TASK_ID}"
test ! -e "$OUT"
test ! -e "$MARKER"
test ! -e "$RUNTIME_EVIDENCE"
test ! -e "$PARTIAL_MANIFEST"
test ! -e "$CACHE_COPY_EVIDENCE"
test ! -e "$JOB_CACHE_DIR"
mkdir -p "$(dirname "$OUT")" "$(dirname "$PARTIAL_DIR")" "$EXPERIMENT_ROOT/markers" "$(dirname "$RUNTIME_EVIDENCE")" "$(dirname "$PARTIAL_MANIFEST")" "$(dirname "$CACHE_COPY_EVIDENCE")"
if [[ -d "$PARTIAL_DIR" ]]; then
  test -f "$IDENTITY"
else
  mkdir "$PARTIAL_DIR"
fi
"${UV_RUN[@]}" python - "$IDENTITY" "$TRAINING_SOURCE_COMMIT" "$APPROVED_TRAINING_MANIFEST_SHA256" "$EVALUATION_SOURCE_COMMIT" "$APPROVED_EVALUATION_MANIFEST_SHA256" "$APPROVED_EVAL_CACHE_SHA256" "$CHECKPOINT_SHA" "$CHECKPOINT_METADATA_SHA" "$TARGET" <<'PY'
import json, os
from pathlib import Path
import sys
path = Path(sys.argv[1])
payload = {
    "schema_version": 2,
    "kind": "flow_math_eval_partial",
    "training_source_commit": sys.argv[2],
    "training_experiment_manifest_sha256": sys.argv[3],
    "evaluation_source_commit": sys.argv[4],
    "evaluation_manifest_sha256": sys.argv[5],
    "eval_cache_manifest_sha256": sys.argv[6],
    "checkpoint_sha256": sys.argv[7],
    "checkpoint_metadata_sha256": sys.argv[8],
    "target": sys.argv[9],
    "batch_size": 16,
    "world_size": 1,
    "limit": None,
    "suite": "math",
}
if path.exists():
    with path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
        if handle.read().strip():
            raise ValueError("partial identity has trailing content")
    if existing != payload:
        raise ValueError("existing partial identity differs from this exact evaluation")
    for entry in path.parent.rglob("*"):
        if entry.is_symlink() or (not entry.is_file() and not entry.is_dir()):
            raise ValueError(f"partial directory contains unsafe entry: {entry}")
else:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
PY
chmod 0444 "$IDENTITY"

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
  --approved-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" --target "$TARGET"

export GCW_STRICT_FLASH=1
export GCW_EVAL_SKIP_TASKS="theoremqa"
export GCW_EVAL_REFERENCE_GENERATE=0
export PYTHONPATH=
export WANDB_MODE=offline
export GCW_EVAL_PARTIAL_DIR="$PARTIAL_DIR"
unset GCW_EVAL_PROGRESS_PATH
export HF_HOME="$JOB_CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$JOB_CACHE_DIR/huggingface/datasets"
export HF_HUB_CACHE="$JOB_CACHE_DIR/huggingface/hub"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

"${UV_RUN[@]}" python scripts/flow_math_eval_runner.py \
  --target "$TARGET" \
  --checkpoint "$CHECKPOINT" \
  --batch-size 16 \
  --out "$OUT" \
  --runtime-evidence "$RUNTIME_EVIDENCE" \
  --partial-identity "$IDENTITY" \
  --eval-cache-dir "$JOB_CACHE_DIR" \
  --array-task-id "$TASK_ID"

"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-cache-copy \
  --cache-dir "$JOB_CACHE_DIR" --manifest "$EVAL_CACHE_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" --target "$TARGET" \
  --out "$CACHE_COPY_EVIDENCE"

"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-evaluation-context \
  --repo-root "$PWD" \
  --evaluation-manifest "$EVALUATION_MANIFEST" \
  --evaluation-manifest-sha256 "$APPROVED_EVALUATION_MANIFEST_SHA256" \
  --evaluation-source-commit "$EVALUATION_SOURCE_COMMIT" \
  --training-experiment-manifest "$TRAINING_EXPERIMENT_MANIFEST" \
  --training-experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256" \
  --training-source-commit "$TRAINING_SOURCE_COMMIT"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-cache-manifest \
  --cache-dir "$EVAL_CACHE_DIR" \
  --manifest "$EVAL_CACHE_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" \
  --require-read-only
if [[ "$TASK_ID" != "0" ]]; then
  "${UV_RUN[@]}" python scripts/flow_math_experiment.py check-run-marker \
    --marker "$TRAIN_MARKER" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --source-commit "$TRAINING_SOURCE_COMMIT" \
    --experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256"
fi

"${UV_RUN[@]}" python - "$PARTIAL_DIR" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
for path in root.rglob("*"):
    if path.is_symlink() or (not path.is_file() and not path.is_dir()):
        raise SystemExit(f"invalid partial evidence entry: {path}")
    if path.is_file():
        path.chmod(0o444)
for path in sorted(root.rglob("*"), reverse=True):
    if path.is_dir():
        path.chmod(0o555)
root.chmod(0o555)
PY
"${UV_RUN[@]}" python scripts/flow_math_experiment.py build-cache-manifest \
  --cache-dir "$PARTIAL_DIR" --out "$PARTIAL_MANIFEST"

"${UV_RUN[@]}" python scripts/flow_math_experiment.py extract-metrics \
  --result "$OUT" \
  --target "$TARGET" \
  --training-source-commit "$TRAINING_SOURCE_COMMIT" \
  --training-experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256" \
  --evaluation-source-commit "$EVALUATION_SOURCE_COMMIT" \
  --evaluation-manifest-sha256 "$APPROVED_EVALUATION_MANIFEST_SHA256" \
  --eval-cache-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" \
  --eval-cache-dir "$JOB_CACHE_DIR" \
  --array-task-id "$TASK_ID" \
  --cache-copy-evidence "$CACHE_COPY_EVIDENCE" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --checkpoint-metadata-sha256 "$CHECKPOINT_METADATA_SHA" \
  --runtime-evidence "$RUNTIME_EVIDENCE" \
  --partial-identity "$IDENTITY" \
  --partial-manifest "$PARTIAL_MANIFEST" \
  "${TRAIN_MARKER_ARGS[@]}" \
  --out "$MARKER"



printf 'FLOW_MATH_EVAL_OK target=%s training=%s evaluation=%s marker=%s\n' "$TARGET" "$TRAINING_SOURCE_COMMIT" "$EVALUATION_SOURCE_COMMIT" "$MARKER"
