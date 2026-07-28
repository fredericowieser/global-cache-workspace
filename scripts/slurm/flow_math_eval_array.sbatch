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

SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
EXPERIMENT_MANIFEST="${EXPERIMENT_MANIFEST:?EXPERIMENT_MANIFEST is required}"
APPROVED_MANIFEST_SHA256="${APPROVED_MANIFEST_SHA256:?APPROVED_MANIFEST_SHA256 is required}"
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
test -f "$EXPERIMENT_MANIFEST"
test -f "$EVAL_CACHE_MANIFEST"
test -f "$MATRIX"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
read -r ACTUAL_MANIFEST_SHA _ < <(sha256sum "$EXPERIMENT_MANIFEST")
[[ "$ACTUAL_MANIFEST_SHA" == "$APPROVED_MANIFEST_SHA256" ]]
UV_RUN=("$UV" run --locked --extra gpu --extra eval)
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-context \
  --repo-root "$PWD" \
  --manifest "$EXPERIMENT_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_MANIFEST_SHA256" \
  --source-commit "$SOURCE_COMMIT"
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-cache-manifest \
  --cache-dir "$EVAL_CACHE_DIR" \
  --manifest "$EVAL_CACHE_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" \
  --require-read-only
"${UV_RUN[@]}" python scripts/flow_math_experiment.py check-eval-pilot \
  --repo-root "$PWD" \
  --experiment-manifest "$EXPERIMENT_MANIFEST" \
  --experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256" \
  --source-commit "$SOURCE_COMMIT" \
  --eval-cache-dir "$EVAL_CACHE_DIR" \
  --eval-cache-manifest "$EVAL_CACHE_MANIFEST" \
  --eval-cache-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" \
  --marker "$EXPERIMENT_ROOT/markers/eval-pilot-${SOURCE_COMMIT:0:12}.json"

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
  TRAIN_MARKER="$EXPERIMENT_ROOT/markers/train-${VARIANT}-seed${SEED}-${SOURCE_COMMIT:0:12}.json"
  "${UV_RUN[@]}" python scripts/flow_math_experiment.py check-run-marker \
    --marker "$TRAIN_MARKER" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --source-commit "$SOURCE_COMMIT" \
    --experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256"
  CHECKPOINT_SHA="$("${UV_RUN[@]}" python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_model_sha256"])' "$TRAIN_MARKER")"
fi
CHECKPOINT_METADATA_SHA="$("${UV_RUN[@]}" python -c 'import sys; from pathlib import Path; from gcw.flow_math_experiment import checkpoint_metadata_sha256; print(checkpoint_metadata_sha256(Path(sys.argv[1])))' "$CHECKPOINT")"

test -d "$CHECKPOINT"
OUT="$EXPERIMENT_ROOT/eval/raw/${TARGET}-${SOURCE_COMMIT:0:12}-${APPROVED_MANIFEST_SHA256:0:12}.json"
MARKER="$EXPERIMENT_ROOT/markers/eval-${TARGET}-${SOURCE_COMMIT:0:12}.json"
PARTIAL_DIR="$EXPERIMENT_ROOT/eval/partials/${TARGET}/${SOURCE_COMMIT:0:12}-${APPROVED_MANIFEST_SHA256:0:12}-${CHECKPOINT_SHA:0:12}-${CHECKPOINT_METADATA_SHA:0:12}"
IDENTITY="$PARTIAL_DIR/identity.json"
test ! -e "$OUT"
test ! -e "$MARKER"
mkdir -p "$(dirname "$OUT")" "$PARTIAL_DIR" "$EXPERIMENT_ROOT/markers"
"${UV_RUN[@]}" python - "$IDENTITY" "$SOURCE_COMMIT" "$APPROVED_MANIFEST_SHA256" "$APPROVED_EVAL_CACHE_SHA256" "$CHECKPOINT_SHA" "$CHECKPOINT_METADATA_SHA" <<'PY'
import json, os
from pathlib import Path
import sys
path = Path(sys.argv[1])
payload = {
    "source_commit": sys.argv[2],
    "experiment_manifest_sha256": sys.argv[3],
    "eval_cache_manifest_sha256": sys.argv[4],
    "checkpoint_sha256": sys.argv[5],
    "checkpoint_metadata_sha256": sys.argv[6],
}
if path.exists():
    if json.loads(path.read_text()) != payload:
        raise SystemExit("stale partial evaluation identity")
else:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
PY

export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"
export GCW_EVAL_SKIP_TASKS="theoremqa"
export GCW_EVAL_PARTIAL_DIR="$PARTIAL_DIR"
export HF_HOME="$EVAL_CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$EVAL_CACHE_DIR/huggingface/datasets"
export HF_HUB_CACHE="$EVAL_CACHE_DIR/huggingface/hub"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${UV_RUN[@]}" python scripts/check_cuda_env.py
"${UV_RUN[@]}" python -m torch.distributed.run --standalone --nproc-per-node=2 -m gcw.cli eval math \
  --checkpoint "$CHECKPOINT" \
  --batch-size 16 \
  --out "$OUT"

"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-context \
  --repo-root "$PWD" \
  --manifest "$EXPERIMENT_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_MANIFEST_SHA256" \
  --source-commit "$SOURCE_COMMIT"
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
    --source-commit "$SOURCE_COMMIT" \
    --experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256"
fi

"${UV_RUN[@]}" python scripts/flow_math_experiment.py extract-metrics \
  --result "$OUT" \
  --target "$TARGET" \
  --source-commit "$SOURCE_COMMIT" \
  --experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256" \
  --eval-cache-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --checkpoint-metadata-sha256 "$CHECKPOINT_METADATA_SHA" \
  --out "$MARKER"

printf 'FLOW_MATH_EVAL_OK target=%s source=%s marker=%s\n' "$TARGET" "$SOURCE_COMMIT" "$MARKER"
