#!/usr/bin/env bash
#SBATCH --job-name=flow-math-eval-cache
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
EXPERIMENT_ROOT="/mnt/jfs-01/shared/gcw/experiments/flow_math_llama32_1b_v1"
UV="$HOME/.local/bin/uv"
IDENTITY="train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}"
WARMUP_RESULT="$EXPERIMENT_ROOT/eval/cache-warmup/tf-backbone-$IDENTITY.json"
CACHE_MARKER="$EXPERIMENT_ROOT/markers/eval-cache-$IDENTITY.json"
LOCK_INVENTORY="$EXPERIMENT_ROOT/eval/cache-lock-inventory-$IDENTITY.json"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
test -x "$UV"
test ! -e "$EVAL_CACHE_DIR"
test ! -e "$EVAL_CACHE_MANIFEST"
test ! -e "$WARMUP_RESULT"
test ! -e "$CACHE_MARKER"
test ! -e "$LOCK_INVENTORY"
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

mkdir -p "$EVAL_CACHE_DIR/huggingface/datasets" "$EVAL_CACHE_DIR/huggingface/hub" "$(dirname "$WARMUP_RESULT")" "$EXPERIMENT_ROOT/markers"
export GCW_STRICT_FLASH=1
export GCW_EVAL_SKIP_TASKS="theoremqa"
export GCW_EVAL_REFERENCE_GENERATE=0
export PYTHONPATH=
unset GCW_EVAL_PARTIAL_DIR
unset GCW_EVAL_PROGRESS_PATH
export HF_HOME="$EVAL_CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$EVAL_CACHE_DIR/huggingface/datasets"
export HF_HUB_CACHE="$EVAL_CACHE_DIR/huggingface/hub"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

"${UV_RUN[@]}" python -m gcw.cli eval math \
  --checkpoint /mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf \
  --batch-size 16 --limit 1 --out "$WARMUP_RESULT"

chmod 0444 "$WARMUP_RESULT"
"${UV_RUN[@]}" python - "$EVAL_CACHE_DIR" "$LOCK_INVENTORY" "$TRAINING_SOURCE_COMMIT" "$EVALUATION_SOURCE_COMMIT" <<'PY'
from pathlib import Path
import hashlib
import json
import os
import sys
root = Path(sys.argv[1])
inventory_path = Path(sys.argv[2])
paths = sorted(root.rglob("*"))
locks = []
for path in paths:
    relative = path.relative_to(root)
    if path.is_symlink():
        raise SystemExit(f"evaluation cache contains symlink: {relative}")
    if path.is_dir():
        continue
    if not path.is_file():
        raise SystemExit(f"evaluation cache contains special entry: {relative}")
    if any(part.endswith((".incomplete", ".tmp")) for part in relative.parts):
        raise SystemExit(f"evaluation cache contains incomplete file: {relative}")
    if path.suffix == ".lock" or ".locks" in relative.parts:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        locks.append({"path": relative.as_posix(), "size_bytes": path.stat().st_size, "sha256": digest})
payload = {
    "schema_version": 1,
    "kind": "flow_math_eval_cache_lock_inventory",
    "training_source_commit": sys.argv[3],
    "evaluation_source_commit": sys.argv[4],
    "quiescence": "warmup evaluator exited before lock inventory and cleanup",
    "locks": locks,
}
inventory_path.parent.mkdir(parents=True, exist_ok=True)
fd = os.open(inventory_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
with os.fdopen(fd, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
for item in locks:
    (root / item["path"]).unlink()
for path in reversed(paths):
    relative = path.relative_to(root)
    if path.is_dir() and ".locks" in relative.parts:
        path.rmdir()
paths = sorted(root.rglob("*"))
for path in paths:
    if path.is_file():
        path.chmod(0o444)
for path in reversed(paths):
    if path.is_dir():
        path.chmod(0o555)
root.chmod(0o555)
PY
"${UV_RUN[@]}" python scripts/flow_math_experiment.py build-cache-manifest \
  --cache-dir "$EVAL_CACHE_DIR" --out "$EVAL_CACHE_MANIFEST"
read -r CACHE_SHA _ < <(sha256sum "$EVAL_CACHE_MANIFEST")
"${UV_RUN[@]}" python scripts/flow_math_experiment.py verify-cache-manifest \
  --cache-dir "$EVAL_CACHE_DIR" --manifest "$EVAL_CACHE_MANIFEST" \
  --approved-manifest-sha256 "$CACHE_SHA" --require-read-only
"${UV_RUN[@]}" python scripts/flow_math_experiment.py finalize-eval-cache \
  --repo-root "$PWD" --evaluation-manifest "$EVALUATION_MANIFEST" \
  --evaluation-manifest-sha256 "$APPROVED_EVALUATION_MANIFEST_SHA256" \
  --evaluation-source-commit "$EVALUATION_SOURCE_COMMIT" \
  --training-experiment-manifest "$TRAINING_EXPERIMENT_MANIFEST" \
  --training-experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256" \
  --training-source-commit "$TRAINING_SOURCE_COMMIT" \
  --eval-cache-dir "$EVAL_CACHE_DIR" --eval-cache-manifest "$EVAL_CACHE_MANIFEST" \
  --eval-cache-manifest-sha256 "$CACHE_SHA" --warmup-result "$WARMUP_RESULT" \
  --lock-inventory "$LOCK_INVENTORY" --out "$CACHE_MARKER"
printf 'FLOW_MATH_EVAL_CACHE_OK evaluation=%s cache_sha256=%s marker=%s\n' "$EVALUATION_SOURCE_COMMIT" "$CACHE_SHA" "$CACHE_MARKER"
