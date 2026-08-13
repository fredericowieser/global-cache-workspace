#!/usr/bin/env bash
#SBATCH --job-name=gcw-qwen-gsm8k
#SBATCH --partition=agent-xlong-15,agent-xlong
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00

set -euo pipefail

SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
PROJECT="$HOME/code/global-cache-workspace"
PYTHON="$PROJECT/.venv/bin/python"
MODEL_PATH="/mnt/jfs-01/shared/gcw/hf/Qwen3-0.6B-base-v1"
RUN_ROOT="$HOME/gcw-runs/qwen3-06b-gsm8k-$SOURCE_COMMIT-$SLURM_JOB_ID"

cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT" || {
  printf 'Wrong source commit: expected %s, found %s\n' "$SOURCE_COMMIT" "$(git rev-parse HEAD)" >&2
  exit 3
}
test -z "$(git status --porcelain)" || {
  printf 'Checkout is dirty; refusing GPU evaluation\n' >&2
  git status --short >&2
  exit 3
}
test -x "$PYTHON" || { printf 'Missing interpreter: %s\n' "$PYTHON" >&2; exit 3; }
test -f "$MODEL_PATH/model.safetensors" || { printf 'Missing model: %s\n' "$MODEL_PATH" >&2; exit 3; }
test ! -e "$RUN_ROOT" || { printf 'Run root already exists: %s\n' "$RUN_ROOT" >&2; exit 3; }
mkdir -p "$RUN_ROOT"

printf 'job=%s node=%s source=%s run_root=%s started=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$SOURCE_COMMIT" "$RUN_ROOT" "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
"$PYTHON" - <<'PY'
import torch
print("torch", torch.__version__, "compiled_cuda", torch.version.cuda)
assert torch.cuda.is_available()
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("cuda_tensor", torch.ones(1, device="cuda").item())
PY

"$PYTHON" scripts/analysis/run_qwen_gsm8k.py \
  --model-path "$MODEL_PATH" \
  --output-dir "$RUN_ROOT/smoke-off" \
  --limit 8 \
  --batch-size 1 \
  --modes thinking_off

"$PYTHON" scripts/analysis/run_qwen_gsm8k.py \
  --model-path "$MODEL_PATH" \
  --output-dir "$RUN_ROOT/smoke-on" \
  --limit 8 \
  --batch-size 1 \
  --modes thinking_on

"$PYTHON" scripts/analysis/run_qwen_gsm8k.py \
  --model-path "$MODEL_PATH" \
  --output-dir "$RUN_ROOT/full" \
  --batch-size 16

printf 'evaluation_complete=%s\n' "$(date -u +%FT%TZ)"
printf 'Persistent allocation remains available; use srun --jobid=%s --overlap for reviewed follow-up work.\n' "$SLURM_JOB_ID"
exec sleep infinity
