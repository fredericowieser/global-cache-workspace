#!/usr/bin/env bash
#SBATCH --job-name=gcw-flow-smoke
#SBATCH --partition=agent-dev-15,agent-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail

: "${GCW_PROJECT_DIR:?submit with GCW_PROJECT_DIR set to the staged project directory}"
: "${GCW_SOURCE_COMMIT:?submit with GCW_SOURCE_COMMIT set to the tested source commit}"
cd "$GCW_PROJECT_DIR"
PYTHON="$GCW_PROJECT_DIR/.venv/bin/python"
test -x "$PYTHON" || { printf 'Missing interpreter: %s\n' "$PYTHON" >&2; exit 1; }

printf 'job=%s node=%s source_commit=%s gpus=%s started=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$GCW_SOURCE_COMMIT" "${SLURM_JOB_GPUS:-unknown}" "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader

srun "$PYTHON" - <<'PY'
import torch

print("torch", torch.__version__, "compiled_cuda", torch.version.cuda, flush=True)
assert torch.cuda.is_available(), "CUDA unavailable to the staged interpreter"
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
value = torch.ones(1, device="cuda")
print("cuda_tensor", value.item(), flush=True)
PY

srun "$PYTHON" scripts/flow_smoke.py --device cuda --flow-steps 2

SUCCESS_PATH="$GCW_PROJECT_DIR/gpu-smoke-$SLURM_JOB_ID.ok"
set -o noclobber
printf 'source_commit=%s completed=%s\n' "$GCW_SOURCE_COMMIT" "$(date -u +%FT%TZ)" > "$SUCCESS_PATH"
printf 'success_marker=%s\n' "$SUCCESS_PATH"
