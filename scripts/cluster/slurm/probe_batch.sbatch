#!/usr/bin/env bash
#SBATCH --job-name=gcw-batch-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_out/batch-probe-%j.out

set -euo pipefail
if [[ $# -eq 0 ]]; then
  echo "Usage: sbatch $0 [probe options] config.yaml [config.yaml ...]" >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR:?}"
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
if [[ -d "$PWD/.venv/include/python3.12" ]]; then
  export CPATH="$PWD/.venv/include/python3.12:$PWD/.venv/include${CPATH:+:$CPATH}"
fi

printf 'job=%s node=%s started=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run --locked --no-sync --extra gpu --extra qwen35 python scripts/cluster/check_cuda_env.py
uv run --locked --no-sync --extra gpu --extra qwen35 python scripts/cluster/probe_batch.py "$@"
