#!/usr/bin/env bash
#SBATCH --job-name=gcw-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

CONFIG="${CONFIG:-configs/train/smollm360m_tf_pretrain.yaml}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PATH="$HOME/.local/bin:$PATH"
export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
mkdir -p "$WANDB_DIR"

NPROC="${GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
UV_EXTRAS="${UV_EXTRAS:---extra gpu}"
read -r -a UV_EXTRA_ARGS <<< "$UV_EXTRAS"
UV_RUN=(uv run --locked "${UV_EXTRA_ARGS[@]}")
"${UV_RUN[@]}" python scripts/check_cuda_env.py
if [[ "$NPROC" == "1" ]]; then
  "${UV_RUN[@]}" gcw train --config "$CONFIG"
else
  "${UV_RUN[@]}" torchrun --standalone --nproc_per_node="$NPROC" -m gcw.cli train --config "$CONFIG"
fi
