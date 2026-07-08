#!/usr/bin/env bash
#SBATCH --job-name=gcw-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

CONFIG="${CONFIG:-configs/train/smollm360m_tf_pretrain.yaml}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PATH="$HOME/.local/bin:$PATH"
export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
mkdir -p "$WANDB_DIR"
uv run --extra gpu torchrun --nproc_per_node="${SLURM_NTASKS_PER_NODE:-8}" -m gcw.cli train --config "$CONFIG"
