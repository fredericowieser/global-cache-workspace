#!/usr/bin/env bash
#SBATCH --job-name=gcw-bt-pe512-20b
#SBATCH --partition=agent-xlong-15,agent-xlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm_out/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PATH="$HOME/.local/bin:$PATH"
export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
mkdir -p "$WANDB_DIR" slurm_out
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
uv run --extra gpu torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" -m gcw.cli train --config configs/train/smollm360m_matched_bt_20b_pe512.yaml
