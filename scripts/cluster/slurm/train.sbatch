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
if [[ -d "$PWD/.venv/include/python3.12" ]]; then
  export CPATH="$PWD/.venv/include/python3.12:$PWD/.venv/include${CPATH:+:$CPATH}"
fi

config_max_steps() {
  awk '$1 == "max_steps:" {print $2; exit}' "$1"
}

positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
    echo "$name must be a positive integer; got '$value'" >&2
    exit 1
  fi
}

export PATH="$HOME/.local/bin:$PATH"
export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
KEEP_STEP_CHECKPOINTS="${KEEP_STEP_CHECKPOINTS:-auto}"
KEEP_STEP_CHECKPOINT_EVERY="${KEEP_STEP_CHECKPOINT_EVERY:-1000}"
positive_int KEEP_STEP_CHECKPOINT_EVERY "$KEEP_STEP_CHECKPOINT_EVERY"
if [[ "$KEEP_STEP_CHECKPOINTS" == "auto" ]]; then
  max_steps="$(config_max_steps "$CONFIG")"
  if [[ "$max_steps" =~ ^[0-9]+$ ]] && [[ "$max_steps" -gt "$KEEP_STEP_CHECKPOINT_EVERY" ]]; then
    KEEP_STEP_CHECKPOINTS=1
  else
    KEEP_STEP_CHECKPOINTS=0
  fi
fi
export GCW_KEEP_STEP_CHECKPOINTS="$KEEP_STEP_CHECKPOINTS"
export GCW_STEP_CHECKPOINT_EVERY="$KEEP_STEP_CHECKPOINT_EVERY"
mkdir -p "$WANDB_DIR"

NPROC="${GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
UV_EXTRAS="${UV_EXTRAS:---extra gpu}"
read -r -a UV_EXTRA_ARGS <<< "$UV_EXTRAS"
UV_RUN=(uv run --locked "${UV_EXTRA_ARGS[@]}")
"${UV_RUN[@]}" python scripts/cluster/check_cuda_env.py
if [[ "$NPROC" == "1" ]]; then
  "${UV_RUN[@]}" gcw train --config "$CONFIG"
else
  "${UV_RUN[@]}" torchrun --standalone --nproc_per_node="$NPROC" -m gcw.cli train --config "$CONFIG"
fi
