#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/train/bottleneck_llm/llama32_1b_icarus_from_sft_frozen.yaml}"
JOB_NAME="${JOB_NAME:-gcw-llama32-1b-icarus}"
PARTITION="${PARTITION:-agent-xlong-15,agent-xlong}"
TIME="${TIME:-24:00:00}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-180G}"
WANDB_MODE="${WANDB_MODE:-online}"

OPENMATH_DATA="/mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl"
SFT_BACKBONE="/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf/model.safetensors"
TOKENIZER_DIR="/mnt/jfs-01/shared/bottleneck_llm/sft/llama-3.2-1b/final_model"

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
}

require_path "$CONFIG"
require_path "$OPENMATH_DATA"
require_path "$SFT_BACKBONE"
require_path "$TOKENIZER_DIR"

mkdir -p slurm_out

sbatch \
  --job-name="$JOB_NAME" \
  --partition="$PARTITION" \
  --time="$TIME" \
  --nodes=1 \
  --ntasks-per-node="$GPUS" \
  --gres="gpu:$GPUS" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM" \
  --output="slurm_out/llama32-1b-icarus-%j.out" \
  --export="ALL,CONFIG=$CONFIG,WANDB_MODE=$WANDB_MODE" \
  scripts/slurm/train.sbatch
