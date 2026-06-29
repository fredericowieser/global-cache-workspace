#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/train/bottleneck_llm/llama32_1b_icarus_from_sft_frozen.yaml}"
GPUS="${GPUS:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"

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

export PATH="$HOME/.local/bin:$PATH"
export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_MODE
mkdir -p "$WANDB_DIR"

if [[ "$GPUS" == "1" ]]; then
  uv run --extra gpu gcw train --config "$CONFIG"
else
  uv run --extra gpu torchrun --standalone --nproc_per_node="$GPUS" -m gcw.cli train --config "$CONFIG"
fi
