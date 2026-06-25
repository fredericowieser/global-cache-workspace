#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/train/bottleneck_llm/llama32_1b_icarus_from_sft_frozen_shared.yaml}"
JOB_NAME="${JOB_NAME:-gcw-llama32-1b-icarus}"
PARTITION="${PARTITION:-agent-xlong-15,agent-xlong}"
TIME="${TIME:-24:00:00}"
GPUS="${GPUS:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-180G}"
WANDB_MODE="${WANDB_MODE:-online}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
UV_EXTRAS="${UV_EXTRAS:---extra gpu --extra eval}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-1}"
EVAL_SUITE="${EVAL_SUITE:-paper7}"
EVAL_GPUS="${EVAL_GPUS:-1}"
EVAL_CPUS_PER_TASK="${EVAL_CPUS_PER_TASK:-8}"
EVAL_MEM="${EVAL_MEM:-120G}"
EVAL_TIME="${EVAL_TIME:-12:00:00}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GCW_EVAL_SKIP_TASKS="${GCW_EVAL_SKIP_TASKS:-theoremqa}"

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

OUTPUT_DIR="${OUTPUT_DIR:-$(awk '$1 == "output_dir:" {print $2; exit}' "$CONFIG")}"
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Could not determine output_dir from $CONFIG" >&2
  exit 1
fi
RUN_NAME="$(basename "$OUTPUT_DIR")"

train_job="$(
  sbatch \
    --parsable \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --time="$TIME" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres="gpu:$GPUS" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --output="slurm_out/llama32-1b-icarus-%j.out" \
    --export="ALL,CONFIG=$CONFIG,WANDB_MODE=$WANDB_MODE,GPUS=$GPUS,PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF,UV_EXTRAS=$UV_EXTRAS" \
    scripts/slurm/train.sbatch
)"
echo "Submitted training job $train_job"

if [[ "$EVAL_AFTER_TRAIN" == "1" ]]; then
  CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/checkpoint}"
  OUT="${OUT:-results/${RUN_NAME}_${EVAL_SUITE}.json}"
  mkdir -p "$(dirname "$OUT")"
  eval_job="$(
    sbatch \
      --parsable \
      --dependency="afterok:$train_job" \
      --job-name="${JOB_NAME}-eval" \
      --partition="$PARTITION" \
      --time="$EVAL_TIME" \
      --nodes=1 \
      --ntasks-per-node="$EVAL_GPUS" \
      --gres="gpu:$EVAL_GPUS" \
      --cpus-per-task="$EVAL_CPUS_PER_TASK" \
      --mem="$EVAL_MEM" \
      --output="slurm_out/${RUN_NAME}-${EVAL_SUITE}-%j.out" \
      --export="ALL,CHECKPOINT=$CHECKPOINT,SUITE=$EVAL_SUITE,BATCH_SIZE=$EVAL_BATCH_SIZE,OUT=$OUT,GCW_EVAL_SKIP_TASKS=$GCW_EVAL_SKIP_TASKS" \
      scripts/slurm/eval.sbatch
  )"
  echo "Submitted eval job $eval_job after training job $train_job"
  echo "Eval checkpoint: $CHECKPOINT"
  echo "Eval output: $OUT"
fi
