#!/usr/bin/env bash
#SBATCH --job-name=gcw-head-loss-dump
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --partition=agent-xlong-15,agent-xlong
#SBATCH --array=0-3
#SBATCH --output=slurm_out/%x-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PATH="$HOME/.local/bin:$PATH"
export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/jfs-01/shared/gcw/hf_datasets_eval_v3}"

ROOT="${ROOT:-/mnt/jfs-01/shared/gcw}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT/results/head_loss_dump/$RUN_ID}"
TASKS="${TASKS:-gsm8k_openmath svamp gsm8k_hard logiqa agieval_gaokao_mathqa}"
FAMILIES="${FAMILIES:-llama32_1b llama32_3b llama31_8b qwen3_06b}"
VARIANTS="${VARIANTS:-sft bt bgt}"
NUM_EXAMPLES="${NUM_EXAMPLES:-256}"
NUM_CALIBRATION="${NUM_CALIBRATION:-512}"
NUM_SPLITS="${NUM_SPLITS:-64}"
GROUP_FRACTION="${GROUP_FRACTION:-0.5}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_TARGET_TOKENS="${MAX_TARGET_TOKENS:-512}"
SEED="${SEED:-42}"
LIMIT_CHECKPOINTS="${LIMIT_CHECKPOINTS:-}"
EXAMPLES_JSONL="${EXAMPLES_JSONL:-}"

read -r -a family_array <<< "$FAMILIES"
read -r -a task_array <<< "$TASKS"
read -r -a variant_array <<< "$VARIANTS"
family="${family_array[$SLURM_ARRAY_TASK_ID]}"

mkdir -p slurm_out "$OUT_DIR"

args=(
  -m gcw.analysis.head_loss_dump
  --root "$ROOT"
  --out-dir "$OUT_DIR"
  --tasks "${task_array[@]}"
  --families "$family"
  --variants "${variant_array[@]}"
  --num-examples "$NUM_EXAMPLES"
  --num-calibration "$NUM_CALIBRATION"
  --num-splits "$NUM_SPLITS"
  --group-fraction "$GROUP_FRACTION"
  --max-length "$MAX_LENGTH"
  --max-target-tokens "$MAX_TARGET_TOKENS"
  --seed "$SEED"
  --device cuda
)

if [[ -n "$LIMIT_CHECKPOINTS" ]]; then
  args+=(--limit-checkpoints "$LIMIT_CHECKPOINTS")
fi
if [[ -n "$EXAMPLES_JSONL" ]]; then
  args+=(--examples-jsonl "$EXAMPLES_JSONL")
fi

uv run --extra gpu --extra eval python "${args[@]}"
chmod -R a+rX "$OUT_DIR"
echo "Wrote $OUT_DIR/$family"
