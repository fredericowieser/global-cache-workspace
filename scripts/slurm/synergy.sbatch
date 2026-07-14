#!/usr/bin/env bash
#SBATCH --job-name=gcw-head-synergy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --partition=agent-xlong-15,agent-xlong
#SBATCH --output=slurm_out/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PATH="$HOME/.local/bin:$PATH"
export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/jfs-01/shared/gcw/hf_datasets_eval_v3}"

ROOT="${ROOT:-/mnt/jfs-01/shared/gcw}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT/results/head_synergy/$RUN_ID}"
TASKS="${TASKS:-gsm8k_openmath svamp gsm8k_hard logiqa agieval_gaokao_mathqa}"
FAMILIES="${FAMILIES:-llama32_1b llama32_3b llama31_8b qwen3_06b}"
VARIANTS="${VARIANTS:-sft bt bgt}"
NUM_EXAMPLES="${NUM_EXAMPLES:-256}"
NUM_CALIBRATION="${NUM_CALIBRATION:-512}"
NUM_SPLITS="${NUM_SPLITS:-64}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
NORM_EPS="${NORM_EPS:-1e-6}"
GROUP_FRACTION="${GROUP_FRACTION:-0.5}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_TARGET_TOKENS="${MAX_TARGET_TOKENS:-512}"
SEED="${SEED:-42}"
LIMIT_CHECKPOINTS="${LIMIT_CHECKPOINTS:-}"

mkdir -p slurm_out "$OUT_DIR"
read -r -a task_array <<< "$TASKS"
read -r -a family_array <<< "$FAMILIES"
read -r -a variant_array <<< "$VARIANTS"

args=(
  -m gcw.analysis.head_synergy
  --root "$ROOT"
  --out-dir "$OUT_DIR"
  --tasks "${task_array[@]}"
  --families "${family_array[@]}"
  --variants "${variant_array[@]}"
  --num-examples "$NUM_EXAMPLES"
  --num-calibration "$NUM_CALIBRATION"
  --num-splits "$NUM_SPLITS"
  --bootstrap-samples "$BOOTSTRAP_SAMPLES"
  --norm-eps "$NORM_EPS"
  --group-fraction "$GROUP_FRACTION"
  --max-length "$MAX_LENGTH"
  --max-target-tokens "$MAX_TARGET_TOKENS"
  --seed "$SEED"
  --device cuda
)

if [[ -n "$LIMIT_CHECKPOINTS" ]]; then
  args+=(--limit-checkpoints "$LIMIT_CHECKPOINTS")
fi

uv run --extra gpu --extra eval python "${args[@]}"
chmod -R a+rX "$OUT_DIR"
echo "Wrote $OUT_DIR"
