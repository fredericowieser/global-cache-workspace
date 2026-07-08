#!/bin/bash
#SBATCH --job-name=gcw-longshot-gsm8k-iters
#SBATCH --partition=agent-long-15,agent-long
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --output=slurm_out/longshot-gsm8k-iters-%j.out

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.cache/hf_datasets}"
export MPLBACKEND=Agg

CHECKPOINT="${CHECKPOINT:-/mnt/jfs-01/shared/gcw-best_longshot}"
OUT_DIR="${OUT_DIR:-/mnt/jfs-01/shared/gcw/results/longshot_gsm8k_iterations/$(date -u +%Y%m%d-%H%M%S)}"
NUM_EXAMPLES="${NUM_EXAMPLES:-256}"
NUM_CALIBRATION="${NUM_CALIBRATION:-256}"
NUM_SPLITS="${NUM_SPLITS:-8}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-500}"
BATCH_FLAGS="${BATCH_FLAGS:-}"

mkdir -p "$OUT_DIR" slurm_out

echo "checkpoint=$CHECKPOINT"
echo "out_dir=$OUT_DIR"
echo "num_examples=$NUM_EXAMPLES num_calibration=$NUM_CALIBRATION num_splits=$NUM_SPLITS max_iterations=$MAX_ITERATIONS"

uv run --extra gpu --extra eval python -m gcw.analysis.longshot_gsm8k_iterations \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT_DIR" \
  --num-examples "$NUM_EXAMPLES" \
  --num-calibration "$NUM_CALIBRATION" \
  --num-splits "$NUM_SPLITS" \
  --max-iterations "$MAX_ITERATIONS" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  $BATCH_FLAGS

