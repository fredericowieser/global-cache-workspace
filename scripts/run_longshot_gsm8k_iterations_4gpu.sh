#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.local/bin:$PATH"

JOB_ID="${JOB_ID:-$(squeue -u "$USER" -n gcw-longshot-4gpu -t R -h -o %i | head -n1)}"
if [[ -z "$JOB_ID" ]]; then
  echo "No running gcw-longshot-4gpu allocation found. Submit scripts/slurm/longshot_4gpu_allocation.sbatch first." >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/mnt/jfs-01/shared/gcw/results/longshot_gsm8k_iterations/${RUN_ID}_4gpu}"
CHECKPOINT="${CHECKPOINT:-/mnt/jfs-01/shared/gcw-best_longshot}"
NUM_SHARDS="${NUM_SHARDS:-4}"
NUM_EXAMPLES="${NUM_EXAMPLES:-256}"
NUM_CALIBRATION="${NUM_CALIBRATION:-256}"
NUM_SPLITS="${NUM_SPLITS:-8}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"
ITERATION_LIST="${ITERATION_LIST:-}"
OLD_BOTTLENECK_PROTOCOL="${OLD_BOTTLENECK_PROTOCOL:-0}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-500}"
CPUS_PER_SHARD="${CPUS_PER_SHARD:-4}"

mkdir -p "$OUT_ROOT"
export OUT_ROOT CHECKPOINT NUM_SHARDS NUM_EXAMPLES NUM_CALIBRATION NUM_SPLITS MAX_ITERATIONS ITERATION_LIST OLD_BOTTLENECK_PROTOCOL BOOTSTRAP_SAMPLES
echo "job_id=$JOB_ID"
echo "out_root=$OUT_ROOT"
echo "checkpoint=$CHECKPOINT"
echo "num_shards=$NUM_SHARDS num_examples=$NUM_EXAMPLES num_calibration=$NUM_CALIBRATION num_splits=$NUM_SPLITS"
echo "old_bottleneck_protocol=$OLD_BOTTLENECK_PROTOCOL iteration_list=${ITERATION_LIST:-<dense>}"

if ! srun --jobid="$JOB_ID" --overlap --ntasks="$NUM_SHARDS" --gpus-per-task=1 --gpu-bind=single:1 --cpus-per-task="$CPUS_PER_SHARD" \
  bash -lc '
    set -euo pipefail
    shard="${SLURM_PROCID}"
    shard_dir="${OUT_ROOT}/shard_${shard}"
    mkdir -p "$shard_dir"
    exec > "${OUT_ROOT}/shard_${shard}.log" 2>&1
    cd '"$ROOT"'
    export PATH="$HOME/.local/bin:$PATH"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-'"$ROOT"'/.cache/hf_datasets}"
    export MPLBACKEND=Agg
    echo "shard=${shard} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset} host=$(hostname)"
    extra_args=()
    if [ "${OLD_BOTTLENECK_PROTOCOL}" = "1" ]; then
      extra_args+=(--old-bottleneck-protocol)
    fi
    if [ -n "${ITERATION_LIST}" ]; then
      extra_args+=(--iteration-list "${ITERATION_LIST}")
    fi
    uv run --extra gpu --extra eval python -m gcw.analysis.longshot_gsm8k_iterations \
      --checkpoint "$CHECKPOINT" \
      --out-dir "$shard_dir" \
      --num-examples "$NUM_EXAMPLES" \
      --num-calibration "$NUM_CALIBRATION" \
      --num-splits "$NUM_SPLITS" \
      --max-iterations "$MAX_ITERATIONS" \
      --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard" \
      "${extra_args[@]}"
  '; then
  echo "At least one shard failed. See $OUT_ROOT/shard_*.log" >&2
  exit 1
fi

merge_args=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  merge_args+=("$OUT_ROOT/shard_${shard}")
done

uv run --extra gpu --extra eval python -m gcw.analysis.longshot_gsm8k_iterations \
  --merge-shards "${merge_args[@]}" \
  --merge-out-dir "$OUT_ROOT/combined" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES"

echo "combined=$OUT_ROOT/combined"
