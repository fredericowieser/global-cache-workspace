#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/jfs-01/shared/gcw}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT/results/head_loss_dump/$RUN_ID}"
PARTITION="${PARTITION:-agent-xlong-15,agent-xlong}"
TIME="${TIME:-48:00:00}"
FAMILIES="${FAMILIES:-llama32_1b llama32_3b llama31_8b qwen3_06b}"

mkdir -p slurm_out

family_count=$(wc -w <<< "$FAMILIES")
array_end=$((family_count - 1))

echo "Writing raw head-loss dump to: $OUT_DIR"
sbatch \
  --partition="$PARTITION" \
  --time="$TIME" \
  --array="0-$array_end" \
  --job-name="${JOB_NAME:-gcw-head-loss-dump}" \
  --export=ALL,ROOT="$ROOT",RUN_ID="$RUN_ID",OUT_DIR="$OUT_DIR",FAMILIES="$FAMILIES" \
  scripts/slurm/head_loss_dump.sbatch
