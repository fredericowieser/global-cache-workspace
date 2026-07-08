#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/jfs-01/shared/gcw}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
PARTITION="${PARTITION:-agent-xlong-15,agent-xlong}"
TIME="${TIME:-48:00:00}"

mkdir -p slurm_out

sbatch \
  --partition="$PARTITION" \
  --time="$TIME" \
  --job-name="${JOB_NAME:-gcw-head-synergy}" \
  --export=ALL,ROOT="$ROOT",RUN_ID="$RUN_ID" \
  scripts/slurm/synergy.sbatch

