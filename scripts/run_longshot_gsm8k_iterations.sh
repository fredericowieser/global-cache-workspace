#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p slurm_out

CHECKPOINT="${CHECKPOINT:-/mnt/jfs-01/shared/gcw-best_longshot}" \
OUT_DIR="${OUT_DIR:-/mnt/jfs-01/shared/gcw/results/longshot_gsm8k_iterations/$(date -u +%Y%m%d-%H%M%S)}" \
NUM_EXAMPLES="${NUM_EXAMPLES:-256}" \
NUM_CALIBRATION="${NUM_CALIBRATION:-256}" \
NUM_SPLITS="${NUM_SPLITS:-8}" \
MAX_ITERATIONS="${MAX_ITERATIONS:-10}" \
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-500}" \
sbatch scripts/slurm/longshot_gsm8k_iterations.sbatch

