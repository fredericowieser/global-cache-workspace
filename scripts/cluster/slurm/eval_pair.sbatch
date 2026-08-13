#!/usr/bin/env bash
# Run two gcw evals sequentially on one GPU.

#SBATCH --job-name=gcw-eval-pair
#SBATCH --partition=agent-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_out/%x-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"

SUITE="${SUITE:-math}"
LIMIT="${LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_out "$(dirname "${FIRST_OUT:?FIRST_OUT is required}")" "$(dirname "${SECOND_OUT:?SECOND_OUT is required}")"

run_eval() {
  local checkpoint="$1"
  local out="$2"
  local cmd=(uv run --extra gpu --extra eval gcw eval "$SUITE" --checkpoint "$checkpoint")
  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi
  if [[ -n "$BATCH_SIZE" ]]; then
    cmd+=(--batch-size "$BATCH_SIZE")
  fi
  cmd+=(--out "$out")
  "${cmd[@]}"
  echo "Wrote $out"
}

run_eval "${FIRST_CHECKPOINT:?FIRST_CHECKPOINT is required}" "$FIRST_OUT"
run_eval "${SECOND_CHECKPOINT:?SECOND_CHECKPOINT is required}" "$SECOND_OUT"
