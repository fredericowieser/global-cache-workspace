#!/usr/bin/env bash
#SBATCH --job-name=gcw-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm_out/%x-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"

CHECKPOINT="${CHECKPOINT:-runs/smoke_tf/checkpoint}"
SUITE="${SUITE:-core}"
LIMIT="${LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
DATA_DIR="${DATA_DIR:-}"
EVAL_TOKENS="${EVAL_TOKENS:-}"
OUT="${OUT:-}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_out "$(dirname "${OUT:-results/tmp.json}")"

cmd=(uv run --extra gpu --extra eval gcw eval "$SUITE" --checkpoint "$CHECKPOINT")
if [[ -n "$LIMIT" && "$SUITE" != "fineweb" ]]; then
  cmd+=(--limit "$LIMIT")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  cmd+=(--batch-size "$BATCH_SIZE")
fi
if [[ "$SUITE" == "fineweb" && -n "$DATA_DIR" ]]; then
  cmd+=(--data-dir "$DATA_DIR")
fi
if [[ "$SUITE" == "fineweb" && -n "$EVAL_TOKENS" ]]; then
  cmd+=(--eval-tokens "$EVAL_TOKENS")
fi
if [[ -n "$OUT" ]]; then
  cmd+=(--out "$OUT")
fi

"${cmd[@]}"
if [[ -n "$OUT" ]]; then
  echo "Wrote $OUT"
else
  true
fi
