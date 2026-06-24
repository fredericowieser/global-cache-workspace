#!/usr/bin/env bash
#SBATCH --job-name=gcw-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
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

gcw_args=(eval "$SUITE" --checkpoint "$CHECKPOINT")
if [[ -n "$LIMIT" && "$SUITE" != "fineweb" ]]; then
  gcw_args+=(--limit "$LIMIT")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  gcw_args+=(--batch-size "$BATCH_SIZE")
fi
if [[ "$SUITE" == "fineweb" && -n "$DATA_DIR" ]]; then
  gcw_args+=(--data-dir "$DATA_DIR")
fi
if [[ "$SUITE" == "fineweb" && -n "$EVAL_TOKENS" ]]; then
  gcw_args+=(--eval-tokens "$EVAL_TOKENS")
fi
if [[ -n "$OUT" ]]; then
  gcw_args+=(--out "$OUT")
fi

UV_RUN=(uv run --locked --extra gpu --extra eval)
"${UV_RUN[@]}" python scripts/check_cuda_env.py

NGPU="${SLURM_GPUS_ON_NODE:-1}"
if [[ "$SUITE" != "fineweb" && "$NGPU" -gt 1 ]]; then
  "${UV_RUN[@]}" torchrun --standalone --nproc-per-node="$NGPU" -m gcw.cli "${gcw_args[@]}"
else
  "${UV_RUN[@]}" gcw "${gcw_args[@]}"
fi
if [[ -n "$OUT" ]]; then
  echo "Wrote $OUT"
else
  true
fi
