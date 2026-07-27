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
EVAL_MANIFEST="${EVAL_MANIFEST:-}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_out

UV_RUN=(uv run --locked --extra gpu --extra eval)
"${UV_RUN[@]}" python scripts/check_cuda_env.py

run_eval() {
  local checkpoint="$1"
  local out="$2"
  local ngpu="${SLURM_GPUS_ON_NODE:-1}"
  local gcw_args=(eval "$SUITE" --checkpoint "$checkpoint")
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
  if [[ -n "$out" ]]; then
    mkdir -p "$(dirname "$out")"
    gcw_args+=(--out "$out")
  fi

  if [[ "$SUITE" != "fineweb" && "$ngpu" -gt 1 ]]; then
    "${UV_RUN[@]}" torchrun --standalone --nproc-per-node="$ngpu" -m gcw.cli "${gcw_args[@]}" || return $?
  else
    "${UV_RUN[@]}" gcw "${gcw_args[@]}" || return $?
  fi
  if [[ -n "$out" ]]; then
    echo "Wrote $out"
  fi
}

if [[ -n "$EVAL_MANIFEST" ]]; then
  while IFS=$'\t' read -r manifest_checkpoint manifest_out; do
    if [[ -z "$manifest_checkpoint" ]]; then
      continue
    fi
    run_eval "$manifest_checkpoint" "$manifest_out"
  done < "$EVAL_MANIFEST"
else
  run_eval "$CHECKPOINT" "$OUT"
fi
