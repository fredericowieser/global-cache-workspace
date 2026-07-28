#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
ACTION="${1:-}"
shift || true

SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
EXPERIMENT_MANIFEST="${EXPERIMENT_MANIFEST:?EXPERIMENT_MANIFEST is required}"
APPROVED_MANIFEST_SHA256="${APPROVED_MANIFEST_SHA256:?APPROVED_MANIFEST_SHA256 is required}"
UV="$HOME/.local/bin/uv"

mkdir -p slurm_out
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
read -r ACTUAL_MANIFEST_SHA _ < <(sha256sum "$EXPERIMENT_MANIFEST")
[[ "$ACTUAL_MANIFEST_SHA" == "$APPROVED_MANIFEST_SHA256" ]]
"$UV" run --locked --extra gpu python scripts/flow_math_experiment.py verify-context \
  --repo-root "$PWD" \
  --manifest "$EXPERIMENT_MANIFEST" \
  --approved-manifest-sha256 "$APPROVED_MANIFEST_SHA256" \
  --source-commit "$SOURCE_COMMIT"

submit() {
  local job_name="$1"
  shift
  if [[ -n "$(squeue -h -u "$USER" -n "$job_name" -o '%i')" ]]; then
    printf 'Refusing duplicate active job name: %s\n' "$job_name" >&2
    exit 2
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN sbatch --job-name=%q ' "$job_name"
    printf '%q ' "$@"
    printf '\n'
  else
    sbatch --parsable --job-name="$job_name" "$@"
  fi
}

COMMON_EXPORT="ALL,SOURCE_COMMIT=$SOURCE_COMMIT,EXPERIMENT_MANIFEST=$EXPERIMENT_MANIFEST,APPROVED_MANIFEST_SHA256=$APPROVED_MANIFEST_SHA256"
case "$ACTION" in
  pilot)
    CONFIG="${1:?pilot requires an approved config path}"
    BASE="$(basename "$CONFIG" .yaml)"
    submit "flow-pilot-$BASE" \
      --export="$COMMON_EXPORT,CONFIG=$CONFIG" \
      scripts/slurm/flow_math_pilot.sbatch
    ;;
  train)
    submit "flow-math-train-$SOURCE_COMMIT" \
      --array=0-8%3 \
      --export="$COMMON_EXPORT" \
      scripts/slurm/flow_math_train_array.sbatch
    ;;
  eval-pilot)
    : "${EVAL_CACHE_DIR:?EVAL_CACHE_DIR is required}"
    : "${EVAL_CACHE_MANIFEST:?EVAL_CACHE_MANIFEST is required}"
    : "${APPROVED_EVAL_CACHE_SHA256:?APPROVED_EVAL_CACHE_SHA256 is required}"
    JOB_NAME="flow-math-eval-pilot-${SOURCE_COMMIT:0:12}"
    submit "$JOB_NAME" \
      --export="$COMMON_EXPORT,EVAL_CACHE_DIR=$EVAL_CACHE_DIR,EVAL_CACHE_MANIFEST=$EVAL_CACHE_MANIFEST,APPROVED_EVAL_CACHE_SHA256=$APPROVED_EVAL_CACHE_SHA256" \
      scripts/slurm/flow_math_eval_pilot.sbatch
    ;;
  eval)
    : "${EVAL_CACHE_DIR:?EVAL_CACHE_DIR is required}"
    : "${EVAL_CACHE_MANIFEST:?EVAL_CACHE_MANIFEST is required}"
    : "${APPROVED_EVAL_CACHE_SHA256:?APPROVED_EVAL_CACHE_SHA256 is required}"
    "$UV" run --locked --extra gpu --extra eval python scripts/flow_math_experiment.py check-eval-pilot \
      --repo-root "$PWD" --experiment-manifest "$EXPERIMENT_MANIFEST" \
      --experiment-manifest-sha256 "$APPROVED_MANIFEST_SHA256" --source-commit "$SOURCE_COMMIT" \
      --eval-cache-dir "$EVAL_CACHE_DIR" --eval-cache-manifest "$EVAL_CACHE_MANIFEST" \
      --eval-cache-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" \
      --marker "/mnt/jfs-01/shared/gcw/experiments/flow_math_llama32_1b_v1/markers/eval-pilot-${SOURCE_COMMIT:0:12}.json"
    submit "flow-math-eval-$SOURCE_COMMIT" \
      --array=0-9%2 \
      --export="$COMMON_EXPORT,EVAL_CACHE_DIR=$EVAL_CACHE_DIR,EVAL_CACHE_MANIFEST=$EVAL_CACHE_MANIFEST,APPROVED_EVAL_CACHE_SHA256=$APPROVED_EVAL_CACHE_SHA256" \
      scripts/slurm/flow_math_eval_array.sbatch
    ;;
  *)
    printf 'usage: %s [--dry-run] {pilot CONFIG|train|eval-pilot|eval}\n' "$0" >&2
    exit 2
    ;;
esac
