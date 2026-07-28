#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
ACTION="${1:-}"
shift || true
UV="$HOME/.local/bin/uv"
EXPERIMENT_ROOT="/mnt/jfs-01/shared/gcw/experiments/flow_math_llama32_1b_v1"

mkdir -p slurm_out

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

training_preflight() {
  SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
  EXPERIMENT_MANIFEST="${EXPERIMENT_MANIFEST:?EXPERIMENT_MANIFEST is required}"
  APPROVED_MANIFEST_SHA256="${APPROVED_MANIFEST_SHA256:?APPROVED_MANIFEST_SHA256 is required}"
  [[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
  [[ -z "$(git status --porcelain --untracked-files=no)" ]]
  read -r ACTUAL_MANIFEST_SHA _ < <(sha256sum "$EXPERIMENT_MANIFEST")
  [[ "$ACTUAL_MANIFEST_SHA" == "$APPROVED_MANIFEST_SHA256" ]]
  "$UV" run --locked --extra gpu python scripts/flow_math_experiment.py verify-context \
    --repo-root "$PWD" --manifest "$EXPERIMENT_MANIFEST" \
    --approved-manifest-sha256 "$APPROVED_MANIFEST_SHA256" --source-commit "$SOURCE_COMMIT"
  TRAINING_EXPORT="ALL,SOURCE_COMMIT=$SOURCE_COMMIT,EXPERIMENT_MANIFEST=$EXPERIMENT_MANIFEST,APPROVED_MANIFEST_SHA256=$APPROVED_MANIFEST_SHA256"
}

evaluation_context_preflight() {
  TRAINING_SOURCE_COMMIT="${TRAINING_SOURCE_COMMIT:?TRAINING_SOURCE_COMMIT is required}"
  TRAINING_EXPERIMENT_MANIFEST="${TRAINING_EXPERIMENT_MANIFEST:?TRAINING_EXPERIMENT_MANIFEST is required}"
  APPROVED_TRAINING_MANIFEST_SHA256="${APPROVED_TRAINING_MANIFEST_SHA256:?APPROVED_TRAINING_MANIFEST_SHA256 is required}"
  EVALUATION_SOURCE_COMMIT="${EVALUATION_SOURCE_COMMIT:?EVALUATION_SOURCE_COMMIT is required}"
  EVALUATION_MANIFEST="${EVALUATION_MANIFEST:?EVALUATION_MANIFEST is required}"
  APPROVED_EVALUATION_MANIFEST_SHA256="${APPROVED_EVALUATION_MANIFEST_SHA256:?APPROVED_EVALUATION_MANIFEST_SHA256 is required}"
  [[ "$(git rev-parse HEAD)" == "$EVALUATION_SOURCE_COMMIT" ]]
  [[ -z "$(git status --porcelain --untracked-files=no)" ]]
  "$UV" run --locked --extra gpu --extra eval python scripts/flow_math_experiment.py verify-evaluation-context \
    --repo-root "$PWD" --evaluation-manifest "$EVALUATION_MANIFEST" \
    --evaluation-manifest-sha256 "$APPROVED_EVALUATION_MANIFEST_SHA256" \
    --evaluation-source-commit "$EVALUATION_SOURCE_COMMIT" \
    --training-experiment-manifest "$TRAINING_EXPERIMENT_MANIFEST" \
    --training-experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256" \
    --training-source-commit "$TRAINING_SOURCE_COMMIT"
  EVALUATION_CONTEXT_EXPORT="ALL,TRAINING_SOURCE_COMMIT=$TRAINING_SOURCE_COMMIT,TRAINING_EXPERIMENT_MANIFEST=$TRAINING_EXPERIMENT_MANIFEST,APPROVED_TRAINING_MANIFEST_SHA256=$APPROVED_TRAINING_MANIFEST_SHA256,EVALUATION_SOURCE_COMMIT=$EVALUATION_SOURCE_COMMIT,EVALUATION_MANIFEST=$EVALUATION_MANIFEST,APPROVED_EVALUATION_MANIFEST_SHA256=$APPROVED_EVALUATION_MANIFEST_SHA256"
}

evaluation_preflight() {
  evaluation_context_preflight
  EVAL_CACHE_DIR="${EVAL_CACHE_DIR:?EVAL_CACHE_DIR is required}"
  EVAL_CACHE_MANIFEST="${EVAL_CACHE_MANIFEST:?EVAL_CACHE_MANIFEST is required}"
  APPROVED_EVAL_CACHE_SHA256="${APPROVED_EVAL_CACHE_SHA256:?APPROVED_EVAL_CACHE_SHA256 is required}"
  EVALUATION_EXPORT="$EVALUATION_CONTEXT_EXPORT,EVAL_CACHE_DIR=$EVAL_CACHE_DIR,EVAL_CACHE_MANIFEST=$EVAL_CACHE_MANIFEST,APPROVED_EVAL_CACHE_SHA256=$APPROVED_EVAL_CACHE_SHA256"
}

case "$ACTION" in
  pilot)
    training_preflight
    CONFIG="${1:?pilot requires an approved config path}"
    BASE="$(basename "$CONFIG" .yaml)"
    submit "flow-pilot-$BASE" --export="$TRAINING_EXPORT,CONFIG=$CONFIG" scripts/slurm/flow_math_pilot.sbatch
    ;;
  train)
    training_preflight
    submit "flow-math-train-$SOURCE_COMMIT" --array=0-8%3 --export="$TRAINING_EXPORT" scripts/slurm/flow_math_train_array.sbatch
    ;;
  eval-cache)
    evaluation_context_preflight
    EVAL_CACHE_DIR="${EVAL_CACHE_DIR:?EVAL_CACHE_DIR is required}"
    EVAL_CACHE_MANIFEST="${EVAL_CACHE_MANIFEST:?EVAL_CACHE_MANIFEST is required}"
    submit "flow-math-eval-cache-${EVALUATION_SOURCE_COMMIT:0:12}" \
      --export="$EVALUATION_CONTEXT_EXPORT,EVAL_CACHE_DIR=$EVAL_CACHE_DIR,EVAL_CACHE_MANIFEST=$EVAL_CACHE_MANIFEST" \
      scripts/slurm/flow_math_eval_cache.sbatch
    ;;
  eval-pilot)
    evaluation_preflight
    IDENTITY="train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}"
    "$UV" run --locked --extra gpu --extra eval python scripts/flow_math_experiment.py check-eval-cache \
      --repo-root "$PWD" --evaluation-manifest "$EVALUATION_MANIFEST" \
      --evaluation-manifest-sha256 "$APPROVED_EVALUATION_MANIFEST_SHA256" \
      --evaluation-source-commit "$EVALUATION_SOURCE_COMMIT" \
      --training-experiment-manifest "$TRAINING_EXPERIMENT_MANIFEST" \
      --training-experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256" \
      --training-source-commit "$TRAINING_SOURCE_COMMIT" \
      --eval-cache-dir "$EVAL_CACHE_DIR" --eval-cache-manifest "$EVAL_CACHE_MANIFEST" \
      --eval-cache-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" \
      --warmup-result "$EXPERIMENT_ROOT/eval/cache-warmup/tf-backbone-$IDENTITY.json" \
      --lock-inventory "$EXPERIMENT_ROOT/eval/cache-lock-inventory-$IDENTITY.json" \
      --marker "$EXPERIMENT_ROOT/markers/eval-cache-$IDENTITY.json"
    submit "flow-math-eval-pilot-${EVALUATION_SOURCE_COMMIT:0:12}" \
      --export="$EVALUATION_EXPORT" scripts/slurm/flow_math_eval_pilot.sbatch
    ;;
  eval)
    evaluation_preflight
    EVAL_ARRAY_SPEC="${1:?eval requires a canonical chunk or one task ID}"
    case "$EVAL_ARRAY_SPEC" in
      0-3%2|4-7%2|8-9%2|[0-9]) ;;
      *) printf 'invalid evaluation chunk: %s\n' "$EVAL_ARRAY_SPEC" >&2; exit 2 ;;
    esac
    "$UV" run --locked --extra gpu --extra eval python scripts/flow_math_experiment.py check-eval-pilot \
      --repo-root "$PWD" --evaluation-manifest "$EVALUATION_MANIFEST" \
      --evaluation-manifest-sha256 "$APPROVED_EVALUATION_MANIFEST_SHA256" \
      --evaluation-source-commit "$EVALUATION_SOURCE_COMMIT" \
      --training-experiment-manifest "$TRAINING_EXPERIMENT_MANIFEST" \
      --training-experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256" \
      --training-source-commit "$TRAINING_SOURCE_COMMIT" \
      --eval-cache-dir "$EVAL_CACHE_DIR" --eval-cache-manifest "$EVAL_CACHE_MANIFEST" \
      --eval-cache-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" \
      --cache-copy-evidence "$EXPERIMENT_ROOT/eval/cache-copy-evidence/pilot-train-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json" \
      --marker "$EXPERIMENT_ROOT/markers/eval-pilot-${TRAINING_SOURCE_COMMIT:0:12}-eval-${EVALUATION_SOURCE_COMMIT:0:12}.json"
    CHUNK_NAME="${EVAL_ARRAY_SPEC//%/-}"
    submit "flow-math-eval-${EVALUATION_SOURCE_COMMIT:0:12}-$CHUNK_NAME" \
      --array="$EVAL_ARRAY_SPEC" --export="$EVALUATION_EXPORT" scripts/slurm/flow_math_eval_array.sbatch
    ;;
  ledger)
    TRAINING_SOURCE_COMMIT="${TRAINING_SOURCE_COMMIT:?TRAINING_SOURCE_COMMIT is required}"
    APPROVED_TRAINING_MANIFEST_SHA256="${APPROVED_TRAINING_MANIFEST_SHA256:?APPROVED_TRAINING_MANIFEST_SHA256 is required}"
    EVALUATION_SOURCE_COMMIT="${EVALUATION_SOURCE_COMMIT:?EVALUATION_SOURCE_COMMIT is required}"
    APPROVED_EVALUATION_MANIFEST_SHA256="${APPROVED_EVALUATION_MANIFEST_SHA256:?APPROVED_EVALUATION_MANIFEST_SHA256 is required}"
    APPROVED_EVAL_CACHE_SHA256="${APPROVED_EVAL_CACHE_SHA256:?APPROVED_EVAL_CACHE_SHA256 is required}"
    EVAL_MARKERS_DIR="${EVAL_MARKERS_DIR:?EVAL_MARKERS_DIR is required}"
    CACHE_MARKER="${CACHE_MARKER:?CACHE_MARKER is required}"
    PILOT_MARKER="${PILOT_MARKER:?PILOT_MARKER is required}"
    EVAL_LEDGER="${EVAL_LEDGER:?EVAL_LEDGER is required}"
    "$UV" run --locked --extra cpu python scripts/flow_math_experiment.py build-eval-ledger \
      --markers-dir "$EVAL_MARKERS_DIR" --cache-marker "$CACHE_MARKER" --pilot-marker "$PILOT_MARKER" \
      --training-source-commit "$TRAINING_SOURCE_COMMIT" \
      --training-experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256" \
      --evaluation-source-commit "$EVALUATION_SOURCE_COMMIT" \
      --evaluation-manifest-sha256 "$APPROVED_EVALUATION_MANIFEST_SHA256" \
      --eval-cache-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" --out "$EVAL_LEDGER"
    ;;
  summarise)
    evaluation_preflight
    EVAL_LEDGER="${EVAL_LEDGER:?EVAL_LEDGER is required}"
    APPROVED_LEDGER_SHA256="${APPROVED_LEDGER_SHA256:?APPROVED_LEDGER_SHA256 is required}"
    SUMMARY_OUT="${SUMMARY_OUT:?SUMMARY_OUT is required}"
    "$UV" run --locked --extra gpu --extra eval python scripts/flow_math_experiment.py summarise \
      --markers-dir "$EXPERIMENT_ROOT/markers" --repo-root "$PWD" \
      --evaluation-manifest "$EVALUATION_MANIFEST" \
      --evaluation-manifest-sha256 "$APPROVED_EVALUATION_MANIFEST_SHA256" \
      --evaluation-source-commit "$EVALUATION_SOURCE_COMMIT" \
      --training-experiment-manifest "$TRAINING_EXPERIMENT_MANIFEST" \
      --training-experiment-manifest-sha256 "$APPROVED_TRAINING_MANIFEST_SHA256" \
      --training-source-commit "$TRAINING_SOURCE_COMMIT" \
      --eval-cache-dir "$EVAL_CACHE_DIR" --eval-cache-manifest "$EVAL_CACHE_MANIFEST" \
      --eval-cache-manifest-sha256 "$APPROVED_EVAL_CACHE_SHA256" \
      --ledger "$EVAL_LEDGER" --approved-ledger-sha256 "$APPROVED_LEDGER_SHA256" \
      --out "$SUMMARY_OUT"
    ;;
  *)
    printf 'usage: %s [--dry-run] {pilot CONFIG|train|eval-cache|eval-pilot|eval CHUNK|ledger|summarise}\n' "$0" >&2
    exit 2
    ;;
esac
