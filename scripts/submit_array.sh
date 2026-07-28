#!/usr/bin/env bash
set -euo pipefail

# Submit a small Slurm worker array that consumes a list of configs.
#
# This keeps Slurm submission count low: one sbatch call creates N array workers,
# and each worker runs train followed by eval inline for every config it claims.

repo_root="${GCW_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
cd "$repo_root"

JOB_NAME="${JOB_NAME:-gcw-train-eval-array}"
PARTITION="${PARTITION:-agent-xlong-15,agent-xlong}"
ARRAY_WORKERS="${ARRAY_WORKERS:-1}"
ARRAY_TIME="${ARRAY_TIME:-5-00:00:00}"
EXCLUDE="${EXCLUDE:-}"
DEPENDENCY="${DEPENDENCY:-}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-180G}"
WANDB_MODE="${WANDB_MODE:-online}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
UV_EXTRAS="${UV_EXTRAS:---extra gpu --extra eval}"
TRAIN_UV_EXTRAS="${TRAIN_UV_EXTRAS:-$UV_EXTRAS}"
EVAL_UV_EXTRAS="${EVAL_UV_EXTRAS:---extra gpu --extra eval}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-1}"
EVAL_ONLY="${EVAL_ONLY:-0}"
EVAL_SUITE="${EVAL_SUITE:-paper7}"
EVAL_GPUS="${EVAL_GPUS:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GCW_EVAL_SKIP_TASKS="${GCW_EVAL_SKIP_TASKS:-theoremqa}"
KEEP_STEP_CHECKPOINTS="${KEEP_STEP_CHECKPOINTS:-auto}"
KEEP_STEP_CHECKPOINT_EVERY="${KEEP_STEP_CHECKPOINT_EVERY:-1000}"
OUT_DIR="${OUT_DIR:-results}"

OPENMATH_DATA="${OPENMATH_DATA:-/mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl}"
SFT_BACKBONE="${SFT_BACKBONE:-/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf/model.safetensors}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/mnt/jfs-01/shared/bottleneck_llm/sft/llama-3.2-1b/final_model}"

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/submit_array.sh [--workers N] [--configs-file FILE] [--dry-run] CONFIG [CONFIG ...]

Options:
  --workers N          Number of Slurm array workers to submit. Default: ARRAY_WORKERS or 1.
  --configs-file FILE  File containing one config path per line. Blank lines and # comments are ignored.
  --dry-run            Print the sbatch command without submitting.

Common environment overrides:
  JOB_NAME, PARTITION, ARRAY_TIME, GPUS, CPUS_PER_TASK, MEM, WANDB_MODE
  EXCLUDE, TRAIN_UV_EXTRAS, EVAL_UV_EXTRAS
  EVAL_AFTER_TRAIN, EVAL_ONLY, EVAL_SUITE, EVAL_GPUS, EVAL_BATCH_SIZE, OUT_DIR
  KEEP_STEP_CHECKPOINTS, KEEP_STEP_CHECKPOINT_EVERY
EOF
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
}

positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
    echo "$name must be a positive integer; got '$value'" >&2
    exit 1
  fi
}

config_output_dir() {
  awk '$1 == "output_dir:" {print $2; exit}' "$1"
}

config_max_steps() {
  awk '$1 == "max_steps:" {print $2; exit}' "$1"
}

worker_uv_run_train() {
  read -r -a uv_extra_args <<< "$TRAIN_UV_EXTRAS"
  uv run --locked "${uv_extra_args[@]}" "$@"
}

worker_uv_run_eval() {
  read -r -a uv_extra_args <<< "$EVAL_UV_EXTRAS"
  uv run --locked "${uv_extra_args[@]}" "$@"
}

run_eval() {
  local checkpoint="$1"
  local out="$2"
  local ngpu="$EVAL_GPUS"
  local gcw_args=(eval "$EVAL_SUITE" --checkpoint "$checkpoint")

  if [[ -n "${LIMIT:-}" && "$EVAL_SUITE" != "fineweb" ]]; then
    gcw_args+=(--limit "$LIMIT")
  fi
  if [[ -n "$EVAL_BATCH_SIZE" ]]; then
    gcw_args+=(--batch-size "$EVAL_BATCH_SIZE")
  fi
  if [[ "$EVAL_SUITE" == "fineweb" && -n "${DATA_DIR:-}" ]]; then
    gcw_args+=(--data-dir "$DATA_DIR")
  fi
  if [[ "$EVAL_SUITE" == "fineweb" && -n "${EVAL_TOKENS:-}" ]]; then
    gcw_args+=(--eval-tokens "$EVAL_TOKENS")
  fi
  if [[ -n "$out" ]]; then
    mkdir -p "$(dirname "$out")"
    gcw_args+=(--out "$out")
  fi

  if [[ "$EVAL_SUITE" != "fineweb" && "$ngpu" -gt 1 ]]; then
    worker_uv_run_eval torchrun --standalone --nproc-per-node="$ngpu" -m gcw.cli "${gcw_args[@]}" || return $?
  else
    worker_uv_run_eval gcw "${gcw_args[@]}" || return $?
  fi
  if [[ -n "$out" ]]; then
    echo "Wrote $out"
  fi
}

run_eval_manifest() {
  local manifest="$1"
  while IFS=$'\t' read -r manifest_checkpoint manifest_out; do
    if [[ -z "$manifest_checkpoint" ]]; then
      continue
    fi
    run_eval "$manifest_checkpoint" "$manifest_out"
  done < "$manifest"
}

run_one_config() {
  local config="$1"
  local output_dir max_steps keep_step_checkpoints run_name manifest step out nproc

  require_path "$config"
  require_path "$OPENMATH_DATA"

  output_dir="$(config_output_dir "$config")"
  if [[ -z "$output_dir" ]]; then
    echo "Could not determine output_dir from $config" >&2
    exit 1
  fi
  max_steps="$(config_max_steps "$config")"
  if [[ -z "$max_steps" ]]; then
    echo "Could not determine max_steps from $config" >&2
    exit 1
  fi

  positive_int KEEP_STEP_CHECKPOINT_EVERY "$KEEP_STEP_CHECKPOINT_EVERY"
  keep_step_checkpoints="$KEEP_STEP_CHECKPOINTS"
  if [[ "$keep_step_checkpoints" == "auto" ]]; then
    if [[ "$max_steps" =~ ^[0-9]+$ ]] && [[ "$max_steps" -gt "$KEEP_STEP_CHECKPOINT_EVERY" ]]; then
      keep_step_checkpoints=1
    else
      keep_step_checkpoints=0
    fi
  fi

  run_name="$(basename "$output_dir")"
  if [[ "$EVAL_ONLY" == "1" ]]; then
    echo "=== Evaluating $config ==="
  else
    echo "=== Training $config ==="
  fi
  echo "Output dir: $output_dir"

  export PATH="$HOME/.local/bin:$PATH"
  export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"
  if [[ -d "$PWD/.venv/include/python3.12" ]]; then
    export CPATH="$PWD/.venv/include/python3.12:$PWD/.venv/include${CPATH:+:$CPATH}"
  fi
  export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
  export WANDB_MODE
  export PYTORCH_CUDA_ALLOC_CONF
  export GCW_KEEP_STEP_CHECKPOINTS="$keep_step_checkpoints"
  export GCW_STEP_CHECKPOINT_EVERY="$KEEP_STEP_CHECKPOINT_EVERY"
  mkdir -p "$WANDB_DIR" slurm_out

  if [[ "$EVAL_ONLY" != "1" ]]; then
    worker_uv_run_train python scripts/check_cuda_env.py || return $?
    nproc="$GPUS"
    if [[ "$nproc" == "1" ]]; then
      worker_uv_run_train gcw train --config "$config" || return $?
    else
      worker_uv_run_train torchrun --standalone --nproc_per_node="$nproc" -m gcw.cli train --config "$config" || return $?
    fi

    if [[ "$EVAL_AFTER_TRAIN" != "1" ]]; then
      return
    fi

    echo "=== Evaluating $config ==="
  fi

  export GCW_EVAL_SKIP_TASKS
  worker_uv_run_eval python scripts/check_cuda_env.py || return $?
  if [[ -n "${CHECKPOINT:-}" || -n "${OUT:-}" ]]; then
    run_eval "${CHECKPOINT:-$output_dir/checkpoint}" "${OUT:-${OUT_DIR}/${run_name}_${OUT_SUFFIX:-final}_${EVAL_SUITE}.json}"
  elif [[ "$keep_step_checkpoints" == "1" ]]; then
    manifest="${EVAL_MANIFEST:-slurm_out/${run_name}_${EVAL_SUITE}_checkpoints.tsv}"
    : > "$manifest"
    for ((step = KEEP_STEP_CHECKPOINT_EVERY; step <= max_steps; step += KEEP_STEP_CHECKPOINT_EVERY)); do
      out="${OUT_DIR}/${run_name}_step${step}_${EVAL_SUITE}.json"
      printf "%s\t%s\n" "$output_dir/checkpoint-$step" "$out" >> "$manifest"
    done
    run_eval_manifest "$manifest"
  else
    run_eval "$output_dir/checkpoint" "${OUT_DIR}/${run_name}_final_${EVAL_SUITE}.json"
  fi
}

claim_next_config() {
  local config
  (
    flock -x 200
    touch "$CLAIMED_FILE"

    config="$(
      awk '
        ARGIND == 1 { claimed[$0] = 1; next }
        /^[[:space:]]*($|#)/ { next }
        !($0 in claimed) { print; exit }
      ' "$CLAIMED_FILE" "$CONFIG_LIST"
    )"
    if [[ -n "$config" ]]; then
      echo "$config" >> "$CLAIMED_FILE"
      echo "$config"
      return
    fi

    echo "STOP"
  ) 200>"$LOCK_FILE"
}

worker_main() {
  CONFIG_LIST="${CONFIG_LIST:?CONFIG_LIST is required in worker mode}"
  STATE_DIR="${STATE_DIR:?STATE_DIR is required in worker mode}"
  CLAIMED_FILE="$STATE_DIR/claimed_configs"
  COMPLETED_FILE="$STATE_DIR/completed_configs"
  FAILED_FILE="$STATE_DIR/failed_configs.tsv"
  LOCK_FILE="$STATE_DIR/queue.lock"
  local claim_result rc failures=0

  mkdir -p "$STATE_DIR" slurm_out "$OUT_DIR"
  require_path "$CONFIG_LIST"
  touch "$COMPLETED_FILE" "$FAILED_FILE"

  echo "Worker ${SLURM_ARRAY_TASK_ID:-0} consuming configs from $CONFIG_LIST"
  while true; do
    if ! claim_result="$(claim_next_config)"; then
      echo "Failed to claim a config from $CONFIG_LIST" >&2
      return 1
    fi
    if [[ "$claim_result" == "STOP" ]]; then
      echo "No unclaimed configs remain; exiting"
      break
    fi
    if run_one_config "$claim_result"; then
      (
        flock -x 200
        printf "%s\n" "$claim_result" >> "$COMPLETED_FILE"
      ) 200>"$LOCK_FILE"
    else
      rc="$?"
      failures=$((failures + 1))
      echo "Config failed with exit code $rc; continuing: $claim_result" >&2
      (
        flock -x 200
        printf "%s\t%s\n" "$rc" "$claim_result" >> "$FAILED_FILE"
      ) 200>"$LOCK_FILE"
    fi
  done

  if [[ "$failures" -gt 0 ]]; then
    echo "Worker completed with $failures failed config(s). See $FAILED_FILE" >&2
    return 1
  fi
}

write_config_list() {
  local list_path="$1"
  local configs_file="$2"
  shift 2

  : > "$list_path"
  if [[ -n "$configs_file" ]]; then
    require_path "$configs_file"
    while IFS= read -r config; do
      if [[ -z "$config" || "$config" =~ ^[[:space:]]*# ]]; then
        continue
      fi
      require_path "$config"
      printf "%s\n" "$config" >> "$list_path"
    done < "$configs_file"
  fi
  for config in "$@"; do
    require_path "$config"
    printf "%s\n" "$config" >> "$list_path"
  done

  if [[ ! -s "$list_path" ]]; then
    echo "No configs provided" >&2
    usage
    exit 1
  fi
}

launcher_main() {
  local configs_file="" dry_run=0 state_dir config_list workers_arg
  local alloc_gpus
  local args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --configs-file)
        configs_file="${2:-}"
        if [[ -z "$configs_file" ]]; then
          echo "--configs-file requires a path" >&2
          exit 1
        fi
        shift 2
        ;;
      --workers)
        ARRAY_WORKERS="${2:-}"
        positive_int --workers "$ARRAY_WORKERS"
        shift 2
        ;;
      --dry-run)
        dry_run=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        args+=("$@")
        break
        ;;
      -*)
        echo "Unknown option: $1" >&2
        usage
        exit 1
        ;;
      *)
        args+=("$1")
        shift
        ;;
    esac
  done

  positive_int ARRAY_WORKERS "$ARRAY_WORKERS"
  positive_int GPUS "$GPUS"
  positive_int EVAL_GPUS "$EVAL_GPUS"
  alloc_gpus="$GPUS"
  if [[ "$EVAL_AFTER_TRAIN" == "1" && "$EVAL_GPUS" -gt "$alloc_gpus" ]]; then
    alloc_gpus="$EVAL_GPUS"
  fi

  mkdir -p slurm_out
  state_dir="$repo_root/slurm_out/${JOB_NAME}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$state_dir"
  config_list="$state_dir/configs.txt"
  write_config_list "$config_list" "$configs_file" "${args[@]}"

  workers_arg="0-$((ARRAY_WORKERS - 1))"
  echo "Config list: $config_list"
  echo "Submitting $ARRAY_WORKERS worker(s) as array $workers_arg"

  sbatch_cmd=(
    sbatch
    --parsable
    --job-name="$JOB_NAME"
    --partition="$PARTITION"
    --time="$ARRAY_TIME"
    --nodes=1
    --ntasks-per-node=1
    --gres="gpu:$alloc_gpus"
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$MEM"
    --array="$workers_arg"
    --output="slurm_out/${JOB_NAME}-%A_%a.out"
    --export="ALL,GCW_ARRAY_MODE=worker,GCW_REPO_ROOT=$repo_root,CONFIG_LIST=$config_list,STATE_DIR=$state_dir,JOB_NAME=$JOB_NAME,GPUS=$GPUS,EVAL_GPUS=$EVAL_GPUS,EVAL_AFTER_TRAIN=$EVAL_AFTER_TRAIN,EVAL_ONLY=$EVAL_ONLY,EVAL_SUITE=$EVAL_SUITE,EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE,GCW_EVAL_SKIP_TASKS=$GCW_EVAL_SKIP_TASKS,WANDB_MODE=$WANDB_MODE,PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF,UV_EXTRAS=$UV_EXTRAS,TRAIN_UV_EXTRAS=$TRAIN_UV_EXTRAS,EVAL_UV_EXTRAS=$EVAL_UV_EXTRAS,KEEP_STEP_CHECKPOINTS=$KEEP_STEP_CHECKPOINTS,KEEP_STEP_CHECKPOINT_EVERY=$KEEP_STEP_CHECKPOINT_EVERY,OUT_DIR=$OUT_DIR,OPENMATH_DATA=$OPENMATH_DATA,SFT_BACKBONE=$SFT_BACKBONE,TOKENIZER_DIR=$TOKENIZER_DIR"
  )
  if [[ -n "$DEPENDENCY" ]]; then
    sbatch_cmd+=(--dependency="$DEPENDENCY")
  fi
  if [[ -n "$EXCLUDE" ]]; then
    sbatch_cmd+=(--exclude="$EXCLUDE")
  fi
  sbatch_cmd+=("$0")

  if [[ "$dry_run" == "1" ]]; then
    printf '%q ' "${sbatch_cmd[@]}"
    printf '\n'
    return
  fi

  job_id="$("${sbatch_cmd[@]}")"
  echo "Submitted array job $job_id"
}

if [[ "${GCW_ARRAY_MODE:-}" == "worker" ]]; then
  worker_main
else
  launcher_main "$@"
fi
