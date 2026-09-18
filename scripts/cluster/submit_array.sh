#!/usr/bin/env bash
set -euo pipefail

# Submit a small worker array that consumes a list of configs or evaluation tasks.
#
# The Slurm backend uses one array submission. The local backend starts detached,
# GPU-pinned process groups and records their PIDs. With either backend, each worker
# runs train followed by eval inline for every config it claims.

repo_root="${GCW_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")"
cd "$repo_root"

load_environment_file() {
  local path="$1"
  local assignment

  while IFS= read -r -d '' assignment; do
    export "$assignment"
  done < "$path"
}

JOB_NAME="${JOB_NAME:-gcw-train-eval-array}"
BACKEND="${BACKEND:-auto}"
PARTITION="${PARTITION:-}"
ARRAY_WORKERS="${ARRAY_WORKERS:-1}"
ARRAY_TIME="${ARRAY_TIME:-}"
EXCLUDE="${EXCLUDE:-}"
DEPENDENCY="${DEPENDENCY:-}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-}"
MEM="${MEM:-}"
WANDB_MODE="${WANDB_MODE:-online}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
UV_EXTRAS="${UV_EXTRAS:---extra gpu --extra eval --extra qwen35}"
TRAIN_UV_EXTRAS="${TRAIN_UV_EXTRAS:-$UV_EXTRAS}"
EVAL_UV_EXTRAS="${EVAL_UV_EXTRAS:---extra gpu --extra eval --extra qwen35}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-1}"
EVAL_ONLY="${EVAL_ONLY:-0}"
EVAL_SUITE="${EVAL_SUITE:-paper7}"
EVAL_GPUS="${EVAL_GPUS:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_MAX_BATCH_SIZE="${EVAL_MAX_BATCH_SIZE:-64}"
EVAL_CHECKPOINT_STEP="${EVAL_CHECKPOINT_STEP:-}"
EVAL_THINKING_MODES="${EVAL_THINKING_MODES:-}"
GCW_EVAL_TASK_MODE="${GCW_EVAL_TASK_MODE:-0}"
GCW_MIXED_TASK_MODE="${GCW_MIXED_TASK_MODE:-0}"
GENERATION_PROCESS_EVERY="${GENERATION_PROCESS_EVERY:-}"
EVAL_HEAD_SELECTION_METHOD="${EVAL_HEAD_SELECTION_METHOD:-}"
GCW_EVAL_SKIP_TASKS="${GCW_EVAL_SKIP_TASKS:-theoremqa}"
KEEP_STEP_CHECKPOINTS="${KEEP_STEP_CHECKPOINTS:-auto}"
KEEP_STEP_CHECKPOINT_EVERY="${KEEP_STEP_CHECKPOINT_EVERY:-1000}"
OUT_DIR="${OUT_DIR:-results}"

SFT_BACKBONE="${SFT_BACKBONE:-/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf/model.safetensors}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/mnt/jfs-01/shared/bottleneck_llm/sft/llama-3.2-1b/final_model}"
LOCAL_GPUS="${LOCAL_GPUS:-}"
LOCAL_STATE_ROOT="${LOCAL_STATE_ROOT:-$repo_root/slurm_out/.local-array}"
SLURM_STATE_ROOT="${SLURM_STATE_ROOT:-$repo_root/slurm_out}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-$repo_root/slurm_out}"
LOCAL_STOP_WHEN_IDLE="${LOCAL_STOP_WHEN_IDLE:-1}"
LOCAL_IDLE_CHECK_INTERVAL="${LOCAL_IDLE_CHECK_INTERVAL:-30}"
NEBIUS_INSTANCE_ID_FILE="${NEBIUS_INSTANCE_ID_FILE:-/mnt/cloud-metadata/instance-id}"
NEBIUS_STOP_PROFILE="${NEBIUS_STOP_PROFILE:-gpu-self-stop}"
DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL:-}"

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/cluster/submit_array.sh [--backend auto|slurm|local] [--workers N] [--local-gpus LIST]
                                  [--configs-file FILE | --eval-tasks-file FILE | --tasks-file FILE]
                                  [--dry-run] [CONFIG ...]
  scripts/cluster/submit_array.sh status STATE_DIR
  scripts/cluster/submit_array.sh cancel [--force] STATE_DIR

Options:
  --backend BACKEND    Submission backend. Default: BACKEND or auto (Slurm when sbatch exists).
  --workers N          Maximum number of array workers. Default: ARRAY_WORKERS or 1.
  --local-gpus LIST    Comma-separated GPU IDs available to local workers. By default, uses
                       GPU indices reported by nvidia-smi.
  --configs-file FILE  File containing one config path per line. Blank lines and # comments are ignored.
  --eval-tasks-file FILE
                       Tab-separated CHECKPOINT, OUTPUT, and THINKING rows for eval-only workers.
  --tasks-file FILE    Six TSV fields: ACTION, CONFIG, CHECKPOINT, OUTPUT_BASE, THINKING_MODES,
                       SKIP_TASKS. ACTION is train or eval; use - for no CONFIG or SKIP_TASKS.
  --dry-run            Print the submission plan without starting workers.

Common environment overrides:
  BACKEND, JOB_NAME, PARTITION, ARRAY_TIME, GPUS, CPUS_PER_TASK, MEM, WANDB_MODE
  Slurm chooses partition, time, CPUs, and memory when their overrides are unset.
  EXCLUDE, TRAIN_UV_EXTRAS, EVAL_UV_EXTRAS
  EVAL_AFTER_TRAIN, EVAL_ONLY, EVAL_SUITE, EVAL_GPUS, EVAL_BATCH_SIZE, EVAL_MAX_BATCH_SIZE, OUT_DIR
  EVAL_THINKING_MODES (comma-separated true/false values)
  EVAL_CHECKPOINT_STEP, GENERATION_PROCESS_EVERY
  EVAL_HEAD_SELECTION_METHOD
  KEEP_STEP_CHECKPOINTS, KEEP_STEP_CHECKPOINT_EVERY
  SLURM_STATE_ROOT, SLURM_LOG_DIR
  LOCAL_GPUS, LOCAL_STATE_ROOT, LOCAL_STOP_WHEN_IDLE, LOCAL_IDLE_CHECK_INTERVAL
  NEBIUS_INSTANCE_ID_FILE, NEBIUS_STOP_PROFILE

Local backend notes:
  Workers are detached; use the printed status/cancel commands to manage them.
  GPU reservations prevent overlap between local submissions from this repository.
  ARRAY_TIME, CPUS_PER_TASK, and MEM are not enforced locally.
  By default, the VM is stopped after all local workers finish and no GPU compute
  processes remain. Set LOCAL_STOP_WHEN_IDLE=0 to disable this behavior.
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

validate_eval_batching() {
  if [[ -z "$EVAL_BATCH_SIZE" ]]; then
    return
  fi
  if [[ "$EVAL_BATCH_SIZE" != "auto" ]]; then
    positive_int EVAL_BATCH_SIZE "$EVAL_BATCH_SIZE"
    return
  fi
  case "$EVAL_SUITE" in
    paper7|paper7-legacy|all|math|general|tasks) ;;
    *)
      echo "EVAL_BATCH_SIZE=auto is not supported for $EVAL_SUITE evaluation" >&2
      exit 1
      ;;
  esac
  positive_int EVAL_MAX_BATCH_SIZE "$EVAL_MAX_BATCH_SIZE"
}

validate_eval_thinking_modes() {
  local mode
  local -a modes
  if [[ -z "$EVAL_THINKING_MODES" ]]; then
    return
  fi
  if [[ "$EVAL_SUITE" == "fineweb" ]]; then
    echo "EVAL_THINKING_MODES is not supported for fineweb evaluation" >&2
    exit 1
  fi
  IFS=',' read -r -a modes <<< "$EVAL_THINKING_MODES"
  for mode in "${modes[@]}"; do
    if [[ "$mode" != "true" && "$mode" != "false" ]]; then
      echo "EVAL_THINKING_MODES values must be true or false; got '$mode'" >&2
      exit 1
    fi
  done
}

write_slurm_environment() {
  local path="$1"
  local config_list="$2"
  local state_dir="$3"

  env -0 \
    "GCW_ARRAY_MODE=worker" \
    "GCW_REPO_ROOT=$repo_root" \
    "CONFIG_LIST=$config_list" \
    "STATE_DIR=$state_dir" \
    "JOB_NAME=$JOB_NAME" \
    "GPUS=$GPUS" \
    "EVAL_GPUS=$EVAL_GPUS" \
    "EVAL_AFTER_TRAIN=$EVAL_AFTER_TRAIN" \
    "EVAL_ONLY=$EVAL_ONLY" \
    "EVAL_SUITE=$EVAL_SUITE" \
    "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE" \
    "EVAL_MAX_BATCH_SIZE=$EVAL_MAX_BATCH_SIZE" \
    "EVAL_CHECKPOINT_STEP=$EVAL_CHECKPOINT_STEP" \
    "EVAL_THINKING_MODES=$EVAL_THINKING_MODES" \
    "GCW_EVAL_TASK_MODE=$GCW_EVAL_TASK_MODE" \
    "GCW_MIXED_TASK_MODE=$GCW_MIXED_TASK_MODE" \
    "GENERATION_PROCESS_EVERY=$GENERATION_PROCESS_EVERY" \
    "EVAL_HEAD_SELECTION_METHOD=$EVAL_HEAD_SELECTION_METHOD" \
    "GCW_EVAL_SKIP_TASKS=$GCW_EVAL_SKIP_TASKS" \
    "WANDB_MODE=$WANDB_MODE" \
    "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" \
    "UV_EXTRAS=$UV_EXTRAS" \
    "TRAIN_UV_EXTRAS=$TRAIN_UV_EXTRAS" \
    "EVAL_UV_EXTRAS=$EVAL_UV_EXTRAS" \
    "KEEP_STEP_CHECKPOINTS=$KEEP_STEP_CHECKPOINTS" \
    "KEEP_STEP_CHECKPOINT_EVERY=$KEEP_STEP_CHECKPOINT_EVERY" \
    "OUT_DIR=$OUT_DIR" \
    "SFT_BACKBONE=$SFT_BACKBONE" \
    "TOKENIZER_DIR=$TOKENIZER_DIR" \
    "SLURM_EXPORT_ENV=ALL" > "$path"
}

validate_backend() {
  case "$1" in
    auto|slurm|local) ;;
    *)
      echo "Backend must be auto, slurm, or local; got '$1'" >&2
      exit 1
      ;;
  esac
}

process_start_ticks() {
  local pid="$1"
  awk '{print $22}' "/proc/$pid/stat" 2>/dev/null
}

process_matches() {
  local pid="$1"
  local expected_start="$2"
  local actual_start

  if ! [[ "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
    return 1
  fi
  actual_start="$(process_start_ticks "$pid")"
  [[ -n "$actual_start" && "$actual_start" == "$expected_start" ]]
}

read_pid_file() {
  local pid_file="$1"
  WORKER_PID=""
  WORKER_START_TICKS=""
  WORKER_GPUS=""
  if [[ -r "$pid_file" ]]; then
    IFS=$'\t' read -r WORKER_PID WORKER_START_TICKS WORKER_GPUS < "$pid_file" || true
  fi
}

load_discord_webhook() {
  local env_file value

  [[ -n "$DISCORD_WEBHOOK_URL" ]] && return 0
  env_file="$repo_root/.env"
  [[ -r "$env_file" ]] || return 0
  value="$(sed -n 's/^DISCORD_WEBHOOK_URL=//p' "$env_file" | tail -n 1)"
  value="${value%$'\r'}"
  if [[ "$value" == \"*\" && "$value" == *\" ]]; then
    value="${value:1:${#value}-2}"
  elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
    value="${value:1:${#value}-2}"
  fi
  if [[ -n "$value" ]]; then
    DISCORD_WEBHOOK_URL="$value"
    export DISCORD_WEBHOOK_URL
  fi
  return 0
}

discord_json_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf "%s" "$value"
}

discord_notify() {
  local message="$1"
  local escaped payload

  load_discord_webhook
  if [[ -z "$DISCORD_WEBHOOK_URL" ]]; then
    echo "Discord webhook is not configured; notification skipped" >&2
    return 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is unavailable; Discord notification skipped" >&2
    return 1
  fi
  escaped="$(discord_json_escape "$message")"
  payload="{\"content\":\"$escaped\"}"
  if curl --fail --silent --show-error \
      -H "Content-Type: application/json" \
      --data "$payload" \
      --config <(printf 'url = "%s"\n' "$DISCORD_WEBHOOK_URL"); then
    return 0
  fi
  echo "Discord notification failed" >&2
  return 1
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
  local enable_thinking="${3:-}"
  local ngpu="$EVAL_GPUS"
  local gcw_args=(eval "$EVAL_SUITE" --checkpoint "$checkpoint")

  validate_eval_batching

  if [[ -n "${LIMIT:-}" && "$EVAL_SUITE" != "fineweb" ]]; then
    gcw_args+=(--limit "$LIMIT")
  fi
  if [[ -n "$EVAL_BATCH_SIZE" ]]; then
    gcw_args+=(--batch-size "$EVAL_BATCH_SIZE")
  fi
  if [[ "$EVAL_BATCH_SIZE" == "auto" ]]; then
    gcw_args+=(--max-batch-size "$EVAL_MAX_BATCH_SIZE")
  fi
  if [[ -n "$GENERATION_PROCESS_EVERY" ]]; then
    positive_int GENERATION_PROCESS_EVERY "$GENERATION_PROCESS_EVERY"
    gcw_args+=(--generation-process-every "$GENERATION_PROCESS_EVERY")
  fi
  if [[ -n "$EVAL_HEAD_SELECTION_METHOD" ]]; then
    gcw_args+=(--head-selection-method "$EVAL_HEAD_SELECTION_METHOD")
  fi
  if [[ -n "$enable_thinking" ]]; then
    gcw_args+=(--enable-thinking "$enable_thinking")
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
    worker_uv_run_eval torchrun --standalone --nproc-per-node="$ngpu" --no-python \
      bash -c 'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}-rank-${LOCAL_RANK}"; exec python -m gcw.cli "$@"' \
      bash "${gcw_args[@]}" || return $?
  else
    worker_uv_run_eval gcw "${gcw_args[@]}" || return $?
  fi
  if [[ -n "$out" ]]; then
    echo "Wrote $out"
  fi
}

run_eval_modes() {
  local checkpoint="$1"
  local out="$2"
  local mode suffix
  local -a modes
  if [[ -z "$EVAL_THINKING_MODES" ]]; then
    run_eval "$checkpoint" "$out"
    return
  fi
  IFS=',' read -r -a modes <<< "$EVAL_THINKING_MODES"
  for mode in "${modes[@]}"; do
    if [[ "$mode" == "true" ]]; then
      suffix="on"
    else
      suffix="off"
    fi
    run_eval "$checkpoint" "${out%.json}_thinking-${suffix}.json" "$mode" || return $?
  done
}

run_eval_manifest() {
  local manifest="$1"
  while IFS=$'\t' read -r manifest_checkpoint manifest_out; do
    if [[ -z "$manifest_checkpoint" ]]; then
      continue
    fi
    run_eval_modes "$manifest_checkpoint" "$manifest_out" || return $?
  done < "$manifest"
}

prepare_worker_environment() {
  local triton_run_id

  export PATH="$HOME/.local/bin:$PATH"
  export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"
  if [[ -d "$PWD/.venv/include/python3.12" ]]; then
    export CPATH="$PWD/.venv/include/python3.12:$PWD/.venv/include${CPATH:+:$CPATH}"
  fi
  export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
  export WANDB_MODE
  export PYTORCH_CUDA_ALLOC_CONF
  triton_run_id="${SLURM_JOB_ID:-$(basename "$STATE_DIR")}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/gcw-triton-${triton_run_id}-${SLURM_ARRAY_TASK_ID:-0}}"
  mkdir -p "$WANDB_DIR" slurm_out "$OUT_DIR"
}

run_one_config() {
  local config="$1"
  local output_dir max_steps keep_step_checkpoints run_name manifest step out nproc

  require_path "$config"

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

  prepare_worker_environment
  export GCW_KEEP_STEP_CHECKPOINTS="$keep_step_checkpoints"
  export GCW_STEP_CHECKPOINT_EVERY="$KEEP_STEP_CHECKPOINT_EVERY"

  if [[ "$EVAL_ONLY" != "1" ]]; then
    worker_uv_run_train python scripts/cluster/check_cuda_env.py || return $?
    nproc="$GPUS"
    if [[ "$nproc" == "1" ]]; then
      worker_uv_run_train gcw train --config "$config" || return $?
    else
      worker_uv_run_train torchrun --standalone --nproc_per_node="$nproc" --no-python \
        bash -c 'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}-rank-${LOCAL_RANK}"; exec python -m gcw.cli train --config "$1"' \
        bash "$config" || return $?
    fi

    if [[ "$EVAL_AFTER_TRAIN" != "1" ]]; then
      return
    fi

    echo "=== Evaluating $config ==="
  fi

  export GCW_EVAL_SKIP_TASKS
  worker_uv_run_eval python scripts/cluster/check_cuda_env.py || return $?
  if [[ -n "${CHECKPOINT:-}" || -n "${OUT:-}" ]]; then
    run_eval_modes "${CHECKPOINT:-$output_dir/checkpoint}" "${OUT:-${OUT_DIR}/${run_name}_${OUT_SUFFIX:-final}_${EVAL_SUITE}.json}"
  elif [[ -n "$EVAL_CHECKPOINT_STEP" ]]; then
    positive_int EVAL_CHECKPOINT_STEP "$EVAL_CHECKPOINT_STEP"
    out="${OUT_DIR}/${run_name}_step${EVAL_CHECKPOINT_STEP}"
    if [[ -n "${OUT_SUFFIX:-}" ]]; then
      out+="_${OUT_SUFFIX}"
    fi
    out+="_${EVAL_SUITE}.json"
    run_eval_modes "$output_dir/checkpoint-$EVAL_CHECKPOINT_STEP" "$out"
  elif [[ "$keep_step_checkpoints" == "1" ]]; then
    manifest="${EVAL_MANIFEST:-slurm_out/${run_name}_${EVAL_SUITE}_checkpoints.tsv}"
    : > "$manifest"
    for ((step = KEEP_STEP_CHECKPOINT_EVERY; step <= max_steps; step += KEEP_STEP_CHECKPOINT_EVERY)); do
      out="${OUT_DIR}/${run_name}_step${step}_${EVAL_SUITE}.json"
      printf "%s\t%s\n" "$output_dir/checkpoint-$step" "$out" >> "$manifest"
    done
    run_eval_manifest "$manifest"
  else
    run_eval_modes "$output_dir/checkpoint" "${OUT_DIR}/${run_name}_final_${EVAL_SUITE}.json"
  fi
}

run_one_eval_task() {
  local task="$1"
  local checkpoint out enable_thinking extra

  IFS=$'\t' read -r checkpoint out enable_thinking extra <<< "$task"
  require_path "$checkpoint"
  if [[ -z "$out" || -z "$enable_thinking" || -n "$extra" ]]; then
    echo "Invalid evaluation task; expected CHECKPOINT<TAB>OUTPUT<TAB>THINKING: $task" >&2
    return 1
  fi
  case "$enable_thinking" in
    true|false) ;;
    *)
      echo "Evaluation task thinking mode must be true or false: $task" >&2
      return 1
      ;;
  esac

  prepare_worker_environment
  export GCW_EVAL_SKIP_TASKS

  echo "=== Evaluating $checkpoint (thinking=$enable_thinking) ==="
  worker_uv_run_eval python scripts/cluster/check_cuda_env.py || return $?
  run_eval "$checkpoint" "$out" "$enable_thinking"
}

claim_next_config() {
  local config
  (
    flock -x 200
    touch "$CLAIMED_FILE"

    if ! config="$(
      awk '
        FILENAME == ARGV[1] { claimed[$0] = 1; next }
        /^[[:space:]]*($|#)/ { next }
        !($0 in claimed) { print; exit }
      ' "$CLAIMED_FILE" "$CONFIG_LIST"
    )"; then
      echo "Failed to read config queue from $CONFIG_LIST" >&2
      return 1
    fi
    if [[ -n "$config" ]]; then
      echo "$config" >> "$CLAIMED_FILE"
      echo "$config"
      return
    fi

    echo "STOP"
  ) 200>"$LOCK_FILE"
}

run_mixed_task() {
  local action config checkpoint out modes skip_tasks mode suffix target
  local -a thinking_modes
  local EVAL_ONLY=0 EVAL_AFTER_TRAIN=1 EVAL_THINKING_MODES
  local CHECKPOINT OUT GCW_EVAL_SKIP_TASKS

  IFS=$'\t' read -r action config checkpoint out modes skip_tasks <<< "$1"
  EVAL_THINKING_MODES="$modes"
  CHECKPOINT="$checkpoint"
  OUT="$out"
  GCW_EVAL_SKIP_TASKS="$skip_tasks"
  if [[ "$skip_tasks" == "-" ]]; then
    GCW_EVAL_SKIP_TASKS=""
  fi
  IFS=',' read -r -a thinking_modes <<< "$modes"
  for mode in "${thinking_modes[@]}"; do
    suffix=off
    [[ "$mode" != "true" ]] || suffix=on
    target="${out%.json}_thinking-${suffix}.json"
    if [[ -e "$target" ]]; then
      echo "Refusing to overwrite existing evaluation output: $target" >&2
      return 1
    fi
  done
  echo "=== Task action=$action config=$config checkpoint=$checkpoint thinking=$modes skip_tasks=$skip_tasks ==="
  if [[ "$action" == "train" ]]; then
    run_one_config "$config" || return $?
  else
    require_path "$checkpoint"
    prepare_worker_environment
    export GCW_EVAL_SKIP_TASKS
    worker_uv_run_eval python scripts/cluster/check_cuda_env.py || return $?
    run_eval_modes "$checkpoint" "$out" || return $?
  fi
}

write_mixed_task_list() {
  local list_path="$1" tasks_file="$2"
  local line action config checkpoint out modes skip_tasks mode suffix target key
  local line_number=0
  local -a thinking_modes
  local -A outputs=() training_configs=()

  : > "$list_path"
  while IFS= read -r line || [[ -n "$line" ]]; do
    line_number=$((line_number + 1))
    [[ ! "$line" =~ ^[[:space:]]*($|#) ]] || continue
    if ! awk -F '\t' 'NF != 6 {exit 1} {for (i=1;i<=NF;i++) if ($i == "") exit 1}' <<< "$line"; then
      echo "Invalid mixed task on line $line_number: expected six nonempty TSV fields" >&2
      return 1
    fi
    IFS=$'\t' read -r action config checkpoint out modes skip_tasks <<< "$line"
    case "$action" in
      train)
        require_path "$config"
        key="$(realpath "$config")"
        if [[ -n "${training_configs[$key]:-}" ]]; then
          echo "Duplicate training config on line $line_number: $config" >&2
          return 1
        fi
        training_configs[$key]=1
        ;;
      eval)
        if [[ "$config" != "-" ]]; then
          echo "Evaluation task CONFIG must be - on line $line_number" >&2
          return 1
        fi
        if [[ "$BACKEND" != "slurm" || ! "$DEPENDENCY" =~ ^afterok:[0-9]+(:[0-9]+)*$ ]]; then
          require_path "$checkpoint"
        fi
        ;;
      *)
        echo "Invalid mixed task action on line $line_number: $action" >&2
        return 1
        ;;
    esac
    case "$modes" in
      false|true|false,true|true,false) ;;
      *)
        echo "Invalid thinking modes on line $line_number: $modes" >&2
        return 1
        ;;
    esac
    if [[ "$out" != *.json || "$checkpoint" == "-" ]]; then
      echo "Mixed tasks require a checkpoint path and .json output base on line $line_number" >&2
      return 1
    fi
    IFS=',' read -r -a thinking_modes <<< "$modes"
    for mode in "${thinking_modes[@]}"; do
      suffix=off
      [[ "$mode" != "true" ]] || suffix=on
      target="${out%.json}_thinking-${suffix}.json"
      key="$(realpath -m "$target")"
      if [[ -e "$target" || -n "${outputs[$key]:-}" ]]; then
        echo "Existing or duplicate evaluation output on line $line_number: $target" >&2
        return 1
      fi
      outputs[$key]=1
    done
    printf '%s\n' "$line" >> "$list_path"
  done < "$tasks_file"
  if [[ ! -s "$list_path" ]]; then
    echo "No mixed tasks provided in $tasks_file" >&2
    return 1
  fi
}

run_claimed_item() {
  if [[ "$GCW_MIXED_TASK_MODE" == "1" ]]; then
    run_mixed_task "$1"
  elif [[ "$GCW_EVAL_TASK_MODE" == "1" ]]; then
    run_one_eval_task "$1"
  else
    run_one_config "$1"
  fi
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
    if run_claimed_item "$claim_result"; then
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

write_eval_task_list() {
  local list_path="$1"
  local tasks_file="$2"
  local line checkpoint out enable_thinking extra
  local line_number=0

  : > "$list_path"
  while IFS= read -r line || [[ -n "$line" ]]; do
    line_number=$((line_number + 1))
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
      continue
    fi
    IFS=$'\t' read -r checkpoint out enable_thinking extra <<< "$line"
    if [[ -z "$checkpoint" || -z "$out" || -z "$enable_thinking" || -n "$extra" ]]; then
      echo "Invalid evaluation task on line $line_number of $tasks_file; expected CHECKPOINT<TAB>OUTPUT<TAB>THINKING" >&2
      return 1
    fi
    if [[ ! -e "$checkpoint" ]]; then
      echo "Missing checkpoint on line $line_number of $tasks_file: $checkpoint" >&2
      return 1
    fi
    if [[ "$enable_thinking" != "true" && "$enable_thinking" != "false" ]]; then
      echo "Evaluation task thinking mode must be true or false on line $line_number of $tasks_file; got '$enable_thinking'" >&2
      return 1
    fi
    printf "%s\t%s\t%s\n" "$checkpoint" "$out" "$enable_thinking" >> "$list_path"
  done < "$tasks_file"

  if [[ ! -s "$list_path" ]]; then
    echo "No evaluation tasks provided in $tasks_file" >&2
    return 1
  fi
}

discover_local_gpus() {
  local raw="$LOCAL_GPUS"
  local gpu
  local -A seen=()

  LOCAL_GPU_POOL=()
  if [[ -z "$raw" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "Cannot detect local GPUs: nvidia-smi is unavailable. Set --local-gpus LIST." >&2
      return 1
    fi
    mapfile -t LOCAL_GPU_POOL < <(nvidia-smi --query-gpu=index --format=csv,noheader)
  else
    IFS=',' read -r -a LOCAL_GPU_POOL <<< "$raw"
  fi

  for gpu in "${LOCAL_GPU_POOL[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    if [[ -z "$gpu" || "$gpu" == *$'\t'* ]]; then
      echo "Invalid GPU ID in LOCAL_GPUS='$raw'" >&2
      return 1
    fi
    if [[ -n "${seen[$gpu]:-}" ]]; then
      echo "Duplicate GPU ID '$gpu' in LOCAL_GPUS='$raw'" >&2
      return 1
    fi
    seen[$gpu]=1
    LOCAL_GPU_POOL[${#seen[@]}-1]="$gpu"
  done

  if [[ "${#LOCAL_GPU_POOL[@]}" -eq 0 ]]; then
    echo "No local GPUs detected. Set --local-gpus LIST." >&2
    return 1
  fi
}

clean_reservations_locked() {
  local registry="$LOCAL_STATE_ROOT/gpu-reservations.tsv"
  local tmp="$registry.tmp.$$"
  local gpu pid start_ticks reserved_state worker_id

  : > "$tmp"
  if [[ -r "$registry" ]]; then
    while IFS=$'\t' read -r gpu pid start_ticks reserved_state worker_id; do
      if process_matches "$pid" "$start_ticks"; then
        printf "%s\t%s\t%s\t%s\t%s\n" "$gpu" "$pid" "$start_ticks" "$reserved_state" "$worker_id" >> "$tmp"
      fi
    done < "$registry"
  fi
  mv "$tmp" "$registry"
}

gpu_is_reserved_locked() {
  local wanted="$1"
  local gpu pid start_ticks reserved_state worker_id
  while IFS=$'\t' read -r gpu pid start_ticks reserved_state worker_id; do
    [[ "$gpu" == "$wanted" ]] && return 0
  done < "$LOCAL_STATE_ROOT/gpu-reservations.tsv"
  return 1
}

current_boot_id() {
  tr -d '\n' < /proc/sys/kernel/random/boot_id
}

stop_marker_is_current_boot() {
  local marker="$LOCAL_STATE_ROOT/vm-stop-requested"
  local marker_boot_id current
  [[ -r "$marker" ]] || return 1
  read -r marker_boot_id < "$marker" || return 1
  current="$(current_boot_id)"
  [[ -n "$current" && "$marker_boot_id" == "$current" ]]
}

vm_stop_is_inhibited_for_current_boot() {
  local marker="$LOCAL_STATE_ROOT/vm-stop-inhibited"
  local marker_boot_id current
  [[ -r "$marker" ]] || return 1
  read -r marker_boot_id < "$marker" || return 1
  current="$(current_boot_id)"
  [[ -n "$current" && "$marker_boot_id" == "$current" ]]
}

gpu_compute_is_active() {
  local pids
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Cannot confirm GPU idleness because nvidia-smi is unavailable; not stopping the VM" >&2
    return 0
  fi
  if ! pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
    echo "Cannot query active GPU processes; not stopping the VM" >&2
    return 0
  fi
  [[ -n "${pids//[[:space:]]/}" ]]
}

notify_job_finished() {
  local state_dir="$1"
  local marker="$state_dir/discord-job-finished.attempted"
  local completed=0 failed=0
  local host

  [[ -e "$marker" ]] && return
  if [[ -r "$state_dir/completed_configs" ]]; then
    completed="$(awk 'END { print NR + 0 }' "$state_dir/completed_configs")"
  fi
  if [[ -r "$state_dir/failed_configs.tsv" ]]; then
    failed="$(awk 'END { print NR + 0 }' "$state_dir/failed_configs.tsv")"
  fi
  host="$(hostname 2>/dev/null || printf unknown)"
  discord_notify "GCW local job finished on $host: $JOB_NAME ($completed completed configs, $failed failed)." || true
  : > "$marker"
}

notify_auto_stop() {
  local state_dir="$1"
  local vm_id="$2"
  local marker="$state_dir/discord-auto-stop.attempted"
  local host

  [[ -e "$marker" ]] && return
  host="$(hostname 2>/dev/null || printf unknown)"
  discord_notify "GCW local jobs are complete and GPU-idle on $host; auto-stopping Nebius instance $vm_id." || true
  : > "$marker"
}

idle_monitor_main() {
  local state_dir="${IDLE_MONITOR_STATE_DIR:?IDLE_MONITOR_STATE_DIR is required}"
  local rc vm_id boot_id
  local registry="$LOCAL_STATE_ROOT/gpu-reservations.tsv"
  local stop_marker="$LOCAL_STATE_ROOT/vm-stop-requested"

  positive_int LOCAL_IDLE_CHECK_INTERVAL "$LOCAL_IDLE_CHECK_INTERVAL"
  mkdir -p "$LOCAL_STATE_ROOT"
  echo "Idle-stop monitor started for $state_dir"

  while true; do
    if (
      flock -x 9
      clean_reservations_locked
      if vm_stop_is_inhibited_for_current_boot; then
        echo "VM auto-stop inhibited by a user cancellation; keeping the node running"
        exit 0
      fi

      if [[ -s "$registry" ]]; then
        exit 10
      fi
      notify_job_finished "$state_dir"
      if stop_marker_is_current_boot; then
        echo "VM stop has already been requested during this boot"
        exit 0
      fi
      if gpu_compute_is_active; then
        exit 10
      fi
      if [[ ! -r "$NEBIUS_INSTANCE_ID_FILE" ]]; then
        echo "Cannot read Nebius instance ID from $NEBIUS_INSTANCE_ID_FILE; will retry" >&2
        exit 11
      fi
      vm_id="$(cat "$NEBIUS_INSTANCE_ID_FILE")"
      if [[ -z "$vm_id" ]]; then
        echo "Nebius instance ID is empty in $NEBIUS_INSTANCE_ID_FILE; will retry" >&2
        exit 11
      fi
      if ! command -v nebius >/dev/null 2>&1; then
        echo "nebius CLI is unavailable; will retry" >&2
        exit 11
      fi

      notify_auto_stop "$state_dir" "$vm_id"
      echo "No local workers or GPU compute processes remain; stopping Nebius instance $vm_id"
      sync
      if nebius compute instance stop \
          --id "$vm_id" \
          --profile "$NEBIUS_STOP_PROFILE" \
          --async; then
        boot_id="$(current_boot_id)"
        printf "%s\n" "$boot_id" > "$stop_marker"
        echo "Nebius stop request submitted"
        exit 0
      fi
      echo "Nebius stop request failed; will retry" >&2
      exit 11
    ) 9> "$LOCAL_STATE_ROOT/manager.lock"; then
      return
    else
      rc="$?"
    fi

    case "$rc" in
      10|11) sleep "$LOCAL_IDLE_CHECK_INTERVAL" ;;
      *)
        echo "Idle-stop monitor encountered unexpected status $rc; will retry" >&2
        sleep "$LOCAL_IDLE_CHECK_INTERVAL"
        ;;
    esac
  done
}

start_local_worker() {
  local worker_id="$1"
  local gpu_group="$2"
  local log_file="$STATE_DIR/local-worker-$worker_id.out"
  local exit_file="$STATE_DIR/worker-$worker_id.exit"
  local pid start_ticks gpu
  local -a reserved_gpus
  local worker_env=(
    "GCW_ARRAY_MODE=worker"
    "GCW_REPO_ROOT=$repo_root"
    "CONFIG_LIST=$CONFIG_LIST"
    "STATE_DIR=$STATE_DIR"
    "WORKER_EXIT_FILE=$exit_file"
    "SLURM_ARRAY_TASK_ID=$worker_id"
    "CUDA_VISIBLE_DEVICES=$gpu_group"
    "JOB_NAME=$JOB_NAME"
    "GPUS=$GPUS"
    "EVAL_GPUS=$EVAL_GPUS"
    "EVAL_AFTER_TRAIN=$EVAL_AFTER_TRAIN"
    "EVAL_ONLY=$EVAL_ONLY"
    "EVAL_SUITE=$EVAL_SUITE"
    "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
    "EVAL_MAX_BATCH_SIZE=$EVAL_MAX_BATCH_SIZE"
    "EVAL_CHECKPOINT_STEP=$EVAL_CHECKPOINT_STEP"
    "EVAL_THINKING_MODES=$EVAL_THINKING_MODES"
    "GCW_EVAL_TASK_MODE=$GCW_EVAL_TASK_MODE"
    "GCW_MIXED_TASK_MODE=$GCW_MIXED_TASK_MODE"
    "GENERATION_PROCESS_EVERY=$GENERATION_PROCESS_EVERY"
    "EVAL_HEAD_SELECTION_METHOD=$EVAL_HEAD_SELECTION_METHOD"
    "GCW_EVAL_SKIP_TASKS=$GCW_EVAL_SKIP_TASKS"
    "WANDB_MODE=$WANDB_MODE"
    "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
    "UV_EXTRAS=$UV_EXTRAS"
    "TRAIN_UV_EXTRAS=$TRAIN_UV_EXTRAS"
    "EVAL_UV_EXTRAS=$EVAL_UV_EXTRAS"
    "KEEP_STEP_CHECKPOINTS=$KEEP_STEP_CHECKPOINTS"
    "KEEP_STEP_CHECKPOINT_EVERY=$KEEP_STEP_CHECKPOINT_EVERY"
    "OUT_DIR=$OUT_DIR"
    "SFT_BACKBONE=$SFT_BACKBONE"
    "TOKENIZER_DIR=$TOKENIZER_DIR"
  )

  setsid env "${worker_env[@]}" "$script_path" > "$log_file" 2>&1 < /dev/null 9>&- &
  pid=$!
  start_ticks="$(process_start_ticks "$pid")"
  if [[ -z "$start_ticks" ]]; then
    echo "Local worker $worker_id exited before its PID could be recorded. See $log_file" >&2
    return 1
  fi
  printf "%s\t%s\t%s\n" "$pid" "$start_ticks" "$gpu_group" > "$STATE_DIR/worker-$worker_id.pid"
  IFS=',' read -r -a reserved_gpus <<< "$gpu_group"
  for gpu in "${reserved_gpus[@]}"; do
    printf "%s\t%s\t%s\t%s\t%s\n" "$gpu" "$pid" "$start_ticks" "$STATE_DIR" "$worker_id" >> "$LOCAL_STATE_ROOT/gpu-reservations.tsv"
  done
  echo "Worker $worker_id: pid=$pid GPUs=$gpu_group log=$log_file"
}
start_idle_monitor() {
  local state_dir="$1"
  local log_file="$state_dir/idle-stop-monitor.out"
  local pid start_ticks

  if [[ "$LOCAL_STOP_WHEN_IDLE" != "1" ]]; then
    return
  fi

  setsid env \
    "GCW_ARRAY_MODE=idle-monitor" \
    "GCW_REPO_ROOT=$repo_root" \
    "LOCAL_STATE_ROOT=$LOCAL_STATE_ROOT" \
    "LOCAL_IDLE_CHECK_INTERVAL=$LOCAL_IDLE_CHECK_INTERVAL" \
    "NEBIUS_INSTANCE_ID_FILE=$NEBIUS_INSTANCE_ID_FILE" \
    "JOB_NAME=$JOB_NAME" \
    "NEBIUS_STOP_PROFILE=$NEBIUS_STOP_PROFILE" \
    "IDLE_MONITOR_STATE_DIR=$state_dir" \
    "$script_path" > "$log_file" 2>&1 < /dev/null 9>&- &
  pid=$!
  start_ticks="$(process_start_ticks "$pid")"
  if [[ -z "$start_ticks" ]]; then
    echo "Idle-stop monitor exited before its PID could be recorded. See $log_file" >&2
    return 1
  fi
  printf "%s\t%s\n" "$pid" "$start_ticks" > "$state_dir/idle-stop-monitor.pid"
  echo "Idle-stop monitor: pid=$pid log=$log_file"
}


local_launcher() {
  local state_dir="$1"
  local config_list="$2"
  local alloc_gpus="$3"
  local dry_run="$4"
  local gpu group worker available_count capacity worker_count start_index
  local -a available_gpus=()

  STATE_DIR="$state_dir"
  CONFIG_LIST="$config_list"
  discover_local_gpus || return $?
  mkdir -p "$LOCAL_STATE_ROOT"

  (
    flock -x 9
    if stop_marker_is_current_boot; then
      echo "A Nebius stop has already been requested during this boot; refusing to start new workers" >&2
      return 1
    fi
    if [[ -e "$LOCAL_STATE_ROOT/vm-stop-requested" ]]; then
      rm -f "$LOCAL_STATE_ROOT/vm-stop-requested"
    fi
    if [[ "$dry_run" != "1" && "$LOCAL_STOP_WHEN_IDLE" == "1" ]]; then
      rm -f "$LOCAL_STATE_ROOT/vm-stop-inhibited"
    fi

    clean_reservations_locked

    for gpu in "${LOCAL_GPU_POOL[@]}"; do
      if ! gpu_is_reserved_locked "$gpu"; then
        available_gpus+=("$gpu")
      fi
    done

    available_count="${#available_gpus[@]}"
    capacity=$((available_count / alloc_gpus))
    if [[ "$capacity" -le 0 ]]; then
      echo "No unreserved local GPU group of size $alloc_gpus is available from: ${LOCAL_GPU_POOL[*]}" >&2
      return 1
    fi
    worker_count="$ARRAY_WORKERS"
    if [[ "$worker_count" -gt "$capacity" ]]; then
      worker_count="$capacity"
      echo "Only $worker_count worker(s) fit on $available_count unreserved GPU(s); all configs will still be consumed."
    fi

    echo "Backend: local"
    echo "Config list: $config_list"
    echo "State directory: $state_dir"
    echo "Starting $worker_count local worker(s), with $alloc_gpus GPU(s) per worker"
    echo "Stop VM when idle: $LOCAL_STOP_WHEN_IDLE"

    if [[ "$dry_run" == "1" ]]; then
      for ((worker = 0; worker < worker_count; worker += 1)); do
        start_index=$((worker * alloc_gpus))
        group="${available_gpus[$start_index]}"
        for ((gpu = 1; gpu < alloc_gpus; gpu += 1)); do
          group+=",${available_gpus[$((start_index + gpu))]}"
        done
        echo "Worker $worker: assigned GPUs $group"
      done
      return
    fi

    printf "local\n" > "$state_dir/backend"
    printf "%s\n" "$ARRAY_WORKERS" > "$state_dir/requested_workers"
    printf "%s\n" "$worker_count" > "$state_dir/started_workers"
    for ((worker = 0; worker < worker_count; worker += 1)); do
      start_index=$((worker * alloc_gpus))
      group="${available_gpus[$start_index]}"
      for ((gpu = 1; gpu < alloc_gpus; gpu += 1)); do
        group+=",${available_gpus[$((start_index + gpu))]}"
      done
      start_local_worker "$worker" "$group"
    done
    start_idle_monitor "$state_dir"
  ) 9> "$LOCAL_STATE_ROOT/manager.lock"

  if [[ "$dry_run" != "1" ]]; then
    echo "Status: $script_path status $state_dir"
    echo "Cancel: $script_path cancel $state_dir"
  fi
}

status_main() {
  local state_dir="${1:-}"
  local pid_file worker_id status rc
  local running=0 succeeded=0 failed=0 cancelled=0 stopped=0
  local -a pid_files

  if [[ -z "$state_dir" ]]; then
    echo "status requires a state directory" >&2
    return 2
  fi
  require_path "$state_dir"
  shopt -s nullglob
  pid_files=("$state_dir"/worker-*.pid)
  shopt -u nullglob
  if [[ "${#pid_files[@]}" -eq 0 ]]; then
    echo "No local worker PID files found in $state_dir" >&2
    return 1
  fi

  printf "%-8s %-10s %-10s %s\n" WORKER STATUS PID GPUS
  for pid_file in "${pid_files[@]}"; do
    worker_id="${pid_file##*/worker-}"
    worker_id="${worker_id%.pid}"
    read_pid_file "$pid_file"
    status="STOPPED"
    rc="-"
    if process_matches "$WORKER_PID" "$WORKER_START_TICKS"; then
      if [[ -e "$state_dir/worker-$worker_id.cancelled" ]]; then
        status="STOPPING"
      else
        status="RUNNING"
      fi
      ((running += 1))
    elif [[ -e "$state_dir/worker-$worker_id.cancelled" ]]; then
      status="CANCELLED"
      ((cancelled += 1))
    elif [[ -r "$state_dir/worker-$worker_id.exit" ]]; then
      read -r rc < "$state_dir/worker-$worker_id.exit" || true
      if [[ "$rc" == "0" ]]; then
        status="SUCCEEDED"
        ((succeeded += 1))
      else
        status="FAILED($rc)"
        ((failed += 1))
      fi
    else
      ((stopped += 1))
    fi
    printf "%-8s %-10s %-10s %s\n" "$worker_id" "$status" "${WORKER_PID:--}" "${WORKER_GPUS:--}"
  done
  echo "Summary: $running running, $succeeded succeeded, $failed failed, $cancelled cancelled, $stopped stopped"
}


cancel_main() {
  local force=0
  local state_dir=""
  local signal="TERM"
  local pid_file worker_id
  local cancelled_count=0
  local -a pid_files

  if [[ "${1:-}" == "--force" ]]; then
    force=1
    shift
  fi
  if [[ "$#" -ne 1 ]]; then
    echo "cancel requires exactly one state directory" >&2
    return 2
  fi
  state_dir="$1"
  require_path "$state_dir"
  if [[ "$force" == "1" ]]; then
    signal="KILL"
  fi

  mkdir -p "$LOCAL_STATE_ROOT"
  printf "%s\n" "$(current_boot_id)" > "$LOCAL_STATE_ROOT/vm-stop-inhibited"
  echo "VM auto-stop disabled because this job was cancelled"
  shopt -s nullglob
  pid_files=("$state_dir"/worker-*.pid)
  shopt -u nullglob
  for pid_file in "${pid_files[@]}"; do
    worker_id="${pid_file##*/worker-}"
    worker_id="${worker_id%.pid}"
    read_pid_file "$pid_file"
    if process_matches "$WORKER_PID" "$WORKER_START_TICKS"; then
      touch "$state_dir/worker-$worker_id.cancelled"
      if ! kill "-$signal" -- "-$WORKER_PID" 2>/dev/null; then
        kill "-$signal" "$WORKER_PID" 2>/dev/null || true
      fi
      echo "Sent SIG$signal to worker $worker_id process group $WORKER_PID"
      ((cancelled_count += 1))
    fi
  done
  if [[ "$cancelled_count" -eq 0 ]]; then
    echo "No running local workers found in $state_dir"
  fi
}

launcher_main() {
  local configs_file="" eval_tasks_file="" tasks_file="" dry_run=0 state_dir config_list workers_arg
  local alloc_gpus export_file
  local args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --backend)
        BACKEND="${2:-}"
        if [[ -z "$BACKEND" ]]; then
          echo "--backend requires auto, slurm, or local" >&2
          exit 1
        fi
        validate_backend "$BACKEND"
        shift 2
        ;;
      --local-gpus)
        LOCAL_GPUS="${2:-}"
        if [[ -z "$LOCAL_GPUS" ]]; then
          echo "--local-gpus requires a comma-separated list" >&2
          exit 1
        fi
        shift 2
        ;;
      --configs-file)
        configs_file="${2:-}"
        if [[ -z "$configs_file" ]]; then
          echo "--configs-file requires a path" >&2
          exit 1
        fi
        shift 2
        ;;
      --eval-tasks-file)
        eval_tasks_file="${2:-}"
        if [[ -z "$eval_tasks_file" ]]; then
          echo "--eval-tasks-file requires a path" >&2
          exit 1
        fi
        shift 2
        ;;
      --tasks-file)
        tasks_file="${2:-}"
        if [[ -z "$tasks_file" ]]; then
          echo "--tasks-file requires a path" >&2
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
  validate_eval_batching
  validate_eval_thinking_modes
  if [[ -n "$tasks_file" ]]; then
    if [[ -n "$configs_file" || -n "$eval_tasks_file" || "${#args[@]}" -gt 0 ]]; then
      echo "--tasks-file cannot be combined with other task inputs" >&2
      return 1
    fi
    if [[ "$EVAL_ONLY" == "1" || -n "$EVAL_CHECKPOINT_STEP" || -n "${CHECKPOINT:-}" || -n "${OUT:-}" || "$EVAL_SUITE" == "fineweb" ]]; then
      echo "--tasks-file owns checkpoint and train/eval selection and requires a generative evaluation suite" >&2
      return 1
    fi
    require_path "$tasks_file"
    GCW_MIXED_TASK_MODE=1
    EVAL_AFTER_TRAIN=1
  fi
  if [[ -n "$eval_tasks_file" ]]; then
    if [[ -n "$configs_file" || "${#args[@]}" -gt 0 ]]; then
      echo "--eval-tasks-file cannot be combined with configs" >&2
      return 1
    fi
    if [[ "$EVAL_ONLY" != "1" ]]; then
      echo "--eval-tasks-file requires EVAL_ONLY=1" >&2
      return 1
    fi
    if [[ -n "$EVAL_CHECKPOINT_STEP" || -n "$EVAL_THINKING_MODES" ]]; then
      echo "--eval-tasks-file cannot be combined with EVAL_CHECKPOINT_STEP or EVAL_THINKING_MODES" >&2
      return 1
    fi
    require_path "$eval_tasks_file"
    GCW_EVAL_TASK_MODE=1
  fi
  if [[ -n "$GENERATION_PROCESS_EVERY" ]]; then
    positive_int GENERATION_PROCESS_EVERY "$GENERATION_PROCESS_EVERY"
    if [[ "$EVAL_SUITE" == "fineweb" ]]; then
      echo "GENERATION_PROCESS_EVERY is not supported for fineweb evaluation" >&2
      return 1
    fi
  fi
  if [[ -n "$EVAL_HEAD_SELECTION_METHOD" && "$EVAL_HEAD_SELECTION_METHOD" != "random" ]]; then
    echo "EVAL_HEAD_SELECTION_METHOD must be random when set; got '$EVAL_HEAD_SELECTION_METHOD'" >&2
    return 1
  fi
  if [[ -n "$EVAL_HEAD_SELECTION_METHOD" && "$EVAL_SUITE" != "paper7" && "$EVAL_SUITE" != "paper7-legacy" ]]; then
    echo "EVAL_HEAD_SELECTION_METHOD is supported only for paper7 evaluation" >&2
    return 1
  fi
  if [[ -n "$EVAL_CHECKPOINT_STEP" ]]; then
    positive_int EVAL_CHECKPOINT_STEP "$EVAL_CHECKPOINT_STEP"
  fi
  validate_backend "$BACKEND"
  if [[ "$BACKEND" == "auto" ]]; then
    if command -v sbatch >/dev/null 2>&1; then
      BACKEND="slurm"
    else
      BACKEND="local"
    fi
  fi
  if [[ "$BACKEND" == "local" ]]; then
    case "$LOCAL_STOP_WHEN_IDLE" in
      0|1) ;;
      *)
        echo "LOCAL_STOP_WHEN_IDLE must be 0 or 1; got '$LOCAL_STOP_WHEN_IDLE'" >&2
        return 1
        ;;
    esac
    positive_int LOCAL_IDLE_CHECK_INTERVAL "$LOCAL_IDLE_CHECK_INTERVAL"
  fi
  if [[ "$BACKEND" == "slurm" && "$dry_run" != "1" ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "Slurm backend requested, but sbatch is unavailable" >&2
    return 1
  fi
  if [[ "$BACKEND" == "local" && -n "$DEPENDENCY" ]]; then
    echo "DEPENDENCY is not supported by the local backend" >&2
    return 1
  fi
  if [[ "$BACKEND" == "local" && -n "$EXCLUDE" ]]; then
    echo "EXCLUDE is not supported by the local backend" >&2
    return 1
  fi

  alloc_gpus="$GPUS"
  if [[ ( "$EVAL_AFTER_TRAIN" == "1" || "$EVAL_ONLY" == "1" ) && "$EVAL_GPUS" -gt "$alloc_gpus" ]]; then
    alloc_gpus="$EVAL_GPUS"
  fi

  if [[ "$BACKEND" == "slurm" ]]; then
    mkdir -p "$SLURM_STATE_ROOT" "$SLURM_LOG_DIR"
    state_dir="$SLURM_STATE_ROOT/${JOB_NAME}_$(date +%Y%m%d_%H%M%S)_$$"
  else
    mkdir -p slurm_out
    state_dir="$repo_root/slurm_out/${JOB_NAME}_$(date +%Y%m%d_%H%M%S)_$$"
  fi
  mkdir -p "$state_dir"
  config_list="$state_dir/configs.txt"
  if [[ -n "$tasks_file" ]]; then
    write_mixed_task_list "$config_list" "$tasks_file"
  elif [[ -n "$eval_tasks_file" ]]; then
    write_eval_task_list "$config_list" "$eval_tasks_file"
  else
    write_config_list "$config_list" "$configs_file" "${args[@]}"
  fi

  if [[ "$BACKEND" == "local" ]]; then
    local_launcher "$state_dir" "$config_list" "$alloc_gpus" "$dry_run"
    return
  fi

  workers_arg="0-$((ARRAY_WORKERS - 1))"
  export_file="$state_dir/slurm-environment"
  write_slurm_environment "$export_file" "$config_list" "$state_dir"
  chmod 600 "$export_file"
  load_environment_file "$export_file"
  echo "Backend: slurm"
  echo "Config list: $config_list"
  echo "Submitting $ARRAY_WORKERS worker(s) as array $workers_arg"

  sbatch_cmd=(
    sbatch
    --parsable
    --job-name="$JOB_NAME"
    --gres="gpu:$alloc_gpus"
    --array="$workers_arg"
    --output="$SLURM_LOG_DIR/${JOB_NAME}-%A_%a.out"
    --export=ALL
  )
  if [[ -n "$PARTITION" ]]; then
    sbatch_cmd+=(--partition="$PARTITION")
  fi
  if [[ -n "$ARRAY_TIME" ]]; then
    sbatch_cmd+=(--time="$ARRAY_TIME")
  fi
  if [[ -n "$CPUS_PER_TASK" ]]; then
    sbatch_cmd+=(--cpus-per-task="$CPUS_PER_TASK")
  fi
  if [[ -n "$MEM" ]]; then
    sbatch_cmd+=(--mem="$MEM")
  fi
  if [[ -n "$DEPENDENCY" ]]; then
    sbatch_cmd+=(--dependency="$DEPENDENCY")
  fi
  if [[ -n "$EXCLUDE" ]]; then
    sbatch_cmd+=(--exclude="$EXCLUDE")
  fi
  sbatch_cmd+=("$script_path")

  if [[ "$dry_run" == "1" ]]; then
    printf '%q ' "${sbatch_cmd[@]}"
    printf '\n'
    rm -f "$export_file"
    return
  fi

  if job_id="$("${sbatch_cmd[@]}")"; then
    rm -f "$export_file"
  else
    rc="$?"
    rm -f "$export_file"
    return "$rc"
  fi
  echo "Submitted array job $job_id"
}

record_worker_exit() {
  local rc="$?"
  local tmp
  trap - EXIT
  set +e
  if [[ -n "${WORKER_EXIT_FILE:-}" ]]; then
    tmp="$WORKER_EXIT_FILE.tmp.$$"
    printf "%s\n" "$rc" > "$tmp"
    mv "$tmp" "$WORKER_EXIT_FILE"
  fi
  exit "$rc"
}

if [[ "${GCW_ARRAY_MODE:-}" == "worker" ]]; then
  trap record_worker_exit EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
  worker_main
elif [[ "${GCW_ARRAY_MODE:-}" == "idle-monitor" ]]; then
  idle_monitor_main
elif [[ "${1:-}" == "status" ]]; then
  shift
  status_main "$@"
elif [[ "${1:-}" == "cancel" ]]; then
  shift
  cancel_main "$@"
else
  launcher_main "$@"
fi
