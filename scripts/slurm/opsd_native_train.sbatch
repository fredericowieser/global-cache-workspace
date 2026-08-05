#!/usr/bin/env bash
#SBATCH --job-name=gcw-opsd-native-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=5-00:00:00

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"
export WANDB_DIR="${WANDB_DIR:-$HOME/wandb}"

REPO="${REPO:-$HOME/code/global-cache-workspace}"
CONFIG="${CONFIG:?CONFIG is required}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION is required}"
export SOURCE_REVISION

cd "$REPO"
ACTUAL_REVISION="$(git rev-parse HEAD)"
[[ "$ACTUAL_REVISION" == "$SOURCE_REVISION" ]] || {
  printf 'source revision mismatch: expected=%s actual=%s\n' "$SOURCE_REVISION" "$ACTUAL_REVISION" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
  printf 'training requires a clean checkout\n' >&2
  exit 2
}
[[ -f "$CONFIG" ]] || {
  printf 'missing train config: %s\n' "$CONFIG" >&2
  exit 2
}
mkdir -p "$WANDB_DIR"

PREFLIGHT_RAW="$(
  uv run --locked --extra gpu python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from gcw.config import load_train_config
config = load_train_config(sys.argv[1])
distill = config.distillation
if distill is None:
    raise SystemExit("config has no distillation section")
if distill.teacher_trajectory_mode != "native_full" or distill.visible_trajectory_mode != "both":
    raise SystemExit("config is not native_full/both")
if distill.max_optimizer_steps is None:
    raise SystemExit("config has no max_optimizer_steps")
if distill.max_optimizer_steps > 1000:
    raise SystemExit(
        "runs above 1000 steps require the resumable 10k promotion runner and "
        "validated 1k evaluation evidence"
    )
if not config.wandb.enabled or config.wandb.mode != "online":
    raise SystemExit("campaign training requires online W&B")
paths = {
    "OUTPUT_DIR": config.output_dir,
    "BACKBONE_CHECKPOINT": config.backbone_checkpoint,
    "DATA_DIR": config.data.data_dir,
    "TOKENIZER_SOURCE": config.data.tokenizer.removeprefix("hf:"),
    "MAX_OPTIMIZER_STEPS": str(distill.max_optimizer_steps),
    "EFFECTIVE_BATCH_EXAMPLES": str(distill.gradient_accumulation_steps),
    "RECORD_CACHE_DISPLACEMENTS": str(distill.record_cache_displacements).lower(),
    "WANDB_MODE_CONFIG": str(config.wandb.mode),
}
for key, value in paths.items():
    if not value:
        raise SystemExit(f"missing config-derived {key}")
    print(f"{key}={value}")
PY
)"
mapfile -t PREFLIGHT <<<"$PREFLIGHT_RAW"
for assignment in "${PREFLIGHT[@]}"; do
  case "$assignment" in
    OUTPUT_DIR=*|BACKBONE_CHECKPOINT=*|DATA_DIR=*|TOKENIZER_SOURCE=*|MAX_OPTIMIZER_STEPS=*|EFFECTIVE_BATCH_EXAMPLES=*|RECORD_CACHE_DISPLACEMENTS=*|WANDB_MODE_CONFIG=*)
      export "$assignment" ;;
    *) printf 'unexpected config preflight output: %s\n' "$assignment" >&2; exit 2 ;;
  esac
done

if [[ "$MAX_OPTIMIZER_STEPS" -eq 1000 ]]; then
  PROMOTION_EVIDENCE="${PROMOTION_EVIDENCE:?1k requires smoke PROMOTION_EVIDENCE}"
  export PROMOTION_EVIDENCE
  uv run --locked --extra gpu python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
success_path = Path(os.environ["PROMOTION_EVIDENCE"]).resolve(strict=True)
success = json.loads(success_path.read_text())
if success.get("source_revision") != os.environ["SOURCE_REVISION"]:
    raise SystemExit("smoke evidence source revision mismatch")
if success.get("optimizer_steps") != 1 or success.get("effective_batch_examples") != 1:
    raise SystemExit("promotion evidence is not a one-step batch-one smoke")
summary_entry = success.get("files", {}).get("summary.json")
if not isinstance(summary_entry, dict):
    raise SystemExit("smoke evidence does not authenticate summary.json")
summary_path = success_path.parent / "summary.json"
if hashlib.sha256(summary_path.read_bytes()).hexdigest() != summary_entry.get("sha256"):
    raise SystemExit("smoke summary digest mismatch")
summary = json.loads(summary_path.read_text())
if summary.get("seen_prompts", 65) > 64 or summary.get("valid_prompts") != 1:
    raise SystemExit("smoke strict-yield gate failed")
projected_seconds = float(summary["elapsed_seconds"]) * 16_000
if projected_seconds > 4.5 * 24 * 60 * 60:
    raise SystemExit(
        f"projected 1k walltime exceeds 4.5 days: {projected_seconds:.0f} seconds"
    )
print(f"smoke_promotion_projected_seconds={projected_seconds:.3f}")
PY
fi

[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || {
  printf 'refusing to overwrite output: %s\n' "$OUTPUT_DIR" >&2
  exit 2
}
mkdir -p "$(dirname "$OUTPUT_DIR")"
mkdir "$OUTPUT_DIR" || {
  printf 'failed to claim fresh output directory: %s\n' "$OUTPUT_DIR" >&2
  exit 2
}
[[ -d "$BACKBONE_CHECKPOINT" ]] || {
  printf 'missing backbone checkpoint: %s\n' "$BACKBONE_CHECKPOINT" >&2
  exit 2
}
[[ -f "$DATA_DIR" ]] || {
  printf 'missing data file: %s\n' "$DATA_DIR" >&2
  exit 2
}
[[ -d "$TOKENIZER_SOURCE" ]] || {
  printf 'missing tokenizer source: %s\n' "$TOKENIZER_SOURCE" >&2
  exit 2
}

uv run --locked --extra gpu python scripts/check_cuda_env.py
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
export GPU_NAME
[[ "$GPU_NAME" == *H100* ]] || {
  printf 'campaign requires H100; allocated GPU is %s\n' "$GPU_NAME" >&2
  exit 2
}
uv run --locked --extra gpu python - <<'PY'
import wandb
if not wandb.api.api_key:
    raise SystemExit("W&B is not authenticated")
print("wandb_authenticated=true")
PY

uv run --locked --extra gpu gcw train --config "$CONFIG"

[[ "$(git rev-parse HEAD)" == "$SOURCE_REVISION" ]] || {
  printf 'source revision changed during training\n' >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
  printf 'source checkout became dirty during training\n' >&2
  exit 2
}

uv run --locked --extra gpu python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
root = Path(os.environ["OUTPUT_DIR"])
summary_path = root / "summary.json"
metrics_path = root / "metrics.jsonl"
run_config_path = root / "run_config.json"
checkpoint = root / "final"
summary = json.loads(summary_path.read_text())
metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
expected_steps = int(os.environ["MAX_OPTIMIZER_STEPS"])
expected_batch = int(os.environ["EFFECTIVE_BATCH_EXAMPLES"])
expected_examples = expected_steps * expected_batch
if summary.get("optimizer_steps") != expected_steps:
    raise SystemExit(
        f"optimizer step mismatch: {summary.get('optimizer_steps')} != {expected_steps}"
    )
if summary.get("effective_batch_examples") != expected_batch:
    raise SystemExit("effective batch mismatch")
if summary.get("valid_prompts") != expected_examples:
    raise SystemExit(
        f"valid example mismatch: {summary.get('valid_prompts')} != {expected_examples}"
    )
if summary.get("source_revision") != os.environ["SOURCE_REVISION"]:
    raise SystemExit("summary source revision mismatch")
if summary.get("source_clean") is not True:
    raise SystemExit("training source was not clean")
if summary.get("checkpoint_fidelity_verified") is not True:
    raise SystemExit("checkpoint fidelity was not verified")
if not summary.get("wandb_run_id") or not summary.get("wandb_run_url"):
    raise SystemExit("online W&B run identity is missing")
if summary.get("initial_processor_sha256") == summary.get("pre_save_processor_sha256"):
    raise SystemExit("processor state did not change")
if hashlib.sha256(run_config_path.read_bytes()).hexdigest() != summary.get(
    "run_config_sha256"
):
    raise SystemExit("run config digest mismatch")
trained = [row for row in metrics if row.get("status") == "trained"]
if len(trained) != expected_examples:
    raise SystemExit(f"trained metric rows mismatch: {len(trained)} != {expected_examples}")
step_rows = [row for row in trained if row.get("grad_norm") is not None]
if len(step_rows) != expected_steps:
    raise SystemExit(f"optimizer metric rows mismatch: {len(step_rows)} != {expected_steps}")
if [row.get("optimizer_steps") for row in step_rows] != list(range(1, expected_steps + 1)):
    raise SystemExit("optimizer metric step sequence is not exact")
finite_fields = (
    "loss",
    "unclipped_kl",
    "clipping_fraction",
    "student_entropy",
    "teacher_entropy",
    "processor_gate_before_example",
    "processor_gate_after_example",
)
for row in trained:
    for field in finite_fields:
        value = row.get(field)
        if not isinstance(value, (int, float)) or not __import__("math").isfinite(value):
            raise SystemExit(f"non-finite or missing {field} at source row {row.get('source_index')}")
if os.environ["RECORD_CACHE_DISPLACEMENTS"] == "true":
    displacements = [
        abs(float(value))
        for row in trained
        for key in (
            "cache_relative_l2_displacements",
            "teacher_trajectory_cache_relative_l2_displacements",
        )
        for value in row.get(key, [])
    ]
    if not displacements or max(displacements) <= 0.0:
        raise SystemExit("cache displacement evidence is missing or zero")
required = [
    summary_path,
    run_config_path,
    metrics_path,
    root / "teacher-traces.jsonl",
    checkpoint / "model.safetensors",
    checkpoint / "config.json",
    checkpoint / "train_state.json",
]
files = {}
for path in required:
    if not path.is_file():
        raise SystemExit(f"missing required output: {path}")
    files[str(path.relative_to(root))] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
payload = json.dumps(
    {
        "source_revision": os.environ["SOURCE_REVISION"],
        "optimizer_steps": expected_steps,
        "effective_batch_examples": expected_batch,
        "valid_examples": expected_examples,
        "gpu_name": os.environ["GPU_NAME"],
        "wandb_run_id": summary["wandb_run_id"],
        "wandb_run_url": summary["wandb_run_url"],
        "files": files,
    },
    indent=2,
    sort_keys=True,
).encode()
success = root / "SUCCESS.json"
temporary = root / f".SUCCESS.json.{os.getpid()}.tmp"
try:
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, success)
    directory_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    temporary.unlink(missing_ok=True)
PY
