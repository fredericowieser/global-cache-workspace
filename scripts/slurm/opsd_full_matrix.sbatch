#!/usr/bin/env bash
#SBATCH --job-name=gcw-opsd-full-matrix
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export GCW_STRICT_FLASH="${GCW_STRICT_FLASH:-1}"

REPO="${REPO:-$HOME/code/global-cache-workspace}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION is required}"
TOKENIZER_SOURCE="${TOKENIZER_SOURCE:?TOKENIZER_SOURCE is required}"
DATASETS_CACHE="${DATASETS_CACHE:?DATASETS_CACHE is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
TASKS="gsm8k_openmath,svamp,gsm8k_hard,logiqa_opsd_generate,agieval_gaokao_mathqa_opsd_generate"
export CHECKPOINT SOURCE_REVISION OUT_DIR TASKS

cd "$REPO"
ACTUAL_REVISION="$(git rev-parse HEAD)"
[[ "$ACTUAL_REVISION" == "$SOURCE_REVISION" ]] || {
  printf 'source revision mismatch: expected=%s actual=%s\n' "$SOURCE_REVISION" "$ACTUAL_REVISION" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
  printf 'evaluation requires a clean checkout\n' >&2
  exit 2
}
[[ -d "$CHECKPOINT" ]] || {
  printf 'missing checkpoint: %s\n' "$CHECKPOINT" >&2
  exit 2
}
[[ -d "$DATASETS_CACHE" ]] || {
  printf 'missing datasets cache: %s\n' "$DATASETS_CACHE" >&2
  exit 2
}
export TRAINING_STEP=1000
uv run --locked --extra gpu python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
checkpoint = Path(os.environ["CHECKPOINT"]).resolve(strict=True)
root = checkpoint.parent
state_path = checkpoint / "train_state.json"
summary_path = root / "summary.json"
success_path = root / "SUCCESS.json"
state = json.loads(state_path.read_text())
summary = json.loads(summary_path.read_text())
success = json.loads(success_path.read_text())
source = os.environ["SOURCE_REVISION"]
for name, payload in (("checkpoint state", state), ("summary", summary), ("success", success)):
    if payload.get("source_revision") != source:
        raise SystemExit(f"{name} source revision mismatch")
if state.get("optimizer_steps") != 1000 or state.get("effective_batch_examples") != 16:
    raise SystemExit("checkpoint is not exact steps1000/batch16")
if state.get("checkpoint_fidelity_verified") is not True:
    raise SystemExit("checkpoint fidelity is not verified")
if state.get("initial_processor_sha256") == state.get("pre_save_processor_sha256"):
    raise SystemExit("checkpoint processor did not change")
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
for relative in ("final/model.safetensors", "final/config.json", "final/train_state.json"):
    entry = success.get("files", {}).get(relative)
    path = root / relative
    if not isinstance(entry, dict) or sha256_file(path) != entry.get("sha256"):
        raise SystemExit(f"checkpoint success evidence mismatch for {relative}")
PY
uv run --locked --extra gpu python - <<'PY'
import torch
count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(f"full matrix requires exactly four Slurm-visible GPUs; got {count}")
names = [torch.cuda.get_device_name(index) for index in range(count)]
if any("H100" not in name for name in names):
    raise SystemExit(f"full matrix requires H100s; got {names!r}")
print(f"matrix_gpus={names!r}")
PY
[[ ! -e "$OUT_DIR" && ! -L "$OUT_DIR" ]] || {
  printf 'refusing to overwrite evaluation directory: %s\n' "$OUT_DIR" >&2
  exit 2
}
mkdir -p "$(dirname "$OUT_DIR")"
mkdir "$OUT_DIR" || {
  printf 'failed to claim evaluation directory: %s\n' "$OUT_DIR" >&2
  exit 2
}
mkdir "$OUT_DIR/logs"
PIDS=()

run_condition() {
  local name="$1"
  local thinking="$2"
  local processor="$3"
  local output="$OUT_DIR/$name.json"
  local log="$OUT_DIR/logs/$name.log"
  srun --exclusive -N1 -n1 --gres=gpu:1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    bash -lc '
      set -euo pipefail
      export WANDB_MODE=offline
      cd "$1"
      exec uv run --locked --extra gpu --extra eval gcw eval opsd-screen \
        --checkpoint "$2" \
        --source-revision "$3" \
        --tokenizer-source "$4" \
        --datasets-cache "$5" \
        --thinking "$6" \
        --processor "$7" \
        --tasks "$8" \
        --out "$9"
    ' _ "$REPO" "$CHECKPOINT" "$SOURCE_REVISION" "$TOKENIZER_SOURCE" \
      "$DATASETS_CACHE" "$thinking" "$processor" "$TASKS" "$output" \
      >"$log" 2>&1 &
  PIDS+=("$!")
}

run_condition cot-off__cp-off off off
run_condition cot-off__cp-on off on
run_condition cot-on__cp-off on off
run_condition cot-on__cp-on on on
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAILED=1
  fi
done
[[ "$FAILED" -eq 0 ]] || {
  printf 'one or more OPSD condition steps failed\n' >&2
  exit 1
}
[[ "$(git rev-parse HEAD)" == "$SOURCE_REVISION" && -z "$(git status --porcelain)" ]] || {
  printf 'source checkout changed during matrix evaluation\n' >&2
  exit 2
}

uv run --locked --extra gpu --extra eval gcw eval opsd-aggregate \
  --inputs \
    "$OUT_DIR/cot-off__cp-off.json" \
    "$OUT_DIR/cot-off__cp-on.json" \
    "$OUT_DIR/cot-on__cp-off.json" \
    "$OUT_DIR/cot-on__cp-on.json" \
  --out "$OUT_DIR/aggregate.json"

export WANDB_MODE=online
uv run --locked --extra gpu python - <<'PY'
import wandb
if not wandb.api.api_key:
    raise SystemExit("W&B is not authenticated")
PY
SOURCE_REVISION="$SOURCE_REVISION" CHECKPOINT="$CHECKPOINT" OUT_DIR="$OUT_DIR" \
  uv run --locked --extra gpu python scripts/log_opsd_matrix_wandb.py
[[ "$(git rev-parse HEAD)" == "$SOURCE_REVISION" && -z "$(git status --porcelain)" ]] || {
  printf 'source checkout changed during aggregation/publication\n' >&2
  exit 2
}

python - <<'PY'
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
root = Path(os.environ["OUT_DIR"])
condition_names = (
    "cot-off__cp-off",
    "cot-off__cp-on",
    "cot-on__cp-off",
    "cot-on__cp-on",
)
expected_tasks = (
    "gsm8k_openmath",
    "svamp",
    "gsm8k_hard",
    "logiqa_opsd_generate",
    "agieval_gaokao_mathqa_opsd_generate",
)
expected_counts = {
    "gsm8k_openmath": 1319,
    "svamp": 300,
    "gsm8k_hard": 1319,
    "logiqa_opsd_generate": 651,
    "agieval_gaokao_mathqa_opsd_generate": 351,
}
expected_files = [*(f"{name}.json" for name in condition_names), "aggregate.json", "wandb-publication.json"]
payloads = {name: json.loads((root / name).read_text()) for name in expected_files}
for condition in condition_names:
    document = payloads[f"{condition}.json"]
    if tuple(document.get("tasks", ())) != expected_tasks or document.get("limit_per_task") is not None:
        raise SystemExit(f"{condition} is not the canonical unlimited five-task evaluation")
    counts = Counter(str(row["task"]) for row in document.get("sample_manifest", ()))
    if dict(counts) != expected_counts:
        raise SystemExit(f"{condition} sample counts mismatch: {dict(counts)!r}")
    if len(document.get("generation_records", ())) != sum(expected_counts.values()):
        raise SystemExit(f"{condition} generation count mismatch")
aggregate = payloads["aggregate.json"]
if (
    tuple(aggregate.get("tasks", ())) != expected_tasks
    or aggregate.get("limit_per_task") is not None
    or aggregate.get("prompt_count") != sum(expected_counts.values())
):
    raise SystemExit("aggregate is not the canonical full matrix")
publication = payloads["wandb-publication.json"]
if (
    not publication.get("wandb_run_id")
    or not publication.get("wandb_run_url")
    or publication.get("training_step") != 1000
    or tuple(publication.get("tasks", ())) != expected_tasks
):
    raise SystemExit("online W&B publication evidence is incomplete")
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
files = {}
for name in expected_files:
    path = root / name
    files[name] = {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "kind": payloads[name].get("kind"),
    }
payload = json.dumps(
    {
        "source_revision": os.environ["SOURCE_REVISION"],
        "checkpoint": os.environ["CHECKPOINT"],
        "training_step": 1000,
        "tasks": list(expected_tasks),
        "sample_counts": expected_counts,
        "wandb_run_id": publication["wandb_run_id"],
        "wandb_run_url": publication["wandb_run_url"],
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
