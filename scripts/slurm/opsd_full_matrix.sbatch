#!/usr/bin/env bash
#SBATCH --job-name=gcw-opsd-full-matrix
#SBATCH --partition=agent-xlong-15
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
TOKENIZER_SOURCE="$(python3 - "$TOKENIZER_SOURCE" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve(strict=True))
PY
)"
[[ "$TOKENIZER_SOURCE" == "/mnt/jfs-01/shared/gcw/hf/Qwen3-0.6B-base-v1" ]] || {
  printf 'pragmatic matrix tokenizer mismatch: %s\n' "$TOKENIZER_SOURCE" >&2
  exit 2
}
DATASETS_CACHE="${DATASETS_CACHE:?DATASETS_CACHE is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
EXPECTED_PRAGMATIC_WAIVER="${EXPECTED_PRAGMATIC_WAIVER:-false}"
BACKBONE_LABEL="converted_GCW_Qwen3_0.6B_base_not_numerically_identical_to_HF"
TASKS="gsm8k_openmath,svamp,gsm8k_hard,logiqa_opsd_generate,agieval_gaokao_mathqa_opsd_generate"
export CHECKPOINT SOURCE_REVISION TOKENIZER_SOURCE OUT_DIR TASKS EXPECTED_PRAGMATIC_WAIVER BACKBONE_LABEL

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
expected_waiver_raw = os.environ["EXPECTED_PRAGMATIC_WAIVER"]
if expected_waiver_raw not in {"true", "false"}:
    raise SystemExit("EXPECTED_PRAGMATIC_WAIVER must be true or false")
expected_waiver = expected_waiver_raw == "true"
if success.get("pragmatic_parity_waiver") is not expected_waiver:
    raise SystemExit("checkpoint pragmatic waiver status mismatch")
if success.get("parity_gate_remains_failed") is not expected_waiver:
    raise SystemExit("checkpoint parity status label mismatch")
if expected_waiver and success.get("backbone_label") != os.environ["BACKBONE_LABEL"]:
    raise SystemExit("checkpoint backbone label mismatch")
tokenizer = os.environ["TOKENIZER_SOURCE"]
if state.get("tokenizer_source") != tokenizer or summary.get("tokenizer_source") != tokenizer:
    raise SystemExit("checkpoint tokenizer provenance mismatch")
if expected_waiver:
    waiver_entry = success.get("files", {}).get("PRAGMATIC_WAIVER.json")
    waiver_path = root / "PRAGMATIC_WAIVER.json"
    if not isinstance(waiver_entry, dict) or not waiver_path.is_file():
        raise SystemExit("checkpoint lacks authenticated pragmatic waiver evidence")
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
if expected_waiver and sha256_file(waiver_path) != waiver_entry.get("sha256"):
    raise SystemExit("checkpoint pragmatic waiver digest mismatch")
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
uv run --locked --extra gpu python - <<'PY'
import wandb
if not wandb.api.api_key:
    raise SystemExit("W&B is not authenticated")
print("wandb_authenticated=true")
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

python - <<'PY'
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
for condition in condition_names:
    document = json.loads((root / f"{condition}.json").read_text())
    if tuple(document.get("tasks", ())) != expected_tasks or document.get("limit_per_task") is not None:
        raise SystemExit(f"{condition} is not the canonical unlimited five-task evaluation")
    if document.get("source_revision") != os.environ["SOURCE_REVISION"]:
        raise SystemExit(f"{condition} source revision mismatch")
    if document.get("tokenizer_source") != os.environ["TOKENIZER_SOURCE"]:
        raise SystemExit(f"{condition} tokenizer provenance mismatch")
    counts = Counter(str(row["task"]) for row in document.get("sample_manifest", ()))
    if dict(counts) != expected_counts:
        raise SystemExit(f"{condition} sample counts mismatch: {dict(counts)!r}")
    if len(document.get("generation_records", ())) != sum(expected_counts.values()):
        raise SystemExit(f"{condition} generation count mismatch")
aggregate = json.loads((root / "aggregate.json").read_text())
if (
    tuple(aggregate.get("tasks", ())) != expected_tasks
    or aggregate.get("limit_per_task") is not None
    or aggregate.get("prompt_count") != sum(expected_counts.values())
):
    raise SystemExit("aggregate is not the canonical full matrix")
print("canonical_matrix_validated_before_publication=true")
PY

export WANDB_MODE=online
SOURCE_REVISION="$SOURCE_REVISION" CHECKPOINT="$CHECKPOINT" OUT_DIR="$OUT_DIR" \
  WANDB_PRAGMATIC_WAIVER="$EXPECTED_PRAGMATIC_WAIVER" \
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
expected_pragmatic_waiver = os.environ["EXPECTED_PRAGMATIC_WAIVER"] == "true"
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
    or publication.get("pragmatic_parity_waiver") is not expected_pragmatic_waiver
    or publication.get("backbone_label")
    != (os.environ["BACKBONE_LABEL"] if expected_pragmatic_waiver else None)
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
        "pragmatic_parity_waiver": expected_pragmatic_waiver,
        "parity_gate_remains_failed": expected_pragmatic_waiver,
        "backbone_label": (
            os.environ["BACKBONE_LABEL"] if expected_pragmatic_waiver else None
        ),
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
