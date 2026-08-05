#!/usr/bin/env bash
#SBATCH --job-name=gcw-opsd-native-train
#SBATCH --partition=agent-xlong-15
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
REPO="$(python3 - "$REPO" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve(strict=True))
PY
)"
CONFIG="${CONFIG:?CONFIG is required}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION is required}"
BACKBONE_LABEL="converted_GCW_Qwen3_0.6B_base_not_numerically_identical_to_HF"
export SOURCE_REVISION BACKBONE_LABEL

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
CONFIG="$(python3 - "$CONFIG" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve(strict=True))
PY
)"
export CONFIG
case "$CONFIG" in
  "$REPO"/*) ;;
  *)
    printf 'config must resolve inside pinned source checkout: %s\n' "$CONFIG" >&2
    exit 2
    ;;
esac
CONFIG_REL="${CONFIG#"$REPO"/}"
git ls-files --error-unmatch -- "$CONFIG_REL" >/dev/null || {
  printf 'config must be tracked by the pinned source revision: %s\n' "$CONFIG_REL" >&2
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
backbone_checkpoint = str(Path(config.backbone_checkpoint).resolve(strict=True))
tokenizer_source = str(
    Path(config.data.tokenizer.removeprefix("hf:")).resolve(strict=True)
)
data_dir = str(Path(config.data.data_dir).resolve(strict=True))
paths = {
    "OUTPUT_DIR": config.output_dir,
    "BACKBONE_CHECKPOINT": backbone_checkpoint,
    "DATA_DIR": data_dir,
    "TOKENIZER_SOURCE": tokenizer_source,
    "MAX_OPTIMIZER_STEPS": str(distill.max_optimizer_steps),
    "EFFECTIVE_BATCH_EXAMPLES": str(distill.gradient_accumulation_steps),
    "RECORD_CACHE_DISPLACEMENTS": str(distill.record_cache_displacements).lower(),
    "WANDB_MODE_CONFIG": str(config.wandb.mode),
}
for key, value in paths.items():
    if not value:
        raise SystemExit(f"missing config-derived {key}")
    print(f"{key}={value}")
parity_raw = config.metadata.get("required_conversion_parity", "")
diagnostic_raw = config.metadata.get("required_conversion_diagnostic", "")
waiver_raw = config.metadata.get("conversion_parity_waiver", "")
parity = str(Path(parity_raw).resolve(strict=True)) if parity_raw else ""
diagnostic = str(Path(diagnostic_raw).resolve(strict=True)) if diagnostic_raw else ""
waiver = str(Path(waiver_raw).resolve(strict=True)) if waiver_raw else ""
base_checkpoint = str(
    Path("/mnt/jfs-01/shared/gcw/checkpoints/base_backbones/qwen3_06b/tf-v1").resolve(
        strict=True
    )
)
base_tokenizer = str(
    Path("/mnt/jfs-01/shared/gcw/hf/Qwen3-0.6B-base-v1").resolve(strict=True)
)
base_scope = "distinct_base_backbone_fallback_after_sft_zero_yield"
pragmatic_scope = "pragmatic_same_gcw_backbone_sensitivity_after_failed_hf_parity"
base_parity = str((Path(base_checkpoint) / "parity.json").resolve(strict=True))
base_diagnostic = str(
    (Path(base_checkpoint) / "diagnostics/layerwise-distribution-v1-fp64.json").resolve(
        strict=True
    )
)
expected_waiver = str(
    Path("configs/protocols/qwen3_06b_opsd_pragmatic_parity_waiver_v1.json").resolve(
        strict=True
    )
)
uses_base_checkpoint = backbone_checkpoint == base_checkpoint
uses_base_tokenizer = tokenizer_source == base_tokenizer
if uses_base_checkpoint != uses_base_tokenizer:
    raise SystemExit("base fallback checkpoint/tokenizer contract mismatch")
requires_parity = False
requires_waiver = False
if uses_base_checkpoint:
    scope = config.metadata.get("scientific_scope")
    if scope == base_scope:
        if parity != base_parity or diagnostic or waiver:
            raise SystemExit("base parity-gated scope/evidence contract mismatch")
        requires_parity = True
    elif scope == pragmatic_scope:
        if parity != base_parity or diagnostic != base_diagnostic or waiver != expected_waiver:
            raise SystemExit("pragmatic base scope/evidence contract mismatch")
        requires_waiver = True
    else:
        raise SystemExit("base fallback uses an unapproved scientific scope")
elif parity or diagnostic or waiver:
    raise SystemExit("non-base config cannot declare base conversion evidence")
print(f"REQUIRES_CONVERSION_PARITY={str(requires_parity).lower()}")
print(f"REQUIRES_PARITY_WAIVER={str(requires_waiver).lower()}")
print(f"REQUIRED_CONVERSION_PARITY={parity}")
print(f"REQUIRED_CONVERSION_DIAGNOSTIC={diagnostic}")
print(f"PARITY_WAIVER={waiver}")
PY
)"
mapfile -t PREFLIGHT <<<"$PREFLIGHT_RAW"
for assignment in "${PREFLIGHT[@]}"; do
  case "$assignment" in
    OUTPUT_DIR=*|BACKBONE_CHECKPOINT=*|DATA_DIR=*|TOKENIZER_SOURCE=*|MAX_OPTIMIZER_STEPS=*|EFFECTIVE_BATCH_EXAMPLES=*|RECORD_CACHE_DISPLACEMENTS=*|WANDB_MODE_CONFIG=*|REQUIRES_CONVERSION_PARITY=*|REQUIRES_PARITY_WAIVER=*|REQUIRED_CONVERSION_PARITY=*|REQUIRED_CONVERSION_DIAGNOSTIC=*|PARITY_WAIVER=*)
      export "$assignment" ;;
    *) printf 'unexpected config preflight output: %s\n' "$assignment" >&2; exit 2 ;;
  esac
done

if [[ "$REQUIRES_CONVERSION_PARITY" == true ]]; then
  [[ -n "$REQUIRED_CONVERSION_PARITY" && "$REQUIRES_PARITY_WAIVER" == false ]] || {
    printf 'base parity-gated scope has an invalid evidence mode\n' >&2
    exit 2
  }
  uv run --locked --extra gpu --extra convert python - <<'PY'
import json
import os
from pathlib import Path
from scripts.verify_hf_conversion import validate_parity_evidence

parity_path = Path(os.environ["REQUIRED_CONVERSION_PARITY"]).resolve(strict=True)
payload = json.loads(parity_path.read_text())
validate_parity_evidence(
    payload,
    hf_path=Path(os.environ["TOKENIZER_SOURCE"]),
    checkpoint=Path(os.environ["BACKBONE_CHECKPOINT"]),
    source_revision=os.environ["SOURCE_REVISION"],
)
PY
elif [[ "$REQUIRES_PARITY_WAIVER" == true ]]; then
  [[ -n "$PARITY_WAIVER" && -n "$REQUIRED_CONVERSION_PARITY" && -n "$REQUIRED_CONVERSION_DIAGNOSTIC" ]] || {
    printf 'pragmatic scope lacks complete waiver evidence paths\n' >&2
    exit 2
  }
  uv run --locked --extra gpu python scripts/validate_opsd_pragmatic_waiver.py \
    --repo "$REPO" \
    --config "$CONFIG" \
    --waiver "$PARITY_WAIVER" \
    --source-revision "$SOURCE_REVISION" \
    --check-only
elif [[ -n "$REQUIRED_CONVERSION_PARITY" || -n "$REQUIRED_CONVERSION_DIAGNOSTIC" || -n "$PARITY_WAIVER" ]]; then
  printf 'undeclared conversion evidence mode\n' >&2
  exit 2
fi

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
CANARY_CONFIG_SHA256 = "65a2133efaea7614a3a0eea72943bf64593db94b49293d9c49e287831be01b30"
if success.get("source_revision") != os.environ["SOURCE_REVISION"]:
    raise SystemExit("canary evidence source revision mismatch")
if success.get("source_config_sha256") != CANARY_CONFIG_SHA256:
    raise SystemExit("canary config digest mismatch")
if success.get("pragmatic_parity_waiver") is not True:
    raise SystemExit("canary lacks strict pragmatic waiver label")
if success.get("parity_gate_remains_failed") is not True:
    raise SystemExit("canary does not retain failed parity label")
if success.get("backbone_label") != os.environ["BACKBONE_LABEL"]:
    raise SystemExit("canary backbone label mismatch")
if success.get("optimizer_steps") != 1 or success.get("effective_batch_examples") != 1:
    raise SystemExit("promotion evidence is not the exact one-step batch-one canary")
files = success.get("files", {})
summary_entry = files.get("summary.json")
waiver_entry = files.get("PRAGMATIC_WAIVER.json")
if not isinstance(summary_entry, dict) or not isinstance(waiver_entry, dict):
    raise SystemExit("canary evidence does not authenticate summary and waiver")
summary_path = success_path.parent / "summary.json"
waiver_path = success_path.parent / "PRAGMATIC_WAIVER.json"
if hashlib.sha256(summary_path.read_bytes()).hexdigest() != summary_entry.get("sha256"):
    raise SystemExit("canary summary digest mismatch")
if hashlib.sha256(waiver_path.read_bytes()).hexdigest() != waiver_entry.get("sha256"):
    raise SystemExit("canary waiver digest mismatch")
waiver = json.loads(waiver_path.read_text())
if (
    waiver.get("source_revision") != os.environ["SOURCE_REVISION"]
    or waiver.get("config_sha256") != CANARY_CONFIG_SHA256
    or waiver.get("optimizer_steps") != 1
    or waiver.get("effective_batch_examples") != 1
    or waiver.get("parity_gate_remains_failed") is not True
    or waiver.get("backbone_label") != os.environ["BACKBONE_LABEL"]
):
    raise SystemExit("canary run-specific waiver identity mismatch")
summary = json.loads(summary_path.read_text())
if summary.get("seen_prompts", 65) > 64 or summary.get("valid_prompts") != 1:
    raise SystemExit("canary strict-yield gate failed")
projected_seconds = float(summary["elapsed_seconds"]) * 16_000
if projected_seconds > 4.5 * 24 * 60 * 60:
    raise SystemExit(
        f"projected 1k walltime exceeds 4.5 days: {projected_seconds:.0f} seconds"
    )
print(f"smoke_promotion_projected_seconds={projected_seconds:.3f}")
PY
fi

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
GPU_NAME="$(uv run --locked --extra gpu python - <<'PY'
import torch
if torch.cuda.device_count() != 1:
    raise SystemExit(f"training requires exactly one Slurm-visible GPU; got {torch.cuda.device_count()}")
print(torch.cuda.get_device_name(0))
PY
)"
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

[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || {
  printf 'refusing to overwrite output: %s\n' "$OUTPUT_DIR" >&2
  exit 2
}
mkdir -p "$(dirname "$OUTPUT_DIR")"
mkdir "$OUTPUT_DIR" || {
  printf 'failed to claim fresh output directory: %s\n' "$OUTPUT_DIR" >&2
  exit 2
}
PRAGMATIC_WAIVER_EVIDENCE=""
if [[ "$REQUIRES_PARITY_WAIVER" == true ]]; then
  PRAGMATIC_WAIVER_EVIDENCE="$OUTPUT_DIR/PRAGMATIC_WAIVER.json"
  uv run --locked --extra gpu python scripts/validate_opsd_pragmatic_waiver.py \
    --repo "$REPO" \
    --config "$CONFIG" \
    --waiver "$PARITY_WAIVER" \
    --source-revision "$SOURCE_REVISION" \
    --out "$PRAGMATIC_WAIVER_EVIDENCE"
fi
export PRAGMATIC_WAIVER_EVIDENCE
REQUIRED_FRESH_PARITY=""
if [[ "$REQUIRES_CONVERSION_PARITY" == true ]]; then
  REQUIRED_FRESH_PARITY="$OUTPUT_DIR/preflight-parity.json"
  export REQUIRED_FRESH_PARITY
  uv run --locked --extra gpu --extra convert python scripts/verify_hf_conversion.py \
    --hf-path "$TOKENIZER_SOURCE" \
    --checkpoint "$BACKBONE_CHECKPOINT" \
    --out "$REQUIRED_FRESH_PARITY"
fi
export REQUIRED_FRESH_PARITY

uv run --locked --extra gpu gcw train --config "$CONFIG"

[[ "$(git rev-parse HEAD)" == "$SOURCE_REVISION" ]] || {
  printf 'source revision changed during training\n' >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
  printf 'source checkout became dirty during training\n' >&2
  exit 2
}

uv run --locked --extra gpu --extra convert python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from scripts.verify_hf_conversion import validate_parity_evidence

root = Path(os.environ["OUTPUT_DIR"])
summary_path = root / "summary.json"
metrics_path = root / "metrics.jsonl"
run_config_path = root / "run_config.json"
checkpoint = root / "final"
fresh_parity = os.environ.get("REQUIRED_FRESH_PARITY")
if fresh_parity:
    payload = json.loads(Path(fresh_parity).read_text())
    validate_parity_evidence(
        payload,
        hf_path=Path(os.environ["TOKENIZER_SOURCE"]),
        checkpoint=Path(os.environ["BACKBONE_CHECKPOINT"]),
        source_revision=os.environ["SOURCE_REVISION"],
    )
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
    Path(os.environ["CONFIG"]),
]
if os.environ.get("REQUIRED_CONVERSION_PARITY"):
    required.append(Path(os.environ["REQUIRED_CONVERSION_PARITY"]))
if os.environ.get("REQUIRED_CONVERSION_DIAGNOSTIC"):
    required.append(Path(os.environ["REQUIRED_CONVERSION_DIAGNOSTIC"]))
if os.environ.get("PARITY_WAIVER"):
    required.append(Path(os.environ["PARITY_WAIVER"]))
if os.environ.get("PRAGMATIC_WAIVER_EVIDENCE"):
    required.append(Path(os.environ["PRAGMATIC_WAIVER_EVIDENCE"]))
if os.environ.get("REQUIRED_FRESH_PARITY"):
    required.append(Path(os.environ["REQUIRED_FRESH_PARITY"]))
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
files = {}
for path in required:
    if not path.is_file():
        raise SystemExit(f"missing required output: {path}")
    try:
        manifest_name = str(path.relative_to(root))
    except ValueError:
        manifest_name = f"external:{path}"
    files[manifest_name] = {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
payload = json.dumps(
    {
        "source_revision": os.environ["SOURCE_REVISION"],
        "source_config_sha256": sha256_file(Path(os.environ["CONFIG"])),
        "backbone_label": os.environ["BACKBONE_LABEL"],
        "optimizer_steps": expected_steps,
        "effective_batch_examples": expected_batch,
        "valid_examples": expected_examples,
        "gpu_name": os.environ["GPU_NAME"],
        "wandb_run_id": summary["wandb_run_id"],
        "wandb_run_url": summary["wandb_run_url"],
        "pragmatic_parity_waiver": bool(os.environ.get("PRAGMATIC_WAIVER_EVIDENCE")),
        "parity_gate_remains_failed": bool(os.environ.get("PRAGMATIC_WAIVER_EVIDENCE")),
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
