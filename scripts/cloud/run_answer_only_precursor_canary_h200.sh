#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

: "${REPO:?REPO is required}"
: "${HARD_DEADLINE_UTC:?HARD_DEADLINE_UTC is required}"

CANARY_CONFIG="configs/train/process_branch_qwen3_06b/qwen3_06b_answer_only_random_depth_canary_seed24.yaml"
CANARY_CONFIG_SHA256="fbb7a439b88898c0d9a543c08a2042f4bca401c3df219d3d13c9e034290616d8"
INPUT_MANIFEST="artifacts/manifests/answer_only_precursor_inputs_v1.json"
INPUT_MANIFEST_SHA256="49699dbd8d84216160fb4d637ffea5fb30140282c6451c51e9e39f24be18fd25"
CANARY_OUTPUT="artifacts/experiments/process_branch_qwen3_06b_answer_only_depth_v1/canary-random-depth/seed-24"
STATE_DIR="artifacts/cloud_runs/answer_only_precursor_canary_v1"
SUCCESS="$STATE_DIR/SUCCESS.json"
FAILURE="$STATE_DIR/FAILURE.json"

cd "$REPO"
mkdir -p "$STATE_DIR"
[[ ! -e "$SUCCESS" && ! -e "$FAILURE" && ! -e "$CANARY_OUTPUT" ]] || {
  printf 'refusing canary relaunch over existing state\n' >&2
  exit 2
}

on_exit() {
  code=$?
  if [[ "$code" -ne 0 ]]; then
    CODE="$code" FAILURE="$FAILURE" /usr/bin/python3 - <<'PY' || true
import json
import os
from pathlib import Path
path = Path(os.environ["FAILURE"])
payload = {"kind": "gcw.answer_only_precursor.canary_failure.v1", "exit_code": int(os.environ["CODE"])}
try:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
except FileExistsError:
    pass
PY
    sync
    sudo -n systemctl poweroff || true
  fi
}
trap on_exit EXIT

now_epoch="$(date -u +%s)"
deadline_epoch="$(date -u -d "$HARD_DEADLINE_UTC" +%s)"
[[ "$deadline_epoch" -gt "$now_epoch" ]] || { printf 'hard deadline is not in the future\n' >&2; exit 2; }
systemctl is-active --quiet gcw-hard-stop.timer || { printf 'guest hard-stop timer is inactive\n' >&2; exit 2; }

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#gpu_names[@]}" -eq 1 && "${gpu_names[0]}" == *H200* ]] || {
  printf 'expected exactly one H200 GPU\n' >&2
  exit 2
}
[[ "$(sha256sum "$CANARY_CONFIG" | cut -d' ' -f1)" == "$CANARY_CONFIG_SHA256" ]] || {
  printf 'canary config identity mismatch\n' >&2
  exit 2
}
[[ "$(sha256sum "$INPUT_MANIFEST" | cut -d' ' -f1)" == "$INPUT_MANIFEST_SHA256" ]] || {
  printf 'input manifest identity mismatch\n' >&2
  exit 2
}
INPUT_MANIFEST="$INPUT_MANIFEST" /usr/bin/python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["INPUT_MANIFEST"]).read_bytes())
for role, expected in manifest["files"].items():
    if role == "source_arrow":
        continue
    path = Path(role)
    data = path.read_bytes()
    if len(data) != expected["size_bytes"]:
        raise SystemExit(f"input size mismatch: {role}")
    if hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise SystemExit(f"input sha256 mismatch: {role}")
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked --extra gpu --extra convert
uv run --locked --extra gpu python scripts/train.py --config "$CANARY_CONFIG"

export CANARY_CONFIG CANARY_CONFIG_SHA256 CANARY_OUTPUT SUCCESS
uv run --locked --extra gpu python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from scripts.evaluate_process_branch_records import authenticate_canonical_checkpoint

config = Path(os.environ["CANARY_CONFIG"])
output = Path(os.environ["CANARY_OUTPUT"])
checkpoint = output / "final"
summary_bytes = (output / "summary.json").read_bytes()
summary = json.loads(summary_bytes)
if summary.get("optimizer_steps") != 1:
    raise SystemExit("canary optimizer_steps mismatch")
manifest_bytes = (checkpoint / "manifest.json").read_bytes()
model_bytes = (checkpoint / "model.safetensors").read_bytes()
manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
model_sha256 = hashlib.sha256(model_bytes).hexdigest()
authenticate_canonical_checkpoint(
    checkpoint,
    expected_manifest_sha256=manifest_sha256,
    expected_model_sha256=model_sha256,
)
payload = {
    "kind": "gcw.answer_only_precursor.canary_success.v1",
    "config_path": str(config),
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "checkpoint_path": str(checkpoint),
    "checkpoint_manifest_sha256": manifest_sha256,
    "checkpoint_model_sha256": model_sha256,
    "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
    "optimizer_steps": summary["optimizer_steps"],
    "initial_backbone_sha256": summary["initial_backbone_sha256"],
    "final_backbone_sha256": summary["final_backbone_sha256"],
    "frozen_identities": summary["frozen_identities"],
}
if payload["config_sha256"] != os.environ["CANARY_CONFIG_SHA256"]:
    raise SystemExit("canary config changed during execution")
if payload["initial_backbone_sha256"] != payload["final_backbone_sha256"]:
    raise SystemExit("canary backbone changed")
destination = Path(os.environ["SUCCESS"])
with destination.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY

sync
trap - EXIT
printf 'CANARY_SUCCESS %s\n' "$SUCCESS"
