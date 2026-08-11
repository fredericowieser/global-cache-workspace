#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

: "${REPO:?REPO is required}"
: "${HARD_DEADLINE_UTC:?HARD_DEADLINE_UTC is required}"
: "${CANARY_EVIDENCE:?CANARY_EVIDENCE is required}"
: "${CANARY_EVIDENCE_SHA256:?CANARY_EVIDENCE_SHA256 is required}"

P0_CONFIG="configs/train/process_branch_qwen3_06b/qwen3_06b_answer_only_k1_6000_seed24.yaml"
P0_CONFIG_SHA256="f6c180976c7cebec20fedce6339ef7f789638490b1066b08189b0b06d0ad0caa"
P1_CONFIG="configs/train/process_branch_qwen3_06b/qwen3_06b_answer_only_random_depth_6000_seed24.yaml"
P1_CONFIG_SHA256="bff09635e74143253cf457be2621d8dacfb989e1cb72aeeac3e601613fd2cc5e"
BASE_PROTOCOL="configs/eval/process_branch_qwen3_06b/answer_only_precursor_off_init_k_v1.json"
BASE_PROTOCOL_SHA256="8f175aaec1b5f44cbf0711771933c5047d980bc216ce18a1a1f61375afecb572"
DRY_RECEIPT="results/answer_only_precursor/off_init_k_dry_run_receipt_v2.json"
DRY_RECEIPT_SHA256="85cd3a6ead18c855e6a1525fd9f2a7aa8b4e55793b415b4a9ebdca3fcffd9685"
INPUT_MANIFEST="artifacts/manifests/answer_only_precursor_inputs_v1.json"
INPUT_MANIFEST_SHA256="49699dbd8d84216160fb4d637ffea5fb30140282c6451c51e9e39f24be18fd25"
EXECUTION_PROTOCOL="artifacts/protocols/answer_only_precursor_off_init_k_execution_v1.json"
P0_OUTPUT="artifacts/experiments/process_branch_qwen3_06b_answer_only_depth_v1/k1/seed-24"
P1_OUTPUT="artifacts/experiments/process_branch_qwen3_06b_answer_only_depth_v1/random-depth/seed-24"
P0_BUNDLE="results/answer_only_precursor/p0/heldout.bundle"
P1_BUNDLE="results/answer_only_precursor/p1/heldout.bundle"
STATE_DIR="artifacts/cloud_runs/answer_only_precursor_full_v1"
SUCCESS="$STATE_DIR/SUCCESS.json"
FAILURE="$STATE_DIR/FAILURE.json"

cd "$REPO"
mkdir -p "$STATE_DIR"
for forbidden in "$SUCCESS" "$FAILURE" "$EXECUTION_PROTOCOL" "$P0_OUTPUT" "$P1_OUTPUT" "$P0_BUNDLE" "$P1_BUNDLE" results/answer_only_precursor/p0/final.bundle results/answer_only_precursor/p1/final.bundle; do
  [[ ! -e "$forbidden" ]] || { printf 'refusing full relaunch over existing state: %s\n' "$forbidden" >&2; exit 2; }
done

on_exit() {
  code=$?
  if [[ "$code" -ne 0 ]]; then
    CODE="$code" FAILURE="$FAILURE" /usr/bin/python3 - <<'PY' || true
import json
import os
from pathlib import Path
path = Path(os.environ["FAILURE"])
payload = {"kind": "gcw.answer_only_precursor.full_failure.v1", "exit_code": int(os.environ["CODE"])}
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

check_sha() {
  local expected="$1" path="$2"
  [[ "$(sha256sum "$path" | cut -d' ' -f1)" == "$expected" ]] || {
    printf 'identity mismatch: %s\n' "$path" >&2
    exit 2
  }
}
check_sha "$CANARY_EVIDENCE_SHA256" "$CANARY_EVIDENCE"
check_sha "$P0_CONFIG_SHA256" "$P0_CONFIG"
check_sha "$P1_CONFIG_SHA256" "$P1_CONFIG"
check_sha "$BASE_PROTOCOL_SHA256" "$BASE_PROTOCOL"
check_sha "$DRY_RECEIPT_SHA256" "$DRY_RECEIPT"
check_sha "$INPUT_MANIFEST_SHA256" "$INPUT_MANIFEST"
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
CANARY_EVIDENCE="$CANARY_EVIDENCE" /usr/bin/python3 - <<'PY'
import json
import os
from pathlib import Path
payload = json.loads(Path(os.environ["CANARY_EVIDENCE"]).read_bytes())
if payload.get("kind") != "gcw.answer_only_precursor.canary_success.v1":
    raise SystemExit("wrong canary evidence kind")
if payload.get("optimizer_steps") != 1:
    raise SystemExit("canary evidence optimizer_steps mismatch")
if payload.get("initial_backbone_sha256") != payload.get("final_backbone_sha256"):
    raise SystemExit("canary evidence backbone mismatch")
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked --extra gpu --extra convert
uv run --locked --extra gpu python scripts/train.py --config "$P0_CONFIG"
uv run --locked --extra gpu python scripts/train.py --config "$P1_CONFIG"
uv run --locked --extra gpu python scripts/evaluate_process_branch_suite.py \
  --finalize-answer-only-protocol "$BASE_PROTOCOL" \
  --execution-protocol "$EXECUTION_PROTOCOL"
uv run --locked --extra gpu python scripts/evaluate_process_branch_records.py canonical \
  --protocol "$EXECUTION_PROTOCOL" --arm p0 --split heldout --bundle-out "$P0_BUNDLE"
uv run --locked --extra gpu python scripts/evaluate_process_branch_records.py canonical \
  --protocol "$EXECUTION_PROTOCOL" --arm p1 --split heldout --bundle-out "$P1_BUNDLE"

export P0_OUTPUT P1_OUTPUT EXECUTION_PROTOCOL P0_BUNDLE P1_BUNDLE SUCCESS
uv run --locked --extra gpu python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from scripts.evaluate_process_branch_records import authenticate_canonical_checkpoint, load_canonical_ledger

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def checkpoint_evidence(output: Path) -> dict:
    checkpoint = output / "final"
    manifest_sha = sha(checkpoint / "manifest.json")
    model_sha = sha(checkpoint / "model.safetensors")
    authenticate_canonical_checkpoint(
        checkpoint,
        expected_manifest_sha256=manifest_sha,
        expected_model_sha256=model_sha,
    )
    summary_bytes = (output / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    if summary.get("optimizer_steps") != 1500:
        raise SystemExit(f"optimizer step mismatch for {output}")
    if summary.get("initial_backbone_sha256") != summary.get("final_backbone_sha256"):
        raise SystemExit(f"backbone changed for {output}")
    return {
        "checkpoint": str(checkpoint),
        "manifest_sha256": manifest_sha,
        "model_sha256": model_sha,
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "optimizer_steps": summary["optimizer_steps"],
    }

def bundle_evidence(bundle: Path, protocol_sha256: str) -> dict:
    manifest_sha256 = sha(bundle / "manifest.json")
    rows, manifest = load_canonical_ledger(
        bundle,
        expected_manifest_sha256=manifest_sha256,
        expected_protocol_sha256=protocol_sha256,
    )
    return {
        "bundle": str(bundle),
        "manifest_sha256": manifest_sha256,
        "rows_sha256": sha(bundle / "rows.jsonl"),
        "row_count": len(rows),
        "split": manifest["selection"]["split"],
    }

execution_protocol_sha256 = sha(Path(os.environ["EXECUTION_PROTOCOL"]))
payload = {
    "kind": "gcw.answer_only_precursor.full_success.v1",
    "execution_protocol": os.environ["EXECUTION_PROTOCOL"],
    "execution_protocol_sha256": execution_protocol_sha256,
    "arms": {
        "p0": checkpoint_evidence(Path(os.environ["P0_OUTPUT"])),
        "p1": checkpoint_evidence(Path(os.environ["P1_OUTPUT"])),
    },
    "heldout": {
        "p0": bundle_evidence(Path(os.environ["P0_BUNDLE"]), execution_protocol_sha256),
        "p1": bundle_evidence(Path(os.environ["P1_BUNDLE"]), execution_protocol_sha256),
    },
    "final_rows_evaluated": False,
}
if any(item["split"] != "heldout" for item in payload["heldout"].values()):
    raise SystemExit("non-heldout result admitted")
destination = Path(os.environ["SUCCESS"])
with destination.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY

[[ ! -e results/answer_only_precursor/p0/final.bundle && ! -e results/answer_only_precursor/p1/final.bundle ]] || {
  printf 'final split artifact unexpectedly exists\n' >&2
  exit 2
}
sync
trap - EXIT
printf 'FULL_SUCCESS %s\n' "$SUCCESS"
