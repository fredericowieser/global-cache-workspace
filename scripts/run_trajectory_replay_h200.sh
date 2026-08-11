#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

: "${REPO:?Set REPO to the global-cache-workspace checkout}"

CONFIG="configs/train/process_branch_qwen3_06b/qwen3_06b_trajectory_replay_random_depth_6000_seed24.yaml"
DATA="data/process_branch/gsm8k-trajectory-replay-v1.jsonl"
MANIFEST="data/process_branch/gsm8k-trajectory-replay-v1.manifest.json"
OUTPUT="artifacts/experiments/process_branch_qwen3_06b_trajectory_replay_v1/random-depth/seed-24"
EXPECTED_DATA_SHA256="700396881011295689065cff7e2def911e81efe65ee676fc646e7ad3b88247de"
EXPECTED_SOURCE_IDS_SHA256="c0330c4dc7c13d36c87e75c03b34d2c616a0ee945d0c8c7e616353dc6de60e3f"

cd "$REPO"
[[ ! -e "$OUTPUT" ]] || { printf 'refusing to overwrite %s\n' "$OUTPUT" >&2; exit 2; }
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#gpu_names[@]}" -eq 1 && "${gpu_names[0]}" == *H200* ]] || {
  printf 'expected exactly one H200 GPU\n' >&2
  exit 2
}

if [[ ! -f "$DATA" || ! -f "$MANIFEST" ]]; then
  uv run --locked --extra gpu --extra convert python \
    scripts/prepare_gsm8k_trajectory_replay.py
fi
[[ "$(sha256sum "$DATA" | cut -d' ' -f1)" == "$EXPECTED_DATA_SHA256" ]] || {
  printf 'trajectory data identity mismatch\n' >&2
  exit 2
}
DATA="$DATA" MANIFEST="$MANIFEST" EXPECTED_SOURCE_IDS_SHA256="$EXPECTED_SOURCE_IDS_SHA256" \
  /usr/bin/python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["MANIFEST"]).read_bytes())
if manifest["output_sha256"] != hashlib.sha256(Path(os.environ["DATA"]).read_bytes()).hexdigest():
    raise SystemExit("trajectory manifest/output mismatch")
if manifest["ordered_source_ids_sha256"] != os.environ["EXPECTED_SOURCE_IDS_SHA256"]:
    raise SystemExit("trajectory source-ID identity mismatch")
if manifest["accepted_rows"] != 7473 or manifest["rejected_rows"] != 0:
    raise SystemExit("trajectory row-count contract mismatch")
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked --extra gpu --extra convert
uv run --locked --extra gpu --extra convert python scripts/train.py --config "$CONFIG"

OUTPUT="$OUTPUT" /usr/bin/python3 - <<'PY'
import json
import os
from pathlib import Path

output = Path(os.environ["OUTPUT"])
summary = json.loads((output / "summary.json").read_bytes())
if summary.get("objective") != "trajectory_replay":
    raise SystemExit("wrong completed objective")
if summary.get("optimizer_steps") != 1500:
    raise SystemExit("optimizer-step contract mismatch")
if summary.get("initial_backbone_sha256") != summary.get("final_backbone_sha256"):
    raise SystemExit("frozen backbone changed")
if summary.get("initial_processor_sha256") == summary.get("final_processor_sha256"):
    raise SystemExit("cache processor did not update")
if summary.get("checkpoint_reload_verified") is not True:
    raise SystemExit("final checkpoint reload was not verified")
print(output / "final")
PY
