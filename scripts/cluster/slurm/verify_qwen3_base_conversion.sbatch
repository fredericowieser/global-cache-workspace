#!/usr/bin/env bash
#SBATCH --job-name=gcw-qwen-base-parity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

REPO="${REPO:-$HOME/code/global-cache-workspace}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION is required}"
HF_PATH="${HF_PATH:-/mnt/jfs-01/shared/gcw/hf/Qwen3-0.6B-base-v1}"
CHECKPOINT="${CHECKPOINT:-/mnt/jfs-01/shared/gcw/checkpoints/base_backbones/qwen3_06b/tf-v1}"
OUT="${OUT:-$CHECKPOINT/parity.json}"
export SOURCE_REVISION HF_PATH CHECKPOINT OUT

cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$SOURCE_REVISION" ]] || {
  printf 'source revision mismatch\n' >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
  printf 'source checkout is dirty\n' >&2
  exit 2
}
[[ -d "$HF_PATH" && ! -L "$HF_PATH" ]] || {
  printf 'missing or linked HF source: %s\n' "$HF_PATH" >&2
  exit 2
}
[[ -d "$CHECKPOINT" && ! -L "$CHECKPOINT" ]] || {
  printf 'missing or linked converted checkpoint: %s\n' "$CHECKPOINT" >&2
  exit 2
}
[[ ! -e "$OUT" && ! -L "$OUT" ]] || {
  printf 'refusing to overwrite parity evidence: %s\n' "$OUT" >&2
  exit 2
}

export PATH="$HOME/.local/bin:$PATH"
uv run --locked --extra gpu --extra convert python scripts/cluster/check_cuda_env.py
GPU_NAME="$(uv run --locked --extra gpu --extra convert python - <<'PY'
import torch
if torch.cuda.device_count() != 1:
    raise SystemExit(f"parity requires exactly one Slurm-visible GPU; got {torch.cuda.device_count()}")
print(torch.cuda.get_device_name(0))
PY
)"
[[ "$GPU_NAME" == *H100* ]] || {
  printf 'parity requires H100; got %s\n' "$GPU_NAME" >&2
  exit 2
}

uv run --locked --extra gpu --extra convert python scripts/analysis/verify_hf_conversion.py \
  --hf-path "$HF_PATH" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT"

[[ "$(git rev-parse HEAD)" == "$SOURCE_REVISION" && -z "$(git status --porcelain)" ]] || {
  printf 'source changed during parity validation\n' >&2
  exit 2
}
uv run --locked --extra gpu --extra convert python - <<'PY'
import json
import os
from pathlib import Path
from scripts.analysis.verify_hf_conversion import validate_parity_evidence

payload = json.loads(Path(os.environ["OUT"]).read_text())
validate_parity_evidence(
    payload,
    hf_path=Path(os.environ["HF_PATH"]),
    checkpoint=Path(os.environ["CHECKPOINT"]),
    source_revision=os.environ["SOURCE_REVISION"],
)
PY
