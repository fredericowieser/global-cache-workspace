#!/usr/bin/env bash
#SBATCH --job-name=gcw-qwen-base-diagnose
#SBATCH --partition=agent-dev-15,agent-dev
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

REPO="${REPO:-$HOME/code/global-cache-workspace}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION is required}"
HF_PATH="${HF_PATH:-/mnt/jfs-01/shared/gcw/hf/Qwen3-0.6B-base-v1}"
CHECKPOINT="${CHECKPOINT:-/mnt/jfs-01/shared/gcw/checkpoints/base_backbones/qwen3_06b/tf-v1}"
OUT="${OUT:-$CHECKPOINT/diagnostics/layerwise-distribution-v1.json}"
PARENT_PARITY_SHA256="1aca29e6db3099d652d8635d57f8f79a9acccfbec915a378c0477e7104b86216"

canonical_existing() {
  python3 - "$1" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve(strict=True))
PY
}

REPO="$(canonical_existing "$REPO")"
HF_PATH="$(canonical_existing "$HF_PATH")"
CHECKPOINT="$(canonical_existing "$CHECKPOINT")"
cd "$REPO"

[[ "$(git rev-parse HEAD)" == "$SOURCE_REVISION" ]] || {
  printf 'source revision mismatch: expected %s, got %s\n' "$SOURCE_REVISION" "$(git rev-parse HEAD)" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
  printf 'diagnosis requires a clean checkout\n' >&2
  exit 2
}
git ls-files --error-unmatch -- \
  scripts/analysis/diagnose_hf_conversion.py \
  scripts/analysis/verify_hf_conversion.py \
  scripts/cluster/slurm/diagnose_qwen3_base_conversion.sbatch >/dev/null

[[ -d "$HF_PATH" ]] || { printf 'missing HF path: %s\n' "$HF_PATH" >&2; exit 2; }
[[ -d "$CHECKPOINT" ]] || { printf 'missing checkpoint: %s\n' "$CHECKPOINT" >&2; exit 2; }
PARENT_PARITY="$CHECKPOINT/parity.json"
[[ -f "$PARENT_PARITY" ]] || { printf 'missing parent parity evidence: %s\n' "$PARENT_PARITY" >&2; exit 2; }
[[ "$(sha256sum "$PARENT_PARITY" | cut -d' ' -f1)" == "$PARENT_PARITY_SHA256" ]] || {
  printf 'parent parity evidence hash mismatch\n' >&2
  exit 2
}
python3 - "$PARENT_PARITY" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("passed") is not False:
    raise SystemExit("parent parity evidence is not a failure")
if payload.get("policy_id") != "qwen3-06b-bf16-parameters-fp32-comparison-v2":
    raise SystemExit("parent parity policy mismatch")
PY

mkdir -p "$CHECKPOINT/diagnostics"
OUT="$(python3 - "$OUT" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
print(path.parent.resolve(strict=True) / path.name)
PY
)"
case "$OUT" in
  "$CHECKPOINT"/diagnostics/*) ;;
  *) printf 'diagnostic output must be under %s/diagnostics\n' "$CHECKPOINT" >&2; exit 2 ;;
esac
[[ ! -e "$OUT" ]] || { printf 'refusing to overwrite diagnostic output: %s\n' "$OUT" >&2; exit 2; }
[[ ! -L "$OUT" ]] || { printf 'refusing diagnostic output symlink: %s\n' "$OUT" >&2; exit 2; }

GPU_INFO="$(uv run --locked --extra gpu python - <<'PY'
import torch
count = torch.cuda.device_count()
name = torch.cuda.get_device_name(0) if count else ""
print(f"{count}\t{name}")
PY
)"
IFS=$'\t' read -r GPU_COUNT GPU_NAME <<<"$GPU_INFO"
[[ "$GPU_COUNT" -eq 1 ]] || { printf 'diagnosis requires exactly one visible GPU, got %s\n' "$GPU_COUNT" >&2; exit 2; }
[[ "$GPU_NAME" == *H100* ]] || { printf 'diagnosis requires H100, got %s\n' "$GPU_NAME" >&2; exit 2; }
printf 'job=%s node=%s gpu=%s revision=%s output=%s\n' \
  "${SLURM_JOB_ID:-unknown}" "$(hostname)" "$GPU_NAME" "$SOURCE_REVISION" "$OUT"

uv run --locked --extra gpu --extra convert python scripts/analysis/diagnose_hf_conversion.py \
  --hf-path "$HF_PATH" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUT" \
  --source-revision "$SOURCE_REVISION"

uv run --locked --extra gpu --extra convert python scripts/analysis/diagnose_hf_conversion.py \
  --hf-path "$HF_PATH" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUT" \
  --source-revision "$SOURCE_REVISION" \
  --validate-only

[[ "$(git rev-parse HEAD)" == "$SOURCE_REVISION" ]] || {
  printf 'source revision changed during diagnosis\n' >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
  printf 'source checkout became dirty during diagnosis\n' >&2
  exit 2
}
chmod 0444 "$OUT"
printf 'diagnostic_sha256=%s\n' "$(sha256sum "$OUT" | cut -d' ' -f1)"
