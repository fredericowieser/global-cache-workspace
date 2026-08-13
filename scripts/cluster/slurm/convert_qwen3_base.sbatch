#!/usr/bin/env bash
#SBATCH --job-name=gcw-qwen-base-convert
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
OUT="${OUT:-/mnt/jfs-01/shared/gcw/checkpoints/base_backbones/qwen3_06b/tf-v1}"
export SOURCE_REVISION HF_PATH OUT

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
[[ -f "$HF_PATH/model.safetensors" && -f "$HF_PATH/config.json" ]] || {
  printf 'incomplete HF source: %s\n' "$HF_PATH" >&2
  exit 2
}
SOURCE_PREFLIGHT_RAW="$(python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["HF_PATH"]).resolve(strict=True)
config_path = root / "config.json"
model_path = root / "model.safetensors"
config = json.loads(config_path.read_text())
expected = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"HF source config mismatch for {key}: {config.get(key)!r} != {value!r}"
        )

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

print(f"HF_CONFIG_SHA256_BEFORE={sha256(config_path)}")
print(f"HF_MODEL_SHA256_BEFORE={sha256(model_path)}")
PY
)"
mapfile -t SOURCE_PREFLIGHT <<<"$SOURCE_PREFLIGHT_RAW"
for assignment in "${SOURCE_PREFLIGHT[@]}"; do
  case "$assignment" in
    HF_CONFIG_SHA256_BEFORE=*|HF_MODEL_SHA256_BEFORE=*) export "$assignment" ;;
    *) printf 'unexpected HF source preflight output: %s\n' "$assignment" >&2; exit 2 ;;
  esac
done
[[ ! -e "$OUT" && ! -L "$OUT" ]] || {
  printf 'refusing to overwrite conversion output: %s\n' "$OUT" >&2
  exit 2
}
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT" || {
  printf 'failed to claim conversion output: %s\n' "$OUT" >&2
  exit 2
}

export PATH="$HOME/.local/bin:$PATH"
uv run --locked --extra gpu --extra convert gcw convert hf \
  --family qwen3 \
  --model "$HF_PATH" \
  --variant tf \
  --dtype bfloat16 \
  --out "$OUT"

[[ "$(git rev-parse HEAD)" == "$SOURCE_REVISION" && -z "$(git status --porcelain)" ]] || {
  printf 'source changed during conversion\n' >&2
  exit 2
}
uv run --locked --extra gpu python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import tempfile

source = Path(os.environ["HF_PATH"]).resolve(strict=True)
output = Path(os.environ["OUT"]).resolve(strict=True)
state = json.loads((output / "train_state.json").read_text())
if Path(state.get("source", "")).resolve() != source:
    raise SystemExit("converted checkpoint source mismatch")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

source_config_sha256 = sha256(source / "config.json")
source_model_sha256 = sha256(source / "model.safetensors")
if source_config_sha256 != os.environ["HF_CONFIG_SHA256_BEFORE"]:
    raise SystemExit("HF config changed during conversion")
if source_model_sha256 != os.environ["HF_MODEL_SHA256_BEFORE"]:
    raise SystemExit("HF model weights changed during conversion")

files = {}
for path in (
    source / "model.safetensors",
    source / "config.json",
    output / "model.safetensors",
    output / "config.json",
    output / "train_state.json",
):
    if not path.is_file():
        raise SystemExit(f"missing conversion artifact: {path}")
    files[str(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
payload = json.dumps(
    {
        "schema_version": 1,
        "kind": "qwen3_hf_to_gcw_conversion",
        "source_revision": os.environ["SOURCE_REVISION"],
        "hf_path": str(source),
        "checkpoint": str(output),
        "source_identity_verified": True,
        "source_hashes_stable_during_conversion": True,
        "hf_config_sha256": source_config_sha256,
        "hf_model_sha256": source_model_sha256,
        "files": files,
    },
    indent=2,
    sort_keys=True,
).encode()
destination = output / "CONVERSION.json"
fd, temporary_name = tempfile.mkstemp(dir=output, prefix=".CONVERSION.json.", suffix=".tmp")
temporary = Path(temporary_name)
try:
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, destination)
    directory_fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    temporary.unlink(missing_ok=True)
PY
