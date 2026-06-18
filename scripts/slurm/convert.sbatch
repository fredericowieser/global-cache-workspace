#!/usr/bin/env bash
# Convert HF/PPT checkpoints into native gcw safetensors.
# Required env:
#   SOURCE=hf MODEL=<hf-id-or-path> FAMILY=<llama|qwen3> OUT=<output-dir>
#   SOURCE=ppt CHECKPOINT=<model_*.pt> META=<meta_*.json> OUT=<output-dir>

#SBATCH --job-name=gcw-convert
#SBATCH --partition=agent-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_out/%x-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

REPO_ROOT="${REPO_ROOT:-/mnt/jfs-00/team-agent/home/f50056118/global-cache-workspace}"
cd "$REPO_ROOT"
mkdir -p slurm_out

DTYPE="${DTYPE:-bfloat16}"
export HF_HOME="${HF_HOME:-/mnt/jfs-00/team-agent/home/f50056118/.cache/huggingface}"

if [ "${SOURCE:?SOURCE is required}" = "hf" ]; then
  uv run --extra cpu --extra convert gcw convert hf \
    --family "${FAMILY:?FAMILY is required}" \
    --model "${MODEL:?MODEL is required}" \
    --dtype "$DTYPE" \
    --out "${OUT:?OUT is required}"
elif [ "$SOURCE" = "ppt" ]; then
  uv run --extra cpu gcw convert ppt \
    --checkpoint "${CHECKPOINT:?CHECKPOINT is required}" \
    --meta "${META:?META is required}" \
    --dtype "$DTYPE" \
    --out "${OUT:?OUT is required}"
else
  echo "Unknown SOURCE=$SOURCE" >&2
  exit 2
fi
