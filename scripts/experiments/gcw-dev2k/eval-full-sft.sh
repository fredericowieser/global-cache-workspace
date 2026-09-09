#!/usr/bin/env bash
# Run benchmarks on the dev2k full-sft (no cache processor) backbone checkpoint.
# Usage: eval-full-sft.sh <jobid>
# Mirrors launch.sh env-hygiene; runs under nohup, logs to $ROOT/logs/eval-full-sft.log
set -uo pipefail
REPO=/home/f50056118/code/global-cache-workspace
cd "$REPO" || exit 1
ROOT=/mnt/jfs-00/team-agent/home/f50056118/gcw-experiments/qwen3-06b-instruct-math-full-single-seed-v2-matrix11-dev2k
CK="$ROOT/training-state/full-sft/checkpoint"
JOBID=${1:?jobid}
OUT="$ROOT/evaluations-v3/full-sft/eval.json"
mkdir -p "$(dirname "$OUT")"
env -u GCW_ALLOW_UNQUALIFIED_CUDA_GRAPH -u GCW_ALLOW_UNQUALIFIED_TRITON_CACHE \
    -u GCW_ALLOW_REJECTED_ICARUS_COMPILE -u GCW_BT_PROCESSOR_BACKEND \
    -u GCW_CACHE_PROCESSOR_BACKEND -u GCW_COMPILE -u GCW_ICARUS_PROCESSOR_BACKEND \
    -u GCW_STOP_AFTER_CHECKPOINT_STEP -u GCW_STOP_AFTER_CHECKPOINT_UTC \
    PYTHONPATH="$REPO/src:$REPO" PYTHONUNBUFFERED=1 \
    srun --jobid="$JOBID" --overlap -N1 -n1 --gres=gpu:1 \
    "$REPO/.venv/bin/python" -m gcw.cli eval chat \
        --checkpoint "$CK" \
        --tokenizer-path /mnt/jfs-01/shared/gcw/hf/Qwen3-0.6B-base-v1 \
        --processor off \
        --tasks gsm8k_openmath svamp gsm8k_hard \
        --mode thinking_off \
        --batch-size 16 \
        --out "$OUT" \
    >> "$ROOT/logs/eval-full-sft.log" 2>&1