#!/usr/bin/env bash
# Launch one probe arm on a camped slurm allocation (lrc).
# Usage: launch.sh <arm> <jobid>
# Run under nohup so it survives the ssh session; logs to $ROOT/logs/<arm>.log.
set -uo pipefail
REPO=/home/f50056118/code/global-cache-workspace
ROOT=/mnt/jfs-00/team-agent/home/f50056118/gcw-experiments/qwen3-06b-instruct-math-full-single-seed-v2-matrix11-dev2k
ARM=${1:?arm}
JOBID=${2:?jobid}
mkdir -p "$ROOT/logs" "$ROOT/training-state"
env -u GCW_ALLOW_UNQUALIFIED_CUDA_GRAPH -u GCW_ALLOW_UNQUALIFIED_TRITON_CACHE \
    -u GCW_ALLOW_REJECTED_ICARUS_COMPILE -u GCW_BT_PROCESSOR_BACKEND \
    -u GCW_CACHE_PROCESSOR_BACKEND -u GCW_COMPILE -u GCW_ICARUS_PROCESSOR_BACKEND \
    -u GCW_STOP_AFTER_CHECKPOINT_STEP -u GCW_STOP_AFTER_CHECKPOINT_UTC \
    PYTHONPATH="$REPO/src:$REPO" PYTHONUNBUFFERED=1 \
    srun --jobid="$JOBID" --overlap -N1 -n1 --gres=gpu:1 \
    "$REPO/.venv/bin/python" -m gcw.cli train --config "$ROOT/configs/$ARM.yaml" \
    >> "$ROOT/logs/$ARM.log" 2>&1
