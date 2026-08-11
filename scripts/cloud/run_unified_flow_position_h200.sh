#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIG=${CONFIG:-configs/explore/unified-flow-position-qwen3-06b-v1.json}
EVAL_DATA=${EVAL_DATA:-data/eval/gsm8k-test-v1.jsonl}
EVAL_AUDIT=${EVAL_AUDIT:-data/eval/gsm8k-test-v1.audit.json}

command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv sync --extra gpu --extra eval --group dev

if [[ ! -f "$CONFIG" ]]; then
  uv run --extra gpu --extra eval python scripts/generate_unified_flow_position_config.py \
    --output "$CONFIG"
fi
if [[ ! -f "$EVAL_DATA" ]]; then
  uv run --extra gpu --extra eval python scripts/prepare_gsm8k_official_test.py \
    --output "$EVAL_DATA" --audit "$EVAL_AUDIT"
fi

uv run --extra gpu --extra eval python -m pytest -q \
  tests/test_unified_flow_position.py \
  tests/test_continuous_thought.py \
  tests/test_longshot_workspace_flow.py

# One H200, frozen Qwen3-0.6B, one training epoch over 6,000 rows, then exactly
# 1,319 generated answers for each of off/matched/shuffled x thinking off/on.
uv run --extra gpu --extra eval python scripts/run_unified_flow_position.py \
  --config "$CONFIG" --mode both
