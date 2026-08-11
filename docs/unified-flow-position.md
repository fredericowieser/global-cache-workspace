# Unified flow-workspace + continuous-position prototype

The prototype composes existing primitives rather than adding a third transformer path:

1. `GlobalCacheProcessor.encode_source` builds a per-cache-head workspace source.
2. Two tied `deliberate` cycles run before midpoint-2 flow refinement (`max_velocity=0.1`).
3. `write_for_readout` edits the recipient value cache at recipient-selected positions.
4. A bounded (`0.1 * tanh`) workspace adapter conditions the existing `ContinuousThoughtEmitter`.
5. Two cycles x two latent embeddings pass through frozen Qwen blocks and become four ordinary causal K/V positions.

`active=False` returns the exact input hidden tensor and exact cache object. It calls neither processor nor emitter. Shuffled controls transplant donor hidden/workspace content but retain recipient write indices, so donor prompt length cannot change the recipient intervention location. The backbone is never owned by the processor checkpoint and must have every parameter frozen.

## Canonical H200 run

```bash
bash scripts/cloud/run_unified_flow_position_h200.sh
```

Expanded:

```bash
uv run --extra gpu --extra eval python scripts/generate_unified_flow_position_config.py \
  --output configs/explore/unified-flow-position-qwen3-06b-v1.json
uv run --extra gpu --extra eval python scripts/prepare_gsm8k_official_test.py \
  --output data/eval/gsm8k-test-v1.jsonl \
  --audit data/eval/gsm8k-test-v1.audit.json
uv run --extra gpu --extra eval python scripts/run_unified_flow_position.py \
  --config configs/explore/unified-flow-position-qwen3-06b-v1.json \
  --mode both
```

The evaluator rejects any official test artifact that does not contain exactly 1,319 rows. It uses Qwen's actual chat-template `enable_thinking=false|true` switch and generates all six combinations of `off|matched|shuffled` and `thinking_off|thinking_on` (7,914 generations). Selection compares mean exact generated-answer accuracy across both thinking modes for `matched` against `off`; shuffled remains a specificity control.

## Parameter expectation for Qwen3-0.6B config

| Trainable component | Parameters |
|---|---:|
| Longshot workspace processor | 4,460,673 |
| Workspace norm + bounded adapter | 131,328 |
| Continuous emitter | 1,838,080 |
| **Total trainable** | **6,430,081** |

All Qwen backbone parameters are frozen and excluded from `processor.safetensors`.
