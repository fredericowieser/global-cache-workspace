# Evaluate a checkpoint

**Status:** Maintained

1. Install evaluation dependencies.

   ```bash
   uv sync --extra cpu --extra eval
   ```

2. Run a bounded suite.

   ```bash
   uv run --extra cpu --extra eval gcw eval all \
     --checkpoint /path/to/checkpoint \
     --limit 1 \
     --out data/eval-runs/manual-smoke.json
   ```

3. Inspect the JSON output and provenance before increasing the limit.

Evaluation constructs independent state. It does not mutate a training checkpoint. Campaign pages define stricter task sets, controls, and evidence gates.

## Qwen3 GSM8K reproduction matrix

Use `scripts/analysis/run_benchmarks.py` when exact generated rows and protocol identity matter. It supports HF directories and native GCW checkpoints, processor bypass, separate tokenizer ownership, and both maintained scorers.

```bash
uv run python -m scripts.analysis.run_benchmarks \
  --target-kind gcw \
  --model-path /path/to/checkpoint \
  --tokenizer-path /path/to/qwen-tokenizer \
  --processor off \
  --scoring-protocol lm_eval_flexible_exact \
  --output-dir data/eval-runs/gsm8k-cell
```

`numeric_last_answer` and `lm_eval_flexible_exact` are different scientific protocols. The historical Qwen3 BGT cells used flexible exact match. Results must retain scorer, batch size, generation cap, tokenizer hash, checkpoint hash, prompt mode, and processor state.

The maintained fast path uses length-sorted, left-padded Hugging Face generation for HF targets and preallocated static-cache generation for native GCW targets. The default batch size is 64. An authenticated Qwen3-0.6B GSM8K validation completed both 1,319-row modes in 35 minutes and scored 781 thinking-off / 899 thinking-on. The earlier exact-length run scored 782 / 888. Batch 128 was slower and drifted further on thinking-on, so 64 is the calibrated default. Treat these as close downstream-score parity, not token-level parity: batch shape changes floating-point execution and therefore greedy trajectories.
