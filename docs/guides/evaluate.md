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

Evaluation constructs independent state. It does not mutate a training checkpoint. Experiment pages define stricter task sets, controls, and evidence gates.

## Canonical scores versus numeric thinking comparisons

Keep two protocols separate:

- Multiple-choice scores use full LM Evaluation Harness datasets, no chat template, and benchmark-native option likelihood. They are comparable to the corresponding harness protocol and have no thinking dimension.
- Numeric thinking comparisons use the official tokenizer chat template, deterministic generation, numeric-last-answer scoring, and the same 2,048-token cap in both modes.

Generated multiple-choice answers are useful only as supplementary instruction-following diagnostics. Do not substitute them for canonical likelihood scores or schedule them across every experiment arm.

```bash
uv run --extra gpu --extra eval gcw eval general \
  --checkpoint /path/to/checkpoint \
  --prompt-mode plain \
  --processor on \
  --out data/eval-runs/standard-general.json

uv run --extra gpu --extra eval gcw eval chat \
  --checkpoint /path/to/checkpoint \
  --tokenizer-path /path/to/qwen-tokenizer \
  --tasks gsm8k_openmath svamp gsm8k_hard \
  --mode thinking_on \
  --max-new-tokens 2048 \
  --out data/eval-runs/chat-thinking-on.json
```

The maintained Qwen3 experiment evaluates HellaSwag, PIQA, WinoGrande, ARC, MMLU, LogiQA, and Gaokao MathQA in their full no-chat likelihood protocols. Chat generation is reserved for the full GSM8K OpenMath, SVAMP, and GSM8K Hard datasets. Generated results record parse failures, truncations, response tokens, processor state, selected-row hashes, and per-task accuracy. LAMBADA is excluded because next-word completion has no meaningful thinking-on/chat analogue.

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
