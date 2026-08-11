# Spike 001 — continuous thought positions

## Verdict

**FEASIBLE.** A trainable processor can emit a model-width latent embedding, pass it through the frozen GCW Qwen3-compatible decoder as a new causal position, append the resulting per-layer K/V to `WindowedCache`, and receive gradients from terminal answer logits. Repeating the loop appends one position per step without changing old K/V.

This is a throwaway interface spike. It changes no production file.

## What this proves

The prototype uses the real GCW native decoder primitives (`TF`, `Block`, rotary positions, `WindowedCache`, and the language-model head) with a tiny random Qwen3-family configuration. It adds only a standalone `forward_embedding_chunk` analogue of `TF.forward_chunk` because production `TF.forward_chunk` currently accepts token IDs only.

Flow:

1. Prefill four prompt tokens into the frozen decoder/cache.
2. A recurrent `ContinuousThoughtEmitter` emits a 32-wide latent.
3. The frozen decoder consumes that latent at `start_pos == cache.seq_len`.
4. Normal decoder K/V for the latent are appended to every layer's cache.
5. The decoder's latent-position hidden state feeds the next emitter step.
6. A terminal answer-prefix token is decoded after the K latent positions; its logits produce the loss.

| Check | Observed result |
|---|---:|
| K=0 no-op cache identity | `true` |
| K=0 terminal-logit max delta vs ordinary decode | `0.0` |
| Old prompt K/V and scores unchanged after append | `true` |
| Prompt cache length | `4` |
| K=1 thought start / resulting cache length | `4` / `5` |
| K=1 logits delta vs no thought | `0.07144232094287872` |
| K=1 emitted-latent logits delta vs zero latent at same position | `0.0444067120552063` |
| K=3 starts | `[4, 5, 6]` |
| K=3 cache lengths after each thought | `[5, 6, 7]` |
| Final cache length after terminal token | `8` |
| K=3 logits delta vs K=1 | `0.1256105601787567` |
| Emitter gradient L1 | `24.80518341064453` |
| Emitter tensors with nonzero gradient | `4` |
| Frozen backbone tensors with gradients | `0` |

The emitted latent affects logits independently of position growth: emitted-vs-zero latent comparison holds K fixed at one and gives a nonzero `0.0444` max-logit delta.

## Important no-op finding

**A zero embedding is not a no-op if it still occupies a causal position.** The zero-latent branch changed terminal logits by `0.0665399432182312` versus K=0. It both shifts the terminal RoPE position and adds an attention slot. Therefore an identity initialization must skip insertion (K=0) or add explicit masking/position semantics. Multiplying an inserted latent by a zero gate is insufficient.

## Exact commands and observed output

Run from `/home/fred/flow/code/global-cache-workspace` using the existing project virtual environment.

### Executable spike

```bash
.venv/bin/python spikes/001-continuous-thought-positions/continuous_thought_positions.py
```

Observed:

```json
{
  "architecture": "GCW native tiny Qwen3 TF + external recurrent continuous-thought emitter",
  "backbone_parameters_frozen": true,
  "backbone_parameters_with_grad": 0,
  "emitter_grad_l1": 24.80518341064453,
  "emitter_parameters_with_nonzero_grad": 4,
  "no_op_cache_identity": true,
  "no_op_logit_max_abs_delta": 0.0,
  "old_cache_prefix_unchanged_after_append": true,
  "one_step_cache_length": 5,
  "one_step_start_positions": [
    4
  ],
  "one_step_vs_no_thought_logit_max_abs_delta": 0.07144232094287872,
  "one_step_vs_zero_latent_logit_max_abs_delta": 0.0444067120552063,
  "prompt_length": 4,
  "repeated_cache_lengths": [
    5,
    6,
    7
  ],
  "repeated_final_cache_length": 8,
  "repeated_k": 3,
  "repeated_start_positions": [
    4,
    5,
    6
  ],
  "repeated_vs_one_step_logit_max_abs_delta": 0.1256105601787567,
  "terminal_loss": 4.278348445892334,
  "verdict": "FEASIBLE: emitted embeddings can append differentiable causal positions without rewriting old KV.",
  "zero_latent_vs_no_thought_logit_max_abs_delta": 0.0665399432182312
}
```

The same machine-readable payload is in `observed-output.json`.

### Test

```bash
.venv/bin/python -m pytest -q spikes/001-continuous-thought-positions/test_continuous_thought_positions.py
```

Observed:

```text
.                                                                        [100%]
1 passed in 2.60s
```

### Lint and format

```bash
.venv/bin/ruff check spikes/001-continuous-thought-positions
.venv/bin/ruff format --check spikes/001-continuous-thought-positions
```

Observed after formatting:

```text
All checks passed!
2 files already formatted
```

### Existing Longshot/process-branch regression tests

```bash
.venv/bin/python -m pytest -q tests/test_longshot_workspace.py tests/test_process_branch.py
```

Observed:

```text
................                                                         [100%]
16 passed in 3.55s
```

## Scope and next integration seam

This proves execution and autograd mechanics, not model quality. The decoder is tiny and randomly initialized; no pretrained Qwen3 checkpoint or task training is involved.

The narrow production seam would be an `inputs_embeds`/`forward_embedding_chunk` path on `TF` (and corresponding family-specific backbones), followed by process-branch position accounting that starts answer decoding at `prompt_len + K`. Existing Longshot `write_for_readout` rewrites the latest value slot and keeps cache length fixed; this spike instead preserves the old prefix and appends ordinary decoder-generated K/V.
