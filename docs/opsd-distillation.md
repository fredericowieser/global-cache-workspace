# On-policy self-distillation with teacher reasoning

`mode: cot_distillation` trains only a BGT cache processor. The native Qwen3 backbone is both the frozen teacher and the frozen student backbone.

## Objective

For prompt `x`:

1. Render a thinking-on teacher prompt and sample teacher reasoning `z_T`.
2. Render a thinking-off student prompt and sample an on-policy answer `y` with BGT active.
3. Replay `y` through the student with gradients.
4. Score the same `y` with the frozen teacher after context `(x, z_T)`.
5. Apply full-vocabulary forward KL only at answer positions.

For answer position `t` and vocabulary item `v`:

```text
k[t,v] = p_T[t,v] * (log p_T[t,v] - log p_S[t,v])
loss = sum(mask[t] * sum_v min(k[t,v], token_clip)) / sum(mask)
```

Both log-softmax operations run in FP32 and sequence chunks while retaining full-vocabulary normalization. Logits are divided by `temperature`; there is no additional temperature-squared multiplier. Teacher reasoning is context, never a loss position. Source dataset answers are ignored.

## Cache schedule

- User/system prompt: ordinary frozen-backbone prefill; no cache processing.
- Response count `0`: process the prompt cache before feeding the assistant trigger. This affects answer token 1.
- Response counts `64, 128, ...`: process after consuming a complete answer block.
- A recurrent event runs only when another answer distribution follows; terminal refreshes are omitted.
- After a recurrent event, truncate the last token's cached K/V and replay that token at the same position. This refreshes the next-token logits after the edit without appending a duplicate token.

Differentiable replay executes answer blocks as causal chunks. No-grad teacher and student sampling use preallocated static KV caches. Sampling remains autoregressive. Both paths share the same event schedule.

V0 supports standard-attention `qwen3` BGT models only. It rejects Qwen3.5/Gemma linear-state caches because truncation cannot reconstruct their recurrent state safely.

## Data and traces

`data.kind` must be `openmath_prompts`. The reader accepts JSONL or a Hugging Face dataset directory, extracts prompt-like fields, and never reads solution fields into the model input.

Teacher reasoning is cached in append-only `teacher-traces.jsonl`. Every row binds:

- teacher/config fingerprint;
- source row index;
- SHA-256 of the prompt;
- reasoning token IDs.

A fingerprint or prompt mismatch fails closed. Teacher and student use independent per-row RNG streams, so reusing a cached trace does not change student sampling.

Native teacher `</think>` output is preferred. If a direct-answer SFT checkpoint instead terminates its worked solution with EOT, V0 removes that terminal EOT and appends the configured thinking-close delimiter. Budget-exhausted generations with neither terminator remain invalid.

## Pilot

The checked-in cluster pilot processes 128 prompts once:

```bash
uv run --extra gpu --extra convert gcw train --config configs/train/opsd/qwen3_06b_opsd_pilot128.yaml
```

Run it inside a Slurm GPU allocation. Do not run GPU work on a login node.

The output directory contains:

- `run_config.json` — resolved run configuration;
- `teacher-traces.jsonl` — reusable privileged traces;
- `metrics.jsonl` — one terminal status per source prompt;
- `summary.json` — aggregate counts and runtime;
- `final/` — native safetensors checkpoint and train state.

The trainer refuses to overwrite existing metrics or a final checkpoint.

## Quick causal screen

Before 10k training, evaluate the 128-prompt checkpoint on the same paired zero-shot samples under four conditions: CoT off/on × trained cache processor off/on. The quick screen uses the first 64 examples from each of `gsm8k_openmath`, `svamp`, and `gsm8k_hard`, greedy decoding, one common 512-token cap, and the exact training-time cache schedule (prompt boundary and every 64 response tokens). It is an exploratory gate, not a final accuracy estimate.

Use `gcw eval opsd-screen` once per condition with the full Git SHA and explicit tokenizer source, then combine the four JSON files with `gcw eval opsd-aggregate`. The evaluator hashes the tokenizer vocabulary/template identity rather than mutating an older checkpoint that lacks tokenizer metadata. Aggregation fails if checkpoint hashes, tokenizer identity, source revisions, task definitions, ordered prompt hashes, or processor-event evidence differ across conditions.

The primary contrasts are:

- cache effect with CoT off: `(CoT off, CP on) - (CoT off, CP off)`;
- cache effect with CoT on: `(CoT on, CP on) - (CoT on, CP off)`;
- CoT gap with the processor disabled;
- interaction between CoT and the trained cache processor.

## Overfit and checkpoint-fidelity diagnostic

Before scaling a failed 128-prompt pilot, run the explicit `overfit_diagnostic` path on the first 21 OpenMath rows. The prior pilot established that these contain exactly 16 valid source-indexed prompts. Preparation locks each valid teacher trace and initial student response before any optimiser update; repeated epochs optimise the same 16 examples rather than changing the target sample.

The main diagnostic uses `qwen3_06b_opsd_overfit16x20.yaml`: 20 epochs × 16 valid prompts = 320 optimiser steps. A one-prompt, two-epoch config is provided for a real-device smoke. Diagnostic JSONL rows record epoch, unclipped KL, gate before/after the example, event cache lengths, and event-level relative-L2 cache displacement.

Checkpoint publication keeps the frozen backbone in BF16 and the Cache Processor master/checkpoint weights in FP32. CUDA processor matrix operations still run under BF16 autocast and write deployed updates into the BF16 cache; this diagnostic tests weight-publication fidelity, not full-FP32 processor execution. Before publication, the processor state and gate are hashed; the temporary checkpoint is reloaded through the standard loader and must match exactly. The diagnostic additionally compares a recurrent-event-covering student-logit hash and the same locked-corpus metrics immediately before save and after reload.

Proceed to a repeated 128-prompt pilot only if all gates pass:

1. processor state hash, scalar gate, and logit-probe hash match exactly across reload;
2. every diagnostic metric is finite and cache displacement is non-zero;
3. token-weighted unclipped KL after 320 steps is at most 25% of its initial value;
4. all 16 locked examples complete every epoch without skips.

## V0 limitations

- one process and one finite epoch, except for explicit fixed-example overfit diagnostics;
- batch size one, with optional gradient accumulation;
- no exact optimiser resume yet;
- malformed/missing thinking-close traces may be skipped by policy;
- CUDA/kernel/OOM/non-finite-loss failures abort rather than being silently skipped;
- full-dataset training is gated on the 128-prompt pilot and subsequent 10k causal matrix.
