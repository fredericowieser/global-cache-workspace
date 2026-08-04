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
loss = sum(mask[t] * sum_v k[t,v]) / sum(mask)
```

Both log-softmax operations run in FP32 and sequence chunks while retaining full-vocabulary normalization. Logits are divided by `temperature`; there is no additional temperature-squared multiplier. Teacher reasoning is context, never a loss position. Source dataset answers are ignored. `token_clip` must be null: individual vocabulary contributions can be negative even though their sum is non-negative, so independently capping positive contributions creates an invalid surrogate that can decrease while true KL increases. Ordinary gradient clipping handles optimisation spikes without changing the estimand.

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
uv run --extra gpu --extra convert gcw train --config configs/train/opsd/qwen3_06b_opsd_pilot128_exact_kl_v2.yaml
```

Run it inside a Slurm GPU allocation. Do not run GPU work on a login node.

The output directory contains:

- `run_config.json` — resolved run configuration;
- `teacher-traces.jsonl` — reusable privileged traces;
- `metrics.jsonl` — one terminal status per source prompt;
- `summary.json` — aggregate counts and runtime;
- `final/` — native safetensors checkpoint and train state.

The corrected pilot records exact per-example forward KL, event-level relative-L2 cache displacement, initial/final processor hashes and gates, seed/initialization recipe, Torch/tokenizer identities, run-config hash, source revision, and mixed-dtype checkpoint fidelity. Training-stream KL is summarized as an on-policy stream metric, not claimed as a matched pre/post reduction.

The trainer refuses to overwrite existing metrics or a final checkpoint.

## Three-way paired causal screen

After the fixed-example gate passes, evaluate the corrected 128-prompt checkpoint on three CoT-off controls:

1. trained checkpoint with the cache processor disabled;
2. a seed-identical FP32 untrained processor control with the cache processor enabled;
3. trained checkpoint with the cache processor enabled.

Use the first 64 examples from each of `gsm8k_openmath`, `svamp`, and `gsm8k_hard`, greedy decoding, one common 512-token cap, and the exact training-time cache schedule (prompt boundary and every 64 response tokens). This is an exploratory paired gate, not a final accuracy estimate.

Derive the untrained control with `scripts/derive_untrained_cp_control.py`. It reconstructs the seed-24 processor using the exact training initialization order, preserves the trained checkpoint's frozen BF16 backbone, stores the processor in FP32, verifies a standard-loader round trip, and publishes by atomic rename.

Run `gcw eval opsd-screen` once per condition with the full Git SHA, explicit tokenizer source, and common dataset cache. Combine the three JSON files with `gcw eval opsd-three-way`, then bind them to the pilot's `metrics.jsonl` with `gcw eval opsd-overlap`. Aggregation fails if source revisions, model configs, tokenizer identity, task definitions, dataset files, ordered sample/target manifests, rendered prompt IDs, event schedules, or untrained-control provenance differ. The overlap audit reports exact and whitespace-normalized prompt collisions; do not interpret the screen as held-out when either count is non-zero. The aggregate reports task-level exact accuracy, paired wrong→right/right→wrong transitions, exact McNemar p-values, shared-index deterministic bootstrap intervals, response identity, generation length, token-cap frequency, and event counts. The three-task macro is descriptive because GSM-Hard derives from GSM8K.

The primary contrasts are trained minus untrained and trained minus processor-off. Untrained minus processor-off isolates damage or benefit from the cache architecture before learning.

## Matched held-out KL and teacher-consistency audit

When trained-active does not beat the seed-identical untrained control, run `gcw eval opsd-kl-audit --config configs/eval/opsd/matched-heldout-kl-audit-v1.yaml` before changing training dose. The production config admits no CLI protocol overrides. It scans source rows from index 128, excludes exact and whitespace-normalized collisions with all 128 training rows and 192 evaluation prompts, and locks the first 32 valid examples within 64 attempts.

For each locked prompt, the untrained processor generates one student response and one self-generated teacher completion. The audit preserves the raw completion IDs and distinguishes a native thinking-close delimiter from a direct-answer EOT rewritten into a close. Both checkpoints are then scored on the exact same student token IDs; the teacher-logit hash and cache-event schedule must match across sequential passes. The primary estimand is token-weighted exact forward KL under the trained processor minus the seed-identical untrained processor, with prompt-level paired bootstrap intervals. Teacher-context consistency is measured separately by scoring the same normalized dataset-solution tokens under (a) the no-privilege direct-answer context and (b) the self-generated reasoning context. Lower dataset-solution NLL is better, but the normalized `answer` field has no independent verification provenance. This is neither teacher correctness nor answer accuracy.

The output binds the clean audit revision, pilot revision, source dataset, training/evaluation manifests, tokenizer, checkpoint bundles, locked prompts, raw and transformed teacher completions, student responses, unverified dataset-solution IDs, cache events and processor states. It is published once with atomic no-clobber semantics.

Decision rule:

1. trained KL does not decrease credibly: run a 128-row four-epoch optimisation-dose ablation;
2. trained KL decreases but held-out accuracy remains flat/down: redesign the objective rather than adding data;
3. self-generated reasoning does not reduce dataset-solution NLL, or most traces are EOT rewrites: inspect/replace the teacher-trace policy before further distillation.

## Overfit and checkpoint-fidelity diagnostic

Before scaling a failed 128-prompt pilot, run the explicit `overfit_diagnostic` path on the first 21 OpenMath rows. The prior pilot established that these contain exactly 16 valid source-indexed prompts. Preparation locks each valid teacher trace and initial student response before any optimiser update; repeated epochs optimise the same 16 examples rather than changing the target sample.

The passing diagnostic uses `qwen3_06b_opsd_overfit16x40.yaml`: 40 epochs × 16 valid prompts = 640 optimiser steps. A one-prompt, two-epoch config is provided for a real-device smoke. Diagnostic JSONL rows record epoch, exact forward KL, gate before/after the example, event cache lengths, and event-level relative-L2 cache displacement.

Checkpoint publication keeps the frozen backbone in BF16 and the Cache Processor master/checkpoint weights in FP32. CUDA processor matrix operations still run under BF16 autocast and write deployed updates into the BF16 cache; this diagnostic tests weight-publication fidelity, not full-FP32 processor execution. Before publication, the processor state and gate are hashed; the temporary checkpoint is reloaded through the standard loader and must match exactly. The diagnostic additionally compares a recurrent-event-covering student-logit hash and the same locked-corpus metrics immediately before save and after reload.

Proceed to a repeated 128-prompt pilot only if all gates pass:

1. processor state hash, scalar gate, and logit-probe hash match exactly across reload;
2. every diagnostic metric is finite and cache displacement is non-zero;
3. token-weighted exact forward KL after 640 steps is at most 25% of its initial value;
4. all 16 locked examples complete every epoch without skips.

## V0 limitations

- one process and one finite epoch, except for explicit fixed-example overfit diagnostics;
- batch size one, with optional gradient accumulation;
- no exact optimiser resume yet;
- malformed/missing thinking-close traces may be skipped by policy;
- CUDA/kernel/OOM/non-finite-loss failures abort rather than being silently skipped;
- full-dataset training is gated on the 128-prompt pilot and subsequent 10k causal matrix.
