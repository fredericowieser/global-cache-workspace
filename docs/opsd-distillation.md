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

## V0 limitations

- one process and one finite epoch;
- batch size one, with optional gradient accumulation;
- no exact optimiser resume yet;
- malformed/missing thinking-close traces may be skipped by policy;
- CUDA/kernel/OOM/non-finite-loss failures abort rather than being silently skipped;
- full-dataset training is gated on the 128-prompt pilot and subsequent 10k causal matrix.
