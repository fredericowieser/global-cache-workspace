# Icarus feature parity with Longshot

**Status:** Implementation groups ordered; linear-state design pending

**Initial inventory:** 2026-10-23

This document tracks features implemented in Longshot that are missing from Icarus, so we can decide which to port, adapt, defer, or leave out. It is an inventory, not a commitment to bring every feature over. Some differences define the architectures; others depend on particular backbones or belong to experimental paths.

The initial comparison covers the processor implementations, configuration validation, shared runtime, tests, and documentation. Experimental qualifications below describe documented evidence, not an assumption that every implemented feature has demonstrated a benefit.

## Tracking status

The tables below are the single source of truth for each item's status and rationale. Keep item numbers stable and update the relevant row directly as work progresses, including implementation and validation links when available. **On hold** identifies deferred work and **N/A** identifies an architectural distinction to preserve. For planned work, the group number gives the implementation order; items in the same group are developed together. **Not started** means implementation has not begun, and **Design needed** means the decisions described in that row must be resolved first. Validate each group before building on it, and record progress in these rows rather than in a separate log.

## Routing, recurrence, and update controls

| # | Feature Longshot has | What is missing from Icarus | Status |
|---|---|---|---|
| 1 | Internal recurrence | Repeated passes through the same processor layers before one cache write, controlled by `iterations` and `iteration_mode: internal`. | On hold |
| 2 | External recurrence | Repeated selection, processing, and cache writeback on the same prefix. Icarus currently performs one pass per invocation. | On hold |
| 3 | Head reselection between internal iterations | Reselection using attention-output magnitude, cache-head attention scores, projected KV norms, learned contextual scores, or random sampling. | On hold |
| 4 | Learned contextual head selection | A selector that scores each head against a pooled context, with configurable selector width and optional head-identity bias. Icarus uses magnitude ranking. | On hold |
| 5 | Differentiable selector training | All-head warmup, soft exact-budget routing, temperature annealing, and transition to hard top-k, including attention support for soft routing biases. | On hold |
| 6 | Random-routing ablations | Random initial selection and iterative reselection, plus an evaluation override that changes routing without modifying checkpoint weights. | On hold |
| 7 | Context-dependent update gates | `contextual_scalar` gates per example/head and `contextual_vector` gates per output channel. Icarus has only a shared learned scalar. | On hold |
| 8 | Fixed-one output scaling | `output_scale_mode: fixed_one`, removing the learned write scale while requiring zero-initialized output projections. | On hold |
| 9 | Routing diagnostics | Head overlap between iterations and selector statistics such as entropy, selection concentration, unused-head fraction, and top-k margin. | On hold |

Sources: [Longshot processor](../../src/gcw/models/cache_processors/longshot_processor.py), [Longshot invocation and external recurrence](../../src/gcw/models/cache_processors/longshot.py), [selector implementation](../../src/gcw/models/cache_processors/longshot_selection.py), [selector training](longshot-selector-training.md), and [update gates](longshot-update-gating.md).

## Diffusion

The repository labels diffusion experimental. These features should be evaluated together with whether diffusion itself belongs in Icarus.

| # | Feature Longshot has | Scope | Status |
|---|---|---|---|
| 10 | Task-directed diffusion refinement | Noisy latent refinement, conditioning on the original latent and time, shared refinement weights, and deterministic or stochastic inference. Icarus already supports Flow, but does not support diffusion. | On hold |
| 11 | Scheduled diffusion and independent solver depth | Separate noise-schedule horizon, training solver steps, and inference solver steps; perturbation/time-conditioning controls; evaluation-time depth overrides; optional contextual writeback gates. | On hold |
| 12 | Per-example diffusion update diagnostics | Measurements of latent changes, proposed KV and linear-state updates, direction changes between steps, and direction relative to the original cache, with JSONL output and plotting support. | On hold |

Sources: [diffusion documentation](../concepts/task-directed-diffusion.md), [diffusion implementation](../../src/gcw/models/cache_processors/diffusion.py), and [evaluation diagnostics](../../src/gcw/evaluation/update_metrics.py).

## Backbones and cache layouts

| # | Feature Longshot has | Applicability | Status |
|---|---|---|---|
| 13 | Qwen3.5 recurrent linear-state processing | Reads linear-attention state into the processor alongside KV heads, with `per_linear_state`, `shared`, and `global` query projections and a separate linear-head budget. Design needed: choose the temporal KV context used to query recurrent state, how each read enters Icarus's temporal attention, and its position/RoPE treatment, without compressing the KV history. Use magnitude selection with separate budgets; contextual ranking remains on hold under #4. Design jointly with #14. | Group 6: Linear state; Design needed |
| 14 | Linear-state writer alternatives | Outer-product updates or gated-delta updates using backbone dynamics, optional outer gating, and `live_rank1_linear_writer` initialization that preserves an initial zero update while allowing gradients. Design needed: decide which temporal processor outputs produce the recurrent-state update, consistently with #13. First implement the complete read/process/write path with the outer-product writer and live initialization; then add gated-delta writing and shared-backbone/FSDP integration. Contextual outer gates remain on hold under #7. | Group 6: Linear state; Design needed |
| 15 | Mixed KV shapes across layers | Shape-specific normalization, encoders, and projections around a shared processor. Icarus requires uniform stored KV shapes, so it cannot process Gemma4's mixed sliding/full-attention shapes together. Implement with #16 and #20, sharing layout and head-identity handling while preserving Icarus's temporal dimension. | Group 3: Cache layouts; Not started |
| 16 | Width-normalized head scores | `head_score_normalization: rms` makes magnitude scores comparable across different head widths. Primarily useful with #15. | Group 3: Cache layouts; Not started |
| 17 | Native Gemma4 static generation | The optimized `generate_fast` path supports mixed shapes and selected-layer subsets, retaining full selected archives and bounded ring buffers for unselected sliding layers. Gemma4 enables this path for TF and Longshot. Implement after the dynamic cache path is validated; compare static writes, ring-buffer behavior, and absolute positions against dynamic generation. | Group 5: Static generation; Not started |
| 18 | Gemma4 window-segment checkpointing | Checkpointing across recurrent windows with mixed-shape or selected-layer caches. Icarus already shares Qwen3 window checkpointing, but this Gemma4 integration is Longshot-specific. Build on group 3 and verify eager/checkpointed output and gradient parity across window boundaries. | Group 4: Window checkpointing; Not started |

Sources: [linear-state adapter](../../src/gcw/models/cache_processors/longshot_linear_state.py), [linear-state writeback](../../src/gcw/models/cache_processors/longshot.py), [cache-layout support](../concepts/cache-processing.md), [Gemma4 model](../../src/gcw/models/backbones/gemma4/model.py), and [checkpointing support](../guides/configure-training.md#native-activation-checkpointing).

## Memory and execution

These gaps include ordinary improvements and optional experiments. In particular, #20 and #22 are separate paths with different storage layouts and admission rules.

| # | Feature Longshot has | Status and qualification | Status |
|---|---|---|---|
| 19 | Bounded-memory magnitude reductions | Icarus now reuses Longshot's chunked FP32 reductions for per-position KV scores and flattened cache-head norms, preserving reduction order and existing gathering/writeback behavior. Each cast or square is capped at 32 MiB unless one complete reduction row exceeds the cap. [Icarus scoring tests](../../tests/test_icarus_scores.py) verify exact FP32/BF16 CPU selection, output, and gradient parity, ties/near ties, and bounded scoring allocations. The focused scoring/attention suite and existing Icarus regressions passed (75 tests total); CUDA validation was unavailable. | Group 1: Bounded scoring; Complete |
| 20 | Gemma4 native-list cache processing | Scores per-layer native cache views and gathers selected positions before packing features, avoiding full-history stacking. Distinct from the optional dense-cache optimization in #22. | Group 3: Cache layouts; Not started |
| 21 | Sequence-length-adaptive processor checkpointing | Enables processor-layer checkpointing only above a padded-sequence-length threshold, with runtime metrics and resume tracking. Icarus has the basic fixed checkpointing switch. Implement independently, reusing checkpoint policy machinery and preserving existing behavior when the threshold is unset. | Group 2: Adaptive checkpointing; Not started |
| 22 | Optional selected-history gather for dense caches | `performance.longshot_full_history` scores directly from cache storage and gathers only selected history, including bounded inverse-RoPE scoring. Default-off; production screening of the inverse-RoPE path documented slowdowns and gradient discrepancies, and did not admit it to the Qwen3 campaign. Port and qualify separately against the eager reference, retaining existing restrictions initially, including exclusion of linear-state and mixed-shape caches. Keep optional and mutually exclusive with #23. | Group 7: Selected-history optimization; Not started |
| 23 | Incrementally maintained KV-score sidecar | Reuses scores for unchanged cache positions and updates appended/rewritten positions. Default-off, restricted configurations, CPU validation only; no GPU throughput, memory, or BF16 hardware-parity claim. It still retains the full-history KV gather. Port and qualify separately, checking score refresh after writes, checkpoint replay, and resume. Retain existing restrictions initially, including exclusion of linear-state and mixed-shape caches, and mutual exclusion with #22. | Group 8: Score sidecar; Not started |

Sources: [bounded scoring](../../src/gcw/performance/longshot_scores.py), [native-list cache processing](../concepts/cache-processing.md), [adaptive checkpointing](../../src/gcw/performance/checkpointing.py), and [performance experiments and evidence](performance.md).

## Architectural difference

| # | Feature Longshot has | Why it needs separate consideration | Status |
|---|---|---|---|
| 24 | Time compression before head mixing | Linear flatten-and-project encoding, or query-attention encoding with a learned or final-token query. Icarus retains the time dimension through cross-head attention. Preserve this defining distinction: adopting Longshot's time compression would remove the architectural reason to use Icarus instead of Longshot. Features brought over must be adapted to Icarus's temporal representations. | N/A |

Sources: [variant definitions](../concepts/model-variants.md), [Longshot encoders](../../src/gcw/models/cache_processors/longshot_layers.py), and [Icarus implementation](../../src/gcw/models/cache_processors/icarus.py).

## Already present in Icarus

These are not missing-feature candidates:

- Per-cache-head projections.
- Value-only updates.
- Zero output initialization.
- Sigmoid/direct scalar gates.
- Inverse RoPE.
- Self-head KV inclusion.
- Flow.
- Basic processor checkpointing.
- Shared Qwen3 window-segment checkpointing and truncated BPTT.
- Separate generation cadence.

Sources: [Icarus implementation](../../src/gcw/models/cache_processors/icarus.py), [training configuration](../guides/configure-training.md), and [shared generation](../../src/gcw/models/backbones/transformer/generation.py).

## Configuration caveat

Shared configuration fields do not guarantee implementation support. In particular, Icarus currently does not consume the recurrence or alternative head-selection settings, even though those fields exist in `CacheProcessorConfig`. Review both execution and admission when porting a feature, so unsupported settings cannot be mistaken for working functionality.

Sources: [processor configuration](../../src/gcw/config/model.py), [architecture validation](../../src/gcw/architecture/validation.py), and [Icarus implementation](../../src/gcw/models/cache_processors/icarus.py).
