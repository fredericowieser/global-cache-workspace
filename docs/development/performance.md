# Performance architecture

**Status:** Maintained

`gcw.performance` owns explicit execution policies and optimized implementations. Model and training domains retain scientific semantics, state ownership, optimizer rules, and checkpoint persistence.

The package is intentionally divided by responsibility:

- `config.py`: dependency-free typed performance policies;
- `adjacent_pairs.py`: deterministic reservation, planning, and collation;
- `prepared_bt.py`: forward-local vectorized BT parameter stacks;
- `checkpointing.py`: checkpoint admission, decisions, and telemetry;
- `losses.py`: explicit loss-backend admission, bounded reference-loss execution, and provenance;
- `longshot_scores.py`: bounded-memory reductions preserving eager FP32 reduction order;
- `longshot_cache.py`: state-neutral selected-history gathers that avoid flattening or copying full WindowedCache histories;
- `whole_model.py`: legacy whole-model compile policy;
- `telemetry.py`: read-only optimized-region metrics;
- `backends/qwen3.py`: Qwen3 backbone chunk compilation;
- `backends/cache_processors.py`: BT and deterministic BGT-family regional compilation;
- `backends/diffusion.py`: noisy Diffusion-BGT recurrence compilation after eager noise sampling;
- `backends/icarus.py`: deterministic Icarus post-selection regional compilation.

There is no backend registry. Every compiled region owns narrow admission checks, fixed compiler options, provenance, and fatal failure behavior. Requested optimized paths never silently retry eagerly.

Model modules retain cache selection, gather/writeback semantics, recurrent scheduling, and state dictionaries. `training/objectives.py` retains the adjacent-pair mathematical objective. The performance package may choose physical execution groups, but it cannot redefine the objective or optimizer step.

The rejected Icarus benchmark backend illustrates this boundary. Its eager caller computes FP32 scores, exact top-k token/cache-head selection, gather indices, latest write positions, and cache scatter; the candidate compiles only the deterministic RoPE/attention/transform region. External weak runtime ownership, production-layout warmup, checkpoint replay, exact cache events, RNG, and compiled-resume evidence passed. Formal H200 trajectory gates did not: adaptive 16/8/4/2 reached gradient relative L2 `5.81e-4` against the `5e-4` limit, and fixed batch 2 exceeded parameter tolerance (`atol=2e-5`, `rtol=2e-4`) after the first Adam update. Ordinary compile requests therefore fail before allocation. Reproduction requires the separate `GCW_ALLOW_REJECTED_ICARUS_COMPILE=1` benchmark override, whose provenance records `campaign_admitted: false`. Campaign Icarus remains eager.

## Adaptive adjacent-pair batching

Adaptive batching is default-off, training-only, and single-process in version 1. The source stream remains physical batch two. Each optimizer step reserves exactly 32 consecutive source pairs, preserving all 64 rows and the original cursor transition. The planner greedily chooses only contiguous groups from the configured 16/8/4/2-row bands. It never sorts, buckets, moves rows, or splits pairs.

For every physical group, training computes each original pair's combined-token mean and backpropagates the group's pair-loss sum divided by 32. There is one zero-gradient operation, clip, schedule update, and AdamW step for all groups. Evaluation remains on the fixed reference path.

The policy schema, ordered bands, ordering rule, weighting rule, and single-process restriction enter run metadata and the exact-resume fingerprint. Noisy Diffusion, nonzero processor/LoRA dropout, and distributed execution fail before reservation or model execution.

## Adaptive V2 admission blocker

There is no configurable or admissible adaptive batching V2. H200 evidence showed that ordinary contiguous grouping changes BT cache decisions at the first optimizer step: 258 of 690 semantic `select` events differed from the fixed batch-two eager oracle. Pair losses differed by 0.35–0.56% and gradient relative L2 by 3.5–6.0%. Row order, pair boundaries, and objective weighting are therefore insufficient parity conditions. Changing the physical batch shape can change BF16 kernels and rounding before discontinuous top-k cache selection.

The CUDA parity harness is the admission gate. Semantic cache selection/write events must be exactly equal; all existing pair-loss, gradient, parameter, optimizer-state, RNG, checkpoint-replay, and trajectory gates remain unchanged. Numerical tolerances cannot override an event mismatch. Until an H200 receipt passes those gates, `adaptive_batching_v2` is intentionally absent from `PerformanceConfig`, so attempted V2 configuration fails closed during parsing.

A future experiment may preserve the oracle only by retaining a physical shape of two for every model invocation. Candidate mechanisms include independently dispatched pair kernels or CUDA streams followed by a merge in source-pair order. Such an experiment must demonstrate that each pair uses the same kernels and algorithm choices as fixed eager execution, and that gradients are accumulated into each parameter in exactly the reference order. A batched/vmapped forward, concurrent backward into shared `.grad` tensors, unordered reductions, padding to a common width, and grouped GEMMs with changed algorithms are not admissible assumptions. Concurrency also requires proof that model state, compiled runners, checkpoint recomputation, allocator lifetimes, and RNG are stream-safe. Sequential batch-two execution is the only currently established exact design, and it offers no batching speedup.

## Bounded reference-loss execution

The default `performance.reference_loss.mode: legacy` preserves the existing 256-position CUDA projection chunks and does not enter the training fingerprint. Opt-in `mode: bounded` still projects every prompt, answer, and padding position through the full vocabulary and still uses the `reference` loss backend. It changes only the physical subchunk width by limiting `batch_rows * projected_positions`; losses are reassembled into the legacy logical chunks before FP32 row reduction.

`checkpoint_chunks: true` may recompute individual projection/cross-entropy chunks during backward to reduce retained activation memory. The policy adds no parameters or buffers. Its budget must cover the maximum physical row count, so the projected-position limit remains a hard bound. The budget, checkpoint choice, implementation identity, Torch version, and derived chunk rule are authenticated in run metadata and exact-resume fingerprints. This first implementation fails closed with distributed execution, a non-reference loss backend, or legacy whole-model compilation.

Evaluation and configurations that omit the policy remain on the legacy reference path.

## Compiled Qwen3 full windows

`backbone_chunk_backend: compile` retains the version-1 behavior: the initial and partial windows stay eager, while complete 64-token windows with past cache use one dynamic full-graph region. `compile_full_windows` is a separate version-2 policy. It adds one static full-graph region for the initial complete window and retains the existing dynamic region for later complete windows.

The initial region accepts tensors only, closes over the frozen Qwen3 layers without registering state, preserves Flash SDPA with `is_causal=True`, and returns hidden states plus stacked K/V and scores. Its requested compiler failure is fatal. Partial windows remain eager. The backend selection, both region identities, compiler options, causal boundaries, and Torch version enter provenance and the exact-resume fingerprint. Version-1 payloads and fingerprints remain unchanged.

The CUDA smoke is opt-in because ordinary CPU CI must not initialize Inductor CUDA state.

Adaptive 16/8/4/2 execution has four expected initial-window specializations. The first observed shape creates the initial graph; the other three batch shapes cause three shape recompiles. The later-window region keeps past-cache length dynamic, but the current compiler still creates one specialization for each physical batch shape. It therefore also has four expected graphs and three batch-shape recompiles. Acceptance tests exercise all four row counts through one model-local runner, vary past-cache lengths within a batch specialization, compare each result with eager execution, and record graph and recompile counts. A changed count is a performance regression even when tensor parity passes.

Compile acceptance telemetry is opt-in and state-neutral. It records observed region signatures and the first synchronized call cost for each shape. That cold cost includes lazy compilation and must be reported separately from steady-state step timing. Runtime shape observations do not estimate Dynamo graph counts; acceptance canaries attach explicit compiler counters. The telemetry object is not a parameter or buffer and does not enter version-1 fingerprints.

Segmented-checkpoint acceptance covers both `compile` and `compile_full_windows`. Backward must preserve processor gradients and allow PyTorch's non-reentrant `_StopRecomputationError` control signal to escape the fail-closed wrapper unchanged.

The CUDA smoke requires strict Flash SDPA and records profiler event names as backend evidence:

```bash
GCW_STRICT_FLASH=1 GCW_RUN_QWEN3_INITIAL_COMPILE_CUDA_SMOKE=1 \
  uv run --extra gpu --group dev pytest -q \
  tests/test_qwen3_backbone_compile.py::test_compiled_initial_full_window_cuda_forward_backward_smoke
```

Running that command on a live H200 (or another BF16 Flash-capable CUDA host) remains the hardware acceptance gate; CPU CI validates only the hook and skips execution.

The regional compilers are **not** admitted for the Qwen3 math campaign. Formal H200 fixed-batch-two trajectories changed discontinuous cache decisions despite preserving physical shape: BT changed 222–241 of roughly 680 selections per step, BGT changed 228–251, and Flow-BGT changed 241–258. Gradient relative L2 reached 6.8%, 8.1%, and 17.3%, respectively. `compile_full_windows` also failed. Throughput from those backends is screening evidence only; unchanged RNG and optimizer-step counters cannot override selection/write, pair-loss, gradient, parameter, and Adam-state failures.

## Longshot selected-history gather

Longshot KV-norm full-history optimization is explicit and default-off:

```yaml
performance:
  longshot_full_history:
    enabled: true
    max_temp_bytes: 33554432
```

The eager layer/head flatten, full-history gather, per-row FP32 scoring expression, and existing top-k are the semantic oracle. The opt-in path computes the same FP32 score rows from contiguous layer-major `WindowedCache` tensors, then gathers only selected positions. Its chunk budget bounds FP32 scoring temporaries. Runtime counters are explicitly named as logical tensor payloads and estimated FP32 working-set bounds; they are not allocator measurements. Measured CUDA peak allocation is the authoritative memory acceptance metric.

The policy, budget, implementation, checkpoint and recurrence contracts, and admission restrictions are authenticated in run metadata and the exact-resume fingerprint. Disabled configurations are omitted from serialized performance payloads, composition receipts, run metadata, and training fingerprints, preserving pre-feature identities.

This first admission supports single-process dense SmolLM/Llama/Qwen3 Longshot training with zero processor dropout, KV-norm scoring, native non-reentrant Longshot layer checkpointing, segmented recomputation, and exact internal or external reselection. Telemetry is recorded only by the original forward and is suppressed during processor-layer and segmented backward recomputation. The optimized policy rejects `unrope_keys: true`: rotating the entire history before chunked scoring would violate the bounded-memory contract, and no chunk-local exact un-RoPE kernel is admitted yet. It also rejects unsupported distributed, compiled, Flow/refinement, linear-state, internal non-KV recurrence, cache type, layout, and shape configurations rather than falling back eagerly.

Real-CUDA BF16 cache-write, selected-index, checkpointed-gradient, per-pair trajectory, resume, and peak-memory evidence is opt-in:

```bash
GCW_RUN_LONGSHOT_CUDA_PARITY=1 pytest -q tests/test_longshot_full_history_cuda.py
```
## Noisy Diffusion-BGT recurrence compilation

Noisy Diffusion-BGT remains eager unless training explicitly requests:

```yaml
performance:
  noisy_diffusion:
    recurrence_backend: compile
```

The model samples `torch.randn_like` eagerly at the existing boundary before dispatch. The compiled full-graph region receives that concrete noisy state and runs only the task-directed recurrence and field evaluations. Thus compilation cannot capture, move, repeat, or fuse the random draw. Runner state lives in an external weak module-identity registry and the compiled function receives the owning diffusion module and field as explicit inputs. Deep copies and serialized models therefore cannot retain stale parameter or field bindings. Each weak execution state also owns a lock that serializes lazy runner creation and shape specialization for that module. Unrelated diffusion modules retain independent locks. The policy preserves the configured recurrence count, time conditioning, cache update equations, state dictionary, and writeback path.

This backend is independent from `GCW_CACHE_PROCESSOR_BACKEND`. The latter continues to reject noisy Diffusion-BGT. The narrow backend requires a noisy BGT model, zero cache-processor attention dropout, and single-process training. It rejects adaptive grouping, records its noise boundary and compiler options in provenance, and enters the exact-resume fingerprint only when enabled. PyTorch non-reentrant checkpoint replay's private early-stop signal passes through unchanged; every other compiler creation or runtime failure is fatal and never retries eagerly.

Run the opt-in CUDA smoke after changing this boundary:

```bash
GCW_RUN_CUDA_DIFFUSION_COMPILE_TESTS=1 \
  uv run --extra gpu pytest -q tests/test_noisy_diffusion_compile_backend.py
```

The smoke covers production field closures, concurrent first dispatch, copy and serialization isolation, refinement and field gradients, repeated noise/RNG events, multi-step eager/compiled AdamW trajectories, uninterrupted/resumed optimizer slots, and alternating shapes. Runner creation and dispatch counts are module-local. PyTorch exposes Dynamo counters process-wide, so their absolute values and since-materialization deltas use the `gcw.noisy_diffusion_compile_telemetry.v2` diagnostic schema and are explicitly excluded from acceptance decisions. Scoped compiler backends or profiler traces provide acceptance evidence instead.

**[PERF INCREASE][PROCESSOR][PRELIMINARY]** On commit `6abe117`, fixed eager batch-2 row-0 screening reduced the warm median from `18.0686` to `17.2364` seconds/step on H200 (`1.048x`) and from `13.2356` to `12.3699` seconds/step on H100 (`1.070x`). Peak allocated memory was unchanged at approximately `31,778` MiB. Both real-Inductor CUDA smoke suites passed (`19/19` on each hardware lineage), but the production BF16 trajectories are not campaign-admitted: the final gradient norm changed by `+2.74%` on H200 and `+2.70%` on H100. The H200 final loss changed from `0.74052975` to `0.74048124`; the H100 final loss changed from `0.74041273` to `0.74025877`. Eager recurrence remains authoritative pending formal production-shape raw-gradient, cache-event, optimizer-state, RNG, and resume parity.


## Change control

Moving code into this package must not change parameter names, buffers, checkpoint payloads, selected indices, or eager defaults. Changes to compiler regions, batching, checkpointing, or loss execution require targeted parity tests and CUDA acceptance canaries. Handwritten kernels remain separately approval-gated.
