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
- `whole_model.py`: legacy whole-model compile policy;
- `telemetry.py`: read-only optimized-region metrics;
- `backends/qwen3.py`: Qwen3 backbone chunk compilation;
- `backends/cache_processors.py`: BT and deterministic BGT-family regional compilation;
- `backends/icarus.py`: deterministic Icarus post-selection regional compilation.

There is no backend registry. Every compiled region owns narrow admission checks, fixed compiler options, provenance, and fatal failure behavior. Requested optimized paths never silently retry eagerly.

Model modules retain cache selection, gather/writeback semantics, recurrent scheduling, and state dictionaries. `training/objectives.py` retains the adjacent-pair mathematical objective. The performance package may choose physical execution groups, but it cannot redefine the objective or optimizer step.

The Icarus backend illustrates this boundary. Its eager caller computes FP32 scores, exact top-k token/cache-head selection, gather indices, latest write positions, and cache scatter. An allocation-preflighted processor module owns a dynamic compiled callable that receives only selected tensors and executes the deterministic RoPE/attention/transform region. The callable is active only inside an admitted training forward/backward scope; direct calls, evaluation, and generation remain eager. Synthetic warmup covers admitted row counts without data or RNG consumption. Admission rejects distributed execution, unqualified adaptive shapes, stochastic dropout, and Flow/Diffusion refinements.

## Adaptive adjacent-pair batching

Adaptive batching is default-off, training-only, and single-process in version 1. The source stream remains physical batch two. Each optimizer step reserves exactly 32 consecutive source pairs, preserving all 64 rows and the original cursor transition. The planner greedily chooses only contiguous groups from the configured 16/8/4/2-row bands. It never sorts, buckets, moves rows, or splits pairs.

For every physical group, training computes each original pair's combined-token mean and backpropagates the group's pair-loss sum divided by 32. There is one zero-gradient operation, clip, schedule update, and AdamW step for all groups. Evaluation remains on the fixed reference path.

The policy schema, ordered bands, ordering rule, weighting rule, and single-process restriction enter run metadata and the exact-resume fingerprint. Noisy Diffusion, nonzero processor/LoRA dropout, and distributed execution fail before reservation or model execution.

## Bounded reference-loss execution

The default `performance.reference_loss.mode: legacy` preserves the existing 256-position CUDA projection chunks and does not enter the training fingerprint. Opt-in `mode: bounded` still projects every prompt, answer, and padding position through the full vocabulary and still uses the `reference` loss backend. It changes only the physical subchunk width by limiting `batch_rows * projected_positions`; losses are reassembled into the legacy logical chunks before FP32 row reduction.

`checkpoint_chunks: true` may recompute individual projection/cross-entropy chunks during backward to reduce retained activation memory. The policy adds no parameters or buffers. Its budget must cover the maximum physical row count, so the projected-position limit remains a hard bound. The budget, checkpoint choice, implementation identity, Torch version, and derived chunk rule are authenticated in run metadata and exact-resume fingerprints. This first implementation fails closed with distributed execution, a non-reference loss backend, or legacy whole-model compilation.

Evaluation and configurations that omit the policy remain on the legacy reference path.

## Change control

Moving code into this package must not change parameter names, buffers, checkpoint payloads, selected indices, or eager defaults. Changes to compiler regions, batching, checkpointing, or loss execution require targeted parity tests and CUDA acceptance canaries. Handwritten kernels remain separately approval-gated.
