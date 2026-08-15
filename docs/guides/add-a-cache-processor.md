# Add a cache processor

**Status:** Maintained

1. Define the scientific intervention: source state, selection rule, learned transition, and writeback destination.
2. Implement `gcw.models.cache_processors.base.CacheProcessorBase`.
3. Keep selection and writeback visible. Shared tensor operations belong in `cache_processors/ops.py`.
4. Declare required cache domains and capabilities in `gcw.architecture`.
5. Add explicit construction to `build_cache_processor`; state remains under `LanguageModel.cache_processor`.
6. Test identity initialization, mutation boundaries, gradients, recurrence events, parameter counts, and runtime FLOPs.
7. Add matched controls before claiming that the mechanism improves a model.

Do not add a generic processor registry. Constructor dispatch remains explicit because each processor has different state and scheduling semantics.

## Forward-local preparation

`LanguageModel.forward` asks a processor for an execution context. The default context calls the processor eagerly. BT overrides it because its layer-parallel kernel needs stacked views of every per-layer weight, norm, bias, and gate. An ordinary forward builds those differentiable stacks lazily, reuses them across cache events, then releases them on return. Segmented activation checkpointing narrows ownership further: each original segment invocation and each backward replay opens and closes an independent context.

Prepared tensors are execution data, not model state. Keep source parameters in their existing modules, never register prepared stacks as parameters or buffers, and never cache them on the processor. Closing an owner is final. Checkpoint replay must create a new owner rather than weakening the closed guard or retaining a forward owner in an autograd closure. Direct processor calls and generation use the eager implementation as the reference fallback. BGT, Icarus, and Longshot retain their existing paths because they do not perform BT's repeated per-layer stacking.

## BT regional compilation

BT uses eager execution by default. H200 canaries may opt into `GCW_BT_PROCESSOR_BACKEND=compile`. This compiles only `run_prepared_batched_cache_processors` with Inductor, `fullgraph=True`, and static shape specialization. Selected lengths such as 64 and 96 therefore receive separate compiled variants. Selection, cache gathering and writeback, language-model recurrence, and the variable-length backbone stay eager.

The compiled backend is fail-closed. An unavailable compiler, disabled Dynamo errors, graph break, or backend/runtime failure raises instead of falling back to eager. `GCW_COMPILE` remains the whole-model switch for other architectures and is rejected for BT training; do not combine it with the BT regional backend. Run metadata records the effective backend, compile options, Torch version, region name, and fingerprint.

The compiled callable owns no model parameters or buffers. Prepared parameter stacks remain execution-local and are released with their context. Checkpoints therefore retain the eager model's state-dict keys and schema. Generation and direct processor calls remain eager. Regional compilation composes with segmented window checkpointing because replay prepares a fresh stack before entering the compiled pure-tensor region.

## BGT regional compilation

BGT, Flow-BGT, and noiseless Diffusion-BGT use eager execution by default. Set `GCW_CACHE_PROCESSOR_BACKEND=compile` to compile only the gathered tensor transform with Inductor, `fullgraph=True`, and static shape specialization. The selector, gather, split, scatter, cache recurrence, and backbone stay eager. Selected lengths 64 and 96 and different batch sizes specialize inside cached compiled callables; no prepared or graph-derived parameter tensors are retained.

The backend fails closed. Compiler creation, graph breaks, and runtime failures raise without eager fallback. Do not combine it with whole-model `GCW_COMPILE`. `GCW_CACHE_PROCESSOR_BACKEND=compile` rejects BT because BT owns `GCW_BT_PROCESSOR_BACKEND`, and rejects Icarus and Longshot because they own different execution boundaries. This is explicit architecture dispatch, not a generic backend registry.

Diffusion-BGT compilation currently requires `noise_scale: 0`. Fullgraph noise equivalence has not been established, so noisy training or stochastic inference is rejected before model construction and before RNG consumption. The eager Diffusion path remains unchanged; the backend never moves, suppresses, or substitutes its random draw.

The regional callable receives the existing `CacheProcessor` module and gathered tensors. Parameters and state-dict keys remain on that module; compiled callables live outside the module tree. Package and checkpoint schemas therefore remain unchanged. Run metadata records architecture, effective backend, regions, compile options, Torch version, noise policy, and a stable fingerprint. Exact resume includes that fingerprint.

Regional BGT compilation composes with `standard_processor_layers` and segmented window activation checkpointing. Processor-layer backward recomputation re-enters the same compiled tensor boundary. Segment replay preserves RNG and calls the eager selector/gather/writeback around that boundary. Changing either checkpointing policy or the regional backend changes the exact-resume fingerprint.

## Icarus post-selection compilation

Ordinary deterministic Icarus remains eager by default. Set `GCW_ICARUS_PROCESSOR_BACKEND=compile` to compile only the pure tensor transform after FP32 cache scores, exact top-k token indices, and cache-head indices have been selected. RoPE, selected-head attention, processor projections, MLP transforms, and the gated residual update enter one dynamic-shape full-graph Inductor region. Cache-score reduction, selection, gathering, latest-position choice, key re-rotation, and K/V writeback remain eager and exact.

Compilation is activated only by the admitted training loop around one forward and its backward. Direct processor calls, evaluation, and both generation paths remain eager even when the environment variable requests compilation. Each allocated processor module owns its compiled callable; there is no process-global runner or cross-model specialization cache. Startup preflight admits the physical training row counts, then synthetic zero-valued forward/backward warmup allocates their kernels without advancing the data stream, changing parameter gradients, or consuming RNG state. Dynamic batches cover the CUDA-qualified adaptive 16/8/4/2 row bands.

This backend is intentionally narrow. It accepts only `variant: icarus`, `processor_mode: standard`, zero attention dropout, and no Flow or Diffusion refinement. DDP, FSDP, non-CUDA adaptive execution, and custom adaptive row bands are rejected before model execution. Compiler unavailability fails before allocation; warmup and runtime graph failures include shape/device diagnostics and are fatal without eager fallback. PyTorch checkpoint's private early-stop exception passes through unchanged so non-reentrant backward replay retains its control flow. The module-owned callable adds no parameters or buffers, so state-dict keys stay unchanged. Run metadata and exact-resume fingerprints record this scope and policy. Run `GCW_RUN_CUDA_COMPILE_ACCEPTANCE=1 pytest -m gpu tests/test_icarus_compile_backend.py tests/test_icarus_compile_cuda_acceptance.py` on CUDA to exercise real Inductor checkpoint, gradient, adaptive-shape, optimizer-trajectory, and process-restart parity.
