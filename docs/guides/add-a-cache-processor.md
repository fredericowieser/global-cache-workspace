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
