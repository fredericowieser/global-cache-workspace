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

`LanguageModel.forward` asks a processor for a forward execution context. The default context calls the processor eagerly. BT overrides it because its layer-parallel kernel needs stacked views of every per-layer weight, norm, bias, and gate. BT builds those differentiable stacks lazily at the first cache event, reuses them for later events in the same forward, then releases them when the forward returns.

Prepared tensors are execution data, not model state. Keep source parameters in their existing modules, never register prepared stacks as parameters or buffers, and never cache them on the processor. A prepared tensor belongs to one autograd graph; carrying it into another forward can produce stale values or a second-backward error. Direct processor calls and generation use the eager implementation as the reference fallback. BGT, Icarus, and Longshot retain their existing paths because they do not perform BT's repeated per-layer stacking.
