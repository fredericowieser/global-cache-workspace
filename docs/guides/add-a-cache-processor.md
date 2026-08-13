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
