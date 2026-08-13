# Add a backbone

**Status:** Maintained

1. Implement the `gcw.models.backbones.base.Backbone` contract under `src/gcw/models/backbones/`.
2. Preserve explicit cache domains. Do not coerce recurrent or shared state into dense KV tensors.
3. Add the family to typed config validation.
4. Add a side-effect-free capability profile in `gcw.architecture`.
5. Extend `build_backbone` only after admission can reject unsupported compositions.
6. Add fixed-seed forward, generation, loss, gradient, state-key, and checkpoint tests.
7. Document cache layout, positional frame, dtypes, supported kernels, and unsupported generation paths.

A backbone profile certifies behavior for a concrete architecture and version. Similar tensor shapes do not prove cache compatibility.
