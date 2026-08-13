# Testing

**Status:** Maintained

Tests protect four layers:

1. Unit tests check tensor operations, validation, and pure decisions.
2. Integration tests check model execution, checkpoints, resume, and CLI paths.
3. Provenance tests check identity framing, parent closure, and fail-closed admission.
4. Compatibility tests check historical imports, configs, checkpoints, and evidence readers.

A processor refactor requires fixed-seed output, gradient, mutation, state-key, parameter, and FLOP parity. A distributed change requires timeout-bounded multiprocess tests. A docs change must rebuild generated pages and pass Sphinx with warnings as errors.
