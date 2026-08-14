# Testing

**Status:** Maintained

Tests protect four layers:

1. Unit tests check tensor operations, validation, and pure decisions.
2. Integration tests check model execution, checkpoints, resume, and CLI paths.
3. Provenance tests check identity framing, parent closure, and fail-closed admission.
4. Compatibility tests check historical imports, configs, checkpoints, and evidence readers.

A processor refactor requires fixed-seed output, gradient, mutation, state-key, parameter, and FLOP parity. A distributed change requires timeout-bounded multiprocess tests. A docs change must rebuild generated pages and pass Sphinx with warnings as errors.

Loss-backend tests use a CPU fake to verify fused-call shape, options, vocabulary truncation, masking, all-ignored behavior, tied/untied gradients, and state neutrality. Real Liger BF16 parity is opt-in: `GCW_RUN_LIGER_CUDA_SMOKE=1 uv run --extra gpu --group dev pytest -m gpu tests/test_loss_backends.py`.
