# Glossary

**Backbone**
: The causal language model that owns embeddings, transformer layers, native cache construction, loss, and generation.

**Cache processor**
: A learned module that selects, transforms, and writes explicit model cache state.

**Workspace**
: Persistent shared processor state. A workspace is not a synonym for every cache or hidden tensor.

**Recurrence**
: Repeated execution of shared processor parameters across deliberate steps.

**Flow**
: Continuous latent refinement by a learned vector field and explicit numerical solver.

**Writeback**
: The operation that maps a processor update to declared cache destinations.

**Experiment identity**
: The stable identity of scientific inputs and protocol.

**Execution ID**
: The identity of one concrete run attempt.

**Evidence admission**
: Review that freezes a result and its complete provenance closure as immutable scientific evidence.
