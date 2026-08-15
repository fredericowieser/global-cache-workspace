# Qualification state substrate

**Status:** Q1 implemented; private and not integrated with production training

`gcw.qualification.state` is the narrow shared representation used to capture and
compare complete training state in qualification work. Capture is an explicit
operation outside timed regions. It does not install hooks, own a mutable registry,
or change model, optimizer, gradient, module-mode, or random-number-generator
state. Production training does not import it.

## Q1 scope

A capture binds all registered module, parameter, and buffer slots to stable
qualified names. This includes `None` slots, persistent versus nonpersistent
buffers, mixed module train/eval modes, module `get_extra_state()` payloads, frozen
parameters, tied object aliases, storage aliases, tensor view metadata, and exact
cloned logical tensor bytes. Raw and post-clip gradients are separate
caller-supplied snapshots over the same canonical parameter alias inventory; one
is never inferred from the other. A populated gradient on a frozen parameter is
captured rather than suppressed, so accidental frozen-state mutation remains
observable.

The record also covers optimizer class/defaults, ordered groups and membership,
recursive optimizer state, Python/NumPy/Torch CPU RNG state, and either every CUDA
RNG state or an exact unavailable/omitted disposition. Real cursor, reservation,
and adaptive-plan dataclasses are represented directly rather than requiring
caller-side type-erasing conversion. Step fields, schedule values, reservation
data, and backend runtime data are included. Runtime is split into `semantic` and
`authority_local`; semantic comparison and hashing exclude the latter in the
substrate API itself.

Values use explicit tagged records. Mappings require string keys and are sorted.
Dataclasses, named tuples, and enums retain class identity and field/member
structure. Python floating-point and NumPy scalar values retain exact dtype and
bit patterns, including signed zero, infinities, and NaNs. Unsupported values,
cycles, foreign or ambiguously duplicated optimizer parameters, and sparse,
quantized, or meta tensors fail closed. Tensor content is copied explicitly to
CPU from logical elements, so capture does not read unused portions of a view's
storage. Serialization uses deterministic JSON metadata plus a separately framed
binary blob archive; tensors are not stringified. Deserialization revalidates
scalar types and invariants and rejects noncanonical archive encodings. Blob
digests are content-integrity checks, not authentication. Comparison reports
field paths and classifies structural, metadata, alias, byte, and ordinary value
differences.

## Threat boundary

This substrate detects accidental or adversarial modification of captured content
when the trusted comparison side or expected digest is already available. It is
not evidence authentication, signing, provenance attestation, process isolation,
or protection against an attacker who can replace both state and its expected
hash. Q1 archives are ordinary in-memory or byte records and are not sealed
artifacts. Capture also cannot make concurrent mutation safe; callers must capture
at a quiescent training boundary.

Q1 CUDA state support is representational and its tests are opt-in. Ordinary CPU
CI records CUDA as unavailable or explicitly omitted and runs no GPU workloads.

## Deferred milestones

Q2 remains responsible for sealed artifact envelopes, durable artifact policy,
and any trusted-key or external-authority design. Q3 remains responsible for
fresh-process qualification and orchestration. Graph and wave qualification
backends remain blocked until those later milestones define and validate their
execution boundary; Q1 deliberately does not port them.
