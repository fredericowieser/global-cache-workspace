# Qualification state substrate

**Status:** Q1 state plus initial Q2–Q4 input/process/resource primitives implemented; private and not integrated with production training

`gcw.qualification` contains narrow shared primitives for complete state, captured
inputs, fresh-process orchestration, resource identity, and uninstrumented timing.
All capture is explicit and outside timed regions. The package does not install
hooks, own a mutable model registry, or change model, optimizer, gradient,
module-mode, or random-number-generator state. Production training does not import
it.

## Q1 scope

A capture binds the prefix-closed, depth-first module inventory and every registered
parameter and buffer slot to stable qualified names. Local registration order is
canonicalized, and impossible module/parameter/buffer namespace overlaps fail
closed. The inventory includes `None` slots, persistent versus nonpersistent
buffers, mixed module train/eval modes, module `get_extra_state()` payloads, frozen
parameters, tied object aliases, storage aliases, storage extents, tensor view
metadata, and exact cloned logical tensor bytes. Raw and post-clip gradients are
separate caller-supplied snapshots over the same canonical parameter alias inventory;
one is never inferred from the other. A populated gradient on a frozen parameter is
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

Values use explicit tagged records. Mappings require nonempty string keys and are
sorted. Dataclasses, named tuples, and enums retain class identity and canonical
field/member structure. One encoding context spans module extra state, optimizer
groups/state/defaults, RNG data, cursors, schedules, reservations, and runtime
state. It assigns deterministic object and storage identities, retains strong
references while capturing to prevent Python identity reuse, preserves repeated
container/tensor/NumPy-array aliases, and rejects cycles or inconsistent repeated
objects. Tensor and NumPy views include storage size, offset, shape, and stride;
shared-storage records must have compatible storage metadata.

Python floating-point and NumPy scalar values retain exact dtype and bit patterns,
including signed zero, infinities, and NaNs. Unsupported values, foreign or
ambiguously duplicated optimizer parameters, and sparse, quantized, or meta tensors
fail closed. Tensor content is copied explicitly to CPU from logical elements, so
capture does not read unused portions of a view's storage. Serialization uses
deterministic JSON metadata plus a separately framed binary blob archive; tensors
are not stringified. Deserialization revalidates scalar types, alias and inventory
invariants, canonical field order, storage bounds, and the complete reserialized
archive, rejecting reordered inventories, alias references, or blob permutations
that could otherwise encode the same represented state more than once. Blob digests
are content-integrity checks, not authentication. Comparison reports field paths and
classifies structural, metadata, alias, byte, and ordinary value differences.

## Q2 single-read inputs and integrity

`inputs.py` captures regular files through one descriptor, verifies their identity
did not change during the read, and retains the authoritative bytes in memory.
Configuration snapshots bind both the exact authored YAML and canonical resolved
`TrainConfig`. Source snapshots use a sorted closed member set, reject traversal or
symlink components, and materialize only captured bytes into an exclusive directory.
Callers must use the captured bytes rather than reopen the original path.

`integrity.py` provides strict canonical JSON envelopes, exclusive atomic writes,
and SHA-256 consistency checks. The API deliberately says integrity rather than
authentication. An unkeyed digest becomes authoritative only when its expected value
is independently anchored.

## Q3 fresh-process boundary

`process.py` launches workers without a shell, with explicit arguments,
environment, working directory, timeout, and exclusive output. The producer must
exit completely before a distinct resumer process starts. The parent captures each
worker output once and records its identity. Model-specific producer/resumer workers
must still use production checkpoint save/open/restore and Q1 state comparison; the
generic process primitive does not claim that merely spawning two processes proves
resume parity.

## Q4 device and timing primitives

`resources.py` separates semantic accelerator identity from authority-local PID,
host, Slurm, CUDA visibility, UUID, PCI, and driver fields. Its closed GPU admission
recognizes H100 and H200 separately and can require exactly one visible CUDA device
and complete UUID/PCI authority. `timing.py` rejects instrumented execution, excludes
warmup, synchronizes complete operation boundaries, retains every raw sample, and
reports allocator peaks. H100 and H200 remain distinct lineages.

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

Trusted-key authentication, external authority, and OS-enforced immutable execution
remain deferred; read-only modes and unkeyed digests are not substitutes. The generic
fresh-process primitive still needs model-specific production checkpoint workers and
full Q1 comparison before it can qualify graph or wave execution. Private-pool,
stream/event, process-lifetime, and parent-observed GPU release evidence must be
attached to each executor. CUDA Graph and pair-wave candidates therefore remain
benchmark-only until those complete hardware protocols pass.
