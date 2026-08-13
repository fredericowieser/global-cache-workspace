# Internal architecture

**Status:** Maintained

GCW separates reusable code, human-authored configuration, mutable execution state, provisional results, and admitted evidence.

```text
configuration -> capability admission -> construction
             -> training or evaluation -> local run record
             -> provisional result -> reviewed evidence
```

`gcw.architecture` resolves model and experiment compatibility without model allocation, downloads, or writes. `gcw.models` owns numerical execution. `gcw.training` owns optimization and resume. `gcw.evaluation` owns independent scoring. `gcw.provenance` owns canonical identity. Scripts remain operational wrappers.

Mechanisms stay explicit: select, encode, transition, optional recurrence or Flow, decode, and writeback.
