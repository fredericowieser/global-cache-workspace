# Package boundaries

**Status:** Maintained

The package uses these domain directories:

- `models/backbones`: native causal language models;
- `models/cache_processors`: cache state and processing mechanisms;
- `data`: datasets, tokenization, and streams;
- `training`: objectives, optimizer orchestration, and exact resume;
- `performance`: explicit execution policies and optimized implementations;
- `evaluation` and `analysis`: independent scoring and rebuildable interpretation.
- `qualification`: private, state-neutral capture and comparison used only by qualification tooling.

There is no `utils` package. Runtime I/O belongs to `runtime.py`. Identity belongs to `provenance.py`. Tensor cache operations belong to `cache_processors/ops.py`. Performance policy must not own model state or scientific objectives.

Training must not import evaluation or analysis. Evaluation must not mutate training state. Performance modules may consume narrow model primitives but must remain state-neutral and fail closed. Scripts must not contain reusable scientific logic.
Production training must not import `qualification`; capture belongs outside timing at a quiescent boundary.
