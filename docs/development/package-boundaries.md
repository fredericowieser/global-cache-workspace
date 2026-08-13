# Package boundaries

**Status:** Maintained

The package uses five domain directories:

- `models/backbones`: native causal language models;
- `models/cache_processors`: cache state and processing mechanisms;
- `data`: datasets, tokenization, and streams;
- `training`: optimization and exact resume;
- `evaluation` and `analysis`: independent scoring and rebuildable interpretation.

There is no `utils` package. Runtime I/O belongs to `runtime.py`. Identity belongs to `provenance.py`. Tensor cache operations belong to `cache_processors/ops.py`.

Training must not import evaluation or analysis. Evaluation must not mutate training state. Scripts must not contain reusable scientific logic.
