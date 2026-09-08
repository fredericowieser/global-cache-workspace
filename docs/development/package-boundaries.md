# Package boundaries

**Status:** Maintained

## Distribution and research references

`src/gcw/` is the installed library. The wheel contains only this package,
excluding `src/gcw/spikes`. The source distribution uses the same source
boundary while retaining the deliberately selected `scripts/`, `examples/`,
`configs/`, `docs/`, `README.md`, `LICENSE`, `pyproject.toml`, and `uv.lock`.
Other siblings under `src/` are excluded even when populated locally.

The following submodules are retained as reference-only research repositories:

- `src/PhiID`: an information-decomposition reference.
- `src/integrated-info-decomp`: an information-decomposition reference.
- `src/bottleneck_llm`: a comparison/reference implementation.

These repositories are not GCW runtime dependencies or integrated GCW features.
They need not be initialized to install, build, or run GCW. The upstream paper
submodule, `src/Global-Cache-Workspace-Theory-ICLR-2027-`, is also retained and
excluded from distributions; packaging does not require access to it.

`tests/test_package_quality.py` checks actual sdist and normal wheel contents
using a temporary source tree populated with dummy submodule files and an
unlisted future sibling. These offline artifact tests require the existing
Hatchling build backend to be importable in the test environment and explicitly
skip otherwise; runtime environments need not include build tooling.

## Runtime domains

The package uses these domain directories:

- `models/backbones`: native causal language models;
- `models/cache_processors`: cache state and processing mechanisms;
- `data`: datasets, tokenization, and streams;
- `training`: objectives, optimizer orchestration, and exact resume;
- `performance`: explicit execution policies and optimized implementations;
- `evaluation` and `analysis`: independent scoring and rebuildable interpretation;
- `qualification`: private state, input, process, resource, and timing primitives used only by qualification tooling.

There is no `utils` package. Runtime I/O belongs to `runtime.py`. Identity belongs to `provenance.py`. Tensor cache operations belong to `cache_processors/ops.py`. Performance policy must not own model state or scientific objectives.

Training must not import evaluation or analysis. Evaluation must not mutate training state. Performance modules may consume narrow model primitives but must remain state-neutral and fail closed. Scripts must not contain reusable scientific logic.
Production training must not import `qualification`; state/input capture belongs outside timing at a quiescent boundary, and scripts own orchestration rather than reusable qualification semantics.
