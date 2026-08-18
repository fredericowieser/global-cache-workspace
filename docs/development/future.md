# Future work: repository fat-cut and refactor

**Status:** In progress — PR 1 landed.

> **PR 1 note (landed 2026-08):** the dead-source deletions below are done. Two
> corrections to the original plan, discovered at deletion time:
> - `qualification/state.py` had already been split into the `qualification/state/`
>   package; the whole package was removed.
> - `qualification/inputs.py`, `integrity.py`, and `resources.py` are **live** —
>   they are imported by `scripts/benchmarks/harness.py` and
>   `scripts/run_math_training_lane.py` (and, via `inputs`, by `receipts.py`), so
>   they were **kept**. Only `state/`, `process.py`, and `timing.py` were dead.
> Net reduction: `src/gcw` 26,745 → 23,675 lines, `tests` 20,256 → 19,142.

The repository currently holds ~25k lines of Python under `src/gcw/`, ~20k in
`tests/`, and ~7.9k in `scripts/`, plus hundreds of historical experiment run
records committed to Git. The scientific core (cache processors, backbones,
compile/Triton backends, exact resume) is compact and healthy; the weight comes
from verified-unreachable QA/campaign scaffolding, one-off campaign scripts,
repo-hygiene debt, intentional-duplication surfaces, and a few oversized
monoliths. This page records the proposed trimming plan for future PRs.

Every "dead" claim below was verified by grepping live importers across
`src/` and `scripts/` (excluding tests) at the time this page was written.
Re-verify before deleting: code moves.

## Guiding rule

Cut only what is unreachable, unmaintained, or duplicated. The implemented
model variants (TF/BT/BGT/Icarus/Longshot/Flow), the performance backends, the
benchmarks that produce the paper's negative results, and the exact-resume
machinery are the contribution. None of the items below removes a feature.

## PR 1 — verified-dead source deletions (no behavior change)  ✅ landed

These modules have no live importer; each is referenced only by its own test
file (which is deleted in the same PR):

- ~~`src/gcw/qualification/state.py`~~ `src/gcw/qualification/state/` (package) —
  training-state capture/digest/comparison framework. Only importer was
  `tests/test_qualification_state.py` and the package `__init__` re-export.
  Largest single dead region.
- `src/gcw/qualification/process.py` — only `tests/test_qualification_process.py`.
- `src/gcw/qualification/timing.py` — only `tests/test_qualification_timing.py`.
- `src/gcw/analysis/` (whole package) — only import was its own `__init__.py`.
- `src/gcw/performance/ragged.py` — only `tests/test_ragged_pair_plans.py`.
- `src/gcw/performance/workspaces.py` — only `tests/test_static_workspaces.py`.
- `src/gcw/performance/activation_lifetime.py` — only `tests/test_activation_lifetime.py`.

> **Kept, not dead** (contrary to the original plan):
> `qualification/inputs.py`, `integrity.py`, `resources.py`, and `receipts.py` are
> the live receipt-bound admission chain (`execution_plan.py` → `receipts` →
> `inputs`; scripts → `integrity`/`resources`).

Dead functions (defined, never called): `config.load_yaml`,
`backbones/transformer.cosine_lr`, `runtime.setup_logging`, `models.lora_parameters`,
`evaluation/scoring.numeric_answers_match`, `backbones/qwen3.qwen3_config`,
`cache_processors/state.append_scores`, `qualification/state.require_equal`.

Verification gates after deletion: `uv run ruff check src`, `uv run pytest -q`.

## PR 2 — retire one-off scripts and test harnesses

- `scripts/analysis/diagnose_hf_conversion.py` (751) — debug diagnostic for a
  past failed conversion; zero references. Keep the verifier core
  (`verify_hf_conversion.py`), which is the pipeline gate.
- `scripts/prepare_openr1_bgt_mixture.py` (409) — unused mixture-builder
  variant; adopted protocol is `prepare_math_experiment.py`.
- `scripts/analysis/bench_cache_processor.py` (401) — orphan microbench,
  superseded by the documented `scripts/benchmarks/harness.py`.
- `scripts/run_math_experiment_eval.py` (44), `run_math_experiment_lr_screen.py`
  (81), `select_math_experiment_lrs.py` (49), `build_math_experiment_launch.py`
  (66) — historical campaign glue; rates are baked into generated configs.
- Test subprocess harnesses that only exist to exercise removed modules:
  `tests/qualification_process_worker.py`, and the resume-worker twins
  (`icarus_compile_restart_worker.py`, `kv_score_sidecar_resume_worker.py`)
  only if their coverage is folded elsewhere first.

Keep `scripts/benchmarks/` and the math-experiment lane suite intact — they
are the documented rejection evidence and paper pipeline.

## PR 3 — repo hygiene: stop committing run records

`data/training-runs/*` (385 records) and `data/eval-runs/*` (41) are tracked
in Git despite `.gitignore` rules (`data/training-runs/*`, `!.../.gitkeep`).
They are ~2.3 MB of historical evidence that should live in an archive branch,
not the paper repo.

- `git rm --cached` the 426 records; keep the `.gitkeep` stubs.
- Optionally move the records to a `data-archive` branch before deletion so
  provenance history is not lost.

## PR 4 — consolidation (intentional duplication stays)

Do **not** merge `performance/backends/*`, `performance/triton/*`, and
`models/cache_processors/*`. They are three orthogonal strategies (torch.compile
regional backends, handwritten Triton kernels, eager reference) and the
adaptive-parity negative-result narrative depends on their coexistence.

Real duplication worth merging:

- Answer-token cross-entropy implemented at `models/backbones/transformer/tf.py`
  (`answer_only_loss_from_hidden`, `loss_from_hidden_reference`),
  `models/backbones/transformer/losses.py`, and `training/objectives.py`.
- Head-selection top-k triplicated:
  `cache_processors/longshot_layers.py:_select_heads_from_head_scores`,
  `cache_processors/icarus.py:_select_heads_from_merged`,
  `cache_processors/ops.py:select_heads_from_per_cache_head_scores`.
- `_topk_linear_heads` duplicated (`cache_processors/longshot_linear_state.py`
  and `flops.py:_longshot_topk_linear_heads`).
- KV-score sidecar reimplemented both in `performance/kv_score_sidecar.py` and
  inline in `cache_processors/longshot.py`.

## PR 5 — split oversized monoliths (keep behavior identical)  ✅ landed

The oversized monoliths were split into sibling modules with natural owners:
`config.py` → `config/`, `models/backbones/transformer.py` →
`models/backbones/transformer/`, `models/cache_processors/longshot.py` →
`longshot{_layers,_processor,_linear_state,_selection,_workspace}.py`,
`training/checkpoint.py` → `training/checkpoint/`, and `convert.py` →
`convert/`. `training/core.py` was reduced to the step loop plus the
plan-builder hand-off to `training/assembly.py`.

## Verification gates for every structural change

`uv run ruff check src scripts tests`, `uv run pytest -q`,
`uv run mypy` on the audited targets, and
`uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html`
(from `docs/development/refactor-plan.md`). Processor changes additionally
require the fixed-seed loss, gradient, cache-mutation, generation,
parameter-count, FLOP, GPU-memory, and throughput checks.