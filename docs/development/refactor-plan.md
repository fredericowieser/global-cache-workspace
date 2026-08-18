# Architecture direction

**Status:** Maintained

GCW uses explicit composition rather than model-specific wrapper classes:

```text
ModelConfig
├── family: smollm | llama | qwen3 | qwen3_5 | qwen3_5_moe | gemma4
├── variant: tf | bt | bgt | icarus | longshot
├── cache_processor: CacheProcessorConfig | null
└── flow_processor: FlowProcessorConfig | null

build_model
├── build_backbone
├── build_cache_processor, when variant != tf
└── LanguageModel(backbone, cache_processor)
```

## Ownership

- `models/backbones/` owns causal language models and native cache production.
- `models/cache_processors/` owns cache state, selection, transformation, recurrence, and writeback.
- `models/model.py` owns composition and the shared processed-model execution path.
- `data/` owns FineWeb and SFT streams.
- `training/` owns optimization and exact resume.
- `evaluation/` owns independent benchmark scoring.
- `provenance.py` owns scientific identity.
- `runtime.py` owns atomic I/O and distributed lifecycle.

There is no `utils` package and no dynamic plugin registry.

## Independent experiment axes

1. **Backbone:** architecture and dimensions.
2. **Processor:** none, BT, BGT, Icarus, or Longshot; Flow modifies supported processors.
3. **Initialization:** random, backbone-only checkpoint, complete model checkpoint, or exact resume.
4. **Data/objective:** FineWeb next-token CE, answer-only SFT CE, or full-sequence SFT CE.
5. **Trainable ownership:** backbone, cache processor, or both.
6. **Execution:** single process, DDP, or FSDP where admitted.

Do not combine these axes into a vague `training_method` field.

## Compatibility policy

The old flat modules and variant classes are removed. Canonical imports are the maintained API. Checkpoint tensor ownership remains stable:

- processor-free backbones use native unprefixed keys;
- composed models use `backbone.*` and `cache_processor.*`.

Exact resume requires version-2 training checkpoints. Model-only checkpoints initialize a new run but cannot restore optimizer, cursor, or RNG state.

## Execution-plan resolution

Cross-feature admission lives in exactly one place: `resolve_execution_plan`
(`architecture/execution.py`). It decides which single execution strategy a
config selects (`eager_reference`, `backbone_chunk_compile`, `triton_cache_kernel`,
`longshot_selected_history`, `kv_score_sidecar`, `noisy_diffusion_compile`, or
`adaptive_pairs`), checks each strategy's `requires`/`forbids` contract, and
returns a frozen `ExecutionPlan`. `TrainConfig.__post_init__` keeps only field-level
sanity (ranges, enum membership, coercion) and delegates composition admission to
the resolver; `train()` re-runs it to obtain the plan. The `ValueError` messages
are the public admission contract and are asserted verbatim by the tests.

## Scientific boundaries

Flow is deterministic time-conditioned latent integration. It is not diffusion. A diffusion processor requires a corruption schedule, denoising target, noise conditioning, and trained reverse process. Keep that proposal outside executable claims until implemented and tested.

## Verification gates

Every structural change must pass:

```bash
uv run ruff check src scripts tests
uv run pytest -q
uv run mypy src/gcw/config src/gcw/architecture src/gcw/models/model.py \
  src/gcw/data/sft.py src/gcw/data/training.py src/gcw/training/core.py \
  src/gcw/evaluation/core.py src/gcw/cli.py
uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
uv build
```

Processor changes additionally require fixed-seed loss, gradient, cache-mutation, generation, parameter-count, FLOP, GPU-memory, and throughput checks.
