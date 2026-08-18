# GCW overview

**Status:** Maintained

GCW asks whether a small learned module can reorganize a causal transformer's key-value cache and improve later prediction. The backbone remains a decoder-only language model. The new module periodically reads selected cached states, mixes information across time, layers, or cache heads, and writes transformed states back. Training still minimizes next-token cross-entropy.

The shortest accurate mental model is:

> A transformer writes a notebook while reading. GCW occasionally lets a small editor compress or cross-reference selected notebook entries before the transformer continues. The editor learns only from whether later token predictions improve.

The core execution path is:

```{mermaid}
flowchart LR
    A[raw text or token shards] --> B[tokenizer and batch stream]
    B --> C[causal transformer]
    C --> D[key-value cache]
    D --> E{processor due?}
    E -- no --> C
    E -- yes --> F[select cache positions]
    F --> G[BT, BGT, Icarus, or Longshot]
    G --> H[rewrite selected cache state]
    H --> C
    C --> I[next-token logits]
    I --> J[cross-entropy loss]
    J --> K[AdamW update]
    K --> L[artifact or resumable checkpoint]
```

Five boundaries prevent conceptual mistakes:

1. `TF` is the unprocessed causal-transformer baseline. `BT`, `BGT`, `Icarus`, and `Longshot` add cache processors.
2. The cache processor changes internal memory, not the training target. All variants learn through causal token loss.
3. A portable artifact checkpoint supports inference and publication. A version-2 training checkpoint additionally captures optimizer, cursor, and random state for continuation.
4. Evaluation uses native GCW models and tokenizers. Hugging Face backbones enter through one strict conversion path.
5. The JEPA section below explains related representation-prediction theory. No current GCW path implements a JEPA objective.

## Repository layers

| Layer | Main paths | Responsibility |
|---|---|---|
| Configuration | `src/gcw/config/`, `configs/` | Typed experiment definitions and concrete run matrices |
| Data | `src/gcw/data/` | Tokenization, FineWeb shards, SFT rows, resumable batch streams |
| Backbone | `src/gcw/models/backbones/` | Decoder-only attention, MLP, RoPE, loss, and generation |
| Memory | `src/gcw/models/cache_processors/state.py`, `src/gcw/models/cache_processors/windowed.py` | Dynamic and static KV storage plus selection metadata |
| Processors | `src/gcw/models/cache_processors/` | BT, BGT, Icarus, Longshot, and Flow cache transformations |
| Training | `src/gcw/training/core.py`, `src/gcw/training/checkpoint/` | Distributed optimization, schedules, exact resume, W&B |
| Evaluation | `src/gcw/evaluation/core.py`, `src/gcw/evaluation/scoring.py` | Task generation, likelihood, BPB, distributed aggregation |
| Conversion | `src/gcw/convert/` | Strict translation from external checkpoint layouts |
| Operations | `scripts/`, `docs/` | Slurm launch, reproduction, publication, and focused reports |
| Verification | `tests/` | Behavioral, numerical, distributed, configuration, and regression contracts |

A standard autoregressive transformer preserves a key and value for every retained token at every layer. This gives later queries direct access to earlier states, but it has three limitations:

- memory grows linearly with context length;
- cached states remain mostly frozen after creation;
- cross-layer information only mixes through the normal forward stack, not through an explicit global memory workspace.

GCW studies a narrower intervention than replacing the transformer. It inserts a cache processor at a fixed cadence and asks:

> Can a comparatively small module improve future prediction by selecting and rewriting internal memory while preserving the pretrained backbone and its causal interface?

This framing creates clean comparisons:

- `TF` measures the backbone without cache rewriting;
- added-on-top configurations hold the backbone fixed and add processor parameters;
- matched-total configurations shrink the backbone so the full model matches the baseline parameter budget;
- frozen-backbone runs isolate the processor's contribution;
- unfrozen runs test joint adaptation;
- value-only, workspace-broadcast, head-mask, unrope, and zero-initialization variants isolate specific mechanisms.

The central scientific risk is confounding. A processor can appear better because it adds parameters, changes the backbone, uses a different tokenizer, sees different examples, or follows different generation cadence. The configuration families and parity tests try to expose those confounds rather than hide them.
