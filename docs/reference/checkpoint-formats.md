# Checkpoint formats

**Status:** Maintained

GCW uses two explicit checkpoint contracts.

## Model checkpoint

A model checkpoint contains:

```text
checkpoint/
├── config.json
├── model.safetensors
└── train_state.json
```

Use it for inference, evaluation, publication, as `backbone_checkpoint` to load only backbone weights, or as `init_checkpoint` to load the complete backbone-and-processor state. Processor-bearing checkpoints keep `backbone.*` and `cache_processor.*` tensor namespaces. Processor-free checkpoints keep native backbone keys.

## Training checkpoint

A resumable version-2 checkpoint additionally contains optimizer state, RNG state, loader cursors, and an authenticated manifest. Resume requires a matching scientific and optimizer fingerprint.

`train_state.json` records loss-backend provenance. The authenticated training fingerprint records the explicit backend name, so exact resume cannot switch among `reference`, `target_only`, and `liger`. Loss backends are execution policy only; they add no tensors or state-dict namespaces.

`backbone_checkpoint` and `init_checkpoint` start a new optimizer and schedule. `resume.checkpoint` continues an existing execution. The three sources are mutually exclusive.

Initialization compares model configurations after applying the checkpoint
loader's legacy-field normalization, so equivalent legacy generation flags are
accepted while differences in generation policy or architecture are rejected.
Processor-only checkpoints still require the frozen backbone referenced by
their own `train_state.json`, even when the new training config sets
`backbone_checkpoint: null`. When relocating such a checkpoint, use a local
view with the original model/config files and a copied `train_state.json` whose
backbone path points to the matching local weights. Preserve the source
checkpoint; this view is for initialization, not authenticated training resume.

When a new run initialized with `init_checkpoint` saves processor-only weights,
it carries forward the initialization checkpoint's frozen-backbone reference.
If the initialization checkpoint contains the complete model, that checkpoint
itself supplies the frozen backbone. The optimizer and training cursor still
start fresh. Older processor-only outputs missing this reference can recover it
from their recorded `init_checkpoint` when loaded; that initialization source
must remain accessible.

## Hugging Face initialization

Convert a supported pretrained backbone and initialize any selected processor:

```bash
uv run --extra convert gcw convert hf \
  --family qwen3 \
  --model Qwen/Qwen3-0.6B \
  --variant longshot \
  --out data/runs/qwen3-06b-longshot-init
```

The converter is shape-strict and records source identity. It does not support obsolete PPT or bespoke Bottleneck checkpoint layouts.
