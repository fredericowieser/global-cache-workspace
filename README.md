# Global Cache Workspace

`gcw` is a clean CE-only workspace for comparing:

- `TF`: vanilla transformer
- `BT`: bottlenecked transformer with one per-layer Cache Processor
- `BGT`: bottlenecked global transformer with one global Cache Processor

The code is native PyTorch. HF checkpoints are converted into native
`safetensors` checkpoints; HF model wrappers are not used at train time.

## Core Commands

```bash
gcw params make --family smollm360m --processor-fraction 0.05 --out configs/model
gcw train --config configs/train/smoke_tf.yaml
gcw train --config configs/train/smoke_bt.yaml
gcw train --config configs/train/smoke_bgt.yaml
gcw prepare fineweb --out ~/.cache/gcw/fineweb_edu_dedup_cosmo2
gcw eval core --checkpoint runs/smoke_tf/checkpoint --limit 5
```

## Design Rules

- Training loss is always next-token cross entropy.
- Cache Processors fire every 64 tokens.
- Selection uses attention-output residual L2 norm.
- Recent 64 tokens are always selected; top-32 history tokens are added.
- BT and BGT configs share the exact same backbone.
- Generated BT/BGT Cache Processors stay under 5% of backbone non-embedding params.
- Attention runs through PyTorch SDPA and requests Flash Attention on CUDA.

Excluded by design: RL, iterative recurrence sweeps, GKVW, lite variants,
surprise routing, HF Trainer, DeepSpeed, and analysis sprawl.
