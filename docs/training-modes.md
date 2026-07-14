# Training Modes

GCW uses one training entry point:

```bash
gcw train --config <config.yaml>
```

The config controls the mode.

## Scratch Pretraining

Use no checkpoint:

```yaml
mode: pretrain
data:
  kind: fineweb
backbone_checkpoint: null
freeze_backbone: false
```

For `variant: tf`, this trains a normal transformer from random initialization. For `variant: bt`, `variant: bgt`, or `variant: icarus`, this trains the backbone and cache processor together from random initialization.

## SFT Or Continued SFT

Use OpenMathInstruct JSONL:

```yaml
mode: sft
data:
  kind: sft
  dataset: /mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl
  tokenizer: hf
  tokenizer_source: /mnt/jfs-01/shared/bottleneck_llm/sft/llama-3.2-1b/final_model
```

The SFT loader masks prompt tokens and trains on answer tokens.

## Frozen-Backbone Cache Processor Training

Use a native GCW SFT backbone checkpoint and freeze it:

```yaml
model:
  variant: bgt
backbone_checkpoint: /mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf
freeze_backbone: true
```

Only cache processor parameters receive gradients. This is the closest GCW reproduction path for the legacy Bottleneck LLM finetunes.

## Unfrozen Finetuning

Use the same checkpoint but do not freeze:

```yaml
backbone_checkpoint: /mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf
freeze_backbone: false
```

For BT/BGT/Icarus, this trains the backbone and cache processor together. For TF, it continues finetuning the transformer.

## No Cache Processor

Use:

```yaml
model:
  variant: tf
  cache_processor: null
```

`variant: tf` never constructs a cache processor.

## Resume Training

Use a version-2 training checkpoint:

```yaml
resume:
  checkpoint: runs/example/checkpoint
  topology: same
```

`topology: same` requires the original world size. FineWeb DDP runs may use `topology: elastic_ddp` when the global
microbatch, total batch size, gradient accumulation, data inventory, model, optimizer, and schedule remain unchanged.
FSDP, random-data, and SFT topology changes are rejected.

Do not set `backbone_checkpoint` and `resume` together. A backbone checkpoint starts a new optimizer and schedule;
resume restores an existing training state. See [training-resume.md](training-resume.md) for the complete contract.
