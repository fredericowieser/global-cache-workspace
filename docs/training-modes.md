# Training Modes

GCW uses one training entry point:

```bash
gcw train --config <config.yaml>
```

The config controls the mode.

## Precision

Select the parameter and compute precision explicitly:

```yaml
precision: bf16
```

The supported modes are:

- `fp32`: FP32 parameters and compute.
- `bf16_mixed`: FP32 parameter and optimizer storage with BF16 CUDA autocast.
- `bf16`: BF16 parameters, gradients, optimizer moments, and model compute. Numerically sensitive logits, losses, norms, scores, and reductions remain FP32.

The legacy `bf16: true` setting remains supported and maps to `precision: bf16_mixed`. Full BF16 requires a CUDA device with BF16 support for GPU training. Checkpoints retain their parameter dtype; standalone evaluation uses that dtype by default, or it can be overridden explicitly:

```bash
gcw eval core --checkpoint <checkpoint> --dtype bfloat16
```

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
