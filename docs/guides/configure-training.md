# Training Modes

GCW uses one training entry point:

```bash
gcw train --config <config.yaml>
```

The config controls the mode. `configs/train/sft_template.yaml` shows complete-model initialization and both SFT loss scopes.

## Scratch Pretraining

Use no checkpoint:

```yaml
mode: pretrain
data:
  kind: fineweb
backbone_checkpoint: null
trainable: all
```

For `variant: tf`, this trains a normal transformer from random initialization. For `variant: bt`, `bgt`, `icarus`, or `longshot`, this trains the backbone and cache processor together from random initialization.

## SFT Or Continued SFT

Use OpenMathInstruct JSONL:

```yaml
mode: sft
data:
  kind: sft
  dataset: /mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl
  tokenizer: hf
  tokenizer_source: /path/to/tokenizer
  sft_loss_scope: answer_only
```

`answer_only` masks prompt labels. `full_sequence` computes next-token cross-entropy over prompt and answer tokens. Padding remains masked in both modes.

Use `global_batch_rows` when experiments must consume the same examples per optimizer step despite different sequence padding or physical microbatches. It is mutually exclusive with the legacy token-budget field `total_batch_size`.

### Adjacent logical-pair objective

Set `logical_pair_grouping: true` to preserve an objective defined by adjacent two-row groups while using a larger physical batch. GCW computes answer-only cross-entropy as a token mean within each adjacent pair, averages the pair losses in the physical batch, then averages physical groups across gradient accumulation. With `global_batch_rows: 64`, every optimizer step remains the mean of exactly 32 pair losses regardless of whether the physical batch is 2, 4, or 8 rows.

Logical pairing preserves source order and drops an incomplete final physical group instead of pairing the dataset tail with the next epoch. The SFT cursor records the post-group row, so exact-topology resume repeats the same drop-last transition. The objective requires an even physical batch, answer-only SFT, an even row-based global batch divisible by the physical batch, and zero router auxiliary loss. Noisy Diffusion is rejected because changing the physical batch changes RNG assignment; deterministic Diffusion with zero noise is supported.

The option defaults to `false`. Its objective protocol, exact-resume fingerprint, run metadata, training loss, perplexity, and validation loss are distinct from ordinary token-mean SFT.

Set `trainable: lora` with `model.lora` for low-rank backbone adaptation. GCW freezes base weights and trains only `lora_a` and `lora_b` on explicitly named projections. Zero-initialized output factors preserve the starting backbone exactly.

## Frozen-Backbone Cache Processor Training

Use a native GCW backbone checkpoint and select processor-only gradients:

```yaml
model:
  variant: bgt
backbone_checkpoint: /mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf
trainable: cache_processor
```

Only cache processor parameters receive gradients. Use `trainable: backbone` to freeze the cache processor instead.

### Native activation checkpointing

Opt in when standard BT or BGT processor activations exceed device memory:

```yaml
trainable: cache_processor
activation_checkpointing: standard_processor_layers
```

The default, `none`, preserves eager execution. The checkpoint boundary covers each complete standard processor transformer layer, including BT's batched parameter-stack path. Backward recomputes that layer instead of retaining its attention projections, attention output, normalized residual, and gated-MLP intermediates. Input/output projections, cache selection and writeback, cache tensors, loss tensors, and frozen-backbone window activations remain eager.

Checkpointing increases processor-layer compute during backward. It uses PyTorch's non-reentrant implementation with RNG preservation, so Diffusion noise and attention dropout match eager execution. The policy is part of runtime provenance and the exact-resume fingerprint; changing it across resume is rejected.

## Joint Finetuning

Use the same checkpoint and train both components:

```yaml
backbone_checkpoint: /mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf
trainable: all
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

Choose one initialization source. `backbone_checkpoint` loads only backbone weights. `init_checkpoint` loads the complete backbone-and-processor state. Both start a new optimizer and schedule. `resume` restores model, optimizer, cursor, and RNG state. See [resume-training.md](resume-training.md) for the complete contract.
