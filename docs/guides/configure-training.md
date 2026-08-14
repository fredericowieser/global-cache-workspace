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

Two independent policies reduce different activation classes. Standard BT and BGT processors can checkpoint each processor transformer layer:

```yaml
trainable: cache_processor
activation_checkpointing: standard_processor_layers
```

The default, `none`, preserves eager processor layers. The layer boundary includes BT's batched parameter-stack path. Backward recomputes attention and MLP intermediates while cache selection, writeback, backbone windows, and loss remain eager.

Qwen3 recurrent variants can instead, or additionally, checkpoint groups of backbone windows:

```yaml
activation_checkpoint_windows: 2  # zero disables; positive values are windows per segment
```

Each segment exports only hidden output and its tensor-only cache boundary. Backward replays the segment with RNG preservation. BT creates a new prepared-stack owner for every original segment invocation and every replay; closed owners remain unusable, and graph-derived stacks never cross segment or forward lifetimes. Replay suppresses processor call counts and Longshot overlap metrics.

Segment checkpointing runs only in training with gradients enabled. Evaluation accepts the resolved policy but executes eagerly. It supports both regional backends: `GCW_BT_PROCESSOR_BACKEND=compile` for BT and `GCW_CACHE_PROCESSOR_BACKEND=compile` for BGT, Flow-BGT, and noiseless Diffusion-BGT. Both compose with `standard_processor_layers`; segment replay keeps selection, gather, and writeback eager around the compiled tensor region. FSDP layer checkpointing and Qwen3.5 MoE router-loss execution reject this policy explicitly.

Both checkpointing policies increase backward compute. They and the selected regional backend are runtime provenance and exact-resume fingerprint fields; changing any of them across resume is rejected.

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
