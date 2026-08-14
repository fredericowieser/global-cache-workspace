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

### Qwen3 frozen-backbone regional compilation

Qwen3 cache-processor runs remain eager unless the training config opts into the 64-token backbone region:

```yaml
model:
  family: qwen3
  variant: bt
  cache_processor:
    process_every: 64
trainable: cache_processor
backbone_chunk_backend: compile
```

The first window establishes frozen cache state eagerly. Each later full window compiles the tensor cache boundary from chunk embeddings and stacked past K/V through every Qwen3 layer and final norm. PyTorch cannot construct `CausalBias` inside a full Dynamo graph, so the eager caller builds one lower-right causal bias and passes it into the region. This keeps the bias object out of graph construction while preserving Flash SDPA for CUDA BF16. The Python window loop, cache-processor scheduling, `WindowedCache` construction, score concatenation, vocabulary head, first window, and short final chunk remain eager. Past length is dynamic, so repeated 64-token calls reuse one regional compiler dispatch as retained history grows. Keeping the no-gradient first window outside the region also avoids a separate autograd specialization.

The opt-in fails closed. It requires exact `family: qwen3`, a windowed processor with `process_every: 64`, `trainable: cache_processor`, at least 128 training tokens, non-FSDP execution, callable `torch.compile`, and Dynamo error suppression disabled. A graph break or compiler/runtime failure raises; it never silently retries an eligible 64-token call eagerly. `GCW_COMPILE` must be disabled because wrapping the outer language-model loop defeats the regional boundary.

BT processor compilation is independent: `GCW_BT_PROCESSOR_BACKEND=compile` may compile the prepared BT processor region alongside `backbone_chunk_backend: compile`. `activation_checkpointing: standard_processor_layers` may also be enabled; it checkpoints processor layers only. Gradients cross each frozen compiled backbone chunk into cache-processor outputs from earlier windows.

The compiled callable registers no parameters or buffers, so state-dict keys remain unchanged. Run metadata records the selected backend, region, dynamic-shape policy, compile options, Torch version, and fingerprint. Exact resume rejects a changed backend fingerprint.

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

Segment checkpointing runs only in training with gradients enabled. Evaluation accepts the resolved policy but executes eagerly. It supports the regional `GCW_BT_PROCESSOR_BACKEND=compile` path and composes with `standard_processor_layers`. FSDP layer checkpointing and Qwen3.5 MoE router-loss execution reject this policy explicitly.

Both policies increase backward compute. Both are runtime provenance and exact-resume fingerprint fields; changing either across resume is rejected.

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
