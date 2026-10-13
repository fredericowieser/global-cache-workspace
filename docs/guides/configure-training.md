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

For plain-text FineWeb adaptation, see the continued-pretraining section below. `mode: sft` selects the prompt/answer JSONL loader.

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

### Continued pretraining on FineWeb

`mode: pretrain` also supports pretrained initialization. Prepare [tokenizer-matched FineWeb shards](../concepts/data-and-tokenization.md), keep the backbone's model configuration, and use:

```yaml
mode: pretrain
backbone_checkpoint: /path/to/native-backbone-checkpoint
data:
  kind: fineweb
  data_dir: /path/to/fineweb-tokenized
  tokenizer: hf
  tokenizer_source: /path/to/backbone-tokenizer
trainable: backbone
```

For full-backbone fine-tuning, use `model.variant: tf`. For LoRA, retain `tf`, supply the existing `model.lora` configuration for that backbone, and set `trainable: lora`. For processor-only training, retain the desired supported processor configuration and set `trainable: cache_processor`. Existing joint policies are also supported. Architecture and execution compatibility restrictions still apply.

FineWeb always uses full next-token loss without chat formatting or answer masking. SFT-specific settings have no effect on its token stream; do not enable SFT-only logical pairing or adaptive grouping. Match sequence length, global token budget (`total_batch_size`), optimizer steps, and prepared shards when comparing training policies. Each run retains the existing initialization, validation, checkpoint, and resume behavior. Changing tokenizer requires preparing new shards; editing the training tokenizer field does not retokenize stored IDs.

With FSDP, each training microbatch and validation batch is right-padded to the
maximum sequence width across ranks before the model runs. This keeps windowed
cache processing and FSDP collectives in the same order when ranks receive
different-length examples. Added positions use ignored targets (`-100`) and a
false attention mask; example ordering, supervised tokens, and gradient
accumulation are preserved. Single-process and DDP batching retain their existing
padding behavior. This synchronization requires no configuration flag.

Use `global_batch_rows` when experiments must consume the same examples per optimizer step despite different sequence padding or physical microbatches. It is mutually exclusive with the legacy token-budget field `total_batch_size`.

### Adjacent logical-pair objective

Set `logical_pair_grouping: true` to preserve an objective defined by adjacent two-row groups while using a larger physical batch. GCW computes answer-only cross-entropy as a token mean within each adjacent pair, averages the pair losses in the physical batch, then averages physical groups across gradient accumulation. With `global_batch_rows: 64`, every optimizer step remains the mean of exactly 32 pair losses regardless of whether the fixed physical batch is 2, 4, 8, or 16 rows.

Logical pairing preserves source order and drops an incomplete final physical group instead of pairing the dataset tail with the next epoch. The SFT cursor records the post-group row, so exact-topology resume repeats the same drop-last transition. The objective requires an even physical batch, answer-only SFT, an even row-based global batch divisible by the physical batch, and zero router auxiliary loss. Noisy Diffusion is rejected because changing the physical batch changes RNG assignment; deterministic Diffusion with zero noise is supported.

The option defaults to `false`. Its objective protocol, exact-resume fingerprint, run metadata, training loss, perplexity, and validation loss are distinct from ordinary token-mean SFT.

#### Adaptive physical grouping

Adaptive grouping is a separate default-off execution policy for the same objective:

```yaml
batch_size: 2                 # source stream remains one original pair
global_batch_rows: 64
logical_pair_grouping: true
loss_backend: reference
performance:
  adaptive_batching:
    enabled: true
    bands:
      - {max_padded_tokens: 1024, physical_rows: 16}
      - {max_padded_tokens: 2048, physical_rows: 8}
      - {max_padded_tokens: 3072, physical_rows: 4}
      - {max_padded_tokens: 4096, physical_rows: 2}
```

At each optimizer step GCW reserves exactly 32 consecutive two-row source batches. It greedily collates only contiguous pairs into the largest admitted band, without sorting, bucketing, moving rows, or splitting pairs. Each physical group's pair-loss sum is divided by 32 before backward, so group count and shape cannot change objective weights. There remains exactly one gradient clip, schedule update, and AdamW step.

Version 1 requires single-process execution, 64 global rows, a two-row source stream, answer-only logical pairing, `reference` loss, at most 4,096 tokens, and zero processor/LoRA dropout. Evaluation remains fixed at batch two. Noisy Diffusion and other admitted RNG-shape changes fail during configuration admission. The ordered bands and planner/weighting rules are authenticated in run metadata and the exact-resume fingerprint.

The composed optimized executor has one reserved, default-off selector:

```yaml
performance:
  optimized_step:
    enabled: false
    profile: combined_v1
```

This source build does not admit `combined_v1`. When disabled, receipt/key fields must remain absent. Future enablement requires all of `receipt_path`, `key_path`, `key_id`, and `qualification_id`, then a source-, config-, kernel-, runtime-, and GPU-bound authenticated receipt. There is no eager fallback. Do not add these fields to campaign configs until the H100 and H200 qualification receipts are published and the source admission bit changes.

### LoRA targets and joint training

Set `trainable: lora` with `model.lora` for low-rank backbone adaptation. GCW freezes base weights and trains only `lora_a` and `lora_b` on explicitly named backbone projections. Targets are module names, not a request to train all parameters: the ordinary defaults remain `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. `trainable: all` instead unfreezes the complete model, including base weights.

The Qwen3.5 all-linear baseline additionally targets `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, and `out_proj`. Those names cover its linear-attention projections; they do not adapt embeddings, normalization parameters, or convolutions.

For KV-only adaptation on a Qwen3.5 hybrid backbone:

```yaml
trainable: lora
model:
  lora:
    rank: 32
    alpha: 32
    dropout: 0
    targets: [k_proj, v_proj, in_proj_kv]
```

`k_proj` and `v_proj` adapt separate attention projections. The virtual target `in_proj_kv` adds independent rank-r K and V factors to native fused `[Q, K, V]` projections, leaving the native Q rows unchanged. It requires matching fused geometry and cannot be combined with `in_proj_qkv` (which adapts the entire fused projection). On backbones with only separate projections, use `[k_proj, v_proj]` instead. Zero-initialized output factors preserve starting outputs; the virtual target also isolates construction RNG so shared backbone and processor initialization stays matched to an adapter-free control. Ordinary target initialization is unchanged.

Use `trainable: lora_and_cache_processor` with any ordinary or KV-only LoRA target list and a cache-processor variant to jointly train adapters and processor while freezing base backbone parameters. Leave `freeze_backbone: false` and `save_cache_processor_only: false`; full checkpoints must retain both components. This policy does not imply better results than standalone LoRA or processor training.

Runtime non-embedding parameter counts include attached adapters. Analytical training FLOPs still exclude adapter operations (`adapter_flops_included: false` in LoRA run metadata); they are not exact joint-training compute counts.

## Loss backend

`loss_backend` selects the language-model head and cross-entropy implementation:

```yaml
loss_backend: reference  # reference | target_only | liger
```

`reference` is the default. It projects every sequence position in chunks and is the experiment oracle. H200 BGT measurement found `target_only` at 0.978x the original throughput, so the optimization is not a safe default. Benchmark configs may opt into `target_only` to project supervised positions only, or `liger` to use fused linear cross-entropy.

`liger` requires CUDA and the `gpu` extra's exact `liger-kernel==0.8.1` pin. Missing packages, another version, and CPU/ROCm execution fail before model allocation. Liger handles `mean` and `sum`; structured `row_sum` and per-token `none` use the explicit `target_only` fallback recorded in provenance. The benchmark-only per-row Liger sum candidate was about 37x slower than bounded reference on the production-shaped H100 boundary and is not dispatched by model training. All backends retain FP32 loss accumulation, ignored-target behavior, real-vocabulary truncation, tied/untied head gradients, and identical state-dict keys.

The selected backend and implementation metadata enter local run records, W&B config, checkpoint state, and the exact-resume fingerprint. Resume rejects a backend change.

Keep `loss_backend: reference` while bounding the physical full-vocabulary projection with an opt-in execution policy:

```yaml
loss_backend: reference
performance:
  reference_loss:
    mode: bounded
    max_projected_positions: 1024
    checkpoint_chunks: false
```

The budget limits `physical_batch_rows * sequence_positions` per projection subchunk and must be at least the maximum admitted physical row count. GCW still projects ignored prompt and padding positions, rebuilds the legacy logical loss chunks, and computes the same FP32 reductions and objective. Set `checkpoint_chunks: true` only after measuring the projection/backward memory trade-off. The bounded policy is default-off, state-neutral, authenticated for resume, and initially limited to single-process execution; it is incompatible with a non-reference loss backend or legacy whole-model compilation.

## Frozen-Backbone Cache Processor Training

Use a native GCW backbone checkpoint and select processor-only gradients:

```yaml
model:
  variant: bgt
backbone_checkpoint: /mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf
trainable: cache_processor
```

Only cache processor parameters receive gradients. Use `trainable: backbone` to freeze the cache processor instead.

Under FSDP with the reference loss and untied embeddings, a frozen backbone's vocabulary embedding and output projection receive separate wrappers in addition to its layer-level wrappers. This keeps those large frozen weights out of the trainable processor's gradient buffer. Tied embeddings, loss backends that access output weights directly, and Longshot's shared `gated_delta` dynamics retain the existing wrapper layout. Longshot's `outer_product` configuration uses the separate vocabulary wrappers.

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

To compile the initial complete 64-token window as a separate static full graph, use:

```yaml
backbone_chunk_backend: compile_full_windows
```

This version compiles two pure tensor regions. The initial region receives embeddings and rotary tensors, uses Flash SDPA's native causal boundary, and returns stacked K/V and scores. Later full windows retain the dynamic-past compiled region described above. Partial windows remain eager. The older `compile` value and its version-1 fingerprint remain unchanged; `compile_full_windows` uses a distinct version-2 fingerprint. The option is state-dict neutral, frozen-Qwen3-only, fail closed, and incompatible with FSDP and whole-model compilation. Run the opt-in CUDA smoke with `GCW_RUN_QWEN3_INITIAL_COMPILE_CUDA_SMOKE=1` before using a new Torch/CUDA stack.

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
activation_checkpoint_min_padded_sequence_length: 2048  # padded width; zero keeps every training forward eligible
```

The minimum-length threshold uses the physical microbatch's padded width (`input_ids.shape[1]`). Shorter microbatches run eager; rows at or above the threshold use segment checkpointing. The decision is local to each forward and deterministic. A threshold of zero preserves the original always-on behavior whenever `activation_checkpoint_windows` is positive.

Each segment exports only hidden output and its tensor-only cache boundary. Backward replays the segment with RNG preservation. BT creates a new prepared-stack owner for every original segment invocation and every replay; closed owners remain unusable, and graph-derived stacks never cross segment or forward lifetimes. Replay suppresses processor call counts and Longshot overlap metrics. Per-step metrics report checkpointed and eager microbatch counts plus an active flag without adding distributed collectives.

Segment checkpointing runs only in training with gradients enabled. Evaluation accepts the resolved policy but executes eagerly. It supports both regional backends: `GCW_BT_PROCESSOR_BACKEND=compile` for BT and `GCW_CACHE_PROCESSOR_BACKEND=compile` for BGT, Flow-BGT, and noiseless Diffusion-BGT. Both compose with `standard_processor_layers`; segment replay keeps selection, gather, and writeback eager around the compiled tensor region. FSDP layer checkpointing and Qwen3.5 MoE router-loss execution reject this policy explicitly.

Icarus has a rejected benchmark-only post-selection region. It is not a training configuration: ordinary `GCW_ICARUS_PROCESSOR_BACKEND=compile` fails before allocation because fixed batch-2 H200 Adam parameter parity did not meet the formal tolerance. Isolated diagnostic reproduction additionally requires `GCW_ALLOW_REJECTED_ICARUS_COMPILE=1`; provenance records that override and `campaign_admitted: false`. Campaign Icarus remains eager.

Both checkpointing policies increase backward compute. The policies, segment size, minimum padded sequence length, and selected regional backend are runtime provenance and exact-resume fingerprint fields; changing any across resume is rejected.

### Optional truncated BPTT

Cache-processor training can cut autograd history at recurrent cache boundaries without changing the forward recurrence:

```yaml
truncated_bptt_windows: 1  # zero is full BPTT; positive values are the gradient horizon
```

The value counts `process_every` windows retained in one autograd horizon. A value of `1` prevents gradients from crossing a cache-processing boundary; larger values retain longer credit assignment. This policy is implemented at the shared windowed-cache boundary, so BT, BGT, Icarus, Longshot, and compatible backbone families use the same behavior. It is training-only in effect and does not alter evaluation losses. `train/training_method`, `train/truncated_bptt_windows`, and the corresponding W&B run metadata identify the method. The setting is part of the training fingerprint and must match for exact resume. It does not modify run names or user-supplied W&B tags.

Longshot keeps its model-level switch as the fixed default. Set an adaptive training threshold only when `model.cache_processor.gradient_checkpointing` remains `true`:

```yaml
model:
  variant: longshot
  cache_processor:
    gradient_checkpointing: true
longshot_gradient_checkpointing_seq_len_threshold: 2048
```

The adaptive policy runs Longshot processor layers eagerly when the physical microbatch's padded `input_ids.shape[1]` is below the threshold. It checkpoints them at or above the threshold. The override covers one forward, including internal and external recurrence; evaluation remains eager because it disables gradients. The policy does not mutate model state or affect other processors. Runtime metadata records the threshold and boundary. Per-step metrics report checkpointed and eager microbatch counts, checkpointed fraction, and minimum/maximum padded lengths. The threshold is part of the exact-resume fingerprint.

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
