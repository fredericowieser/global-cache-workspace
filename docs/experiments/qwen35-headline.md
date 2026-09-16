# Qwen3.5 headline experiment drafts

## Status and scope

The self-contained recipes are in `configs/train/qwen35_headline/{2b,4b,9b}/`.
They are **drafts, not launch-qualified experiments**. No training, model-weight
conversion/download, or dataset preparation was performed when creating them.
Model/experiment admission and meta-device parameter counts do not establish GPU
fit, throughput, convergence, or scientific superiority.

The proposed protocol is a larger-data successor to the bounded experiments in
[qwen35-ablations.md](qwen35-ablations.md), not a reproduction of their scores.
The data budget, sequence cap, mixed precision, distributed execution, processor
scaling, and matched LoRA ranks below are explicit new protocol choices.
The paper protocol uses **one seed, 24, for every arm**. Report these as
single-seed results: they do not estimate training-seed variability, and any
example-level confidence interval is not a seed-level uncertainty estimate.

### Contextual vector writeback at initialization −2

Every processor recipe opts into scheduled H50/K4 with
`cache_processor.update_gate_type: contextual_vector`, `output_scale_mode: learned`,
`gate_mode: sigmoid`, and `gate_init: -2.0`. SFT and ordinary LoRA remain ungated
backbone controls. These replace the earlier draft `fixed_one` processor files;
run outputs and metadata use the distinct `qwen35-headline-v2` protocol.

The existing fused readout predicts a residual and one gate per output channel
from the final solver latent, once per intervention. Primary KV and linear-state
readouts own separate gates. Gate weights start at zero and biases at −2, giving
initial factor scales of `sigmoid(-2) ≈ 0.1192`. Linear-state vector gates act on
the key/value factors, not individual matrix entries. The writer normalizes the
gated key before the outer product, so uniform positive key scaling cancels
outside the normalization epsilon regime; vector gates can change key direction. Live random-key/zero-value
initialization still gives a zero initial matrix residual and a first value
gradient. A gate on an initially zero update begins learning only once that
update becomes nonzero.

This is **writeback gating, not a gated diffusion solver**. H=50, K=4, shared
field weights, training perturbation, deterministic inference, routing, and
cadence are unchanged. Scheduled `static_scalar` retains its legacy fixed-one
behavior and checkpoint identity; explicit fixed-one scaling still rejects
contextual gates. All scheduled variants require zero readout initialization.
See [Longshot update gating](../development/longshot-update-gating.md).

The earlier ordinary Longshot gate−2 result did not establish a learned-gate or
held-out advantage. The new vector-gated treatment must be tested, not called
best. No joint KV-LoRA, retired untied stages, auxiliary losses, or inference
noise is included in these processor-only arms.

## Proposed comparison matrix

Each size has seven files. `A` is prepared rows `[0, 512000)` and `B` is
`[512000, 1024000)`. Prepare and freeze their ordering once for every method.

| File stem | Initial weights | Data | Steps | LR | Update ownership |
|---|---|---|---:|---:|---|
| `sft` | Native upstream text backbone | A+B | 8,000 | 1e-5 | All backbone parameters |
| `lora` | Same native backbone; new adapters | A+B | 8,000 | 1e-4 | Ordinary all-linear LoRA |
| `diffusion_h50_k4_vector_gate` | Same native backbone; new processor | A+B | 8,000 | 1e-4 | Processor only |
| `sft_stage1` | Same native backbone | A | 4,000 | 1e-5 | All backbone parameters |
| `sft_then_sft` | Checkpoint from same-size `sft_stage1` | B | 4,000 | 1e-5 | All backbone parameters |
| `sft_then_lora` | Same stage-one checkpoint; new adapters | B | 4,000 | 1e-4 | Ordinary all-linear LoRA |
| `sft_then_diffusion_h50_k4_vector_gate` | Same stage-one checkpoint; new processor | B | 4,000 | 1e-4 | Processor only |

This is nine direct recipes, three shared warmups, and nine continuation recipes.
The stage-one checkpoint is trained once per size and reused by all three forks;
its standalone evaluation is a half-budget reference, not another full-budget
competitor. The direct SFT run provides a no-reset comparison; the shared-prefix
`SFT → SFT` run is the **reset-matched control** for `SFT → processor`.

All forks load **weights**, using `backbone_checkpoint`, and start fresh optimizer,
scheduler, and data-cursor state. None uses `resume` or `init_checkpoint`.
Freeze and authenticate the stage-one checkpoint before launching its forks;
rerunning stage one into the same mutable output path would invalidate the
shared-parent comparison.

Equal example exposure is not equal FLOPs or wall time. The staged processor
result also includes the cost of its SFT prefix. A half/half split is a proposed
controlled comparison, **not an established optimal allocation**. Extra context
or data does not guarantee processor superiority.

## Common training protocol

- Official chat template, thinking disabled, full assistant response supervised
  with answer-only CE; no step-marker augmentation or auxiliary objective.
- Sequence cap **2,048**, up from the earlier 512. Audit rendered prompt/answer
  lengths before accepting this cap: right truncation can remove answer targets.
- Physical batch **one row per GPU**, two GPUs per run, global batch **128 rows**:
  **64 accumulation microsteps** per optimizer step. `total_batch_size: 0`
  avoids an ambiguous second batch specification. No logical-pair grouping.
  The trainer averages microbatch token means; preserve the common physical batch
  when comparing methods, not just the global row count.
- **`bf16_mixed`** for every arm: FP32 parameters with BF16 compute/autocast.
  This intentionally differs from historical BF16 parameter storage; it avoids
  silently repeating that storage-rounding treatment. It is not a claimed
  measured precision improvement. FSDP additionally applies its mixed-precision
  sharding policy, so execution is not bitwise identical to DDP.
- AdamW β=(0.9, 0.999), ε=1e-8; zero weight decay, no warmup/warmdown, clipping 1.
  Constant LR avoids an additional schedule-shape change between the direct and
  staged budgets. LR tuning on development data remains a separate comparison.
- Liger loss; eager backbone/solver execution; no truncated BPTT. Do not enable
  `activation_checkpoint_windows` for Qwen3.5: it is currently Qwen3-only.
- Full SFT uses **FSDP**; LoRA and frozen processors use **DDP**. FSDP's existing
  layer-checkpointing default is distinct from the YAML
  `activation_checkpointing` option. Processor checkpointing is enabled with the
  retained sequence threshold 513. These are launch proposals, not GPU evidence.
- Save complete weights every 1,000 steps and at completion; no inline evaluation.
  Publish matched held-out/benchmark results separately rather than treating
  printed training CE EMA as accuracy.

The processor recipes deliberately use **`freeze_backbone: false` together with
`trainable: cache_processor`**. The latter freezes every original backbone weight.
The legacy boolean stays false because saving currently tests
`save_cache_processor_only OR freeze_backbone`; setting it true would force an
incomplete processor-only checkpoint even with `save_cache_processor_only: false`.

**Resume limits:** FSDP saves full model artifacts but does not support training
resume here. DDP full-checkpoint resume is a separate supported path requiring
matching fingerprints/topology. These recipes do not fix or qualify processor-only
resume. Do not assume a five-day allocation guarantees completion or recoverability
of the whole matrix. Current authenticated answer-NLL evaluation also requires a
complete training manifest; FSDP weight artifacts need an appropriately qualified
evaluation path rather than being presented as resumable training checkpoints.

## Constant processor fraction and matched LoRA capacity

The denominator is the **actual native text backbone parameter count**: include
embeddings and the output head, count tied weights once, exclude the unconverted
vision tower and MTP modules. Do not divide by the rounded marketing size or use
the dense non-embedding estimator for this hybrid backbone.

Target fraction: **22,427,520 / 1,881,825,088 = 1.1917962%**, anchored to the
2B H50/K4 processor with contextual vector writeback. Integer dimensions make exact equality impractical;
all three fractions agree within **0.1% relative error** of that target.

| Size | Native backbone parameters | CP hidden / FFN / heads | CP parameters | CP/backbone | LoRA rank | LoRA parameters |
|---|---:|---|---:|---:|---:|---:|
| 2B | 1,881,825,088 | 384 / 1536 / 3 | 22,427,520 | 1.191796% | 21 | 22,075,200 |
| 4B | 4,205,751,296 | 384 / 2720 / 3 | 50,173,824 | 1.192981% | 25 | 50,726,400 |
| 9B | 8,953,803,264 | 768 / 3168 / 6 | 106,619,904 | 1.190778% | 39 | 105,490,944 |

Counts are from instantiated meta-device modules, not allocated giant models.
All processors retain three layers and 128-dimensional attention heads.
Vector readouts add parameters, so the 4B/9B processor FFNs change from
2592/3072 to 2720/3168 to preserve the target fraction. These are the closest
multiples of 32; relative target errors are +0.09943% and −0.08545%. The 2B
hidden/FFN geometry is unchanged. The 4B reader is already larger because it
accesses more native linear-state heads: scaling hidden width alone would
overshoot the parameter budget.
KV head selections are 3/8/8; linear-state head selections are 72/192/192,
preserving one-quarter of each native head population. Time compression,
cadence 64, top-k history 32, H=50, train/eval K=4, R=1, training perturbation 0.1,
live zero-write outer-product initialization, and deterministic inference remain
fixed. Equal parameter fraction still does not mean equal processor compute.

LoRA uses the nearest integer rank to each CP budget, `alpha=rank`, dropout zero,
and the existing twelve named full/linear-attention and MLP projections. It does
not target embeddings, norms, or convolutions and is not KV-only LoRA. Report the
remaining count mismatch (largest approximately 1.57% at 2B), not exact equality.
The revised 4B match is rank 25 rather than 24; 2B/9B remain rank 21/39.

### Verified upstream geometry

Configuration JSONs were read from these upstream revisions; weights were not
downloaded. Metadata records these as **pretrained origins**, not the identity of
the stage-two SFT parent.

| Model | Revision | Hidden / FFN / layers | Full Q/KV heads | Linear K/V heads | Tied embeddings |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 2048 / 6144 / 24 | 8 / 2 | 16 / 16 | Yes |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 2560 / 9216 / 32 | 16 / 4 | 16 / 32 | Yes |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 4096 / 12288 / 32 | 16 / 4 | 16 / 32 | No |

All use head dimension 256, linear key/value dimension 128, convolution width 4,
and every fourth layer is full attention. The model's positional capacity remains
262,144; that is **not** the training sequence cap.

## Dataset audit and proposed preparation

As checked on September 8, 2026, the current cluster file
`/mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl` has **1,000,000 lines**
and **1,392,578,835 bytes**. Its directory contains no preparation manifest or
full parquet shards. Its records expose `prompt`/`answer`, losing original source
columns. A historical manifest records its checksum, but byte identity was not
rehashed during this inventory. Its relationship to official `train_1M` is
**unproven**; do not relabel it as that fair-downsampled split.

The prior recipe skips 128,000 rows and takes 128,000, then performs 2,000 steps
at batch 128: **256,000 presentations over that 128,000-row selection**. This is
nominally two passes over the selection, not the local million or upstream full
corpus. Multiple rows may have the same problem.

At upstream revision `469216e3f46f4dacf476b382e192485ea51a143e`:

| Official split | Solution rows |
|---|---:|
| `train` | 13,972,791 |
| `train_1M` | 1,000,000 |
| `train_2M` | 2,000,000 |
| `train_5M` | 5,000,000 |

The smaller splits are overlapping downsampled alternatives, not additional data.
The full release contains approximately 607.3K distinct questions according to
the paper's rounded table; **14M means solutions, not unique problems**. There is
no official validation/test split. Full shards were not found in the targeted
cluster/cache inventory; this is not a claim about every cluster directory.

The draft path below intentionally differs from the historical raw file and
**has not been created**:

```text
/mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2-headline-v1/train.jsonl
```

Proposed preparation contract, requiring a separate authorized preparation step:

1. Pin `train_2M` at the revision above. Preserve source identity and exclusions;
   map `problem` to `prompt`, `generated_solution` to `answer`, not
   `expected_answer` alone. Metadata in the recipes is a **planned** source,
   not a completed data receipt.
2. Allocate development/test by normalized **problem groups before selecting
   solution rows**. Keep all answers to a held-out problem outside training;
   exclude known historical tuning/probe exposure from confirmatory holdouts.
   Audit benchmark contamination and near duplicates. Do not reuse the earlier
   common dev/test as untouched confirmation after expanding into their source.
3. Audit rendered examples under the pinned tokenizers and the 2,048 cap. Reject
   malformed/empty-answer/zero-target rows. Retain only examples whose full prompt,
   assistant response, and terminal marker fit every size's cap; do not silently
   right-truncate the reasoning or answer. Report length exclusions and use one
   common eligible training population across the sizes.
4. Select **exactly 1,024,000 eligible rows**, with a documented problem-balanced
   policy and deterministic pre-shuffle. Publish A/B selection/order once; their
   row ranges alone do not prove problem disjointness. GCW currently walks file
   order rather than reshuffling each epoch.
5. Record source revision, transforms, retained/excluded counts, file hashes,
   problem overlaps, answer-token statistics, and stage boundaries. If filtering
   cannot supply the declared count, change the budget consistently in all files;
   do not silently cycle a shorter file or substitute the old million-row input.

### Can these run in epochs?

GCW currently stops by **`max_steps`**, not an `epochs` YAML field. The loader
tracks a cursor/epoch and cycles the selected rows. With global batch 128:

- 512,000 selected rows / 128 = **4,000 steps per pass**.
- 1,024,000 selected rows / 128 = **8,000 steps per pass**.
- Unfiltered full `train` / 128 ≈ **109,163 steps per pass**, with tail-policy
  rounding; the last batch can repeat prefix rows. Filtering changes this count.

The proposed prepared-subset pass is **2.097B nominal token slots**, 16 times the
old 2,000×128×512 budget. Actual unpadded and supervised tokens are smaller and
must be measured. Full-data epochs would be another substantial scale increase;
start with the prepared-subset comparison, measure throughput/held-out behavior,
and budget 2M/5M/full-data extensions separately. Do not assume the matrix fits in
the two separate two-GPU/five-day allocations. Each recipe uses one allocation;
these jobs are not a single four-rank distributed run.

## What the bottleneck reference actually establishes

The reference lives in `src/bottleneck_llm/config/` (singular), at submodule
commit `8f66f71`. Its OpenMath processor recipes load native SFT checkpoints
through `backbone_path`, freeze the backbone, and start fresh optimizer state.
This is a precedent for **SFT → processor**, not evidence that a half/half split
wins for Qwen3.5 or Diffusion–Longshot.

For example, `Qwen3-0.6B-bottleneck-v2.yaml` loads a `checkpoint-1000` SFT backbone
and requests one epoch over a 128K selection; the 1B recipe requests two epochs
and `next_data: true`. The 3B/8B processor recipes also reference SFT checkpoints.
Most corresponding OpenMath settings use training length **512**, global batch
128 and LR **1e-4**; evaluation length 2048 is not their training length. Some
baseline truncations are absent, and not all stage-two selections are disjoint.
The code also has an unused `end_step` path and nonfunctional unfreeze callback
logic, so individual YAML fields are not proof of executed budgets.

These are configuration/code observations, not audited paper-result receipts.
The submodule contains no paper manuscript. Do not claim that it proves the
proposed method will beat continued SFT or that it already implements our gates,
constant processor fraction, chat format, or diffusion schedule.

## Preflight and launch after approval

Before launching any draft:

1. Confirm the subset/context/precision/stage budget for the selected runs.
   The single seed and contextual vector writeback are configured, but their
   actual two-rank execution still needs qualification.
2. Materialize and authenticate the prepared data/holdouts. Prepare native 4B/9B
   checkpoints from pinned upstream snapshots. Only the checked 2B native input
   already exists at its configured path. Do not download/convert mutable `main`
   and then label it as the pinned revision.
3. Synchronize a frozen source/dependency snapshot to the allocation; authenticate
   all loaded inputs and use fresh output directories/W&B names. The recipes use
   explicit cluster paths, not portable local defaults.
4. Qualify forward/backward, gradients through recurrent state, optimizer changes,
   checkpoint save/load, and multi-rank parameter agreement at the actual size,
   sequence cap, and physical batch. Check effective FLA/causal-conv/Liger backends
   rather than treating an available import or a silent fallback as qualification.
5. Measure peak memory and step time before budgeting the whole matrix. Freeze the
   evaluation protocol: common native backend, tokenizer/chat/EOS policy, context
   and generation limits, deterministic K=4, complete task counts, and a preregistered
   primary metric. Match stopping rather than mixing historical HF/native scores.

For an already qualified and synchronized environment **inside one two-GPU job**:

```bash
uv run --locked --no-sync --extra gpu --extra qwen35 \
  torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m gcw.cli train --config configs/train/qwen35_headline/2b/sft.yaml
```

This launches two workers; a plain `gcw train` command does not. Four CPUs and
90G host RAM per allocation do not establish GPU fit, especially for 4B/9B full
SFT; qualify each size independently rather than assuming the larger drafts fit. The allocation
holding process does not run training automatically. Stage-two files require
successful same-size `sft_stage1.yaml` completion first. This document is not a
launcher and authorizes no jobs.

Sources: [pinned OpenMathInstruct-2 card](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2/blob/469216e3f46f4dacf476b382e192485ea51a143e/README.md),
[OpenMathInstruct-2 paper](https://arxiv.org/html/2410.01560v2),
[Qwen3.5-2B config](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json),
[Qwen3.5-4B config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json),
[Qwen3.5-9B config](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json).
