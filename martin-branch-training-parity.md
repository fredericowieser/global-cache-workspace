# `martin` branch training parity

**Status:** audited 2026-08-19 against `martin@a2fb602` and `dev@209e105`; findings 2 and 3 fixed 2026-08-20.
Configuration parity is closed. Two code-level divergences remain open (findings 4 and 5), neither of which
affects a currently used config.

See [Change log](#change-log) for what has been closed since the initial audit.

Evaluation parity was closed by `209e105` ("add legacy evaluation of previous checkpoints") and confirmed
byte-identical through the `paper7-legacy` suite. This page covers the remaining question: which parameters
recover the `martin` *training* modes, and what still differs in ways that change training dynamics.

The bar is exact reproducibility, not rough equivalence — with one deliberate exception. Where `martin` was
simply wrong, `dev` keeps the correct behaviour and this page records the difference instead of preserving it.
[Finding 1](#1-sft-rank-sharding--martin-bug-no-legacy-support) is the only such case.

## Method

Three independent checks, because a source diff alone cannot establish numerical equivalence across a
package-layout refactor (`martin:gcw/` to `dev:src/gcw/`).

1. **Configuration resolution diff.** Every one of the 94 `martin` train configs loaded through both branches'
   `load_train_config`, then flattened and compared field by field.
2. **Source read of the training path.** `martin:gcw/train.py` against `dev:src/gcw/training/{core,assembly,model_setup,optimizer,precision,distributed,objectives}.py`.
3. **Same-seed numerical probe.** Build an identical model on each branch, `torch.manual_seed(24)`,
   `init_weights()`, hash all parameters, run one forward/backward on a fixed batch, hash all gradients. See
   [Reproducing the audit](#reproducing-the-audit).

## Verdict

All 94 `martin` train configs load unmodified on `dev`, and every knob added by `dev` resolves to the
`martin`-equivalent value with no edits. Nothing needs to be set to recover the training modes.

| Variant | Init weights | Gradients | Loss |
|---|---|---|---|
| TF | identical | identical | identical |
| BT | identical | identical | identical |
| BGT | identical | identical | 7.6e-8 relative |
| Icarus | identical | identical | 7.6e-8 relative |
| Longshot (`query_attention`) | identical | ~1 ulp | 1.5e-7 relative |

SFT batch tensors (`x`, `y`, attention mask) are bit-identical on every variant.

Only TF and BT reproduce bit-exactly. BGT, Icarus, and Longshot carry float32 reassociation noise; see
[finding 6](#6-loss-reduction-reassociation--closed-dev-behaviour-kept).

**The audit is closed.** All ten findings are resolved: 2, 3, 7, 8 and 9 by changes to `dev`; 1, 4, 5, 6 and 10
by a deliberate decision to keep `dev`'s behaviour, each recorded in its section and summarized under
[Status](#status). The two behavioural
consequences to remember when comparing against `martin` results are that pretraining runs with a nonzero
`warmdown_ratio` are not step-for-step reproducible ([finding 4](#4-_wsd_lr-warmdown-off-by-one--closed-dev-behaviour-kept)),
and that non-TF/BT comparisons hold to about 1e-7 rather than bitwise
([finding 6](#6-loss-reduction-reassociation--closed-dev-behaviour-kept)).

## Parameters that recover the `martin` training modes

Every value below is now pinned explicitly in all 67 `configs/train/bottleneck_llm` configs rather than left to
`dev`'s defaults. Pinning was verified inert: the resolved config is unchanged for 67/67 (see
[Change log](#change-log)).

Defaults are not a safe carrier for the legacy mode. `dev` adds execution policy continuously, and the nine
drifted Longshot configs in [finding 7](#7-config-drift-and-missing-configs--closed) show the same class of
silent regression reaching real configs. An explicit value fails loudly on a rename and shows up in the run
manifest; an omitted key silently follows whatever the default becomes.

| Parameter | Value | `martin` equivalent |
|---|---|---|
| `data.sft_chat_template` | `False` | `martin` never applied a chat template in SFT; `sft_batch` did `encode(prompt) + encode(response_prefix + answer) + [eot_id]` |
| `data.sft_enable_thinking` | `False` | no such concept |
| `data.sft_loss_scope` | `"answer_only"` | hardcoded `[-100] * len(prompt_ids) + answer_ids` |
| `data.prompt_suffix` | `""` | no such concept |
| `data.response_prefix` | from YAML (`"\n"` on the SFT configs) | same field |
| `data.validation_dataset` | `None` | `martin`'s `_eval_loss` used the random loader for non-pretrain modes; `dev` matches when this is `None` |
| `loss_backend` | `"reference"` | `martin`'s `loss_from_hidden`, full-vocabulary projection |
| `performance.reference_loss.mode` | `"legacy"` | same |
| `trainable` | derived from `freeze_backbone` | exact match on 94/94 |
| `save_cache_processor_only` | derived from `freeze_backbone` | `martin` passed `cache_processor_only=config.freeze_backbone`; exact match on 94/94 |
| `model.legacy_bottleneck_generate` | still accepted, aliases to `processor_generation_mode` | correct on 94/94 (51 to `post_prefill_relative`, 43 to `windowed`) |
| `activation_checkpointing` | `"none"` | no such concept |
| `backbone_chunk_backend` | `"eager"` | no such concept |
| `activation_checkpoint_windows` | `0` | no such concept |
| `activation_checkpoint_min_padded_sequence_length` | `0` | no such concept |
| `longshot_gradient_checkpointing_seq_len_threshold` | `null` | `martin` used `model.cache_processor.gradient_checkpointing` alone |
| `global_batch_rows` | `0` | `martin` sized steps by `total_batch_size` only |
| `logical_pair_grouping` | `false` | no such concept |
| `init_checkpoint` | `null` | no such concept |
| `resume` | `null` | no such concept |
| `save_final` | `true` | `martin` always saved the final step when it was not a `save_every` boundary |
| `performance.adaptive_batching.enabled` | `false` | no such concept |
| `performance.longshot_full_history.enabled` | `false` | no such concept |
| `performance.kv_score_sidecar.enabled` | `false` | no such concept |
| `performance.triton_cache_movement.enabled` | `false` | no such concept |

Deliberately not pinned:

- `model.processor_generation_mode`. `martin`'s own `model.legacy_bottleneck_generate` is already explicit in the
  configs that need it, and `dev` honours it as an alias that overrides `processor_generation_mode`. Pinning both
  would create two sources of truth for one setting. It affects generation only, not training.
- `data.canonical_execution` and the `expected_*_sha256` / `experiment_manifest` provenance gates. These add
  validation rather than changing training dynamics.
- `loss`. Dead field on both branches.

Two cosmetic resolution diffs, neither with any effect:

- `loss`: `'next_token_ce'` to `None`. Dead field, referenced nowhere on either branch.
- `model.legacy_bottleneck_generate`: `False` to `None`, but correctly translated to `processor_generation_mode`.

## Divergences

### 1. SFT rank sharding — `martin` bug, no legacy support

**Affects 75/94 configs (56 DDP + 19 FSDP SFT).** `dev`'s behaviour is correct and stays as-is. This entry exists
to record that `martin` multi-GPU SFT results are not like-for-like comparable with `dev` results, not as work to
be done.

`martin:gcw/train.py:_loader` had no rank offset: `rows[(i + j) % len(rows)]`. Every rank received the same
rows. `src/gcw/data/training.py:278` shards rank-disjoint.

The evidence that this is an oversight rather than a deliberate choice is internal to `martin`. Its FineWeb
loader shards correctly in the same codebase — `gcw/data/shards.py:85` computes
`rank_offset = rank * batch_size * sequence_len` — while the SFT branch of `_loader` never references `rank` and
is not passed it. No launcher compensates through `sft_skip_rows`. Pretraining was sharded; SFT was added later
without it.

It is close to invisible in practice, which is why it survived: the runs train normally, losses look ordinary,
and reported throughput is unaffected. The only symptom is that the run consumes `world_size` times fewer unique
rows than its own token accounting claims while duplicating that work across every rank.

Demonstrated against a 64-row dataset, `batch_size=4`:

```
world_size=2, rank=0   martin -> [0-3] [4-7] [8-11]     dev -> [0-3]  [8-11]  [16-19]
world_size=2, rank=1   martin -> [0-3] [4-7] [8-11]     dev -> [4-7]  [12-15] [20-23]
world_size=1           martin == dev (identical)
```

Consequences for a `martin` multi-GPU SFT run: the gradient was an average over `world_size` identical
replicas, making the step mathematically equivalent to a single-GPU step, and the run consumed `world_size`
times fewer unique rows than its own token accounting claimed. `martin` computed
`micro_tokens = batch_size * sequence_len * world_size` but never sharded.

For `qwen35_27b_longshot_from_base.yaml` (8 GPUs, `sft_take_rows: 128000`, 3000 steps): `martin` consumed
about 48k rows, roughly 0.4 epochs; `dev` consumes about 384k rows, roughly 3 epochs.

This is the divergence that will visibly move SFT curves.

**No legacy flag is needed, because an exact equivalence already exists.** Since every `martin` rank computed the
same batch, DDP averaged `world_size` identical gradients, so a `martin` run reduces exactly to a single-process
run. Re-running the same config on one process reproduces it:

> A `martin` run with `world_size` W, `batch_size` B, and `total_batch_size` T is numerically identical to a
> `dev` run with **one process**, the same B, and `total_batch_size` T/W.

The gradient-accumulation arithmetic lands on the same value from both sides — `martin` computes
`grad_accum = T / (B * S * W)` and the one-process `dev` run computes `(T/W) / (B * S)` — so both consume the
same rows in the same order and average over the same number of micro-batches. `dev`'s one-process row order was
verified byte-identical to `martin`'s in the table above.

For `qwen35_2b_longshot_from_sft_step2000_frozen_shared` (W=2, B=16, S=512, T=65536): `martin` ran
`grad_accum = 4` over 64 unique rows per step. The equivalent `dev` command is one process, `batch_size: 16`,
`total_batch_size: 32768`.

Reported `train/tokens_per_step` and epoch counts will differ under this recipe, because `martin`'s were the
inflated ones. The optimizer trajectory is what matches.

### 2. `precision: bf16` non-FSDP trained in fp32 — CLOSED

`src/gcw/training/assembly.py` built the non-FSDP model as `build_model(config.model).to(device)` with no dtype,
and `bf16` disables autocast by design, so the run was fp32 end to end. `_build_training_model`, which does
apply `model_dtype`, was only reached on the FSDP branch.

```
martin      non-FSDP, precision=bf16 -> param dtypes: ['torch.bfloat16']
dev before  non-FSDP, precision=bf16 -> param dtypes: ['torch.float32']
dev after   non-FSDP, precision=bf16 -> param dtypes: ['torch.bfloat16']
```

Fixed by casting after the model is built and any checkpoint loaded, matching `martin`'s ordering. A 2-step
CPU run now saves bfloat16 weights and bfloat16 Adam moments. `bf16_mixed` and `fp32` were unaffected and
still resolve to float32 parameters.

This was a plain `dev` bug rather than a parity-only issue: no shipped config hit it, because both `bf16`
configs are `distributed: fsdp`, but any single-GPU or DDP `bf16` run silently got full fp32 with no warning.

### 3. `precision: bf16` FSDP initialized in bf16 — CLOSED

`src/gcw/training/model_setup.py` cast to `model_dtype` *before* `init_weights()`; `martin` cast after. RNG
draws therefore happened in bf16 rather than fp32-then-round, which is not the same distribution of values.
Confirmed on the audit model: init digests `8edfce53...` (`martin`) versus `17d4f4ac...` (`dev`).

Fixed by moving the cast to the end of `_build_training_model` on both the FSDP rank-0 branch and the non-FSDP
branch, so weights are drawn in fp32 and rounded once. Re-verified on `smoke_tf`: `martin` ordering and both
`dev` branches now agree at `a1926728...`, against `d49cceeba...` for the previous ordering.

This mattered most for `qwen35_27b_longshot_from_base.yaml`, where the backbone loads from a checkpoint but the
cache-processor weights come from `init_weights()`, so the ordering set that run's starting point directly.

**Deliberately not changed.** `src/gcw/training/distributed.py:33` still wraps `precision: bf16` in
`MixedPrecision(bf16, bf16, bf16)` where `martin` passed `None` for anything that was not `bf16_mixed`. This is
inert for every family, not just the ones in use: `_build_training_model` ends in `model.to(dtype=model_dtype)`,
and `nn.Module.to` casts floating buffers as well as parameters, so under `precision: bf16` the whole module is
already bfloat16 before FSDP sees it. `param_dtype` and `buffer_dtype` are then no-ops and `reduce_dtype=bf16`
is what FSDP does anyway at that parameter dtype. Verified across `smollm`, `llama`, `qwen3`, `qwen3_5`, and
`gemma4` — see [finding 8](#8-precision-modes-were-declared-but-not-supported--closed).

`bf16` deliberately still means no fp32 master weights anywhere: bf16 parameters, bf16 optimizer states. That is
what `martin` did and what makes it memory-cheap. Anything that keeps fp32 masters should be a new precision
mode, not a redefinition of this one.

### 4. `_wsd_lr` warmdown off-by-one — CLOSED, `dev` behaviour kept

**19 configs set a nonzero `warmdown_ratio`.**

`src/gcw/training/optimizer.py:29` changed both the boundary (`step <= max_steps - warmdown` to `step <`) and
the interpolation (`(max_steps - step) / warmdown` to `(max_steps - 1 - step) / (warmdown - 1)`).

The boundary change is value-neutral. For any `warmdown >= 2`, `dev` returns exactly `1.0` at
`step == max_steps - warmdown`, the same as `martin`. An earlier version of this entry claimed `dev` begins the
warmdown one step earlier at `1.0`; that was wrong. The value at that step is `1.0` on both branches.

The whole difference is in the tail. `martin` spreads the decay over `warmdown` intervals but stops one
increment short of the floor; `dev` spreads it over `warmdown - 1` and lands on it:

| `max_steps` | `warmdown_ratio` | `final_lr_frac` | last-step `lr_scale` `martin` | `dev` | max delta | total LR mass |
|---|---|---|---|---|---|---|
| 38147 | 0.2 | 0.0 | 0.000131 | 0.0 | 1.31e-4 | -0.0015% |
| 1000 | 0.2 | 0.0 | 0.005 | 0.0 | 5.0e-3 | -0.077% |
| 1000 | 0.1 | 0.1 | 0.109 | 0.1 | 9.0e-3 | -0.047% |

The mass difference is exactly half a step of peak LR in every case.

The 85 configs with `warmdown_ratio: 0.0` are unaffected: `warmdown == 0` and both branches return `1.0` at
every step. The 19 affected are the SmolLM-360M FineWeb pretraining configs, three Qwen3-0.6B diffusion
configs, `sft_template.yaml`, and one OpenMath explore config.

**Decision (2026-08-20): keep `dev`.** `martin`'s schedule never reaches `final_lr_frac`, which is the point of
the decay phase — fourteen of the nineteen set `final_lr_frac: 0.0` and would still be training at
`peak/warmdown` on the final step. `martin` also silently produces no warmdown at all when
`round(warmdown_ratio * max_steps) == 1`, because `step <= max_steps - 1` is true at the last step. `dev`'s
form is a clean interpolation from `1.0` to `final_lr_frac` and guards the `warmdown - 1` division. `martin`
has no advantage here beyond matching checkpoints already trained. The consequence is accepted: pretraining
runs started on `martin` are not step-for-step reproducible on `dev`, though the aggregate difference is far
below seed noise.

### 5. Longshot `_fit_time_length` pads the opposite side — CLOSED, `dev` behaviour kept

- `martin`: `out = x.new_zeros(...); out[:, :, :current, :] = x` — pad on the recent side.
- `dev` `src/gcw/models/cache_processors/longshot_layers.py:79`: `F.pad(x, (0, 0, pad, 0))` — pad on the oldest
  side.

Found because the first Longshot probe showed structurally different gradients. Localized with forward hooks:
`in_norm` output was bit-identical, but `in_proj`'s input had an identical norm with every element different —
a pure permutation of the feature axis, same multiset in a different order. The `in_proj` weight columns
therefore see real data where `martin` saw zeros and vice versa.

#### When the padding path runs

The time axis handed to the linear encoder is always

```
T = min(cache_len, process_every + topk)
```

from `select_l2_indices` (`src/gcw/models/cache_processors/state.py:71`) and its per-cache-head twin
`select_per_cache_head_kv_norm_indices_from_scores` (`src/gcw/models/cache_processors/ops.py:64`). Both take
`min(process_every, seq)` recent positions plus `min(topk, seq - recent)` history positions, so neither can
return more than `process_every + topk`, nor more than the cache holds. The processor is invoked at
`cache_len = process_every, 2*process_every, 3*process_every, ...`, and `encoder_seq_len` defaults to
`process_every + topk` (`src/gcw/models/cache_processors/longshot_processor.py:55`).

Padding therefore fires iff `min(n * process_every, process_every + topk) < encoder_seq_len`. Measured by
instrumenting every call across a full forward:

| `process_every` | `topk` | `encoder_seq_len` | calls `(cache_len -> target)` | padded |
|---|---|---|---|---|
| 6 | 0 | 6 (`smoke_longshot`) | `(6,6) (6,6)` | 0/2 |
| 6 | 3 | default 9 | `(6,9) (9,9) (9,9) (9,9) (9,9)` | 1/5 |
| 64 | 32 | default 96 | `(64,96) (96,96) (96,96)` | 1/3 |
| 6 | 3 | 20 (oversized) | `(6,20) (9,20) (9,20) (9,20) (9,20)` | 5/5 |

With the default `encoder_seq_len` this is exactly one call per sequence — the first — padding exactly `topk`
positions. It is the cache cold start, not an edge case: at the first invocation the cache holds only
`process_every` positions, so there is no history for `topk` to select from. At `process_every: 64, topk: 32`
that zeroes a third of the encoder window on that call. An `encoder_seq_len` above `process_every + topk` pads
every call forever; nothing validates against that today.

#### Scope

`_fit_time_length` is only called on the `encoder_method: linear` branch, and only two configs set it:

- `configs/train/smoke_longshot.yaml` — `process_every: 6`, `topk: 0`, `encoder_seq_len: 6`. Never pads;
  running it on both branches gives an identical loss to 10 decimal places.
- `configs/model/smollm360m_longshot.yaml` — `process_every: 64`, `topk: 32`, `encoder_seq_len: null` -> 96.
  Would pad the first call by 32, but it is referenced by no train config, doc, or script.

All 23 real Longshot configs use `encoder_method: query_attention`, which has no fixed `in_features` and so
never needs a fixed time length. Nothing already trained is affected by the pad side.

**Decision (2026-08-20): keep `dev`.** `dev`'s pad side is the correct one independently of parity. The trim
branch is `x[:, :, -target_len:, :]`, which is right-aligned; `F.pad(x, (0, 0, pad, 0))` is right-aligned too,
so a given `in_proj` weight column means "the k-th most recent position" on every call. `martin` left-aligns
the pad against a right-aligned trim, so on the cold-start call slot *i* denotes a different position than on
every other call. That is a `martin` bug, not a convention difference.

Three follow-ups were considered and declined as out of scope for the merge: rejecting
`encoder_seq_len > process_every + topk` in `CacheProcessorConfig.__post_init__`; restoring `martin`'s
`warnings.warn` on the padding path, which is what would have surfaced this; and deleting or wiring up the
orphan `smollm360m_longshot.yaml`. For the linear encoder with `topk > 0`, one padded call is in any case
structurally unavoidable — `process_every + topk` real positions cannot exist when only `process_every` do —
so the only true never-pad settings are `encoder_seq_len <= process_every` (which costs the `topk` history) or
skipping the first processor call (which changes semantics).

### 6. Loss reduction reassociation — CLOSED, `dev` behaviour kept

`martin`'s `F.cross_entropy(..., reduction="sum")` became `reduction="none"` followed by `.sum()` in
`src/gcw/models/backbones/transformer/losses.py`. Mathematically identical, different float32 accumulation
order. Measured on CPU over 20 trials at 4096x2048: max relative difference 1.4e-7.

This is not a regression. `martin`'s chunked `loss_from_hidden` (`gcw/models/transformer.py:453`) raises
`ValueError` for `reduction="none"` — the chunked path structurally cannot emit per-token or per-row losses.
`dev` needs both: `src/gcw/training/objectives.py` consumes `row_sum` for the paired and adjacent-pair
objectives and for `reference_loss.mode`, and `src/gcw/evaluation/scoring.py` consumes `none`.
`answer_only_loss_from_hidden` has no `martin` equivalent either.

**Decision (2026-08-20): keep `dev`.** Neither form is more accurate — both are float32 tree reductions with no
systematic bias, and the difference sits at the level of float32 rounding. `martin`'s is marginally cheaper (no
per-token vector materialized per chunk), but the saving is kilobytes against logits the backward pass already
holds. Its only real advantage is bitwise equality with old BGT/Icarus/Longshot runs. Record that non-TF/BT
comparisons across the two branches are valid to about 1e-7, not bitwise; TF and BT are unaffected and remain
bit-identical end to end.

A middle option was considered and rejected: branch so that `sum`/`mean` call cross-entropy with
`reduction="sum"` and only `none`/`row_sum` take the per-token path. That would restore bitwise equality on
every `martin`-reachable mode, but it makes `sum` and `row_sum` disagree in the last bits with each other,
which is a worse property than disagreeing with an archived branch.

### 7. Config drift and missing configs — CLOSED

Resolved by restoring 11 files from `martin`. `configs/train` is now byte-identical to `martin`, with an exact
file-set match and no `dev`-only additions.

Two configs were absent:

- `configs/train/bottleneck_llm/qwen35_27b_instruct_tf_sft.yaml`
- `configs/train/bottleneck_llm/qwen35_27b_longshot_from_base.yaml`

Nine Qwen3.5 Longshot configs had silently regressed to pre-final values, losing `martin`'s later tuning
commits (`5877b44`, `1a9a911`, `4ac7d20`, `b5cc1c5`):

| Config | Drifted keys |
|---|---|
| `qwen35_2b_longshot_from_sft_step2000_frozen_shared` | processor `hidden_size`, `intermediate_size`, `num_heads`, `num_kv_heads`, `linear_state_query_mode`, `linear_state_update_mode`, `save_every`, metadata |
| `qwen35_2b_instruct_longshot_from_sft_step2000_frozen_shared` | as above |
| `qwen35_2b_longshot_cp_frac005_…`, `…_frac010_…` | as above, plus `num_layers` on `frac010` |
| `qwen35_2b_instruct_longshot_cp_frac005_…`, `…_frac010_…` | as above |
| `qwen35_4b_base_longshot_from_sft_step2000_frozen_shared` | as above, plus `distributed` (`ddp` to `fsdp`), `batch_size` (2 to 16), `max_steps` (2000 to 3000) |
| `qwen35_4b_base_longshot_cp_frac005_…`, `…_frac010_…` | as above |

Every drifted config had `linear_state_query_mode: global` where `martin` specifies `per_linear_state`, and
omitted `linear_state_update_mode: outer_product` entirely. Any run launched from the pre-restore configs used a
different cache-processor architecture and parameter count from the `martin` results it would be compared with.

`configs/tasks/theoremqa.yaml` is not missing. It moved to `configs/tasks_legacy/paper7/theoremqa.yaml`,
identical apart from an em-dash in a comment.

The two 27B configs are the only `precision: bf16` users. Findings 2 and 3 landed on 2026-08-20, so they now
reproduce `martin`.

## Confirmed equivalent

Checked, no action required.

- FineWeb `data_loader`: byte-identical, same rank offsets and shard ordering.
- `FineWebBatchStream` ordering matches `data_loader`.
- `_build_optimizer` param grouping (`ndim >= 2` decay split) and `initial_lr` capture.
- `_weight_decay` cosine schedule.
- `_clip_grad_norm`, including the FSDP branch.
- Gradient-accumulation normalization and `no_sync` placement on the final micro-step.
- `_autocast_kwargs` against `martin`'s `_uses_bf16_autocast` for `fp32`, `bf16_mixed`, and `bf16`.
- `_validate_precision`.
- `init_distributed` and `cleanup_distributed`.
- FSDP wrap classes, `param_init_fn`, `sync_module_states`, `use_orig_params`.
- `GCW_FSDP_ACTIVATION_CHECKPOINTING` opt-out, `GCW_COMPILE` contract,
  `GCW_KEEP_STEP_CHECKPOINTS` / `GCW_STEP_CHECKPOINT_EVERY`.
- `load_backbone_checkpoint`: adds only a LoRA carve-out, otherwise identical key resolution and strictness.
- `CacheProcessorConfig`: every field default identical, including the `gated_delta` and `linear_state_cache`
  additions from `martin`.
- `torch.manual_seed(config.seed)` placement relative to model construction.

One cosmetic note: `train/epoch` now reads `stream.cursor.epoch`. `martin`'s SFT loader reported `row` and left
`train/epoch` as `None`.

### OpenMathInstruct-2 SFT data path — verified bit-identical

All 67 `configs/train/bottleneck_llm` configs are `mode: sft` over
`OpenMathInstruct-2/train.jsonl`, with `sft_skip_rows: 128000`, `sft_take_rows: 128000`,
`response_prefix: "\n"`, `sft_loss_scope: answer_only`, `sft_chat_template: false`,
`prompt_suffix: ""`, `validation_dataset: null`, `logical_pair_grouping: false`,
`loss_backend: reference`, and no activation checkpointing. Under exactly those values every
`dev`-only knob in the SFT path is inert and the code collapses to `martin`'s.

`dev` rewrote both halves of the path, so both were checked against the real dataset and the real
tokenizer rather than by reading:

- **Row selection.** `martin` materialized the file with `list(iter_jsonl(...))` then sliced
  `rows[skip:][:take]`. `dev` uses `IndexedJSONL` (byte-offset index, rows parsed on demand) wrapped in
  `_RowSlice(start, start + take)`. Both skip blank lines on the same predicate, so the row at a given
  index is the same row.
- **Batch order.** `martin` advanced an unbounded `i` and read `rows[(i + j) % len(rows)]`; `dev` keeps an
  `SFTCursor` storing `row = absolute_next % len(rows)` and reads `rows[(cursor.row + offset) % len(rows)]`.
  Algebraically the same index, including across the wrap. `drop_last` is `config.logical_pair_grouping`,
  which is `false` in all 67, so the wrap-and-truncate branch never runs.
- **Tokenization and masking.** With `chat_template: false` and an empty `prompt_suffix`, `dev`'s
  `sft_batch` takes the `else` branch and produces `prompt_ids + answer_ids` exactly as `martin` did; with
  `loss_scope: answer_only` its `(prefix_mask + labels)[:len(seq)]` is literally `martin`'s
  `([-100] * len(prompt_ids) + answer_ids)[:len(seq)]`. Padding, width, and attention mask are unchanged.

Probe (`/tmp/gcwcmp/loader_probe.py`), real `train.jsonl` and the Llama-3.2-1B SFT tokenizer, `rank=0`,
`world_size=1`, `batch_size=16`, `sequence_len=512`: SHA-256 of `x`, `y`, and `attention_mask` match for
every batch.

| Rows taken | Batches | Result |
|---|---|---|
| 128000 (the production slice) | 8 | identical |
| 100 (forces epoch wrap on a batch boundary) | 25 | identical |
| 97 (forces a wrap mid-batch) | 25 | identical |

Full step (`/tmp/gcwcmp/sft_step_probe.py`), same real rows and tokenizer through a seeded tiny `tf`
model, `loss_reduction="mean"` with the attention mask:

| Quantity | Value on both branches |
|---|---|
| `x` / `y` digest | `052a480601f2d4cd` / `878c71601aa75816` |
| unpadded tokens / supervised tokens | 1384 / 1160 |
| init parameter digest | `3ba774a5048214c0` |
| loss | `11.79229736328125` |
| gradient digest | `df4b7960910e1db2` |
| gradient norm | `0.83220523595809937` |

Bit-identical, consistent with finding 6 recording `tf` as exact end to end. On a cache-processor family
the same probe would differ in the last bits for the reason given there.

Two caveats, neither a `dev` regression:

- Rank sharding (finding 1) still applies. These runs were multi-GPU, so reproducing an old one means a
  one-process `dev` run with `total_batch_size` divided by the original `world_size`.
- `qwen35_27b_instruct_tf_sft.yaml` and `qwen35_27b_longshot_from_base.yaml` carried `martin`'s
  `/mnt/gcw/...` prefix for both `data.dataset` and `backbone_checkpoint`, while the other 65 use
  `/mnt/jfs-01/shared/gcw/...`. `/mnt/gcw` does not exist on this machine at all, so those two configs
  could not run. Realigned to the shared prefix on 2026-08-20. The 27B backbone is still not present under
  `checkpoints/hf_native/`, which is a separate download, not a path problem.

### 8. Precision modes were declared but not supported — CLOSED

Found while auditing the precision matrix beyond the configs actually in use.

This is a `dev` widening, not a `dev` cleanup. `martin:gcw/config.py:27` already read
`TrainingPrecision = Literal["fp32", "bf16_mixed", "bf16"]`. `dev` extended the `Literal` to six modes —
adding `auto`, `fp16`, `fp16_mixed` — without extending `TrainConfig.__post_init__`, which still accepted only
three. The three additions were type-legal and runtime-rejected, and live branches for them survived in
`_autocast_kwargs`, `_model_dtype`, and the FSDP `MixedPrecision` condition. Nothing could reach that code.
`fp16` could not have worked in any case: there is no `GradScaler` anywhere in the package, so fp16 gradients
would underflow silently.

Closed by restoring `martin`'s three-mode `Literal`, deleting the unreachable branches, and deriving the
validation error from the type so the two cannot drift apart again.

The legacy `bf16: bool` alias is unchanged and resolves identically on both branches for all eight
`(bf16, precision)` combinations: `bf16: true` with no `precision` gives `bf16_mixed`, `bf16: false` gives
`fp32`, and `bf16: true` with `precision: fp32` raises. Note that `__post_init__` ends with
`self.bf16 = self.precision != "fp32"`, so after construction `bf16` means "not fp32" rather than "mixed" —
that coercion is what [finding 9](#9-checkpoint-metadata-could-not-distinguish-bf16-from-bf16_mixed--closed)
tripped over.

Behaviour across the full matrix, verified by building each combination rather than reading the code
(`smollm`, `llama`, `qwen3`, `qwen3_5`, `gemma4` x `fp32`, `bf16_mixed`, `bf16`):

| precision | parameters | float buffers | compute |
|---|---|---|---|
| `fp32` | float32 | float32 | float32 |
| `bf16_mixed` | float32 | float32 | autocast bfloat16 (CUDA only) |
| `bf16` | bfloat16 | bfloat16 | bfloat16 |

Uniform across all five families, with no per-family exceptions.

### 9. Checkpoint metadata could not distinguish `bf16` from `bf16_mixed` — CLOSED

`martin` recorded `"precision": config.precision` in `_checkpoint_metadata`. `dev` dropped it and wrote
`"dtype": "bfloat16" if config.bf16 else "float32"` instead. Because `__post_init__` sets
`self.bf16 = self.precision != "fp32"`, a `bf16_mixed` run — whose weights are float32 — recorded
`"dtype": "bfloat16"`.

The saved tensors were never actually wrong: `load_checkpoint` prefers `_infer_floating_dtype(weights)` and
falls back to the recorded name only when inference yields nothing. The damage was to provenance, which is the
one thing a cross-branch comparison depends on.

Closed by restoring `"precision"` and making `"dtype"` report the real parameter dtype.

#### Resume fingerprint — also closed

`_training_fingerprint` recorded `"bf16": config.bf16` and no `"precision"`, so a `bf16` run and a `bf16_mixed`
run produced byte-identical fingerprints and resuming across the two passed the `CheckpointCompatibilityError`
gate silently. This was `martin`'s behaviour as well.

`src/gcw/training/audit.py` now records `"precision": config.precision` in place of `"bf16"`. Because that
changes the fingerprint key set, every checkpoint written before the change would otherwise fail to resume, so
`src/gcw/training/checkpoint/resume.py` gained an explicit backwards-compatibility rule,
`_fingerprint_with_legacy_precision`. Before comparing, a manifest that carries `bf16` and no `precision` is
mapped onto the current key:

| Saved `bf16` | Current `precision` | Result |
|---|---|---|
| `true` | `bf16_mixed` or `bf16` | resumes, with a `RuntimeWarning` naming the ambiguity |
| `false` | `fp32` | resumes silently |
| `true` | `fp32` | rejected, fingerprint differs for `precision` |
| `false` | `bf16_mixed` or `bf16` | rejected, fingerprint differs for `precision` |

The legacy flag was `config.precision != "fp32"`, so it cannot separate `bf16_mixed` from `bf16`. An old
checkpoint therefore resumes into either — exactly the behaviour it was written under — and the warning says so
rather than hiding it. A manifest that already records `precision` is left untouched, so new checkpoints get the
strict comparison.

### 10. `bf16_mixed` silently degrades to fp32 off CUDA — CLOSED, `dev` behaviour kept

`_autocast_kwargs` sets `enabled: device.type == "cuda"`, so `precision: bf16_mixed` on CPU runs full fp32 with
no signal. That is `martin`'s behaviour too and the CPU test suite depends on it, so the behaviour stands. What
was missing was any way to notice. Runs now print the precision actually in effect:

```
precision: bf16_mixed (parameters float32, autocast disabled on cpu, computing in float32)
```

### 11. Icarus and Longshot processor parity — CLOSED

Method: function-by-function comparison on normalized ASTs (docstrings, annotations, and formatting
stripped), so the file reorganization drops out and only logic differences remain. `dev` splits
`martin`'s two files into `icarus.py` + `ops.py` and six `longshot_*` modules.

**Icarus reduces exactly to `martin`.** 15 of 20 `icarus.py` functions and 29 of 30 shared `ops.py`
functions are AST-identical. The six differences are all inert under `martin`'s configs: a `backend`
kwarg on `gather_per_cache_head_kv` whose `"eager"` branch is `martin`'s body verbatim;
`run_icarus_post_selection_region`, whose eager branch is literally `block(x, positions, selected_heads)`;
a `flow_config` that constructs nothing when `None`; and added type guards. Every performance gate
(`TritonCacheMovementConfig`, `KVScoreSidecarConfig`, `LongshotFullHistoryConfig`) defaults to
`enabled=False`, and `GCW_ICARUS_PROCESSOR_BACKEND` defaults to `eager`.

**Longshot: 31 identical, 19 different.** Most differences are inert — `_topk_linear_heads` (condition
inverted, same result), `LinearStateAdapter.__init__` (`int()` on an int), `_scatter_linear_state_updates`
(one guard split in two), `ProcessorAttention.__init__` (`martin` bound `self.forward` as an instance
attribute; `dev` uses a method that dispatches identically), and the recomputation guards on the two
metrics methods. `_gated_delta_linear_state_update` inlines `martin`'s
`Qwen35GatedDeltaDynamics.forward` character-for-character.

Two differences were measured rather than argued:

- **Chunked score reductions.** `dev` replaced `merged.float().pow(2).sum(-1)` and
  `merged.float().flatten(2).norm(-1)` with `chunked_fp32_squared_l2` / `chunked_fp32_l2_norm`, custom
  autograd Functions that chunk over the outer row axis and supply a hand-written `backward`.
  `/tmp/gcwcmp/score_probe.py` compares both against the eager oracle over three shapes, fp32 and bf16,
  and single- and multi-chunk caps: **max forward and gradient difference is exactly 0.0 in all 24
  cases.** The custom backward is bitwise correct.
- **Selected-head ordering.** `dev` dropped `martin`'s `.sort()`, so `_select_heads_from_head_scores`
  returns heads score-descending rather than index-ascending. Every consumer is permutation-invariant:
  `selected_heads` picks keys/values while the output is indexed by the query axis, the returned
  `cache_head_attention_scores` are indexed by absolute head, and the overlap metric is set membership.
  `/tmp/gcwcmp/order_probe.py` permutes `selected_heads` across both `include_self_head_kv` settings and
  all three `internal_iteration_score` modes: output differs by at most `1.5e-07`, and `kv_norm` and
  `cache_head_attention` scores are exactly invariant. Reduction-order noise only.

#### Fixed in `dev`

Four `dev`-only regressions found by walking every path, including ones no current config reaches:

| Issue | `martin` | `dev` before | Now |
|---|---|---|---|
| `_zero_projection` on a projection with no tensor weight | `AttributeError` | silently skipped, leaving `out_proj_init: zero` **randomly initialized** | delegates to `ops.zero_projection`, raising `TypeError` like Icarus |
| `_select_heads_from_head_scores(scores, topk <= 0)` | `assert topk > 0` | returned **every** head | raises `ValueError` |
| gated-delta head count vs dynamics | `ValueError` | dropped — cryptic broadcast, or silent broadcast at `heads == 1` | `ValueError` restored, read off `dynamics.A_log` |
| `_fit_time_length` pad/trim | `warnings.warn` | silent | warning restored |
| `gated_delta` specs without matching dynamics | `ValueError` at construction | deferred to first forward | build-time `ValueError` restored |
| `_gather_cache_heads` shapes | two named `ValueError`s | dropped | restored |

`martin` guarded the gated-delta dynamics at *both* construction and use, and `dev` keeps both. They cover
different points in the lifecycle: `refresh_linear_state_dynamics` reassigns the dict after FSDP wrapping,
long after the constructor ran, so the runtime check is the only guard against an incomplete refresh.

Eight tests added across `tests/test_longshot_attention.py` and `tests/test_models.py`. The restored
`_fit_time_length` warning now fires in `test_longshot_value_only_forward_backward`, which is the
cold-start pad of finding 5.

#### Internal re-selection realigned to `martin`

`dev` had changed `process_hidden` to re-select with separate kv and linear head budgets. That is now
reverted: inside the cache processor the re-selection is one combined `topk_heads + topk_linear_heads`
budget over the merged kv+linear pool, so the two groups compete on score, exactly as `martin` did.
Outside the processor, `_select_combined_heads` keeps the per-group budgets — also as `martin` did. The
asymmetry is deliberate: the outer selection decides how much of each *kind* of state to admit, while the
inner re-selection is free to reallocate across kinds as the iteration refines its estimate.

`process_hidden` is now AST-identical to `martin` apart from the Flow early return and two type guards.
`tests/test_models.py::test_qwen35_longshot_internal_reselection_uses_one_budget_across_kv_and_linear_heads`
pins it: with `topk_heads=1`, `topk_linear_heads=2`, and head scores `[10, 1, 100, 90, 80, 70]` over two
kv and four linear heads, the second iteration selects `[2, 3, 4]` — the three highest scores wherever
they live — and not the quota-preserving `[0, 2, 3]`.

#### Correction: `run_batched_cache_processors` was not missing

An earlier pass reported that `martin`'s `run_batched_cache_processors` and its `_gate_scale` /
`_linear` / `_rms_norm` / `_attention` helpers had no `dev` equivalent. That was wrong — they live in
`src/gcw/performance/prepared_bt.py`, outside the module set that pass compared. All five are
AST-identical to `martin`. See finding 12.

### 12. Standard (BT/BGT) processor parity — CLOSED

`martin`'s `cache_processor.py` + `cache_processor_ops.py` against `dev`'s
`cache_processors/{standard,base,ops}.py` + `performance/prepared_bt.py`: **45 functions AST-identical**,
9 differing, and every difference reduces to `martin` under defaults.

| Function | Difference | Default |
|---|---|---|
| `gather_per_cache_head_kv` | `backend` parameter added | `"eager"` → `martin`'s body, with the `ndim` check hoisted above it |
| `CacheProcessor.__init__` | `flow_config` / `diffusion_config` parameters | both `None` → `self.flow = self.diffusion = None` |
| `CacheProcessor.compute_update` | flow/diffusion dispatch | both `None` → `out_proj(out_norm(_run_layers(...)))` |
| `CacheProcessor._run_layers` | `activation_checkpointing` branch | `TrainingConfig.activation_checkpointing = "none"` → `martin`'s layer loop |
| `GlobalCacheProcessor.forward`, `_forward_head_mask_broadcast`, `_forward_workspace_broadcast` | `run_bgt_processor_region(...)`, Triton movement | `GCW_CACHE_PROCESSOR_BACKEND=eager` → `block(x, positions)` / `block.scaled_update(x, positions)`; `TritonCacheMovementConfig.enabled=False` |
| `PerLayerCacheProcessor.forward` | `_prepared` / `_prepared_backend` parameters, Triton movement | Triton off; prepared path covered below |
| `PerLayerCacheProcessor.__init__` | caches `num_kv_heads` / `head_dim` | `split_layer_kv(updated, self.num_kv_heads, self.head_dim)` equals `martin`'s `keys.shape[2], keys.shape[4]` — `gather_cache_layers` returns `(layers, batch, kv_heads, positions, head_dim)` and `stored_cache_shape` is uniformity-checked |

#### The prepared BT path is bitwise identical

Unlike the other gates, `PerLayerCacheProcessor.prepare_forward_execution` is **on by default**:
`model.py` opens a `PreparedPerLayerCacheProcessorExecution` for every forward, so BT runs
`run_prepared_batched_cache_processors` rather than `martin`'s `run_batched_cache_processors`. The
difference is only *when* the per-layer parameter stacks are built — once per forward instead of once
per call site — and `torch.stack` is exact, so the arithmetic and its order are unchanged.

Verified rather than argued. `/tmp/gcwcmp/bt_prepared_probe.py` runs both implementations over the same
seeded 4-layer BT processor:

| dtype | forward max abs diff | input-grad max abs diff |
|---|---|---|
| float64 | 0.000e+00 | 0.000e+00 |
| float32 | 0.000e+00 | 0.000e+00 |

`martin`'s `run_batched_cache_processors` remains in the tree, AST-identical, as the reference the
prepared path is checked against.

### 13. Backbones — CLOSED, no numeric divergence

`martin/gcw/models/{transformer,qwen3,gemma4}.py` against `dev/src/gcw/models/backbones/{transformer,qwen3,gemma4}/`:
80 functions AST-identical, 38 differing. Every difference is one of:

- **`loss_backend` threading** through `forward`, `forward_full`, `forward_chunk`, `loss_from_hidden` —
  covered by findings 6 and 8; `"reference"` + a legacy manifest routes to `loss_from_hidden_reference`,
  which is `martin`'s body.
- **`activation_checkpoint_windows`** in `forward_windowed_loss` — defaults to `0`, giving `martin`'s
  plain window loop.
- **`backbone_chunk_backend`** in `TF.__init__` — defaults to `"eager"`, skipping both compiled
  `forward_embedding_chunk` branches; the eager body is `martin`'s `forward_chunk`.
- **`kv_score_sidecar` plumbing** — always `None` (finding 14).
- **`int(config.head_dim)` → `config.head_dim_for_layer(idx)` / `_head_dim(config)`** — for every family
  except gemma4 these return `config.head_dim`, and `__post_init__` resolves it before use.
- **`cast(...)` calls and `is None` guards** — typing narrowing with no runtime effect.

Two worth naming:

- `Qwen35GatedDeltaNet.forward` inlines `g = -A_log.float().exp() * softplus(a.float() + dt_bias)` where
  `martin` called `self.dynamics(a)`. `martin`'s `Qwen35GatedDeltaDynamics.forward` body is that exact
  expression; only its head-count check was dropped, and the equivalent check now lives in Longshot's
  `_gated_delta_linear_state_update` (finding 11).
- `qwen35_moe_load_balancing_loss` raises on empty `router_logits` where `martin` returned `0`. Both call
  sites are guarded by `router_aux_loss_coef`, so the branch is unreachable either way.

`martin`'s `cosine_lr` and `qwen3_config` are absent from `dev`; both are dead code in `martin` with no
call sites.

### 14. Cache state and windowed cache — CLOSED

`cache_state.py` → `cache_processors/state.py`: 27 identical, 1 differing. `windowed_cache.py` →
`cache_processors/windowed.py`: 8 identical, 4 differing. Every difference is `kv_score_sidecar`
plumbing — a field threaded through the constructors and invalidated (`= None`) in both `scatter_layer`
implementations. `KVScoreSidecarConfig.enabled` defaults to `False`, so the sidecar is never populated
and the invalidation is a no-op.

`martin`'s `CacheState.append_scores` has no `dev` equivalent and no call site in `martin` either.

### 15. FLOP and parameter accounting — CLOSED, reporting only

`flops.py` and `params.py` were compared function-by-function (12 + 14 identical, 8 + 3 differing) and
then swept end to end: `martin`'s modules loaded side by side with `dev`'s against the same
`ModelConfig`, over **all 101 `configs/train` configs**.

| Quantity | Identical | Differing |
|---|---|---|
| `backbone_flops_per_token` | 101 | 0 |
| `processor_forward_flops_per_sequence` | 99 | 2 |
| `flops_per_token` | 99 | 2 |
| `training_flops_per_token` | 99 | 2 |
| `estimate_processor_params` | 98 | 3 |
| `total_non_embedding_params` | 98 | 3 |

The only differing configs are `qwen3_06b_diffusion_bgt*.yaml` — a `dev`-only architecture `martin`
cannot express. Every `martin`-comparable config reports identical FLOPs and parameters.

#### Latent divergences, unreachable from any config

The user's standing instruction is to consider all paths, so these are recorded even though no config
reaches them. All three are reporting-only — they change no training or evaluation arithmetic — and in
each case `dev` matches what the processor actually executes while `martin` did not.

| Path | `martin` | `dev` | Measured |
|---|---|---|---|
| `include_self_head_kv` with all heads selected | suppressed the `+1` when `selected_heads >= cache_heads` | always adds the self head | Icarus +7.32%, Longshot +1.00% reported processor FLOPs. `IcarusProcessorAttention` binds `forward_with_self_head_kv` unconditionally, so `dev` is right. |
| Longshot `iteration_mode="internal"`, `iterations>1` | multiplied encode + field by `iterations` | encode once, field per iteration | −38.10% at `iterations=3`. `LongshotCacheProcessor.forward` calls `_forward_once` once in internal mode; only the field repeats. `dev` is right. |
| Longshot `linear_state_update_mode="gated_delta"` | `lh·(8·lk·lv + 2·lk + 4)` | `4·lh·lk·lv` | −0.176% on a 768-linear-head config. |

`dev`'s `topk_heads <= 0` branch in `processor_forward_flops_per_sequence` is dead: `__post_init__`
rejects a non-positive `topk_heads`.

### 16. Checkpoint conversion and evaluation — one real gap

`convert.py` → `convert/{core,hf_config,hf_state}.py`: 16 identical, 2 differing (`_load_hf_model`
restructured with the gemma4 `ValueError` retry preserved; `config_from_hf` adds family/variant
validation). Both equivalent for valid inputs.

**The gap:** 14 `martin` functions have no `dev` equivalent anywhere —

```
convert_bottleneck   config_from_bottleneck   config_from_bottleneck_llama1b   load_bottleneck_state
convert_ppt          config_from_ppt_meta     load_ppt_state
_bottleneck_spec     _map_bottleneck_key      _map_processor_subkey   _pad_vocab_tensor
_model_file          _step_from_path          _infer_state_dtype
```

These back `gcw convert bottleneck` and `gcw convert ppt` in `martin`'s CLI, plus `gcw eval ppt`
(`run_ppt`). `dev`'s CLI has neither subcommand. A checkpoint from the original bottleneck-LLM codebase
or from PPT can be ingested by `martin` and cannot be ingested by `dev`.
`scripts/publish_bottleneck_llm_artifacts.py` — already `martin`-only — depends on `convert_bottleneck`.

This does not affect any checkpoint `dev` itself produced, and keeping the `martin` branch preserves the
capability. It is recorded because it is the one place where "evaluation of checkpoints can be fully
backwards compatible" does not hold inside `dev` alone.

#### `gcw eval paper7` is a faithful replica

`dev` splits `martin`'s single `GCWEvalLM` into `GCWEvalLM` (`all`/`math`/`general`/`tasks`/`chat`) and
`LegacyPaper7EvalLM` (`paper7`, aliased `paper7-legacy`). The legacy subclass restores all three pieces
of `martin`'s behaviour: `add_special_tokens_for_generate = _legacy_generate_add_special_tokens(model)`,
`martin`'s exact `apply_chat_template`, and the `_legacy_bottleneck_cache_eval` →
`_loglikelihood_tokens_legacy_boundary` dispatch. `run_paper7_legacy_groups` also reproduces `martin`'s
chat-template decision verbatim and points `TaskManager` at `configs/tasks_legacy/paper7`. Both CLI names
route there, so `gcw eval paper7 --checkpoint ...` scores a legacy checkpoint exactly as `martin` did.

`_should_apply_chat_template(tokenizer, "auto")` is `martin`'s expression; `_load_eval_target` with a
checkpoint and the processor enabled is `martin`'s `load_checkpoint` + `_tokenizer_for`;
`evaluate_nll_bpb` defaults to `loss_backend="reference"` and `run_fineweb` resolves it from the
checkpoint manifest (finding 9).

#### Legacy-checkpoint dispatch narrowed to `paper7`

In `martin`, `run_core` / `run_ppt` / `run_math` / `run_paper7` all funnelled into one `run_task_groups`
and therefore one `GCWEvalLM`, so the legacy boundary and `add_special_tokens_for_generate` applied to
**every** suite. In `dev` they apply to `paper7` only.

A `legacy_bottleneck_generate` bt/bgt checkpoint scored with `gcw eval math` in `dev` will not take the
legacy boundary path, and will not match `martin`'s `gcw eval math` for the same checkpoint. Scored with
`gcw eval paper7` it matches exactly. Since the paper-7 results are the ones being compared across
branches, this is recorded rather than changed — but it is a real behavioural difference, not a
refactor.

## Status

| # | Finding | Resolution |
|---|---|---|
| 1 | SFT rank sharding | `dev` kept — `martin` bug; one-process recipe reproduces old runs |
| 2 | `precision: bf16` non-FSDP trained in fp32 | Fixed in `dev` to match `martin` |
| 3 | `precision: bf16` FSDP initialized in bf16 | Fixed in `dev` to match `martin` |
| 4 | `_wsd_lr` warmdown off-by-one | `dev` kept — `martin` never reaches `final_lr_frac` |
| 5 | Longshot `_fit_time_length` pad side | `dev` kept — `martin` pads against its own trim |
| 6 | Loss reduction reassociation | `dev` kept — enables `row_sum`/`none`, which `martin` raised on |
| 7 | Config drift and missing configs | 11 configs restored from `martin` |
| 8 | Precision modes declared but unsupported | Narrowed to `martin`'s three modes |
| 9 | Checkpoint metadata and resume fingerprint could not distinguish the bf16 modes | `precision` restored in both; legacy manifests handled by an explicit compatibility rule |
| 10 | `bf16_mixed` degrades to fp32 off CUDA | `dev` kept — now reported in the run preamble |
| 11 | Icarus and Longshot processor parity | Icarus exact; Longshot equivalent after fixing four `dev`-only regressions |
| 12 | Standard (BT/BGT) processor parity | Equivalent under defaults; prepared BT path verified bitwise identical |
| 13 | Backbone parity (transformer, qwen3.5, gemma4) | Equivalent — all differences are audited gates, typing, or resolved `head_dim` |
| 14 | Cache state and windowed cache | Equivalent — `kv_score_sidecar` plumbing only, disabled by default |
| 15 | FLOP and parameter accounting | Identical on all 101 configs; three latent reporting-only divergences recorded |
| 16 | Checkpoint conversion and evaluation | `paper7` eval is a faithful replica; bottleneck/PPT conversion is `martin`-only |

Open: the bottleneck/PPT conversion path (finding 16) exists only on `martin`. Keeping the branch
preserves it.

## Change log

**2026-08-21, remaining audits completed.** Compared the standard (BT/BGT) processor, all three
backbones, cache state, windowed cache, FLOP and parameter accounting, checkpoint conversion, and
evaluation against `martin` (findings 12-16). Corrected the earlier claim that
`run_batched_cache_processors` had no `dev` equivalent — it is in `performance/prepared_bt.py` and is
AST-identical. Verified the default prepared BT path bitwise against it. Swept FLOPs and parameters over
all 101 train configs. Found one real gap: `dev` cannot ingest bottleneck or PPT checkpoints.

**2026-08-20, Longshot validations restored.** Put back `martin`'s build-time check that every
`gated_delta` linear-state spec has matching Qwen3.5 dynamics, and the two `_gather_cache_heads` shape
checks. The build-time check immediately caught a test constructing a processor with dynamics for only one
of its two spec layers. Suite 838 -> 840.

**2026-08-20, Longshot internal re-selection realigned.** Reverted `dev`'s split kv/linear head budget in
`process_hidden` to `martin`'s single combined `topk_heads + topk_linear_heads` budget; the outer
`_select_combined_heads` keeps its per-group budgets, matching `martin` on both sides. The existing test
that pinned the split behaviour was rewritten to pin the combined behaviour and renamed.

**2026-08-20, finding 11 — Icarus and Longshot processors audited by AST.** Icarus reduces exactly to
`martin`. Longshot's two measurable differences were probed: the chunked score reductions match the eager
oracle bitwise in forward and gradient, and selected-head ordering is permutation-invariant to 1.5e-07.
Fixed four `dev`-only regressions found on paths no config currently reaches — a silently skipped
zero-init, a non-positive head budget that selected everything, a dropped gated-delta head-count check,
and the lost `_fit_time_length` warning. Six tests added; suite 831 -> 838.

**2026-08-20, 27B config paths realigned.** `qwen35_27b_instruct_tf_sft.yaml` and
`qwen35_27b_longshot_from_base.yaml` inherited `martin`'s `/mnt/gcw/...` prefix for `data.dataset` and
`backbone_checkpoint`; `/mnt/gcw` does not exist here. Both now use `/mnt/jfs-01/shared/gcw/...` like the
other 65 configs. Paths only — no training semantics changed.

**2026-08-20, OpenMathInstruct-2 SFT path verified.** Confirmed all 67 `bottleneck_llm` configs still
finetune on OpenMathInstruct-2 and that `dev`'s rewritten row indexing (`IndexedJSONL` + `_RowSlice` +
`SFTCursor`) and `sft_batch` reproduce `martin`'s batches exactly. Probed against the real dataset and
tokenizer: batches identical over the production slice and across two epoch-wrap cases, and a full
forward/backward on a seeded `tf` model bit-identical in loss, gradients, and gradient norm. Recorded under
Confirmed equivalent. No code change.

**2026-08-20, finding 9 fully closed — resume fingerprint.** Replaced `"bf16"` with `"precision"` in
`_training_fingerprint`, so the two bf16 modes no longer collide, and added
`_fingerprint_with_legacy_precision` in `src/gcw/training/checkpoint/resume.py` as an explicit
backwards-compatibility rule for manifests written before the change. A legacy `bf16: true` manifest resumes
into either bf16 mode with a `RuntimeWarning` naming the ambiguity it cannot resolve; a legacy flag that
disagrees is still rejected, and the error names `precision`. Six tests added across
`tests/test_training_checkpoint.py` and `tests/test_train.py`; the three positive-path legacy tests and the
fingerprint test were each confirmed to fail against the previous code.

**2026-08-20, finding 5 decided — `dev` kept; audit closed.** Traced when the Longshot linear encoder actually
pads: the selected time axis is always `min(cache_len, process_every + topk)`, so with the default
`encoder_seq_len` exactly one call per sequence pads, the cold-start one, by exactly `topk` positions.
Instrumented every `_fit_time_length` call across four configurations to confirm. No config reaches it —
`smoke_longshot.yaml` never pads and `smollm360m_longshot.yaml` is an orphan — so no checkpoint is affected.
Kept `dev`'s pad side on its merits: it is right-aligned like the trim branch it shares a function with, where
`martin` left-aligns the pad against a right-aligned trim. Declined the config validation, the restored
warning, and the orphan cleanup as out of scope. No code changed. This closes the last open finding.

**2026-08-20, findings 4 and 6 decided — `dev` kept for both.** Compared the two WSD warmdown forms and the
two cross-entropy reduction forms on their merits rather than on parity. `dev` wins both: its warmdown actually
reaches `final_lr_frac` and does not silently no-op at `warmdown == 1`, and its `reduction="none"` is
load-bearing for the `row_sum` and `none` reductions that `martin`'s chunked loss raised on. Corrected the
finding 4 entry, which wrongly claimed `dev` starts the warmdown one step earlier at `1.0`; the boundary change
is value-neutral for `warmdown >= 2` and the entire difference is in the tail. No code changed. Accepted
consequence: `martin` pretraining runs are not step-for-step reproducible on `dev`, and BGT/Icarus/Longshot
cross-branch comparisons hold to about 1e-7 rather than bitwise.

**2026-08-20, precision matrix audit (findings 8, 9, 10).** Broadened the precision review past the two configs
that set `precision: bf16`, on the principle that any mode should behave predictably with any model. Narrowed
`TrainingPrecision` to the three supported modes and deleted the unreachable `auto`/`fp16`/`fp16_mixed`
branches; restored `precision` to checkpoint metadata and made the recorded `dtype` truthful; added a
precision line to the run preamble. Verified the parameter/buffer dtype matrix across five families x three
modes by construction. The narrowed `Literal` restores `martin`'s, which was already three modes; `dev` had
widened it. The legacy `bf16: bool` alias was verified unchanged against `martin` across all eight
`(bf16, precision)` combinations. **Correction to the previous entry:** the FSDP `MixedPrecision` wrapper was described
there as inert only for `qwen3_5`, on the grounds that `transformer`/`gemma4` fp32 rotary buffers would be
cast. That was wrong — `model.to(dtype=...)` already casts buffers, so under `precision: bf16` the module is
fully bfloat16 before FSDP wraps it and the policy is inert for every family.

**2026-08-20, findings 2 and 3 closed.** `precision: bf16` now behaves as `martin` did. Two edits:
`src/gcw/training/model_setup.py` moves the `model_dtype` cast to after `init_weights()` and any checkpoint
load on both branches, and `src/gcw/training/assembly.py` applies the same cast on the non-FSDP path, which
previously had none. Three regression tests were added in `tests/test_train.py` covering the end-to-end
checkpoint dtype, the fp32/`bf16_mixed` parameter dtype, and the init ordering against a fp32-then-cast
reference; each was confirmed to fail against the previous code. Full suite: 812 passed, 43 skipped, unchanged.
The FSDP `MixedPrecision` policy was left as-is — see [finding 3](#3-precision-bf16-fsdp-initialized-in-bf16--closed).

**2026-08-19, configuration parity.** Restored 11 files in `configs/train/bottleneck_llm` from `martin`: the two
missing 27B configs and nine drifted Qwen3.5 Longshot configs. Verified afterwards that all 67 configs are
byte-identical to `martin`, that the file set matches exactly, that all 67 load and pass `dev`'s admission gate,
and that all 67 resolve field-for-field to the `martin` values apart from the two known cosmetic diffs (`loss`,
`legacy_bottleneck_generate`). The rest of `configs/train` — `bottleneck_compat`, `resume`, the SmolLM
pretraining configs, and the smoke configs — was already byte-identical and needed no change. Closes finding 7.

**2026-08-19, explicit legacy-mode pinning.** Wrote the legacy training mode explicitly into all 67
`configs/train/bottleneck_llm` configs instead of relying on `dev`'s defaults: five keys in the `data` block, 13
top-level keys, and a `performance` block. `trainable` and `save_cache_processor_only` were written per config
from `freeze_backbone` (43 frozen, 24 not) rather than left to derivation.

Verified inert: the resolved configuration is byte-identical before and after for 67/67, all 67 still load and
pass admission, they still match `martin` field-for-field apart from the two cosmetic diffs, and the full suite
passes (812 passed, 43 skipped). Pinning therefore records current behaviour without changing it, which is the
point — it converts a silent future default change into a visible config diff.

These configs are now byte-different from `martin` by construction. Byte comparison against `martin` is no longer
the parity check; resolved-config comparison is. Re-run the loader in
[Reproducing the audit](#reproducing-the-audit) against `martin` after any change here.

**2026-08-19, `configs/experiments` pinned.** Applied the same pinning to the nine seed-sweep configs in
`configs/experiments`, which is gitignored and therefore never appears in a repository diff. All nine are
frozen-backbone, so all nine took `trainable: cache_processor` and `save_cache_processor_only: true`. Verified
inert: resolved configuration unchanged for 9/9, all nine still load and pass admission.

Two pre-existing defects found there were also fixed. The seed 24 4B-base config had been copied from seed 25
with only `seed:` updated, leaving its `output_dir`, `wandb.name`, and `seed-25` tag pointing at the seed 25 run;
it had not been launched. Its three identity fields were realigned to seed 24, and all nine configs now agree
across filename, `seed`, `output_dir`, `wandb.name`, and tag. Separately, `linear_state_update_mode:
outer_product` was pinned in the three non-`gated_delta` configs, which had been relying on the default — the
same implicit-value exposure as [finding 7](#7-config-drift-and-missing-configs--closed). Pinning it was inert;
the only resolved change across all nine was the seed 24 identity.

**2026-08-19, finding 1 reclassified.** SFT rank sharding was re-examined and confirmed to be a `martin` bug
rather than behaviour to preserve: `martin`'s FineWeb loader shards by rank in the same codebase while its SFT
loader never receives `rank`, and no launcher compensates. `dev`'s corrected sharding stands with no legacy flag.
The section now records the one-process equivalence recipe for reproducing an old run instead. Removed from the
remediation list.

## Reproducing the audit

Create a `martin` worktree, then run each probe under both packages in separate processes — both are imported
as `gcw`, so they cannot share an interpreter.

```bash
git worktree add /tmp/martin_wt martin
```

Configuration resolution diff:

```python
# run twice, with sys.path[0] set to /tmp/martin_wt then to src/
import glob, json, os, sys
from gcw.config import load_train_config, to_dict

root = "/tmp/martin_wt/configs/train"
out = {}
for path in sorted(glob.glob(root + "/**/*.yaml", recursive=True)):
    out[os.path.relpath(path, root)] = to_dict(load_train_config(path))
json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True, default=str)
```

Numerical probe:

```python
# run twice, with sys.path[0] set to /tmp/martin_wt then to src/
import hashlib, sys
import numpy as np
import torch
from gcw.config import ModelConfig
from gcw.models import build_model

model_kwargs = dict(
    variant=sys.argv[1], family="qwen3", sequence_len=128, vocab_size=512,
    hidden_size=64, intermediate_size=128, num_layers=4, num_heads=4,
    num_kv_heads=2, head_dim=16, qk_norm=True, rotary_style="llama",
)
if sys.argv[1] != "tf":
    model_kwargs["cache_processor"] = dict(
        process_every=16, topk=8, topk_heads=4, hidden_size=32,
        intermediate_size=64, num_layers=1, num_heads=2, num_kv_heads=2,
        encoder_method="query_attention",
    )

def digest(tensor):
    array = tensor.detach().to(torch.float64).cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]

torch.manual_seed(24)
model = build_model(ModelConfig(**model_kwargs))
model.init_weights()
print("init", digest(torch.cat([p.detach().reshape(-1) for p in model.parameters()])))

generator = torch.Generator().manual_seed(7)
x = torch.randint(0, 512, (2, 64), generator=generator)
y = x.roll(-1, dims=1).clone()
y[:, -1] = -100
y[0, :10] = -100

model.train()
loss = model(x, y, loss_reduction="mean")
loss.backward()
print("loss", f"{float(loss):.10f}")
print("grad", digest(torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])))
```

To localize a divergence once found, register forward hooks on every named submodule, record a digest per call,
and compare to find the first differing module and call index. That is how finding 5 was traced to
`_fit_time_length`.
