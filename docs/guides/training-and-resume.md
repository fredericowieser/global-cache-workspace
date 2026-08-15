# Training and exact continuation

**Status:** Maintained

## Startup order

Resume correctness depends on ordering. Setup operations can consume random numbers or change parameter wrappers, so training follows this sequence:

1. initialize distributed execution;
2. validate the checkpoint manifest and training fingerprint;
3. build the raw model;
4. load resumed weights, or load a configured backbone for a new run;
5. freeze requested backbone parameters;
6. compile and wrap with DDP or FSDP;
7. build AdamW and restore optimizer state;
8. build the batch stream and restore its cursor;
9. restore Python, NumPy, CPU Torch, and CUDA random states;
10. continue from the completed optimizer step.

Restoring randomness last prevents model construction or wrapper setup from perturbing the resumed trajectory.

## Batch and optimizer semantics

Let

$$
M=(\text{per-rank batch})(\text{sequence length})(\text{world size})
$$

be microbatch tokens. If `total_batch_size` is configured, gradient accumulation uses `total_batch_size / M`; otherwise one microbatch equals one optimizer step. The ratio must divide exactly.

AdamW separates matrix-shaped parameters, which receive weight decay, from vector parameters, which do not. The learning-rate schedule uses warmup, a stable plateau, and warmdown. Weight decay follows a cosine decay to zero. Gradient accumulation uses `no_sync` on nonfinal DDP microsteps, then clips gradients before the optimizer step.

`max_steps` means the total target number of completed optimizer steps:

- saved step below `max_steps`: continue;
- saved step equal to `max_steps`: report completion and exit successfully;
- saved step above `max_steps`: reject the configuration.

## Two checkpoint contracts

Portable artifact checkpoints contain model weights, model configuration, and inference metadata. They support conversion, evaluation, and publication. They do not promise exact training continuation.

Version-2 training checkpoints add:

```text
checkpoint/
├── model.safetensors
├── optimizer.pt
├── rng.pt
├── config.json
├── train_state.json
└── manifest.json
```

The manifest records the format version, completed step, topology, world size, training fingerprint, file hashes, and completeness. The loader rejects missing, extra, truncated, altered, or unsupported files before applying state.

## Atomic save protocol

All ranks enter the save collective. They gather rank-local random states and cursors. Rank zero writes a sibling temporary directory, flushes every file, writes and flushes the manifest last, rotates any old destination to `.previous`, renames the complete temporary directory into place, and flushes the parent directory. It broadcasts success or a serialized error so peers cannot wait forever on a recoverable rank-zero I/O failure. After every rank observes success, the code removes `.previous`.

On load, a broken destination may fall back to a valid `.previous`. A legacy model-only artifact cannot satisfy exact resume and fails with an actionable error. `optimizer.pt` uses PyTorch's restricted `weights_only` loader. `rng.pt` necessarily contains Python and NumPy state and uses the broader loader after manifest hash validation.

## Compatibility fingerprint

Exact continuation requires equality for fields that change training dynamics: model configuration, loss, optimizer hyperparameters, schedules, BF16 mode, frozen parameters, activation-checkpointing policies (including `activation_checkpoint_windows` and `activation_checkpoint_min_padded_sequence_length`), adaptive-batching policy and ordered bands, data kind and identity, sequence length, total batch, gradient accumulation, and `max_steps`. Output paths, checkpoint/evaluation frequency, and display-only W&B metadata may change.

The current data identity records paths, shard names, and sizes, not content hashes. A same-size in-place data mutation can therefore evade the fingerprint. Treat prepared datasets as immutable.

## Same-topology and elastic resume

Same-topology resume restores model parameters and buffers, Adam moments and groups, schedule position, smoothed loss, the next global data location, and every rank's random state. The test target is equivalence with uninterrupted execution.

Adaptive adjacent-pair execution reserves 32 source pairs only inside an optimizer step. Checkpoints remain step-boundary-only, so the saved SFT cursor is either before or after all 64 rows; no partial adaptive plan is serialized. Resume reconstructs the same reservation and grouping from the authenticated policy.

Elastic DDP resume supports only version-2 FineWeb checkpoints. It keeps the global microbatch token count, total batch, accumulation, shard inventory, and next global cursor fixed while allowing world size to change. New ranks repartition the next global batch and receive deterministic seeds derived from the global seed, saved step, new rank, and new world size. This guarantees repeatable elastic resumes and no data skips or duplication. It does not claim bitwise identity with the old topology.

Random data, SFT, and FSDP reject elastic resume. FSDP also rejects exact training resume in this implementation; it can still write portable model artifacts.

## W&B continuation

`WandbConfig` carries an optional run `id` and the official `allow`, `must`, `never`, or `auto` resume mode. `must` requires an ID. Resumed training logs after the checkpoint's completed step, and `eval_on_start` only runs at step zero.
