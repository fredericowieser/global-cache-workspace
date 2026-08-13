# Training Resume

GCW has two checkpoint formats with different purposes.

- Artifact checkpoints contain `model.safetensors`, model configuration, and descriptive training metadata. Conversion,
  publication, and evaluation use this portable format.
- Version-2 training checkpoints additionally contain AdamW state, the next data cursor, smoothed-loss state, and
  per-rank random-number-generator state. Only this format can resume training.

Legacy checkpoints from before version 2 cannot resume exactly because they never stored optimizer moments or RNG
state. GCW rejects them instead of silently restarting those states.

## Configuration

```yaml
max_steps: 38147
resume:
  checkpoint: runs/example/checkpoint
  topology: same
wandb:
  id: existing-run-id
  resume: must
```

`max_steps` is the total target. A checkpoint at step 20,000 continues at step 20,000 and stops after step 38,147. A
checkpoint already at the target exits successfully. A checkpoint beyond the target fails validation.

`wandb.resume: must` requires `wandb.id`. GCW resumes logging after the completed checkpoint step and does not repeat
`eval_on_start`.

## Exact same-topology resume

`topology: same` requires the saved and current world sizes to match. GCW restores:

- model parameters and buffers;
- AdamW moments and parameter-group state;
- optimizer-step and learning-rate/weight-decay schedule position;
- the raw smoothed-loss accumulator;
- the next unconsumed global data position; and
- Python, NumPy, Torch CPU, and Torch CUDA RNG state for every rank.

The model, data identity, loss, frozen-parameter policy, optimizer, schedule, precision, sequence length, global
microbatch, and total batch size must match. Output directories, save/evaluation frequency, and W&B display metadata may
change.

## Elastic FineWeb DDP resume

`topology: elastic_ddp` permits a FineWeb DDP run to change world size when the global microbatch and total batch size
remain fixed. The loader resumes from one global shard cursor and repartitions the next batch across the new ranks.

The old rank-local RNG streams have no one-to-one mapping onto a new topology. GCW therefore derives deterministic new
rank seeds from the configured seed, completed step, rank, and new world size. Repeating the same elastic resume produces
the same result and does not skip or repeat FineWeb microbatches, but it is not a bitwise continuation of the old
topology.

GCW rejects elastic resume for random data, SFT, and FSDP.

## Storage and failure handling

A complete training checkpoint contains:

```text
checkpoint/
├── config.json
├── manifest.json
├── model.safetensors
├── optimizer.pt
├── rng.pt
└── train_state.json
```

Rank zero writes a sibling temporary directory, hashes and flushes every payload, writes the completeness manifest,
then swaps the directory into place. The previous complete directory remains available during the swap. Recoverable
rank-zero I/O errors propagate to every DDP rank so workers fail together instead of waiting at a collective.
