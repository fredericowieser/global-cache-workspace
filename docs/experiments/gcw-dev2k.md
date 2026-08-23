# GCW dev2k experiment

The dev2k experiment is a bounded Qwen3-0.6B math-adaptation protocol. It uses
4,096-token sequences, full-sequence SFT loss, global batch 64, and a native
frozen backbone checkpoint. The configs preserve the original experiment
identities and remote artifact paths.

## Layout

- `configs/experiments/gcw-dev2k/` contains the source manifest and launch
  configs.
- `scripts/experiments/gcw-dev2k/` contains config generators, launch helpers,
  and the GPU smoke script.
- `src/gcw/` contains only reusable model and training logic. The experiment
  does not add dev2k-specific branches to the library.

## Generate configs

Run the generators from any working directory:

```bash
python scripts/experiments/gcw-dev2k/gen_configs.py
python scripts/experiments/gcw-dev2k/make_full_sft_dev2k.py
```

Both scripts write into `configs/experiments/gcw-dev2k/`. The generated files
remain launch records; do not edit their remote experiment paths unless the
experiment identity changes.

## Cluster helpers

`launch.sh` starts one configured arm on an existing Slurm allocation:

```bash
scripts/experiments/gcw-dev2k/launch.sh <arm> <jobid>
```

`eval-full-sft.sh` evaluates the dev2k full-SFT checkpoint on that allocation.
The scripts target the original cluster paths and are not local training
entrypoints.

The smoke helper is intentionally kept under `scripts/`, not `tests/`, because
it constructs a GPU model and performs generation:

```bash
python scripts/experiments/gcw-dev2k/smoke.py \
  configs/experiments/gcw-dev2k/<arm>.yaml
```
