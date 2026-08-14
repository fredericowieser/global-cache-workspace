# Global Cache Workspace

GCW tests whether small learned processors can improve a causal language model by selecting, transforming, and rewriting its internal cache.

Supported backbones: SmolLM, Llama 3/3.1/3.2, Qwen3, Qwen3.5 dense and MoE, and Gemma4.

Implemented model compositions:

- **TF**: causal-transformer baseline.
- **BT**: one cache processor per backbone layer.
- **BGT**: one processor across selected state from all layers.
- **Icarus**: cache-head selection with frontier writeback.
- **Longshot**: time compression, cross-head processing, persistent workspace, and recurrence.
- **Flow**: deterministic, time-conditioned latent refinement inside BGT, Icarus, or Longshot.

GCW supports scratch FineWeb pretraining and supervised fine-tuning with answer-only or full-sequence cross-entropy. Training can update the backbone, cache processor, or both. Capability admission rejects incompatible compositions before model allocation or output writes.

## Install

Install `uv`, then load its environment into the current shell. The installer writes to `~/.local/bin`; without the `source` command, the same shell can still report `uv: command not found`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

GCW requires one Torch runtime extra. On an NVIDIA host:

```bash
uv sync --extra gpu --extra convert --extra eval --group dev
```

Use `--extra cpu` or `--extra rocm` instead of `--extra gpu` on the corresponding runtime. Qwen3.5 optimized kernels also require `--extra qwen35`.

Authenticate Hugging Face before downloading gated model weights. Create a read token at [Hugging Face settings](https://huggingface.co/settings/tokens), then:

```bash
uv run hf auth login
uv run hf auth whoami
```

Authenticate W&B before enabling remote tracking. The local metric stream remains canonical if W&B becomes unavailable.

```bash
uv run wandb login
uv run wandb status
```

Do not place either token in a config file, shell script, run manifest, or Git. Both CLIs store credentials under the user account.

## Verify the installation

```bash
uv run --extra gpu python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
uv run --extra cpu python examples/cpu_smoke.py
uv run --extra cpu --group dev pytest -q
```

The smoke program constructs TF and BGT, computes finite losses, admits Longshot, and demonstrates one fail-closed composition.

## Train

Run a CPU smoke config:

```bash
uv run --extra cpu gcw train --config configs/train/smoke_tf.yaml
```

Run a GPU config inside a Slurm allocation:

```bash
uv run --extra gpu gcw train --config /path/to/experiment.yaml
```

New local run records live under `data/training-runs/`. They contain resolved config, metrics, logs, environment, provenance, and W&B identity. Mutable checkpoints remain separate.

## Evaluate

```bash
uv run --extra cpu --extra eval gcw eval all \
  --checkpoint /path/to/checkpoint \
  --limit 1 \
  --out data/results/core-smoke.json
```

Evaluation creates an independent record under `data/eval-runs/`. It does not mutate training state.

## Documentation

The [documentation index](docs/index.md) is the human knowledge layer.

- [Installation](docs/getting-started/installation.md)
- [GCW overview](docs/concepts/overview.md)
- [Cache processing](docs/concepts/cache-processing.md)
- [Model variants](docs/concepts/model-variants.md)
- [Training configuration](docs/guides/configure-training.md)
- [Checkpoint formats](docs/reference/checkpoint-formats.md)
- [Run format](docs/reference/run-format.md)
- [Internal architecture](docs/development/architecture.md)
- [Active refactor plan](docs/development/refactor-plan.md)

Build the documentation with warnings treated as errors:

```bash
uv run --extra cpu --group docs python scripts/docs/generate_reference.py
uv run --extra cpu --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Repository boundaries

- `src/gcw/`: reusable installed library.
- `configs/`: human-authored experiment definitions.
- `data/`: dataset registries, run records, mutable state, results, evidence, and generated analysis.
- `scripts/`: thin data, analysis, cluster, and publication commands.
- `examples/`: executable examples used by documentation and CI.
- `tests/`: unit, integration, provenance, and compatibility contracts.

Local metric files are canonical. W&B mirrors them. Historical evidence keeps its original paths and identity schemas.
