# Operate the repository

**Status:** Maintained

Select exactly one accelerator extra in every environment or command:

```bash
uv sync --extra cpu --extra convert --extra eval
uv run --extra cpu --group dev pytest -q
uv run --extra cpu --group dev ruff check src/gcw scripts tests examples
uv run --extra cpu --group dev mypy src/gcw/architecture.py src/gcw/provenance.py src/gcw/config.py src/gcw/models/__init__.py src/gcw/distill/__init__.py src/gcw/analysis/__init__.py examples/cpu_smoke.py
```

Use `--extra gpu` for CUDA 12.8 or `--extra rocm` for the declared ROCm index. The extras conflict by design, and the lock constraints reject CUDA 13 contamination.

Representative workflows:

```bash
# Prepare data.
uv run --extra cpu gcw prepare fineweb --out /path/to/fineweb
uv run --extra cpu gcw prepare openmath --source /path/to/source --out /path/to/train.jsonl

# Train locally or inside an allocation.
uv run --extra gpu gcw train --config configs/train/smoke_tf.yaml

# Evaluate a native artifact.
uv run --extra gpu --extra eval gcw eval math --checkpoint /path/to/checkpoint

# Convert a Hugging Face model.
uv run --extra cpu --extra convert gcw convert hf \
  --family llama --model meta-llama/Llama-3.2-1B --variant tf --out /path/to/native

# Run analysis launchers.
bash scripts/analysis/run_bottleneck_synergy.sh
bash scripts/analysis/run_bottleneck_head_loss_dump.sh
bash scripts/analysis/run_longshot_gsm8k_iterations.sh
```

GPU cluster work must run inside Slurm. The generic `scripts/cluster/slurm/train.sbatch` accepts `CONFIG` through the environment. Analysis launchers and batch scripts document their own required paths and overrides.
