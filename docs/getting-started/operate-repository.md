# Operate the repository

**Status:** Maintained

Select exactly one accelerator extra in every environment or command:

```bash
uv sync --extra cpu --extra convert --extra eval
uv run --extra cpu --group dev pytest -q
uv run --extra cpu --group dev ruff check src/gcw scripts tests
uv run --extra cpu --group dev mypy src/gcw scripts/quickstart/cpu_smoke.py scripts/quickstart/flow_smoke.py
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

# Run the authenticated GSM8K reproduction command and analysis launchers.
uv run gcw eval gsm8k-reproduction --help
uv run python scripts/analysis/bench_cache_processor.py --help
```

GPU cluster work must run inside Slurm. The generic `scripts/cluster/slurm/train.sbatch` accepts `CONFIG` through the environment. Analysis launchers and batch scripts document their own required paths and overrides.
