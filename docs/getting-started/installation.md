# Installation

**Status:** Maintained

## Prerequisites

- Python 3.10 or newer.
- `uv` for locked environment management.
- One Torch runtime: CPU, CUDA, or ROCm.

## CPU development install

1. Create the locked environment.

   ```bash
   uv sync --extra cpu --extra convert --extra eval --group dev
   ```

2. Verify the package import.

   ```bash
   uv run --extra cpu python -c "import gcw; print(gcw.__version__)"
   ```

A version number means the source package and CPU Torch runtime imported successfully.

## GPU install

Run GPU setup inside the target cluster environment:

```bash
uv sync --extra gpu --extra convert --extra eval
```

Qwen3.5 optimized kernels require `--extra qwen35`. GCW can use slower fallbacks without that extra, but memory use can increase enough to invalidate a batch-size assumption.
