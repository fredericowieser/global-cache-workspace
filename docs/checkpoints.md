# Shared GCW Checkpoints

Public GCW artifacts live under:

```text
/mnt/jfs-01/shared/gcw
```

Run publication from the cluster checkout:

```bash
cd ~/gcw-paper
uv run --extra cpu --extra convert python scripts/publish_bottleneck_llm_artifacts.py \
  --root /mnt/jfs-01/shared/gcw
```

The script publishes two checkpoint families:

```text
/mnt/jfs-01/shared/gcw/checkpoints/bottleneck_llm/{llama32_1b,llama32_3b,llama31_8b,qwen3_06b}/{bt,bgt}/final
/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/{llama32_1b,llama32_3b,llama31_8b,qwen3_06b}/tf
```

It also writes:

```text
/mnt/jfs-01/shared/gcw/manifest.json
/mnt/jfs-01/shared/gcw/MANIFEST.md
```

Each manifest entry records source path, public tokenizer source, output path, dtype, sequence length, SHA256, file size, and strict-load validation status.

## Eval Smoke

```bash
uv run --extra gpu --extra eval gcw eval paper7 \
  --checkpoint /mnt/jfs-01/shared/gcw/checkpoints/bottleneck_llm/llama32_1b/bgt/final \
  --limit 1 \
  --batch-size 1 \
  --out results/llama32_1b_bgt_limit1.json
```

For two-GPU paper-table eval, submit `scripts/slurm/eval.sbatch` with:

```bash
CHECKPOINT=/mnt/jfs-01/shared/gcw/checkpoints/bottleneck_llm/llama32_1b/bgt/final
SUITE=paper7
BATCH_SIZE=16
OUT=results/llama32_1b_bgt_paper7.json
```

Use `scripts/probe_eval.py` to inspect partial results and rough ETA while the run is active.
