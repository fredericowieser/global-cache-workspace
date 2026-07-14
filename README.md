# Global Cache Workspace

GCW is a PyTorch workspace for training, converting, and evaluating transformer baselines and cache-processor models:

- `TF`: a standard causal transformer.
- `BT`: a bottlenecked transformer with one cache processor per backbone layer.
- `BGT`: a bottlenecked global transformer with one shared cache processor across layers.
- `Icarus`: a global cache-head processor that reads selected cache tokens and rewrites the latest selected cache position.
- `Longshot`: an iterative global cache processor with configurable cache encoding and query sources.

The repo supports three workflows:

1. Scratch pretraining on FineWeb-Edu shards.
2. SFT or continued SFT on OpenMathInstruct-style JSONL.
3. Frozen or unfrozen BT/BGT/Icarus cache-processor finetuning from shared Llama/Qwen SFT backbones.

HF checkpoints are converted into native GCW `safetensors`; training and eval do not use HF model wrappers.

## Setup

```bash
uv sync --extra cpu --extra convert --extra eval
uv run --extra cpu --group dev ruff check gcw scripts tests
uv run --extra cpu --group dev pytest -q
```

On the LRC H100 cluster, use the GPU extra inside Slurm:

```bash
uv sync --extra gpu --extra convert --extra eval
```

GPU compute must run inside a Slurm allocation, for example with `srun --jobid=<job> --overlap ...`.

## Data

Prepare FineWeb-Edu shards for scratch pretraining:

```bash
gcw prepare fineweb --out /mnt/jfs-01/shared/gcw/data/fineweb_edu_dedup_cosmo2
```

Prepare OpenMathInstruct SFT JSONL from a legacy Arrow or JSONL source:

```bash
gcw prepare openmath \
  --source /home/f50056118/bottleneck_llm/data/OpenMathInstruct-2 \
  --out /mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl
```

Rows are normalized to `{"prompt": ..., "answer": ...}`. During SFT, prompt tokens are masked with `-100` and only answer tokens contribute to the loss.

## Checkpoints

Publish the shared GCW-native Bottleneck LLM artifacts:

```bash
uv run --extra cpu --extra convert python scripts/publish_bottleneck_llm_artifacts.py \
  --root /mnt/jfs-01/shared/gcw
```

This writes:

- `/mnt/jfs-01/shared/gcw/checkpoints/bottleneck_llm/...`
- `/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/...`
- `/mnt/jfs-01/shared/gcw/manifest.json`
- `/mnt/jfs-01/shared/gcw/MANIFEST.md`

See [docs/checkpoints.md](docs/checkpoints.md). Training checkpoints use a separate, resumable format described in
[docs/training-resume.md](docs/training-resume.md).

## Training

Scratch SmolLM 360M pretraining:

```bash
sbatch --export=ALL,CONFIG=configs/train/smollm360m_tf_20b_wandb.yaml scripts/slurm/train.sbatch
sbatch --export=ALL,CONFIG=configs/train/smollm360m_matched_bt_20b.yaml scripts/slurm/train.sbatch
sbatch --export=ALL,CONFIG=configs/train/smollm360m_matched_bgt_20b.yaml scripts/slurm/train.sbatch
sbatch --export=ALL,CONFIG=configs/train/smollm360m_matched_icarus_20b.yaml scripts/slurm/train.sbatch
```

Bottleneck LLM reproduction configs live under `configs/train/bottleneck_llm/`.

```bash
sbatch --export=ALL,CONFIG=configs/train/bottleneck_llm/llama32_1b_bgt_from_sft_frozen.yaml scripts/slurm/train.sbatch
sbatch --export=ALL,CONFIG=configs/train/bottleneck_llm/llama32_1b_icarus_from_sft_frozen_shared.yaml scripts/slurm/train.sbatch
```

For the standard Llama 3.2 1B frozen-backbone Icarus run on the LRC cluster:

```bash
bash scripts/martin.sh
```

Run the same configs directly inside an existing Slurm allocation:

```bash
uv run --extra gpu gcw train --config configs/train/bottleneck_llm/llama32_1b_icarus_from_sft_frozen_shared.yaml
```

Icarus OpenMath configs are available for all shared SFT backbones:

```bash
sbatch --export=ALL,CONFIG=configs/train/bottleneck_llm/llama32_1b_icarus_from_sft_frozen_shared.yaml scripts/slurm/train.sbatch
sbatch --export=ALL,CONFIG=configs/train/bottleneck_llm/llama32_3b_icarus_from_sft_frozen.yaml scripts/slurm/train.sbatch
sbatch --export=ALL,CONFIG=configs/train/bottleneck_llm/llama31_8b_icarus_from_sft_frozen.yaml scripts/slurm/train.sbatch
sbatch --export=ALL,CONFIG=configs/train/bottleneck_llm/qwen3_06b_icarus_from_sft_frozen.yaml scripts/slurm/train.sbatch
```

Exact-resume templates live under `configs/train/resume/`. `max_steps` always means the total target step, not the
number of additional steps after resuming.

See [docs/training-modes.md](docs/training-modes.md), [docs/training-resume.md](docs/training-resume.md), and
[docs/reproduce-bottleneck-llm.md](docs/reproduce-bottleneck-llm.md).

## Evaluation

Evaluate the verified Llama 3.2 1B BGT checkpoint from shared storage:

```bash
CHECKPOINT=/mnt/jfs-01/shared/gcw/checkpoints/bottleneck_llm/llama32_1b/bgt/final \
SUITE=paper7 \
GCW_EVAL_SKIP_TASKS=theoremqa \
BATCH_SIZE=16 \
OUT=results/llama32_1b_bgt_paper7.json \
sbatch scripts/slurm/eval.sbatch
```

Flatten a paper-table result:

```bash
uv run --extra eval python scripts/extract_paper_table.py \
  --json results/llama32_1b_bgt_paper7.json \
  --out results/llama32_1b_bgt_paper7.md \
  --csv results/llama32_1b_bgt_paper7.csv
```

## Research analysis

The imported cache-head and Longshot analyses share the same checkpoint tokenization and generation semantics as the
main evaluation path:

```bash
bash scripts/run_bottleneck_synergy.sh
bash scripts/run_bottleneck_head_loss_dump.sh
bash scripts/run_longshot_gsm8k_iterations.sh
```

For sharded Longshot analysis, first submit `scripts/slurm/longshot_4gpu_allocation.sbatch`, then run
`scripts/run_longshot_gsm8k_iterations_4gpu.sh`.

The branch-integration history and recovery hashes are recorded in [docs/unified-dev.md](docs/unified-dev.md).
