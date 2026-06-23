# Reproducing Bottleneck LLM Finetunes

This doc covers the Llama/Qwen OpenMathInstruct finetune path. The public shared artifacts are expected at:

```text
/mnt/jfs-01/shared/gcw
```

## 1. Prepare Data

```bash
gcw prepare openmath \
  --source /home/f50056118/bottleneck_llm/data/OpenMathInstruct-2 \
  --out /mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl
```

The output JSONL contains `prompt` and `answer` fields.

## 2. Publish Native SFT Backbones And Final BT/BGT Checkpoints

```bash
uv run --extra cpu --extra convert python scripts/publish_bottleneck_llm_artifacts.py \
  --root /mnt/jfs-01/shared/gcw
```

The final converted BT/BGT checkpoints are for evaluation and backwards-compatibility checks. The SFT backbone checkpoints are the starting points for reproducing BT/BGT cache-processor training.

## 3. Train From The Shared SFT Backbones

Frozen-backbone BGT, Llama 3.2 1B:

```bash
CONFIG=configs/train/bottleneck_llm/llama32_1b_bgt_from_sft_frozen.yaml \
sbatch scripts/slurm/train.sbatch
```

Frozen-backbone Icarus, Llama 3.2 1B:

```bash
CONFIG=configs/train/bottleneck_llm/llama32_1b_icarus_from_sft_frozen_shared.yaml \
sbatch scripts/slurm/train.sbatch
```

The convenience wrapper for that run is:

```bash
bash scripts/martin.sh
```

Inside an existing GPU allocation, run the raw training entry point directly:

```bash
uv run --extra gpu gcw train --config configs/train/bottleneck_llm/llama32_1b_icarus_from_sft_frozen_shared.yaml
```

Other configs follow the same naming:

```text
configs/train/bottleneck_llm/{llama32_1b,llama32_3b,llama31_8b,qwen3_06b}_tf_sft.yaml
configs/train/bottleneck_llm/{llama32_1b,llama32_3b,llama31_8b,qwen3_06b}_{bt,bgt}_from_sft_frozen.yaml
configs/train/bottleneck_llm/{llama32_1b,llama32_3b,llama31_8b,qwen3_06b}_{bt,bgt}_from_sft_unfrozen.yaml
configs/train/bottleneck_llm/{llama32_1b,llama32_3b,llama31_8b,qwen3_06b}_icarus_from_sft_frozen.yaml
configs/train/bottleneck_llm/{llama32_1b,llama32_3b,llama31_8b,qwen3_06b}_icarus_from_sft_unfrozen.yaml
```

The frozen configs train only the cache processor. The unfrozen configs train both backbone and cache processor. Icarus uses the same shared OpenMathInstruct JSONL and the same native GCW SFT backbone checkpoints as BT/BGT.

Launch all frozen Icarus comparisons:

```bash
sbatch --export=ALL,CONFIG=configs/train/bottleneck_llm/llama32_1b_icarus_from_sft_frozen_shared.yaml scripts/slurm/train.sbatch
sbatch --export=ALL,CONFIG=configs/train/bottleneck_llm/llama32_3b_icarus_from_sft_frozen.yaml scripts/slurm/train.sbatch
sbatch --export=ALL,CONFIG=configs/train/bottleneck_llm/llama31_8b_icarus_from_sft_frozen.yaml scripts/slurm/train.sbatch
sbatch --export=ALL,CONFIG=configs/train/bottleneck_llm/qwen3_06b_icarus_from_sft_frozen.yaml scripts/slurm/train.sbatch
```

## 4. Evaluate A Published Final Checkpoint

```bash
CHECKPOINT=/mnt/jfs-01/shared/gcw/checkpoints/bottleneck_llm/llama32_1b/bgt/final \
SUITE=paper7 \
BATCH_SIZE=16 \
OUT=results/llama32_1b_bgt_paper7.json \
sbatch scripts/slurm/eval.sbatch
```

Flatten the result:

```bash
uv run --extra eval python scripts/extract_paper_table.py \
  --json results/llama32_1b_bgt_paper7.json \
  --out results/llama32_1b_bgt_paper7.md \
  --csv results/llama32_1b_bgt_paper7.csv
```

## Verified Llama 3.2 1B BGT Compatibility Result

This run used the converted legacy BGT checkpoint, fixed legacy BOS prompt handling, and fixed legacy cache processing at the loglikelihood context-continuation boundary. TheoremQA was omitted from the fast verification run, so AVG is over the present tasks.

| Model | GSM8K | SVAMP | GSM-Hd | ThmQA | LogiQA | Gaokao | AVG |
|---|---:|---:|---:|---:|---:|---:|---:|
| GCW BGT | 35.33 | 45.33 | 8.11 |  | 29.03 | 25.64 | 28.69 |
| Target BGT | 35.60 | 46.70 | 7.70 | 9.20 | 29.50 | 25.40 | 28.98 |
| Delta | -0.27 | -1.37 | +0.41 |  | -0.47 | +0.24 | -0.29 |

Small differences remain within likely eval implementation or task-version noise except SVAMP, which should be rechecked if exact paper-table parity is required.
