# Evaluation and conversion

**Status:** Maintained

GCW exposes two benchmark suites:

- `gcw eval math`: GSM8K, SVAMP, GSM-Hard, LogiQA, and Gaokao MathQA;
- `gcw eval general`: LAMBADA, HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge, and MMLU.

`gcw eval all` runs both. `gcw eval fineweb` measures held-out pretraining NLL and bits per byte.

Evaluation uses LM Evaluation Harness through `src/gcw/evaluation/core.py`. Shared tokenizer, truncation, stop-sequence, and scoring behavior lives in `src/gcw/evaluation/scoring.py`.

## Hugging Face conversion

Use one strict conversion path for supported pretrained backbones:

```bash
uv run --extra convert gcw convert hf \
  --family llama \
  --model meta-llama/Llama-3.2-1B \
  --variant bgt \
  --out data/runs/llama32-1b-bgt-init
```

Supported families are `llama`, `qwen3`, `qwen3_5`, `qwen3_5_moe`, and `gemma4`. SmolLM scratch runs use native model configs and FineWeb preparation rather than conversion.

Conversion is shape-strict. It initializes the selected cache processor while loading backbone tensors from the Hugging Face checkpoint.
