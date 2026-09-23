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

## Paper result figures

Generate both main-text figure candidates from the current appendix tables:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gcw-matplotlib \
  .venv/bin/python scripts/plot_paper_results.py
```

The script requires Matplotlib and NumPy from the project environment. It reads
`src/Global-Cache-Workspace-Theory-ICLR-2027-/appendices/D_Extended_Results.tex`
and saves SVG figures with editable text, `plotted_values.csv`, and a source-hash
manifest under `data/analysis/paper_results_figures/`. It does not modify the
paper or run evaluations. Values retain the precision of the paper tables.

Use `--option raw` or `--option relative` to generate one option, `--columns 5`
for a single row of backbone panels, and `--output-dir PATH` to save elsewhere.
`--suite korbench_after_math` selects a single figure suite.
`--source PATH` accepts another appendix with the same table labels and layout.

Each option contains three figures: mathematical benchmark averages after
mathematical adaptation, KOR-Bench after mathematical adaptation, and KOR-Bench
following Reasoning Gym adaptation. Panels represent backbones; the x-axis shows
adaptation methods. Raw plots include the pretrained instruct baseline;
relative plots omit its columns. Thinking-off
accuracy uses filled bars; thinking-on accuracy uses overlapping, unfilled
bars with dashed borders. Token lines use solid circles and dashed open diamonds,
respectively, on the right axis. Axis limits are shared across a figure's panels.

Raw accuracy is expressed as a percentage. Relative accuracy is percentage
change, `100 * (accuracy / pretrained_accuracy - 1)`, using the baseline for
that backbone, evaluation suite, and thinking mode. Zero means unchanged
accuracy; negative values indicate a decrease. In relative plots, response-token counts use the same percentage-change
formula against pretrained response-token counts, matched by backbone and
thinking mode. Both relative axes share the same scale and zero. Raw plots
retain absolute mean response-token counts. Missing results are omitted, not converted to zero; `N/A` marks
methods without either thinking-mode result. A missing or zero baseline for a
reported score or token count causes an error rather than an undefined relative value.

For the mathematical suite, accuracy averages all six table benchmarks,
including LogiQA, whereas response tokens are pooled over the four generation
tasks. For KOR-Bench, accuracy averages the five equally sized categories and
response tokens average all 1,250 responses. Reasoning Gym currently has only
thinking-off KOR-Bench results. These plots show aggregate point estimates
without uncertainty intervals; raw per-example results are not inputs.
