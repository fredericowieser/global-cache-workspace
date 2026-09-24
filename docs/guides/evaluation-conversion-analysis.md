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
and saves SVG figures with editable text, vector PDFs, `plotted_values.csv`,
and a source-hash manifest under `data/analysis/paper_results_figures/`.
Copy selected figures into the paper submodule when ready.
The script does not edit LaTeX source or run evaluations. Values retain the
precision of the paper tables.

Use `--option raw` or `--option relative` to generate one option, `--columns 5`
for a single row of backbone panels, and `--output-dir PATH` to save elsewhere.
`--suite korbench_after_math` selects a single figure suite.
`--source PATH` accepts another appendix with the same table labels and layout.

Each option contains four figures: mathematical/LogiQA and KOR-Bench
averages after mathematical adaptation and after Reasoning Gym adaptation. Panels represent backbones; the x-axis shows
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
thinking mode. Relative bar plots use independent ranges for accuracy and
token change, each including zero and shared across the figure's backbone
panels. The left axis measures accuracy and the right axis measures tokens.
The ranges expand as needed to align zero on both axes, preserving a single
horizontal baseline while retaining independent scale factors. Raw plots
retain absolute mean response-token counts. Missing results are omitted, not converted to zero; `N/A` marks
methods without either thinking-mode result. A missing or zero baseline for a
reported score or token count causes an error rather than an undefined relative value.

For the mathematical suite, accuracy averages all six table benchmarks,
including LogiQA, whereas response tokens are pooled over the four generation
tasks. For KOR-Bench, accuracy averages the five equally sized categories and
response tokens average all 1,250 responses. The original three-suite views
include thinking-off KOR-Bench results for Reasoning Gym; the paired-benchmark
view below also reads its newly reported mathematical evaluations. These plots show aggregate point estimates
without uncertainty intervals; raw per-example results are not inputs.

### Compact point plots

The same script also offers a relative point view with two vertically stacked
panels: accuracy change above and response-token change below. Generate it with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gcw-matplotlib \
  .venv/bin/python scripts/plot_paper_results.py --layout points --option relative
```

The new `plot_points()` function reuses the table parser and baseline
normalisation. It saves three `*_relative_points.svg` files and a separate
`manifest_points.json`, preserving the existing bar figures and their manifest.
`--suite` can select a single benchmark setting. `--columns` applies only to
the bar view.

Each method column contains the available backbones, encoded by both colour
and marker shape. Filled markers indicate thinking off; hollow markers indicate
thinking on. Fixed horizontal offsets separate backbones and match across both
panels. Thinking off/on share the same horizontal position for each backbone;
hollow markers have transparent interiors to preserve overlapping filled points.
All reported backbones are retained, including
Qwen3.5-2B. Unavailable evaluations have no marker. Both panels show percentage
change against the same backbone's instruct baseline in the same thinking mode;
the instruct columns are omitted. The panels use separate y-axis ranges.

The compact artwork omits the long explanatory footer. When used in the paper,
the caption should state the baseline matching, missing-result convention,
and aggregation: mathematical accuracy averages six benchmarks including
LogiQA, while mathematical token means pool only the four generation tasks;
KOR-Bench accuracy and token means cover all 1,250 questions. Positive accuracy
change means higher accuracy; negative token change means shorter responses.

### Alternative main-text layouts

Three additional functions in the same script provide relative figure
candidates. Select one with `--layout compact`, `--layout tradeoff`, or
`--layout heatmap`, always together with `--option relative`. Each generates
all three benchmark settings, or the suites selected with `--suite`, as
`*_relative_<layout>.svg`, with its own `manifest_<layout>.json`. Existing
figures are preserved.

- `plot_compact()`: four backbone panels in a 2-by-2 grid, reusing the bar
  plotting logic with shared legends, method labels only on the bottom row,
  and separate percentage scales for accuracy bars (left) and token lines
  (right), labelled on the outside of the grid. The
  maths-trained settings omit Qwen3.5-2B only in this view; Reasoning Gym
  retains all four reported backbones. This selection is recorded in the
  manifest and must be disclosed in the paper caption.
- `plot_tradeoff()`: accuracy change versus response-token change in one panel
  per backbone, retaining all backbones. Colours identify methods; filled
  circles and hollow triangles identify thinking off/on. A cross at the
  origin marks instruct. The upper-left quadrant denotes higher accuracy
  with fewer tokens. Limits are shared across panels; points are not jittered
  or connected, so similar results can overlap.
- `plot_heatmap()`: paired accuracy and token matrices, retaining all
  backbones, with methods as rows and backbone/thinking combinations as
  columns. Each panel has its own symmetric colour scale centred on zero.
  Green denotes higher accuracy or fewer tokens; red denotes lower accuracy
  or more tokens. Grey N/A cells mark unavailable evaluations. Thinking
  modes without any reported adapted results for a backbone are omitted.
  Cell text rounds changes to whole percentages, while colours and the CSV
  retain full calculated precision. A displayed zero can therefore be a
  small nonzero change. Caption labels should identify Llama as Llama 3.2
  and Qwen as Qwen3.5, since the compact column headers omit family versions.

All alternatives use the existing matched-mode pretrained baseline and
aggregation definitions. They are candidates for review, not automatically
inserted into the paper.

### Paired-benchmark adaptation heatmaps

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gcw-matplotlib \
  .venv/bin/python scripts/plot_paper_results.py --layout paired_heatmap --option relative
```

`plot_paired_heatmap()` generates two new figures,
`mathematical_relative_paired_heatmap.svg` and
`reasoning_relative_paired_heatmap.svg`. Each pairs mathematical/LogiQA accuracy
with KOR-Bench accuracy, without token panels. Select one training setting with
`--adaptation mathematical` or `--adaptation reasoning`; `--suite` does not
apply to this view. Existing figures and data exports remain unchanged. This
view writes `plotted_values_paired_heatmap.csv` and `manifest_paired_heatmap.json`.

Both panels use the same method rows, backbone/thinking columns, and one
symmetric terracotta/cream/forest-green scale centred on zero. The columns include every
backbone and thinking mode with an available adapted accuracy in either
benchmark; unavailable cells are grey and marked N/A. The common scale compares
the sizes of changes across benchmarks, so small changes can appear pale.
The paired view uses terracotta losses, forest-green gains, and a cream zero midpoint,
with neutral grey missing cells. Family headers group the model sizes;
lettered panel headings and a compact shared legend replace the repeated
headline and footer. Numeric annotations are 7.5 pt at the native 7.2-inch
width. These styling choices preserve the underlying values, linear symmetric
colour limits, and zero midpoint. The adaptation setting belongs in the paper
caption. Accuracy/token heatmaps use the same formatting, with separate colour
scales and legends for each metric. Their token palette is reversed so that
fewer tokens are green and more tokens are terracotta.

The shared palette is defined in `HEATMAP_COLORS` near the top of
`scripts/plot_paper_results.py`:
`["#AC4035", "#E6AB8F", "#FCF6EA", "#A8C8A6", "#236548"]`.
The experimental palette choices and their generated files have been removed.
All heatmaps use this palette with the grouped-header layout and larger
numbers. The script exports both SVG and PDF, with matching filenames.

The reasoning-adaptation maths panel reads
`tab:paper7-reasoning-gym-results`, including that table's pretrained baseline.
These results currently cover Qwen3.5-4B with thinking disabled and average
five tasks, excluding TheoremQA. The mathematical-adaptation maths panel
averages six tasks including TheoremQA. Both include LogiQA, and the panel
titles disclose the task counts. Each cell uses its own suite/backbone/thinking
baseline; the five-task results are never normalised against the six-task
pretrained average. This aggregation difference must remain explicit in the
paper caption.

The paper includes the SVG using `\includesvg`, with `inkscapelatex=false`
to preserve its typography and `inkscapepath=svgdir` to keep generated vector
PDFs beside the source. The local
Tectonic build can reuse `mathematical_relative_paired_heatmap_svg-raw.pdf`
without Inkscape; this is a vector conversion of the SVG, not a raster image.
Regenerate figures in the analysis directory, then copy the selected SVG and
its PDF into the paper. For the existing `\includesvg` setup, copy the PDF
as `<figure-name>_svg-raw.pdf` to refresh the local build cache. The script
exports a vector PDF directly, so a separate conversion is no longer needed
for this workflow.
Overleaf can regenerate the conversion through the `svg` package. With
Inkscape installed, the conversion can also be refreshed explicitly:

```bash
inkscape figures/mathematical_relative_paired_heatmap.svg \
  --export-area-page --export-type=pdf \
  --export-filename=figures/mathematical_relative_paired_heatmap_svg-raw.pdf
```

The local conversion was produced using librsvg/Cairo because the Inkscape
executable is unavailable here. Both formats retain the same source artwork.

All four table-level suites, including mathematical/LogiQA transfer after
reasoning adaptation, are available through `--layout heatmap --option relative`.
The paper includes both paired accuracy heatmaps in the main text and all four
accuracy/token heatmaps in the corresponding appendix subsections. The reasoning
adaptation maths view averages five accuracy tasks and pools response lengths
over three generation tasks, using that table’s matched instruct baseline.

For example, refresh the mathematical-adaptation paired heatmap with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gcw-matplotlib \
  .venv/bin/python scripts/plot_paper_results.py --layout paired_heatmap --option relative
cp data/analysis/paper_results_figures/mathematical_relative_paired_heatmap.svg \
  src/Global-Cache-Workspace-Theory-ICLR-2027-/figures/
cp data/analysis/paper_results_figures/mathematical_relative_paired_heatmap.pdf \
  src/Global-Cache-Workspace-Theory-ICLR-2027-/figures/mathematical_relative_paired_heatmap_svg-raw.pdf
```

### Sequential domain adaptation

The sequential-adaptation tables use five mathematical/LogiQA tasks and five
KOR-Bench tasks, with thinking disabled. Generate matching individual and
paired heatmaps with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gcw-matplotlib \
  .venv/bin/python scripts/plot_paper_results.py --layout heatmap --option relative \
  --suite math_after_sequential --suite korbench_after_sequential
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gcw-matplotlib \
  .venv/bin/python scripts/plot_paper_results.py --layout paired_heatmap \
  --option relative --adaptation sequential
```

Both commands save SVG and PDF files in the analysis output directory.
The paired heatmap is included in the main-text sequential-adaptation subsection.
All changes use the matched pretrained instruct baseline, not the
mathematics-adapted checkpoint. Heatmap columns retain backbones and modes
with reported baselines even when adapted results are pending; those cells
remain N/A. Entirely missing panels use a symmetric fallback colour limit.

The main-text paired heatmaps omit Diffusion Longshot, which is presented
separately in the behavioural-reasoning subsection. The individual
accuracy/token heatmaps and appendix tables retain it. The paired manifest
records this exclusion; CSV exports still contain all methods.

### Longshot diffusion comparison

Generate the main-text difference heatmap with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gcw-matplotlib \
  .venv/bin/python scripts/plot_paper_results.py --layout diffusion_heatmap --option difference
```

`plot_diffusion_heatmap()` saves `diffusion_difference_heatmap.svg` and `.pdf`,
with its own CSV and manifest. Rows are backbones and columns are Maths/KOR-Bench
pairs grouped by the three training regimes. Values are Diffusion Longshot
minus Longshot accuracy in percentage points, using thinking-off results and
requiring both methods to be reported. Missing comparisons are grey N/A cells.
The shared linear colour scale is symmetric around zero, and annotations use
one decimal place. The six-task versus five-task maths aggregation follows the
source tables and is disclosed in the paper caption. The full absolute-score
table remains in the matching appendix subsection.


### Mean performance across campaigns

Generate the method-by-campaign overview with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gcw-matplotlib \
  .venv/bin/python scripts/plot_paper_results.py --layout campaign_heatmap --option relative
```

`plot_campaign_heatmap()` saves `campaign_relative_heatmap.svg` and `.pdf`.
It shares the diffusion comparison's layout and palette, with SFT, LoRA, BT,
BGT, and Longshot as rows and the same six Maths/KOR-Bench columns grouped by
training regime. Thinking-off accuracy is first normalised to the matching
pretrained instruct backbone using `100 * (accuracy / baseline_accuracy - 1)`.
Each cell is the arithmetic mean of those percentage changes across available
backbones, with equal weight per backbone. This is not a percentage change of
pooled mean accuracies. Diffusion Longshot is excluded.

Cell annotations show the mean with ± standard deviation beneath it. The SD
is computed across the same per-backbone percentage changes, using population
SD (`ddof=0`) to describe the available backbones; a single result has SD zero.
Colours encode the mean. Missing results are omitted rather than treated as
zero; cells without any results show N/A. Backbone coverage can differ between cells. The source tables use
six Maths tasks after mathematical adaptation and five after reasoning or
sequential adaptation. `campaign_heatmap_averages.csv` records each plotted
mean, SD, count, and contributing backbone list. Counts remain in the CSV
rather than the figure. The view also writes its own
source-values CSV and manifest. Existing figures and paper inclusions are
unchanged by this command.


For the matching token-usage overview, use:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gcw-matplotlib \
  .venv/bin/python scripts/plot_paper_results.py --layout campaign_heatmap --option relative --metric tokens
```

This saves `campaign_relative_tokens_heatmap.svg` and `.pdf`, with separate
`campaign_tokens_heatmap_averages.csv`, source-values CSV, and manifest files.
The cells report mean ± population SD of each backbone's percentage change in
mean response tokens against its matching pretrained instruct baseline.
Thinking is off, and each available backbone has equal weight. Green indicates
fewer tokens and terracotta more tokens; negative values are reductions.
The Maths columns use the source tables' pooled generation-task token means
(four tasks after mathematical adaptation, three after the other campaigns).
The accuracy overview is preserved. The `--metric` option applies only to the
campaign overview; the default is accuracy.
