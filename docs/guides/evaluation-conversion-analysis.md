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
`src/Global-Cache-Workspace-Theory-ICLR-2027-/appendices/C_Extended_Results.tex`
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
adaptation methods. Raw plots include the backbone baseline;
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
change against the same backbone's baseline in the same thinking mode;
the backbone columns are omitted. The panels use separate y-axis ranges.

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
  origin marks the backbone. The upper-left quadrant denotes higher accuracy
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

All six table-level suites are available through `--layout heatmap --option relative`. The main text uses the pooled stacked cross-campaign overview, and the corresponding appendix subsections contain the six individual accuracy/token heatmaps. The reasoning and sequential adaptation Maths views average five accuracy tasks and pool response lengths over three generation tasks, using each table’s matched backbone baseline.

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
The per-campaign paired heatmaps remain available as analysis figures; the main text uses the pooled stacked cross-campaign overview.
Both individual accuracy/token heatmaps are included in the matching appendix
subsection.
All changes use the matched backbone baseline, not the
mathematics-adapted checkpoint. Heatmap columns retain backbones and modes
with reported baselines even when adapted results are pending; those cells
remain N/A. Entirely missing panels use a symmetric fallback colour limit.

All submission tables and figures now omit Diffusion Longshot. Its methods, results, and original figures are preserved in the optional internal appendix `appendices/E_Diffusion_Archive.tex`, excluded by default through `\includeResearchArchivefalse` in the paper's `MAIN.tex`. Individual accuracy/token heatmaps include only methods present in the source table, so the ordinary `--layout heatmap --option relative` command now produces the submission views without an empty diffusion row. Paired and campaign heatmaps also exclude diffusion.

### Longshot diffusion comparison

The difference heatmap is retained only in the internal archive. The [archive maintenance notes](../../src/Global-Cache-Workspace-Theory-ICLR-2027-/appendices/diffusion/README.md#regenerating-the-archived-comparison) provide a command using the archived tables and their separate labels; the default submission tables no longer contain diffusion results.

`plot_diffusion_heatmap()` compares Diffusion Longshot minus Longshot accuracy in percentage points, with thinking disabled and both methods required for each cell. Rows are backbones and columns are Maths/KOR-Bench pairs grouped by the three training regimes. Missing comparisons are grey N/A cells. The shared linear colour scale is symmetric around zero, and annotations use one decimal place. Mathematical adaptation averages six tasks; direct Reasoning Gym and sequential adaptation average five, excluding TheoremQA. The archive retains the absolute-score comparison table and original SVG/PDF assets.


### Mean performance across campaigns

Generate the method-by-campaign overview with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gcw-matplotlib \
  .venv/bin/python scripts/plot_paper_results.py --layout campaign_heatmap --option relative
```

`plot_campaign_heatmap()` saves `campaign_relative_heatmap.svg` and `.pdf`. It shares the diffusion comparison's layout and palette, with FFT, LoRA, BT, BGT, and GCW as rows and the same six Maths/KOR-Bench columns grouped by training regime. Thinking-off accuracy is first normalised to the matching backbone using `100 * (accuracy / baseline_accuracy - 1)`. Each cell is the arithmetic mean of those percentage changes across available backbones, with equal weight per backbone. This is not a percentage change of pooled mean accuracies. Diffusion Longshot is excluded.

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
mean response tokens against its matching backbone baseline.
Thinking is off, and each available backbone has equal weight. Green indicates
fewer tokens and terracotta more tokens; negative values are reductions.
The Maths columns use the source tables' pooled generation-task token means
(four tasks after mathematical adaptation, three after the other campaigns).
The accuracy overview is preserved. The `--metric` option applies only to the
campaign overview; the default is accuracy.


### Thinking-enabled cross-campaign overviews

Use `--thinking-view pooled` to average the available thinking-off and thinking-on observations into the two existing Maths-training cells. Each backbone/mode observation contributes equal weight after normalisation to the backbone baseline for that same suite, backbone, and thinking mode. Thus, a backbone with both modes contributes twice, and a backbone with only one reported mode contributes once. The cell SD is the population SD across these observations, not a standard error or a standard deviation of backbone means. Reasoning Gym and sequential cells still use thinking off.

Use `--thinking-view split` to show eight columns: Maths off, Maths on, KOR-Bench off, and KOR-Bench on under Maths training, followed by the existing two thinking-off columns for each other campaign. Each cell shows mean ± population SD across the available backbones for its mode. Both views retain all available backbones, omit diffusion, and preserve the thinking-off-only figures.

```bash
for metric in accuracy tokens; do
  for view in pooled split; do
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gcw-matplotlib \
      .venv/bin/python scripts/plot_paper_results.py --layout campaign_heatmap \
      --option relative --metric "$metric" --thinking-view "$view"
  done
done
```

The new SVG/PDF filenames append `_thinking_pooled` or `_thinking_split` to the existing figure stems, with separate CSVs and manifests. CSVs record both distinct backbone counts and backbone/mode observation counts and identities. Missing observations are omitted rather than treated as zero. The default `--thinking-view off` retains the original view. This option applies only to campaign heatmaps.


### Side-by-side performance and token overview

The preferred cross-campaign summary pools the maths-training thinking modes. To place performance and token usage side by side in a single figure, use `--metric both`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gcw-matplotlib \
  .venv/bin/python scripts/plot_paper_results.py --layout campaign_heatmap \
  --option relative --metric both --thinking-view pooled
```

This saves `campaign_relative_overview_heatmap_thinking_pooled.svg` and `.pdf`, with separate overview CSVs and a manifest. Both panels reuse the existing aggregation and rendering logic and show means only; population standard deviations remain in the averages CSV. Performance is on the left and token usage on the right, each with its own benchmark headings and symmetric colour scale. Green denotes improved accuracy or reduced token usage. The 7.2-by-2.6-inch figure uses 9.5-point cell labels and 11.5-point panel titles (approximately 7.3 and 8.8 points at the paper’s text width), consistent with the other paper figures. The caption describes the averaging and thinking-mode coverage. The GCW row label is bold. `--thinking-view split` remains available with a wider figure for separate thinking-mode columns. Standalone accuracy and token heatmaps retain their mean ± SD annotations.

Paper method labels are read from the `\newcommand{\method...}{\textsc{...}}` definitions in `MAIN.tex`: `\methodSFT{}` renders as FFT and `\methodLongshot{}` as GCW. Both the aggregate plotting script and `data/analysis/korbench_overlap/paper_figures.py` use these names, with built-in FFT/GCW defaults if `MAIN.tex` or the corresponding command is absent. Without the paper submodule, the aggregate plotting script still needs an available results table supplied through `--source`; backend method names remain `SFT` and `Longshot` in parsed records, CSVs, and manifests, as do experiment and checkpoint identifiers. `METHOD_LABELS` applies paper aliases only when rendering plot labels. Table parsing accepts both `\methodSFT{}` and historical literal `SFT` rows. The unadapted checkpoint is labelled Backbone in tables and plots; parsers also accept historical `Pretrained instruct` rows, while parsed records retain the internal `Instruct` method identifier. GCW labels are bold in the appendix heatmaps and correctness plots. Individual, paired, and campaign heatmaps display the N/A legend only when their plotted cells contain missing values.
