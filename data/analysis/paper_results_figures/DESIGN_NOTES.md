# Main-text figure candidates

## Paired-benchmark adaptation heatmaps

Added `mathematical_relative_paired_heatmap.svg` and
`reasoning_relative_paired_heatmap.svg`, using `plot_paired_heatmap()` in the
existing script (`--layout paired_heatmap --option relative`). Each pairs
maths/LogiQA accuracy with KOR-Bench accuracy and omits token changes. Both
panels share a colour scale, rows, and backbone/thinking columns; unreported
results remain grey N/A cells. Earlier figures are preserved.
The paired view now uses terracotta losses and forest-green gains, a cream
zero midpoint, clean white dividers, 7.5-pt numeric labels, and bold charcoal
headings. Backbone families group the model sizes; lettered panel labels and
a compact shared legend replace the repeated headline and colour footer.
The adaptation setting is supplied by the paper caption. The scale remains
linear and symmetric around zero.
The paper's SVG and local vector conversion use this updated styling.

Final palette selection: terracotta/forest green, retaining the grouped
backbone headers, compact legend, and larger numbers. The alternate palettes,
CLI selector, generated figures, and manifests have been removed.

The reasoning-adaptation maths data are present in the updated appendix under
`tab:paper7-reasoning-gym-results`. They cover all five backbones through Qwen3.5-27B and average five tasks without TheoremQA, unlike the six-task maths-trained average. Reasoning and sequential figures use thinking off for every backbone; mathematical-adaptation figures retain their existing thinking-mode coverage. The regenerated figures and source manifests are stored in `data/analysis/rg27b_paper_results/figures/`. Each uses its own matching pretrained baseline. Keep this distinction
in the paper caption.

The mathematical-adaptation paired heatmap is included using `\includesvg`
in the paper's results subsection. Its SVG is stored in the paper's `figures/`
directory, alongside a preconverted `_svg-raw.pdf` for local Tectonic builds
without Inkscape. The vector conversion retains the SVG artwork. PNG spot
checks and PNG export support have been removed; the script exports SVG and vector PDF.

## Earlier candidates

The preferred alternatives are the relative compact bars and heatmaps. Both
are retained for all three benchmark settings, alongside the earlier figures.
The relative trade-off SVGs and their manifest have been removed at the
user's request; the script still supports regenerating them.

1. Compact bar-chart panels (`*_relative_compact.svg`): shared legends and
   outer axis labels, caption notes outside the artwork, and a 2-by-2 backbone
   selection. Qwen3.5-2B is omitted from the maths-trained figures in this view
   only; Reasoning Gym retains all four backbones.
   Both compact and original relative bar views use independent accuracy and
   token-change scales, with each quantity's range shared across backbones.
   Both axes align zero at the same height to share one horizontal baseline.
2. Accuracy versus response-token scatter plots: one panel per backbone,
   method colours and thinking-mode markers, with relative changes on both
   axes and the pretrained instruct baseline at the origin
   (`*_relative_tradeoff.svg`). This alternative was rejected and its generated
   files removed.
3. Paired heatmaps: methods as rows, backbones and thinking modes as columns,
   separate accuracy and response-token panels, and missing results marked
   (`*_relative_heatmap.svg`). All backbones are retained.

## First candidate: generated point plots

The user's new proposal consolidates backbones into categorical point plots.
Instead of one subplot per backbone, use separate accuracy-change and
response-token-change panels. The x-axis contains adaptation methods; each
method column contains a point for every available backbone and thinking mode.

Approved encoding: colour and marker shape identify the
backbone; filled versus open markers identify thinking off versus on. Small,
fixed horizontal offsets within method columns separate backbones. Thinking
off/on share the same horizontal position, with larger markers and transparent
interiors for thinking-on markers to preserve overlapping filled points.
Use matching offsets in both panels, a shared method axis, and a zero reference
line. Retain all backbones initially. No connecting lines are needed.

Both quantities remain percentage changes from the pretrained instruct
baseline, matched by backbone, benchmark suite, and thinking mode. Missing
evaluations remain absent. Preserve the existing raw and relative SVG figures.

The user confirmed that method names such as Longshot and BT belong in their
own x-axis columns; the points within a column distinguish backbone and
thinking mode.

Implemented as `plot_points()` in `scripts/plot_paper_results.py`, selected
with `--layout points --option relative`. The three generated
`*_relative_points.svg` files preserve every reported backbone. The long
footer is omitted from the artwork for paper use; baseline and aggregation
details belong in the eventual LaTeX caption. The three alternatives above are
implemented as `plot_compact()`, `plot_tradeoff()`, and `plot_heatmap()` in the
same script. Select them with `--layout compact`, `--layout tradeoff`, or
`--layout heatmap`, together with `--option relative`. Compact bars and
heatmaps are the retained alternatives for further paper layout work. Both paired accuracy heatmaps are now in the main text; all four individual
accuracy/token heatmaps are in the appendix.

All accuracy/token heatmaps now share the paired-view palette, grouped family
headers, lettered panel headings, and 7.5-pt annotations. Each metric retains
its own symmetric colour scale and compact legend. The token palette is
reversed so that green denotes fewer tokens, while green denotes higher
accuracy in performance panels. Edit HEATMAP_COLORS in the script to update
the palette for every heatmap.

The fourth individual view, math_after_reasoning_relative_heatmap.svg, uses
the five-task accuracy and three-generation-task token averages from the
Reasoning Gym mathematical-transfer table. Both metrics use its matched
pretrained instruct baseline.

## Output location

Figures, plotted-value CSVs, and manifests default to
`data/analysis/paper_results_figures/`. Every plot is saved as both SVG and
vector PDF. Use `--output-dir PATH` for an explicit override. Only figures
used by the paper remain in its figures directory. Copy selected SVGs and PDFs
there when updating the paper; for the existing `\includesvg` setup, name
the copied PDF `<figure-name>_svg-raw.pdf`.
