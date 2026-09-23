# Main-text figure candidates

The user wants to see all three alternatives below in a future comparison.
They have not yet been generated:

1. Compact bar-chart panels: shared legends and outer axis labels, caption
   notes outside the artwork, and a possible 2-by-2 backbone selection with
   Qwen3.5-2B retained in the appendix.
2. Accuracy versus response-token scatter plots: one panel per backbone,
   method colours and thinking-mode markers, with relative changes on both
   axes and the pretrained instruct baseline at the origin.
3. Paired heatmaps: methods as rows, backbones and thinking modes as columns,
   separate accuracy and response-token panels, and missing results marked.

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
details belong in the eventual LaTeX caption. The three other alternatives
listed above remain pending.
