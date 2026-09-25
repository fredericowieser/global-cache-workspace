# Cache processor scaling data

The three `BGT - ...csv` files are unchanged source exports. The cleaned CSVs retain seed-level results for groups with all three target backbone fractions: 0.015, 0.05 and 0.1 (1.5%, 5% and 10%). These are target fractions, not measured parameter counts.

| Cleaned file | Source file | Selected heads | Training steps | Observations |
| --- | --- | --- | --- | --- |
| `llama_3.2_1B.csv` | `BGT - L3.2-1B.csv` | 32 | 2000 | 9 |
| `llama_3.2_3B.csv` | `BGT - L3.2-3B.csv` | 32 | 2000 | 9 |
| `llama_3.2_3B.csv` | `BGT - L3.2-3B.csv` | 56 | 1000, 2000, 3000 | 27 |
| `qwen_35_2B.csv` | `BGT - Qwen3.5-2B-IT.csv` | 32 | 1000, 2000, 3000 | 27 |

Each row records `topk_heads`, `training_steps`, `seed`, `target_backbone_fraction`, `paper7_avg_no_theoremqa_pct` and `source_cell`. Scores are the supplied averages over five benchmarks, in percentage points (0–100), using the same Paper7-without-TheoremQA interpretation as the workspace-scaling data. Source cells refer to the original spreadsheet positions, including blank rows and spacer columns. Keep head count and training steps fixed when comparing processor sizes; aggregate only across seeds 24, 25 and 26.

## Extraction rules

- The initial seed block uses the default target fraction 0.015. Only blocks explicitly labelled `CP 0.05` or `CP 0.10` in the relevant experiment columns override it. Labels in unrelated columns do not apply across the whole row.
- Training steps come from the column's `n. steps` entry; an absent entry means 2000. This default applies to Llama 1B.
- Later processor-size blocks inherit the same column settings and seed order (24, 25, 26) as the initial block.
- Qwen columns D–F form one experiment with successive checkpoints. Settings listed in D apply to E and F as well; blank cells in those continuation columns do not define different settings. The named Qwen experiments starting at H, K, N and R are separate groups.
- Only raw seed scores are extracted. Existing rounded averages, standard deviations, error cells and backbone-only results are omitted.

## Matched settings and source locations

All retained series use `kv_norm` scoring/token selection, `Self-attn=FALSE`, a `learned sigmoid scalar` gate and `KVs` as the update stream. Llama series additionally specify `None` for training iteration, evaluation iteration and re-selection. Qwen's selected group lists `N/A` for linear-state query mode, linear-state update mode and linear-state delta gating; its backbone-source and other upper configuration fields are unspecified and are not inferred from adjacent experiments.

| Model | Source columns | 0.015 seed rows | 0.05 seed rows | 0.1 seed rows |
| --- | --- | --- | --- | --- |
| Llama 1B | J | 11–13 | 20–22 | 29–31 |
| Llama 3B | B, D, E, F | 12–14 | 21–23 | 30–32 |
| Qwen 2B | D, E, F | 22–24 | 31–33 | 40–42 |

Llama 3B column B is a distinct 32-head comparison and must not be pooled with the 56-head series in D–F, even at the shared 2000-step checkpoint. The other Llama 1B ablations and the named Qwen experiments do not contain matching 0.05 and 0.1 results, so they are excluded.

## Plot

Run `python scripts/plot_cache_processor_scaling.py` from the repository root to produce `plots/cache_processor_scaling_steps_2000.svg` and `.pdf`. Both output formats are ignored by Git in this analysis directory. The compact plot overlays all three models at 2000 steps, selecting 32 heads for Llama 1B and Qwen 2B and 56 heads for Llama 3B. The x-axis, "Target processor size (% of backbone parameters)", converts target fractions to 1.5%, 5% and 10%. For each backbone, its baseline is the arithmetic mean of the nine selected scores (three seeds × three sizes). Each score is normalised as `100 * (score / baseline - 1)`, so zero is that backbone's overall mean. Lines average these relative scores across seeds 24, 25 and 26; shading shows mean ± one sample standard deviation (`ddof=1`) on the same relative scale, rather than standard error. The single baseline per backbone is shared across all sizes and seeds; no other checkpoints or head-count groups enter it. The plot does not include a compute-complexity reference, since the target parameter fraction is not a measured compute cost.

The figure's PDF and SVG are copied to the paper's `figures/` directory. The Results subsection "Scaling the Cache Processor" includes the PDF at 90% of the text width. Refresh both copies after regenerating the plot, then compile `MAIN.tex` with `C:/ProgramData/Miniconda3/envs/gcw-writeup/Library/bin/tectonic.exe`.
