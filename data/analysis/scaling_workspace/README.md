# Workspace scaling data

These files prepare the imported results for later plotting. Scores are Paper7 averages excluding TheoremQA, in percentage points (0–100). Each score is a benchmark average for one seed, not an average over seeds.

## Llama 3.2 1B

`llama_3.2_1B.csv` is the original table converted from tabs to commas, with its first column correctly labelled `seed`. Rows are seeds 24, 25 and 26; the remaining column names are `topk_heads` values: 1, 6, 13, 32, 64, 96 and 128. All 21 supplied scores are preserved. The import does not specify the training checkpoint or other settings.

`previous_example.png` is the historical reference plot. Its relative-complexity curve is an illustrative `y = x` reference, not measured compute data. For a later percentage axis, Llama's full workspace is 128 heads; use the actual head counts rather than rounded labels from the image.

## Qwen3.5 4B

`BGT - Qwen3.5-4B-IT.csv` is the unchanged source export. `qwen_35_4B.csv` contains 54 curve rows representing 45 unique source observations, with routing method, training steps, seed, `topk_heads`, score, source experiment name, source spreadsheet cell and a `shared_dense` flag. `head_routing` identifies the scaling series and applies to both training and evaluation at partial workspace sizes. At 100%, all heads are used and no selection is performed. The same nine dense observations appear in both series, marked `shared_dense=true` and retaining their original experiment name and source cell. These are shared endpoints, not independent observations; do not pool them across routing methods as additional samples. Cell addresses use the original export's row and column positions, including empty spacer columns.

Only groups covering all three workspace sizes (25%, 50% and 100%) are retained. Dense results can be shared between series differing only in head selection, provided all other recorded settings match:

| Head routing | Source experiment | `topk_heads` | Workspace percentage |
| --- | --- | --- | --- |
| magnitude | RAW-GD | 8 | 25 |
| magnitude | RAW-GD-MAG50 | 16 | 50 |
| magnitude | RAW-GD-DENSE | 32 | 100 |
| learned_contextual | RAW-GD-CTX25 | 8 | 25 |
| learned_contextual | RAW-GD-CTX50 | 16 | 50 |
| learned_contextual | RAW-GD-DENSE (shared) | 32 | 100 |

The percentage mapping follows the source's 25/50/dense experiment labels, with 32 heads representing the full workspace. Each retained routing/checkpoint group contains all three sizes for seeds 24, 25 and 26, at 1,000, 2,000 and 3,000 training steps. Keep routing method and training steps fixed when plotting a scaling curve; aggregate only over seeds. The learned-contextual groups qualify by sharing the dense endpoint at the same training step and seed. There is no 4,000-step scaling result in the export.

Selection compares all 23 configuration rows (source rows 2–24), allowing `topk_heads` at row 14 to vary. Training and evaluation head routing (rows 12–13) must match within each partial-workspace series but are ignored when sharing the dense endpoint. All other settings must match exactly. Experiment names identify provenance and are not settings. The fixed settings shared by the selected groups are:

| Setting | Source value |
| --- | --- |
| experiment_type | longshot_train_eval |
| backbone_source | raw_instruct |
| starting_backbone_sft_step | N/A |
| starting_backbone_sft_chat_template | N/A |
| sft_chat_template | TRUE |
| freeze_backbone | TRUE |
| precision | bf16 |
| training_rows | 132266:40:00 |
| inference_cadence | post_prefill_then_every_64 |
| scoring / token selection | kv_norm |
| Self-attn | FALSE |
| Gate | gated delta |
| update stream | KVs |
| linear_state_query_mode | per_linear_state |
| linear_state_update_mode | gated_delta |
| gate_linear_state_delta | FALSE |
| training iter. | every 64 tokens |
| eval iter. | post-prefill + every 64 generated tokens |
| re-selection | every 64 tokens |

The source's `training_rows` field appears time-formatted; its literal value is retained here without inferring a row count. Matching establishes equality of the exported settings only.

Scores come from source rows 27–29 (1,000 steps), 36–38 (2,000 steps) and 45–47 (3,000 steps). The latter blocks inherit the initial seed order 24, 25, 26. Source averages and standard deviations are omitted so later plots can compute summaries directly from the seed scores. The source labels its aggregate as five benchmarks; the cleaned score name uses the supplied Paper7-without-TheoremQA interpretation.

Other experiments change backbone source, gate or update settings, or lack a complete 25%/50%/100% series even after sharing dense results. In particular, the random-routing experiments have no matching 50% result and remain excluded. The later repeated blocks contain copies of the same three contextual-gate experiments at eight heads and supply no additional scaling observations.

## Plots

Run `python scripts/plot_workspace_scaling.py` from the repository root to regenerate the four figures in `plots/`, each as SVG and PDF. SVG and PDF files under this analysis directory are ignored by Git. `llama_3.2_1B` uses all seven supplied workspace sizes; `qwen_35_4B_steps_1000`, `qwen_35_4B_steps_2000` and `qwen_35_4B_steps_3000` overlay magnitude and learned-contextual routing with shared dense endpoints. Qwen figures share the same performance-axis limits across checkpoints.

Lines show the arithmetic mean across seeds 24, 25 and 26. Shading is mean ± one standard error, computed as the sample standard deviation (`ddof=1`) divided by the square root of three. Use `--band std --output-dir data/analysis/scaling_workspace/plots_std` for sample-standard-deviation bands instead. Shared dense observations are used once per series, retaining three seeds at 100%.

Workspace percentage is computed as `100 * topk_heads / total_heads`: Llama uses 128 total heads, giving 0.78125%, 4.6875%, 10.15625%, 25%, 50%, 75% and 100%; Qwen uses 32 total heads, giving 25%, 50% and 100%. The dashed line uses the right axis to show the illustrative linear complexity relation `y = x`, not measured runtime or compute.

The Llama figure uses a paper layout without a title or footer, with uncertainty shading identified in the legend and the x-axis labelled `Selected Heads (%)`. Its PDF is copied to the paper's `figures/workspace_scaling_llama_3_2_1b.pdf` and included directly in the Results subsection "Scaling the Workspace", avoiding an Inkscape conversion during compilation. An SVG copy is also retained alongside it. After regenerating the Llama figure, refresh these copies to update the paper. The local Tectonic executable is `C:/ProgramData/Miniconda3/envs/gcw-writeup/Library/bin/tectonic.exe`; run it on `MAIN.tex` from the paper directory.
