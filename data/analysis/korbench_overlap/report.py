"""Summarize paired correctness, keeping diffusion analysis separate."""

import csv
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import OUT

rows = list(csv.DictReader((OUT / "summary.csv").open()))
methods = ["sft", "lora", "bt", "bgt", "longshot"]
names = dict(zip(methods, ["SFT", "LoRA", "BT", "BGT", "Longshot"]))
names["diffusion_longshot"] = "Diffusion Longshot"
regimes = {
    "math": "Mathematical",
    "reasoning": "Reasoning Gym",
    "sequential": "Sequential",
}


def aggregate_table(selected):
    lines = [
        "| Training | Method | Retained (%) | Recovered (%) | Correctness changed (%) | Accuracy change (pp) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for regime, title in regimes.items():
        for method in selected:
            rr = [
                r
                for r in rows
                if r["regime"] == regime and r["method"] == method
            ]
            numbers = [
                mean(float(r[k]) for r in rr)
                for k in (
                    "retained_pct",
                    "recovered_pct",
                    "changed_correctness_pct",
                    "accuracy_change_pp",
                )
            ]
            lines.append(
                f"| {title} | {names[method]} | "
                + " | ".join(f"{v:.2f}" for v in numbers)
                + " |"
            )
    return "\n".join(lines)


for selected, filename in [
    (methods, "overlap"),
    (["longshot", "diffusion_longshot"], "diffusion_overlap"),
]:
    conditions = list(
        dict.fromkeys((r["regime"], r["model"], r["mode"]) for r in rows)
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 5.8 if len(selected) == 5 else 4.5),
        constrained_layout=True,
    )
    labels = [
        m.replace("llama32_", "Llama ")
        .replace("qwen35_", "Qwen ")
        .upper()
        .replace("LLAMA", "Llama")
        .replace("QWEN", "Qwen")
        + (" on" if mode == "on" else " off")
        for _, m, mode in conditions
    ]
    for ax, metric, title, upper in zip(
        axes,
        ("retained_pct", "recovered_pct"),
        (
            "Backbone-correct answers retained (%)",
            "Backbone-wrong answers recovered (%)",
        ),
        (100, 25),
    ):
        data = np.array(
            [
                [
                    float(
                        next(
                            r[metric]
                            for r in rows
                            if (r["regime"], r["model"], r["mode"]) == condition
                            and r["method"] == method
                        )
                    )
                    for condition in conditions
                ]
                for method in selected
            ]
        )
        im = ax.imshow(data, vmin=0, vmax=upper, cmap="Blues", aspect="auto")
        ax.set_yticks(range(len(selected)), [names[m] for m in selected])
        ax.set_xticks(
            range(len(conditions)), labels, rotation=45, ha="right", fontsize=8
        )
        ax.set_title(title, loc="left", fontsize=12, pad=30)
        for i in range(len(selected)):
            for j in range(len(conditions)):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if data[i, j] > upper * 0.55 else "black",
                )
        for x in (7.5, 11.5):
            ax.axvline(x, color="black", linewidth=1.2)
        fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    for x, label in (
        (3.5, "Mathematical training"),
        (9.5, "Reasoning Gym"),
        (13.5, "Sequential training"),
    ):
        axes[0].text(x, -0.9, label, ha="center", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"{filename}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)

report = (
    """# KOR-Bench correctness overlap

All six appendix tables have two added columns. Diffusion appears only in the separate diffusion appendix; its analysis is in [diffusion.md](diffusion.md). All original accuracy and token entries are preserved.

A [follow-up analysis accounting for accuracy](normalization.md) adds shared/new-success shares and category-normalized overlap, with worked examples and plots. It qualifies the raw-retention interpretation below: higher retention alone does not establish closer behavior independently of overall accuracy.

## Definitions

For the same question, let B indicate that the pretrained instruct backbone was correct and F that the adapted model was correct. Retained = 100 × count(B and F) / count(B). Recovered = 100 × count(not B and F) / count(not B). Baselines compared with themselves therefore have 100% retained and 0% recovered. Thinking off and on are paired separately. Scores pool all 1,250 questions, with the denominator taken over the backbone's correct or incorrect subset, rather than averaging category percentages.

The complete paired contingency table is retained: both correct, backbone only correct (lost), adapted only correct (gained), and both wrong. Correctness changed = 100 × (lost + gained) / 1,250, while accuracy change = 100 × (gained − lost) / 1,250. If p is backbone accuracy and r and g are retained and recovered fractions, adapted accuracy is p r + (1 − p) g. Thus high retention and low recovery indicate closeness in correctness; high recovery alone does not imply better overall accuracy.

## Main-method comparison

These are equal-weight means across eight backbone/thinking conditions for mathematical training and four thinking-off backbones for each other regime. They are descriptive summaries of the available runs, not repeated independent trials or evidence of statistical significance. The mathematical conditions comprise Llama 1B/3B off and Qwen 2B/4B/27B off/on; the other regimes comprise Llama 1B/3B and Qwen 2B/4B off. Compare methods within a regime; the mathematical aggregate has a different model/mode composition.

"""
    + aggregate_table(methods)
    + """

![Per-condition retention and recovery](overlap.png)

- Longshot has the highest mean retention and the smallest mean correctness-change rate in all three training regimes. It has the highest retention in 9 of the 16 individual backbone/regime/mode conditions. The exceptions matter: BT leads for mathematical Llama 3B and Qwen 4B thinking-on, SFT for mathematical Qwen 27B thinking-on and Reasoning Gym Qwen 2B, and BGT for Qwen 4B after Reasoning Gym or sequential training. BT also leads for sequential Llama 3B.
- Divergence is usually dominated by lost successes: gained answers exceed lost answers in only 2 of the 80 main-method comparisons. Mathematical Qwen 27B SFT with thinking on retains 694/774 = 89.66% of backbone successes and recovers 103/476 = 21.64% of its errors. It loses 80 successes but gains 103, improving accuracy by 1.84 points. Reasoning Gym Llama 3B Longshot loses 41 and gains 43, a 0.16-point improvement. These small net improvements are descriptive, without a significance claim.
- Similar accuracy can hide considerable turnover. Mathematical Llama 1B Longshot scores 16.88% versus the backbone's 17.44%, yet loses 61 correct answers and gains 54. It preserves only 72.02% of backbone successes, and correctness changes on 115/1,250 = 9.20% of questions. After Reasoning Gym, Llama 3B Longshot has almost unchanged accuracy (33.44% versus 33.28%), with correctness changing on 84/1,250 = 6.72% of questions.
- The affected categories differ markedly. Mathematical Llama 3B Longshot retains only 7.43% of the backbone's counterfactual successes, losing 187 and gaining 1 in that category. Mathematical Qwen 2B Longshot retains 95.98% of counterfactual successes but only 12.21% of operation successes (187 lost and 7 gained). A single aggregate retention percentage conceals these differences; see `categories.csv`.
- Formatting failures explain part, but not most, of the lost successes. Across main-method comparisons, missing/empty answer delimiters occur in 28.46% of lost-success events after mathematical training, 26.46% after Reasoning Gym, and 25.39% after sequential training. These are pooled run-question events, not unique questions. The benchmark's parse-failure flag is narrower than all possible grading or answer-format problems.

Restricting mathematical training to the same four thinking-off backbones used in the other regimes gives Longshot 60.05% retained and 5.96% recovered, versus retention of 20.32% for SFT, 18.54% for LoRA, 46.17% for BT, and 32.76% for BGT. Its raw mean-retention lead therefore also holds in this matched subset; this still does not adjust for the adapted models' different accuracies.

## Coverage and interpretation

The audit covers 104 distinct complete evaluations (130,000 saved responses), reproducing every category accuracy and overall accuracy in all 144 reported table/mode entries. Repeated baseline and Longshot rows across the appendices are counted only once per comparison condition. There are 96 adapted-model comparisons (80 main methods and 16 diffusion), each paired on all 1,250 questions. Question identities and complete source documents match exactly. No model generation or training was rerun.

Some Llama comparisons have different `Today Date` strings in the system prompt: mathematical Llama 1B BT, BGT, and diffusion, and both Llama backbones after Reasoning Gym or sequential adaptation. All prompts match after replacing that date field, checked on every question. These differences are a confound for attributing changes solely to training. All other paired prompts match exactly. Thinking mode, chat formatting, generation caps, source data, and scoring identity were also checked.

These metrics describe overlap of correctness, not semantic equivalence of wrong answers, similarity of reasoning traces, internal representations, or distributional divergence. The `both_wrong` group may contain different wrong answers. `summary.csv` additionally records exact response-string agreement and exact nonempty extracted-answer agreement over all questions; these are literal equality measures, not semantic judgments. Category-specific denominators can be small, so consult counts alongside percentages. The experiments use one available training seed per condition and do not support a universal ranking of adaptation methods.

## Paper presentation

The paper presents the main-method findings in `appendices/E_Extended_Analysis.tex`, immediately after the extended results. Diffusion findings remain in the separate archive. The section motivates absolute retention/recovery, success composition, and category-normalised overlap in sequence, and qualifies the interpretation using the measure definitions and evaluation limitations.

Regenerate the four paper figures (PDF and SVG) from the validated count summaries with:

```bash
MPLCONFIGDIR=/tmp/gcw-overlap-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python data/analysis/korbench_overlap/paper_figures.py
```

Main-method assets are written to the paper's `figures/` directory; diffusion assets go only to `appendices/diffusion/figures/`. The paper includes the PDFs directly. The existing exploratory plots remain separate.

## Reproduction and files

Run from the repository root with the existing evaluation environment:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python data/analysis/korbench_overlap/analyze.py
.venv/bin/python data/analysis/korbench_overlap/update_tables.py
MPLCONFIGDIR=/tmp/gcw-overlap-mpl .venv/bin/python data/analysis/korbench_overlap/report.py
```

`analyze.py` uses the existing KOR-Bench scorer to rescore eight historical Qwen 27B evaluation files offline, bringing them to the same scoring convention as the paper. All other runs use their recorded per-example scores. Raw source files are unchanged. `provenance.json` records source paths, SHA256 hashes, original and effective scoring metadata, score-change counts, and the mapping to paper rows. The three complete original 27B thinking-on backbone/SFT/LoRA files were copied read-only from `nb_gpu:/opt/soperator-home/martin/global-cache-workspace/results/qwen27b-korbench-concurrent-20260920_162752/` into `data/analysis/local/korbench-overlap-20260924/sources/`; their names and hashes are in the manifest. Other raw files remain at their existing local/shared-storage paths. Reproduction requires those saved results and the evaluation dependencies.

- `normalization.md` and `diffusion_normalization.md`: follow-up explanations, results, and plots accounting for accuracy, reproduced with `normalize.py`.
- `normalization.csv`: per-condition success shares and category-normalized overlap with their supporting counts and reference values.
- `summary.csv`: paired counts and percentages for each regime/backbone/method/mode, including self-comparisons for baselines.
- `categories.csv`: the same contingency analysis for each KOR-Bench category.
- `paired_correctness.csv.gz`: source-question identities and paired binary correctness, sufficient to recompute all contingency statistics without model files or scoring dependencies.
- `provenance.json`: source and paper-row audit trail.
- `overlap.png` / `overlap.pdf`: main-method condition-level heatmaps.
- `diffusion.md` and `diffusion_overlap.png` / `.pdf`: separate diffusion comparisons.

`update_tables.py` inserts the two columns only into KOR-Bench tables and leaves a table unchanged if it already contains them. Existing unavailable thinking modes remain unavailable. The original accuracy and response-token columns are preserved.
"""
)
(OUT / "README.md").write_text(
    report
    + "\n## Validation\n\nIndependent recomputation from the compressed paired labels confirms every contingency count and conditional percentage. All 117 expanded KOR-Bench table rows preserve their original accuracy and token entries; all other tables are unchanged. The existing paper-figure parser accepts both updated appendices. The complete paper, including the diffusion archive, compiled successfully with the cached Tectonic toolchain, and the rendered table pages were checked. The build retains the pre-existing bibliography and underfull-box warnings. Analysis scripts pass Ruff checks, table updates are idempotent, and both repository diffs pass whitespace checks.\n"
)
(OUT / "diffusion.md").write_text(
    """# Separate diffusion overlap analysis

Diffusion results were added only to the three KOR-Bench tables in `E_Diffusion_Archive.tex`. A [separate follow-up](diffusion_normalization.md) accounts for accuracy using success composition and category-normalized overlap; the raw comparisons below alone do not establish greater behavioral divergence. Definitions, provenance, and limitations are shared with [the main report](README.md). Means below give equal weight to the eight mathematical backbone/thinking conditions and the four conditions in each other training regime.

"""
    + aggregate_table(["longshot", "diffusion_longshot"])
    + """

![Separate Longshot and diffusion comparison](diffusion_overlap.png)

Diffusion retains fewer pretrained successes than Longshot in 15 of 16 conditions. The exception is mathematical Llama 3B: diffusion retains 241/416 = 57.93%, versus Longshot's 155/416 = 37.26%, while recovering 21/834 = 2.52% versus 52/834 = 6.24%. Diffusion has lower mean recovery and greater mean correctness turnover in every regime. In all 16 diffusion conditions, lost successes exceed recovered errors.

The strongest separation is after direct Reasoning Gym adaptation: mean retention is 27.08% for diffusion versus 76.56% for Longshot, with mean recovery of 1.77% versus 4.31%. These results characterize the recorded checkpoints and evaluation settings, including the Llama system-date differences described in the main report.

The Qwen 27B thinking-on comparison uses the complete 1,250-question backbone evaluation, not the earlier partial recovery snapshot. Diffusion retains 496/774 = 64.08% and recovers 56/476 = 11.76%; it loses 278 successes and gains 56, for a 17.76-point accuracy decline.
"""
)
