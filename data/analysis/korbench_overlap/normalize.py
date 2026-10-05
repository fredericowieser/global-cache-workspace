"""Compare the locations of successes while accounting for accuracy."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
IDENTITY = ("regime", "model", "method", "mode")
METHODS = {
    "sft": "SFT",
    "lora": "LoRA",
    "bt": "BT",
    "bgt": "BGT",
    "longshot": "Longshot",
    "diffusion_longshot": "Diffusion Longshot",
}
REGIMES = {
    "math": "Mathematical",
    "reasoning": "Reasoning Gym",
    "sequential": "Sequential",
}


def normalize(summary, categories):
    grouped = defaultdict(list)
    for row in categories:
        grouped[tuple(row[k] for k in IDENTITY)].append(row)
    normalized = []
    for row in summary:
        key = tuple(row[k] for k in IDENTITY)
        cells = grouped[key]
        assert len(cells) == 5
        assert len({c["category"] for c in cells}) == 5
        for field in ("n", "both_correct", "lost", "gained", "both_wrong"):
            assert sum(int(c[field]) for c in cells) == int(row[field])
        expected = 0.0
        maximum = 0
        minimum = 0
        for cell in cells:
            n = int(cell["n"])
            b = int(cell["backbone_correct"])
            f = int(cell["both_correct"]) + int(cell["gained"])
            expected += b * f / n
            maximum += min(b, f)
            minimum += max(0, b + f - n)
        shared = int(row["both_correct"])
        gained = int(row["gained"])
        successes = shared + gained
        assert minimum <= shared <= maximum
        normalized.append(
            {k: row[k] for k in IDENTITY}
            | dict(
                n=int(row["n"]),
                backbone_correct=int(row["backbone_correct"]),
                method_correct=successes,
                both_correct=shared,
                gained=gained,
                shared_success_pct=100 * shared / successes
                if successes
                else None,
                new_success_pct=100 * gained / successes if successes else None,
                expected_shared=expected,
                maximum_shared=maximum,
                minimum_shared=minimum,
                category_normalized_overlap=(shared - expected)
                / (maximum - expected)
                if maximum > expected
                else None,
            )
        )
    return normalized


def aggregate_table(rows, methods):
    lines = [
        "| Training | Method | Shared successes (%) | New successes (%) | Category-normalized overlap |",
        "|---|---|---:|---:|---:|",
    ]
    for regime, title in REGIMES.items():
        for method in methods:
            group = [
                r
                for r in rows
                if r["regime"] == regime and r["method"] == method
            ]
            shared, new, overlap = [
                mean(r[field] for r in group if r[field] is not None)
                for field in (
                    "shared_success_pct",
                    "new_success_pct",
                    "category_normalized_overlap",
                )
            ]
            lines.append(
                f"| {title} | {METHODS[method]} | {shared:.2f} | {new:.2f} | {overlap:.3f} |"
            )
    return "\n".join(lines)


def plot_comparison(rows, methods, prefix):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 3.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for ax, (regime, title) in zip(axes, REGIMES.items()):
        shares = [
            mean(
                r["shared_success_pct"]
                for r in rows
                if r["regime"] == regime and r["method"] == method
            )
            for method in methods
        ]
        novel = [100 - share for share in shares]
        y = np.arange(len(methods))
        ax.barh(y, shares, color="#4c78a8", label="Already correct in backbone")
        ax.barh(
            y,
            novel,
            left=shares,
            color="#e7a24a",
            label="Wrong in backbone (new)",
        )
        for i, (old, new) in enumerate(zip(shares, novel)):
            ax.text(
                old / 2,
                i,
                f"{old:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
            ax.text(
                old + new / 2,
                i,
                f"{new:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
            )
        ax.set_yticks(y, [METHODS[m] for m in methods])
        ax.set_ylim(len(methods) - 0.5, -0.5)
        ax.tick_params(axis="y", labelleft=ax is axes[0])
        ax.set_xlim(0, 100)
        ax.set_title(title)
        ax.set_xlabel("Share of correct answers (%)")
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=2, frameon=False
    )
    fig.suptitle(
        "Where do the adapted model's successes come from?", fontsize=14
    )
    for extension in ("png", "pdf"):
        fig.savefig(
            OUT / f"{prefix}_success_composition.{extension}",
            dpi=180,
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close(fig)

    conditions = list(
        dict.fromkeys((r["regime"], r["model"], r["mode"]) for r in rows)
    )
    indexed = {tuple(r[k] for k in IDENTITY): r for r in rows}
    values = np.array(
        [
            [
                indexed[(regime, model, method, mode)][
                    "category_normalized_overlap"
                ]
                for regime, model, mode in conditions
            ]
            for method in methods
        ],
        dtype=float,
    )
    labels = [
        model.replace("llama32_", "Llama ")
        .replace("qwen35_", "Qwen ")
        .replace("b", "B")
        + " "
        + mode
        for _, model, mode in conditions
    ]
    fig, ax = plt.subplots(
        figsize=(15, 4 if len(methods) > 2 else 3.2), constrained_layout=True
    )
    im = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(
        range(len(conditions)), labels, rotation=45, ha="right", fontsize=9
    )
    ax.set_yticks(range(len(methods)), [METHODS[m] for m in methods])
    ax.set_title(
        "Shared successes beyond the category-matched random expectation",
        loc="left",
        pad=34,
    )
    for i in range(len(methods)):
        for j in range(len(conditions)):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if values[i, j] > 0.55 else "black",
            )
    for x in (7.5, 11.5):
        ax.axvline(x, color="black", linewidth=1.2)
    for x, title in (
        (3.5, "Mathematical"),
        (9.5, "Reasoning Gym"),
        (13.5, "Sequential"),
    ):
        ax.text(
            x,
            1.025,
            title,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    bar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    bar.set_label("Normalized overlap\n0: random; 1: maximum", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(
            OUT / f"{prefix}_normalized_overlap.{extension}",
            dpi=180,
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    with (OUT / "summary.csv").open() as f:
        summary = list(csv.DictReader(f))
    with (OUT / "categories.csv").open() as f:
        categories = list(csv.DictReader(f))
    rows = normalize(summary, categories)
    with (OUT / "normalization.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    methods = [m for m in METHODS if m != "diffusion_longshot"]
    plot_comparison(rows, methods, "main")
    plot_comparison(rows, ["longshot", "diffusion_longshot"], "diffusion")
    (OUT / "normalization.md").write_text(
        """# Retention and recovery after accounting for accuracy

This follow-up qualifies the [original analysis](README.md): Longshot's higher raw retention establishes that it preserves more backbone successes, but does not by itself establish greater similarity independently of overall accuracy. A model with more correct answers can have both more retained successes and more recovered errors. This analysis uses the existing paired counts; it changes no paper tables. Diffusion comparisons are [reported separately](diffusion_normalization.md).

## First measure: where do the adapted model's successes come from?

Take only the questions that the adapted model got right. What percentage did the backbone also get right, and what percentage did the backbone get wrong?

- Shared-success share = 100 × both correct / adapted correct.
- New-success share = 100 × adapted correct but backbone wrong / adapted correct.

These percentages add to 100. Their denominator is the adapted model's successes. The original Retained column divides by the backbone's successes; the original Recovered column divides by the backbone's errors. Thus new-success share and Recovered answer different questions. “New” denotes a success relative to the backbone on this question, not evidence of a new reasoning mechanism.

For mathematical Qwen 4B with thinking disabled:

| Method | Adapted successes | Shared with backbone | New successes | Original retention | Shared-success share | New-success share |
|---|---:|---:|---:|---:|---:|---:|
| BT | 522 | 506 | 16 | 66.58% | 96.93% | 3.07% |
| Longshot | 597 | 567 | 30 | 74.61% | 94.97% | 5.03% |

Longshot preserves more successes overall, while BT's successes are more concentrated on questions already solved by the backbone. Neither description contradicts the other. A larger new-success share indicates greater complementarity among successful answers; it does not imply higher accuracy or greater total recovery.

![Composition of adapted-model successes](main_success_composition.png)

Bars are equal-weight means of per-condition percentages, not percentages obtained by pooling every run's answers. Mathematical training has eight conditions (Llama 1B/3B off; Qwen 2B/4B/27B off/on); Reasoning Gym and sequential training each have four (Llama 1B/3B and Qwen 2B/4B off). Compare methods within a training regime. Shares can be unstable when a method has few successes. Across backbones, the available fraction of backbone-wrong questions also changes. With independently allocated successes, expected new-success share equals the backbone's error rate, rather than zero.

## Second measure: does the overlap exceed what the scores already predict?

Imagine shuffling the adapted model's correct/incorrect labels among questions within each KOR-Bench category. Keep the backbone's answers fixed. The adapted model keeps exactly the same number of correct answers in every category; only the identities of its successful questions change. This is a reference calculation, not a simulation of model behavior or a new evaluation.

The measure asks how far the real overlap lies between the overlap expected after this shuffle and the largest overlap possible with those same scores.

### A small worked example

Suppose one category contains 100 questions. The backbone gets 60 right and the adapted model gets 40 right. Of those 40 successes, 30 are shared with the backbone and 10 are new.

1. Randomly placing 40 successes among 100 questions would put 60% of them in the backbone's successful set. Expected shared successes: 40 × 0.60 = 24.
2. The most shared successes possible is 40, since all 40 adapted successes could fit inside the backbone's 60 successes.
3. The observed 30 shared successes are 6 above the expectation, out of a possible excess of 16. The normalized score is (30 − 24) / (40 − 24) = 0.375.

This means the overlap is 37.5% of the way from the random expectation to the maximum feasible overlap. It does **not** mean 37.5% of backbone answers were retained: ordinary retention here is 30/60 = 50%, and new-success share is 10/40 = 25%.

For multiple categories, add each category's expected overlap and maximum overlap before normalizing. If category c contains n_c questions, b_c backbone successes and f_c adapted successes, and O is the observed total of shared successes:

```text
E = sum_c (b_c * f_c / n_c)       expected shared successes
U = sum_c min(b_c, f_c)           maximum feasible shared successes
S = (O - E) / (U - E)            category-normalized overlap
```

S = 1 means the largest possible overlap at the observed category accuracies. S = 0 means overlap equal to the category-matched random expectation. Negative values are possible and mean less overlap than expected; the negative range need not stop at −1. If U = E, there is no overlap freedom under this normalization and S is undefined. Every reported comparison here has U > E and positive S. The heatmaps therefore use a shared 0–1 color scale. This is a proposed descriptive normalization, not a significance test or a claim of an established metric name.

The category adjustment matters because two models may overlap simply because both do well on counterfactual questions and poorly on puzzles. It holds those category accuracies fixed, while still allowing shared question difficulty within categories to explain some overlap.

![Category-normalized overlap for each condition](main_normalized_overlap.png)

## Results and revised interpretation

The two success shares below are percentages; the final column is a dimensionless score. All entries are equal-weight means of per-condition values, with the same condition sets as the first plot.

"""
        + aggregate_table(rows, methods)
        + """

- After mathematical training, category-normalized overlap is essentially tied between BT (0.630) and Longshot (0.626), with BGT close behind (0.614). The raw-retention lead does not establish a clear normalized lead for Longshot here.
- After Reasoning Gym, Longshot's normalized overlap remains higher (0.741 versus 0.611 for the next-highest method, BT). Only 8.97% of its successful answers are new relative to the backbone, versus 16.67–18.69% for the other methods. This supports stronger concentration on backbone successes in the recorded runs.
- After sequential training, Longshot's normalized lead is modest (0.614 versus 0.591 for BGT). Its mean new-success share is 14.32%, versus 16.88–18.93% for the other methods.

Raw retained and recovered rates remain useful measures of absolute preservation and recovery. The new measures describe how the available successes are distributed. At fixed backbone accuracy p and adapted accuracy q, retention r and recovery g obey q = p r + (1 − p) g. Consequently, after fixing performance, greater retention necessarily trades off against recovery; they cannot be treated as independent abilities. The second measure also accounts for successes that must be new whenever adapted accuracy in a category exceeds backbone accuracy.

No normalization completely isolates preservation of internal model behavior. The shared-difficulty explanation, system-date differences in some Llama comparisons, single-seed coverage, and possible answer-format effects still apply. The first measure remains constrained by the two accuracies, and the second depends on the chosen category-based reference. Stronger follow-up checks would hold accuracy fixed within individual rules and estimate uncertainty by resampling rules. Neither is performed here, and no statistical-significance claim is made. Diffusion has its own comparison rather than being included in the main-method ranking.

## Reproduction

From the repository root:

```bash
MPLCONFIGDIR=/tmp/gcw-overlap-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python data/analysis/korbench_overlap/normalize.py
```

This reads `summary.csv` and `categories.csv`, validates that category contingency counts sum to the corresponding overall counts, and writes `normalization.csv`, this report, the separate diffusion report, and both pairs of PNG/PDF plots. `normalization.csv` contains all 112 comparison conditions, including baseline self-comparisons, and exposes the success counts, both success shares, expected overlap, minimum/maximum feasible overlap, and normalized overlap. The plots exclude baseline self-comparisons. Empty CSV metric cells indicate an undefined denominator. Original score and sample provenance is in `provenance.json`; no raw results or model execution are needed for this follow-up.
"""
    )
    (OUT / "diffusion_normalization.md").write_text(
        """# Separate diffusion analysis after accounting for accuracy

Definitions and a worked example are in [the main normalization analysis](normalization.md). This comparison uses the same saved evaluations as the [original diffusion analysis](diffusion.md). It changes no paper tables.

![Composition of Longshot and diffusion successes](diffusion_success_composition.png)

![Category-normalized Longshot and diffusion overlap](diffusion_normalized_overlap.png)

Means give equal weight to eight backbone/thinking conditions after mathematical training and four thinking-off conditions in each other training regime. Shared and new success shares sum to 100%; category-normalized overlap is a dimensionless score.

"""
        + aggregate_table(rows, ["longshot", "diffusion_longshot"])
        + """

Accounting for accuracy substantially weakens the raw-retention contrast after mathematical training: category-normalized overlap is 0.615 for diffusion and 0.626 for Longshot, while new-success shares are 12.82% and 13.78%, respectively. Diffusion's successes are therefore not generally more novel simply because its raw retention is lower. After Reasoning Gym the normalized gap remains large (0.514 versus 0.741); after sequential training it is smaller (0.579 versus 0.614). The two normalizations need not rank methods identically because one describes success composition and the other also conditions on category accuracies.

These are descriptive results, without significance tests. Single-seed coverage, shared question difficulty, and the documented Llama system-date differences remain limitations. All numeric inputs and derived quantities are in `normalization.csv`; regeneration uses `normalize.py` as described in the main report.
"""
    )
    print(
        f"Wrote normalization for {len(rows)} comparisons, two reports, and four PNG/PDF plot pairs."
    )


if __name__ == "__main__":
    main()
