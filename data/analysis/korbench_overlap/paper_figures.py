"""Render the paired-correctness analysis at the paper's figure dimensions."""

import csv
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from normalize import IDENTITY, METHODS, OUT, REGIMES, normalize

PAPER = OUT.parents[2] / "src/Global-Cache-Workspace-Theory-ICLR-2027-"


def main():
    with (OUT / "summary.csv").open() as f:
        summary = list(csv.DictReader(f))
    with (OUT / "categories.csv").open() as f:
        rows = normalize(summary, list(csv.DictReader(f)))
    plt.rcParams.update(
        {"font.size": 8, "font.family": "DejaVu Sans", "svg.fonttype": "none"}
    )
    conditions = list(
        dict.fromkeys((r["regime"], r["model"], r["mode"]) for r in rows)
    )
    indexed = {tuple(r[k] for k in IDENTITY): r for r in rows}
    for methods, folder in [
        ([m for m in METHODS if m != "diffusion_longshot"], PAPER / "figures"),
        (
            ["longshot", "diffusion_longshot"],
            PAPER / "appendices/diffusion/figures",
        ),
    ]:
        folder.mkdir(parents=True, exist_ok=True)
        labels = [
            METHODS[m].replace("Diffusion Longshot", "Diffusion\nLongshot")
            for m in methods
        ]
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(6.5, 2.4 if len(methods) > 2 else 1.9),
            sharey=True,
            constrained_layout=True,
        )
        for ax, (regime, title) in zip(axes, REGIMES.items()):
            shared = [
                mean(
                    r["shared_success_pct"]
                    for r in rows
                    if r["regime"] == regime and r["method"] == m
                )
                for m in methods
            ]
            novel = [100 - s for s in shared]
            y = np.arange(len(methods))
            ax.barh(y, shared, color="#4c78a8", label="Shared with backbone")
            ax.barh(
                y,
                novel,
                left=shared,
                color="#e7a24a",
                label="New relative to backbone",
            )
            for i, value in enumerate(novel):
                ax.text(
                    102, i, f"{value:.1f}", ha="left", va="center", fontsize=7
                )
            ax.set_yticks(y, labels)
            ax.set_ylim(len(methods) - 0.5, -0.5)
            ax.set_xlim(0, 122)
            ax.set_xticks([0, 50, 100])
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Success share (%)", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", labelleft=ax is axes[0], length=0)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="outside lower center",
            ncol=2,
            frameon=False,
            fontsize=8,
        )
        for extension in ("pdf", "svg"):
            fig.savefig(
                folder / f"korbench_success_composition.{extension}",
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(fig)

        values = np.array(
            [
                [
                    indexed[(regime, model, m, mode)][
                        "category_normalized_overlap"
                    ]
                    for regime, model, mode in conditions
                ]
                for m in methods
            ]
        )
        fig, ax = plt.subplots(
            figsize=(6.5, 2.9 if len(methods) > 2 else 1.8),
            constrained_layout=True,
        )
        im = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_yticks(range(len(methods)), labels)
        tick_labels = [
            model.replace("llama32_", "L")
            .replace("qwen35_", "Q")
            .replace("b", "B")
            + " "
            + mode
            for _, model, mode in conditions
        ]
        ax.set_xticks(
            range(len(conditions)),
            tick_labels,
            rotation=60,
            ha="right",
            fontsize=7,
        )
        for i in range(len(methods)):
            for j in range(len(conditions)):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if values[i, j] > 0.55 else "black",
                )
        for x in (7.5, 11.5):
            ax.axvline(x, color="black", linewidth=0.8)
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
                fontsize=8,
            )
        bar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
        bar.set_label("Normalised overlap $S$", fontsize=8)
        bar.ax.tick_params(labelsize=7)
        for extension in ("pdf", "svg"):
            fig.savefig(
                folder / f"korbench_normalized_overlap.{extension}",
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(fig)


if __name__ == "__main__":
    main()
