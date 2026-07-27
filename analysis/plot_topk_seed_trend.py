#!/usr/bin/env python3
"""Plot top-k heads average performance across seeds."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "analysis" else SCRIPT_DIR
DEFAULT_CSV_PATH = ROOT / "topk_seed_averages.csv"
OUT_DIR = SCRIPT_DIR if SCRIPT_DIR.name == "analysis" else SCRIPT_DIR / "analysis"
SVG_PATH = OUT_DIR / "topk_seed_average_trend.svg"
PDF_PATH = OUT_DIR / "topk_seed_average_trend.pdf"

os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".matplotlib-cache"))

import matplotlib.pyplot as plt  # noqa: E402


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.titleweight": "semibold",
        "axes.labelsize": 12,
        "axes.labelweight": "normal",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.15,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "svg.fonttype": "none",
    }
)

EMBEDDED_HEADS = np.array([1, 6, 13, 32, 64, 96, 128], dtype=int)
EMBEDDED_SEED_RESULTS = np.array(
    [
        # seed 24
        [27.28, 27.46, 27.61, 28.71, 28.29, 27.88, 28.53],
        # seed 25
        [27.21, 27.77, 27.90, 27.36, 28.36, 27.94, 27.82],
        # seed 26
        [27.66, 27.94, 28.23, 27.66, 27.58, 28.39, 28.22],
    ],
    dtype=float,
)


def is_seed(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def load_seed_results(path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if path is None:
        return EMBEDDED_HEADS, EMBEDDED_SEED_RESULTS

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        heads = np.array([int(name) for name in reader.fieldnames[1:]], dtype=int)
        rows = []
        for row in reader:
            seed = (row.get("seed") or "").strip()
            if not is_seed(seed):
                continue
            rows.append([float(row[str(head)]) for head in heads])

    if not rows:
        raise ValueError(f"No numeric seed rows found in {path}")

    return heads, np.array(rows, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--individual-seeds",
        action="store_true",
        help="Plot each seed as a low-alpha trend line instead of the standard-deviation band.",
    )
    parser.add_argument(
        "--hide-complexity",
        action="store_true",
        help="Do not overlay the relative linear complexity curve.",
    )
    parser.add_argument(
        "--normalize-seeds",
        action="store_true",
        help="Mean-center each seed before averaging, preserving within-seed trends while removing seed-level offsets.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=f"Optional CSV to load instead of the embedded seed results, e.g. {DEFAULT_CSV_PATH}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heads, values = load_seed_results(args.csv)
    head_percentages = heads / heads.max() * 100.0
    plot_values = values
    ylabel = "Average performance"
    mean_label = "Mean"
    if args.normalize_seeds:
        seed_offsets = values.mean(axis=1, keepdims=True)
        grand_mean = values.mean()
        plot_values = values - seed_offsets + grand_mean
        ylabel = "Normalized average performance"
        mean_label = "Normalized mean"

    mean = plot_values.mean(axis=0)
    stderr = (
        plot_values.std(axis=0, ddof=1) / np.sqrt(plot_values.shape[0])
        if plot_values.shape[0] > 1
        else np.zeros_like(mean)
    )

    OUT_DIR.mkdir(exist_ok=True)

    performance_color = "#1F957A"
    seed_color = "#1F957A"
    complexity_color = "#D6AB43"
    grid_color = "#D8DDE3"

    fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor="white")
    ax.set_facecolor("white")
    if args.individual_seeds:
        for seed_values in plot_values:
            ax.plot(
                head_percentages,
                seed_values,
                color=seed_color,
                alpha=0.32,
                linewidth=1.65,
                solid_capstyle="round",
            )

    ax.plot(
        head_percentages,
        mean,
        color=performance_color,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor=performance_color,
        markeredgewidth=2.2,
        linewidth=3.0,
        solid_capstyle="round",
        label=mean_label,
    )
    if not args.individual_seeds:
        ax.fill_between(
            head_percentages,
            mean - stderr,
            mean + stderr,
            color=performance_color,
            alpha=0.14,
            linewidth=0,
            label="Std. err.",
        )

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if not args.hide_complexity:
        ax_complexity = ax.twinx()
        relative_complexity = head_percentages
        complexity_line = ax_complexity.plot(
            head_percentages,
            relative_complexity,
            color=complexity_color,
            marker="D",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=complexity_color,
            markeredgewidth=1.5,
            linestyle="--",
            dashes=(5, 4),
            linewidth=2.0,
            alpha=0.9,
            label="Relative complexity",
        )[0]
        ax_complexity.set_ylabel("Relative complexity")
        ax_complexity.set_ylim(0, 105)
        ax_complexity.spines["top"].set_visible(False)
        ax_complexity.spines["right"].set_color(complexity_color)
        ax_complexity.tick_params(axis="y", colors=complexity_color, width=1.1)
        ax_complexity.yaxis.label.set_color(complexity_color)
        legend_handles.append(complexity_line)
        legend_labels.append("Relative complexity")

    ax.set_xlabel("Spotlight Size")
    ax.set_ylabel(ylabel)
    ax.set_xticks(head_percentages)
    ax.set_xticklabels([f"{x:.0f}%" if x >= 10 else f"{x:.1f}%" for x in head_percentages])
    y_min = float(np.min(mean - stderr))
    y_max = float(np.max(mean + stderr))
    y_pad = max((y_max - y_min) * 0.22, 0.08)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(True, axis="y", color=grid_color, linewidth=0.9)
    ax.grid(True, axis="x", color=grid_color, linewidth=0.7, alpha=0.35)
    ax.tick_params(axis="both", colors="#222222", width=1.1, length=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(legend_handles),
        handlelength=2.6,
        columnspacing=1.4,
    )
    fig.tight_layout()

    fig.savefig(SVG_PATH)
    fig.savefig(PDF_PATH)
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
