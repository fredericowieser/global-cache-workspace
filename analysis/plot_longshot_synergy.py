#!/usr/bin/env python3
"""Plot Longshot performance and synergy across iterative steps."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = SCRIPT_DIR / "Longshot_synergy.csv"
OUT_DIR = (
    SCRIPT_DIR if SCRIPT_DIR.name == "analysis" else SCRIPT_DIR / "analysis"
)
SVG_PATH = OUT_DIR / "longshot_synergy_trend.svg"
PDF_PATH = OUT_DIR / "longshot_synergy_trend.pdf"

os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".matplotlib-cache"))

import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 18,
        "axes.titleweight": "semibold",
        "axes.labelsize": 14,
        "axes.labelweight": "normal",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "svg.fonttype": "none",
    }
)


def load_longshot_results(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required_columns = {
        "iterations",
        "gsm8k_acc",
        "raw_synergy_nats_per_token",
        "raw_synergy_ci_low",
        "raw_synergy_ci_high",
    }
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required columns in {path}: {missing}")

        iterations = []
        performance = []
        synergy = []
        synergy_low = []
        synergy_high = []
        for row in reader:
            iterations.append(int(row["iterations"]))
            performance.append(float(row["gsm8k_acc"]))
            synergy.append(float(row["raw_synergy_nats_per_token"]))
            synergy_low.append(float(row["raw_synergy_ci_low"]))
            synergy_high.append(float(row["raw_synergy_ci_high"]))

    if not iterations:
        raise ValueError(f"No rows found in {path}")

    return (
        np.array(iterations, dtype=int),
        np.array(performance, dtype=float),
        np.array(synergy, dtype=float),
        np.array(synergy_low, dtype=float),
        np.array(synergy_high, dtype=float),
    )


def padded_limits(
    values: np.ndarray, pad_fraction: float = 0.16, minimum_pad: float = 0.08
) -> tuple[float, float]:
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    y_pad = max((y_max - y_min) * pad_fraction, minimum_pad)
    return y_min - y_pad, y_max + y_pad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"CSV file to plot. Defaults to {DEFAULT_CSV_PATH}.",
    )
    parser.add_argument(
        "--hide-synergy-ci",
        action="store_true",
        help="Do not show the raw synergy confidence interval band.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iterations, performance, synergy, synergy_low, synergy_high = (
        load_longshot_results(args.csv)
    )

    OUT_DIR.mkdir(exist_ok=True)

    performance_color = "#1F957A"
    synergy_color = "#D6AB43"
    grid_color = "#D8DDE3"

    fig, ax = plt.subplots(figsize=(7.6, 3.2), facecolor="white")
    ax.set_facecolor("white")

    performance_line = ax.plot(
        iterations,
        performance,
        color=performance_color,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor=performance_color,
        markeredgewidth=2.2,
        linewidth=3.0,
        solid_capstyle="round",
        label="GSM8K accuracy",
    )[0]

    ax_synergy = ax.twinx()
    synergy_line = ax_synergy.plot(
        iterations,
        synergy,
        color=synergy_color,
        marker="D",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor=synergy_color,
        markeredgewidth=1.5,
        linestyle="--",
        dashes=(5, 4),
        linewidth=2.0,
        alpha=0.9,
        label="Raw synergy",
    )[0]
    synergy_ci = None
    if not args.hide_synergy_ci:
        synergy_ci = ax_synergy.fill_between(
            iterations,
            synergy_low,
            synergy_high,
            color=synergy_color,
            alpha=0.16,
            linewidth=0,
            label="Synergy 95% CI",
        )

    ax.set_xlabel("Iterative steps")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax_synergy.set_ylabel("Raw synergy (nats/token)")

    ax.set_xticks(iterations)
    ax.set_ylim(*padded_limits(performance, pad_fraction=0.18, minimum_pad=1.0))
    ax_synergy.set_ylim(
        *padded_limits(
            np.concatenate([synergy_low, synergy_high]),
            pad_fraction=0.16,
            minimum_pad=0.12,
        )
    )

    ax.grid(True, axis="y", color=grid_color, linewidth=0.9)
    ax.grid(True, axis="x", color=grid_color, linewidth=0.7, alpha=0.35)
    ax.tick_params(axis="both", colors="#222222", width=1.1, length=4)
    ax_synergy.tick_params(axis="y", colors=synergy_color, width=1.1)
    ax_synergy.yaxis.label.set_color(synergy_color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax_synergy.spines["top"].set_visible(False)
    ax_synergy.spines["right"].set_color(synergy_color)

    legend_handles = [performance_line, synergy_line]
    legend_labels = ["GSM8K accuracy", "Raw synergy"]
    if synergy_ci is not None:
        legend_handles.append(synergy_ci)
        legend_labels.append("Synergy 95% CI")
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
