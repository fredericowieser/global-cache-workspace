#!/usr/bin/env python3
"""Plot relative performance per relative complexity for top-k heads."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "topk_seed_averages.csv"
OUT_DIR = ROOT / "analysis"
SVG_PATH = OUT_DIR / "topk_efficiency_ratio.svg"
PDF_PATH = OUT_DIR / "topk_efficiency_ratio.pdf"

os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".matplotlib-cache"))

import matplotlib.pyplot as plt  # noqa: E402


def is_seed(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def load_seed_results(path: Path) -> tuple[np.ndarray, np.ndarray]:
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
    if heads[-1] != heads.max():
        raise ValueError("Expected the largest topk_heads value to be the final CSV column")

    return heads, np.array(rows, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear y-axis instead of the default log-scaled efficiency axis.",
    )
    parser.add_argument(
        "--hide-individual-seeds",
        action="store_true",
        help="Hide low-alpha per-seed efficiency curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heads, values = load_seed_results(CSV_PATH)

    full_head_performance = values[:, [-1]]
    relative_performance = values / full_head_performance
    relative_complexity = heads / heads.max()
    efficiency = relative_performance / relative_complexity

    mean = efficiency.mean(axis=0)
    std = efficiency.std(axis=0, ddof=1) if efficiency.shape[0] > 1 else np.zeros_like(mean)

    OUT_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    if not args.hide_individual_seeds:
        for seed_efficiency in efficiency:
            ax.plot(heads, seed_efficiency, color="#2ca02c", alpha=0.22, linewidth=1.4)

    ax.plot(heads, mean, color="#2ca02c", marker="o", linewidth=2.25, label="Mean efficiency")
    ax.fill_between(
        heads,
        mean - std,
        mean + std,
        color="#2ca02c",
        alpha=0.16,
        linewidth=0,
        label="Std. dev.",
    )
    ax.axhline(1.0, color="#777777", linestyle=":", linewidth=1.4, label="128-head baseline")

    if not args.linear_y:
        ax.set_yscale("log")

    ax.set_title("Top-k Cache Heads Efficiency")
    ax.set_xlabel("topk_heads")
    ax.set_ylabel("Relative performance / relative complexity")
    ax.set_xticks(heads)
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    fig.savefig(SVG_PATH)
    fig.savefig(PDF_PATH)
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
