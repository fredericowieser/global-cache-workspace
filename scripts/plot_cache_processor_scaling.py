#!/usr/bin/env python3
"""Compare cache processor size at 2,000 steps with seed standard deviations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/analysis/scaling_cp"
MODELS = (
    ("llama_3.2_1B.csv", "Llama 3.2 1B", 32, "#159D83", "o"),
    ("llama_3.2_3B.csv", "Llama 3.2 3B", 56, "#9863AC", "D"),
    ("qwen_35_2B.csv", "Qwen3.5 2B", 32, "#4268B3", "s"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=DATA / "plots")
    args = parser.parse_args()
    fractions = (0.015, 0.05, 0.1)
    seeds = (24, 25, 26)
    expected = {(seed, fraction) for seed in seeds for fraction in fractions}
    curves = []
    for filename, label, heads, color, marker in MODELS:
        with (args.data_dir / filename).open(newline="", encoding="utf-8") as stream:
            rows = [
                row for row in csv.DictReader(stream)
                if int(row["training_steps"]) == 2000 and int(row["topk_heads"]) == heads
            ]
        samples = {
            (int(row["seed"]), float(row["target_backbone_fraction"])):
            float(row["paper7_avg_no_theoremqa_pct"])
            for row in rows
        }
        if len(rows) != 9 or set(samples) != expected:
            raise ValueError(f"Incomplete or duplicate scaling series: {label}")
        values = np.array([[samples[seed, fraction] for fraction in fractions] for seed in seeds])
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite seed score: {label}")
        baseline = values.mean()
        if baseline <= 0:
            raise ValueError(f"Non-positive normalisation baseline: {label}")
        relative = 100 * (values / baseline - 1)
        mean = relative.mean(axis=0)
        std = relative.std(axis=0, ddof=1)
        curves.append((label, color, marker, mean, std))
        print(f"{label}: baseline={baseline:.6f}, relative mean={mean.round(6).tolist()}, sample std={std.round(6).tolist()}")

    plt.rcParams.update({"font.family": "DejaVu Serif", "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.24, top=0.82)
    x = 100 * np.array(fractions)
    handles = []
    for label, color, marker, mean, std in curves:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14)
        line, = ax.plot(
            x, mean, color=color, marker=marker, linewidth=2,
            markersize=5, markerfacecolor="white", markeredgewidth=1.4, label=label,
        )
        handles.append(line)
    handles.append(Patch(facecolor="#848B95", alpha=0.25, label="Std. dev."))
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.99),
        ncol=4, frameon=False, fontsize=10, columnspacing=1.2, handlelength=1.8,
    )
    ax.set_xticks(x, labels=["1.5", "5", "10"])
    ax.set_xlim(1.1, 10.4)
    ax.set_xlabel("Target processor size (% of backbone parameters)", labelpad=8, fontsize=11)
    ax.set_ylabel("Relative performance (%)", labelpad=8, fontsize=12)
    extent = max(float(np.max(np.abs(mean) + std)) for _, _, _, mean, std in curves)
    ax.set_ylim(-1.1 * extent, 1.1 * extent)
    ax.axhline(0, color="#848B95", linewidth=0.9, linestyle=(0, (4, 4)), zorder=1)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(labelsize=10.5)
    ax.grid(axis="y", color="#DDE2E8", linewidth=0.8)
    ax.grid(axis="x", color="#EEF0F4", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#848B95")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        fig.savefig(args.output_dir / f"cache_processor_scaling_steps_2000.{suffix}", facecolor="white")
    plt.close(fig)
    print(f"Saved SVG and PDF in {args.output_dir}")


if __name__ == "__main__":
    main()
