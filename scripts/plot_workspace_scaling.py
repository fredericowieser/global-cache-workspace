#!/usr/bin/env python3
"""Plot workspace scaling with uncertainty across training seeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/analysis/scaling_workspace"
STYLES = {
    "Mean": ("#159D83", "o"),
    "Magnitude": ("#159D83", "o"),
    "Learned contextual": ("#4268B3", "s"),
}


def plot_scaling(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    total_heads: int,
    title: str,
    output: Path,
    band: str,
    ylim: tuple[float, float] | None = None,
    paper_style: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 2.8) if paper_style else (7.2, 4.3))
    fig.subplots_adjust(
        left=0.13 if paper_style else 0.11,
        right=0.86 if paper_style else 0.88,
        bottom=0.24 if paper_style else 0.22,
        top=0.82 if paper_style else 0.78,
    )
    if not paper_style:
        fig.suptitle(title, y=0.97, fontsize=13)
    handles = []
    for label, (heads, values) in curves.items():
        if values.shape != (3, len(heads)) or not np.isfinite(values).all():
            raise ValueError(f"Expected three finite seed scores per size: {label}")
        x = 100 * heads / total_heads
        mean = values.mean(axis=0)
        spread = values.std(axis=0, ddof=1)
        if band == "sem":
            spread /= np.sqrt(values.shape[0])
        color, marker = STYLES[label]
        ax.fill_between(x, mean - spread, mean + spread, color=color, alpha=0.14)
        line, = ax.plot(
            x, mean, color=color, marker=marker, linewidth=2,
            markersize=5, markerfacecolor="white", markeredgewidth=1.4,
            label=label,
        )
        handles.append(line)
        if paper_style:
            handles.append(Patch(
                facecolor=color, alpha=0.14,
                label="Std. err." if band == "sem" else "Std. dev.",
            ))

    complexity = ax.twinx()
    reference, = complexity.plot(
        [0, 100], [0, 100], color="#D9A126", linestyle=(0, (5, 5)),
        linewidth=1.4, label="Linear complexity",
    )
    complexity.set_ylim(0, 105)
    complexity.set_yticks([0, 25, 50, 75, 100])
    complexity.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    complexity.set_ylabel(
        "Relative complexity", color="#A67816", labelpad=9,
        fontsize=12 if paper_style else 10,
    )
    complexity.tick_params(axis="y", colors="#A67816", labelsize=10.5 if paper_style else 9)
    complexity.spines["right"].set_color("#D9A126")
    complexity.spines[["top", "left", "bottom"]].set_visible(False)
    ax.set_xlim(-3 if total_heads == 128 else 20, 104)
    ax.set_xticks([0, 10, 25, 50, 75, 100] if total_heads == 128 else [25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel(
        "Selected Heads (%)" if paper_style else "Workspace size", labelpad=8,
        fontsize=12 if paper_style else 10,
    )
    ax.set_ylabel("Average performance (%)", labelpad=8, fontsize=12 if paper_style else 10)
    ax.yaxis.set_major_locator(MaxNLocator(4 if paper_style else 6))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#DDE2E8", linewidth=0.8)
    ax.grid(axis="x", color="#EEF0F4", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#848B95")
    ax.tick_params(labelsize=10.5 if paper_style else 9)
    handles.append(reference)
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.99 if paper_style else 0.91),
        ncol=len(handles), frameon=False, fontsize=10.5 if paper_style else 9,
    )
    uncertainty = "standard error" if band == "sem" else "sample standard deviation"
    note = f"Mean across seeds 24, 25, 26; shading: ±1 {uncertainty}."
    if len(curves) > 1:
        note += "\n100% uses the shared dense run."
    if not paper_style:
        fig.text(0.5, 0.035, note, ha="center", va="bottom", fontsize=8, color="#505863")
    for suffix in ("svg", "pdf"):
        fig.savefig(output.parent / f"{output.name}.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=DATA / "plots")
    parser.add_argument("--band", choices=("sem", "std"), default="sem")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})

    with (args.data_dir / "llama_3.2_1B.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        heads = np.array([int(name) for name in reader.fieldnames if name != "seed"])
        rows = sorted(reader, key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in rows] != [24, 25, 26]:
        raise ValueError("Expected Llama seeds 24, 25, 26")
    values = np.array([[float(row[str(head)]) for head in heads] for row in rows])
    plot_scaling(
        {"Mean": (heads, values)}, 128, "Llama 3.2 1B",
        args.output_dir / "llama_3.2_1B", args.band,
        paper_style=True,
    )

    with (args.data_dir / "qwen_35_4B.csv").open(newline="", encoding="utf-8") as stream:
        qwen = list(csv.DictReader(stream))
    checkpoints = {}
    for step in (1000, 2000, 3000):
        curves = {}
        for method, label in (("magnitude", "Magnitude"), ("learned_contextual", "Learned contextual")):
            selected = [r for r in qwen if int(r["training_steps"]) == step and r["head_routing"] == method]
            samples = {(int(r["seed"]), int(r["topk_heads"])): float(r["paper7_avg_no_theoremqa_pct"]) for r in selected}
            expected = {(seed, head) for seed in (24, 25, 26) for head in (8, 16, 32)}
            if len(selected) != 9 or set(samples) != expected:
                raise ValueError(f"Incomplete or duplicate Qwen series: {method}, {step}")
            values = np.array([[samples[seed, head] for head in (8, 16, 32)] for seed in (24, 25, 26)])
            curves[label] = (np.array([8, 16, 32]), values)
        if not np.array_equal(curves["Magnitude"][1][:, -1], curves["Learned contextual"][1][:, -1]):
            raise ValueError(f"Shared dense scores differ at {step} steps")
        checkpoints[step] = curves

    bounds = []
    for curves in checkpoints.values():
        for _, values in curves.values():
            mean = values.mean(axis=0)
            spread = values.std(axis=0, ddof=1)
            if args.band == "sem":
                spread /= np.sqrt(3)
            bounds.extend((float((mean - spread).min()), float((mean + spread).max())))
    ylim = (min(bounds) - 0.18, max(bounds) + 0.18)
    for step, curves in checkpoints.items():
        plot_scaling(
            curves, 32, f"Qwen3.5 4B · {step:,} training steps",
            args.output_dir / f"qwen_35_4B_steps_{step}", args.band, ylim,
        )
    print(f"Saved four plots as SVG and PDF with {args.band} bands in {args.output_dir}")


if __name__ == "__main__":
    main()
