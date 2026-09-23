#!/usr/bin/env python3
"""Plot aggregate accuracy and response lengths from the paper's tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "src/Global-Cache-Workspace-Theory-ICLR-2027-"
TABLES = {
    "math_after_math": (
        "tab:results-overview",
        "Mathematical benchmarks after mathematical adaptation",
        "Accuracy: mean of six tasks (including LogiQA). "
        "Tokens: pooled mean across the four generation tasks.",
    ),
    "korbench_after_math": (
        "tab:korbench-results",
        "KOR-Bench after mathematical adaptation",
        "Accuracy: mean of five categories. "
        "Tokens: mean across all 1,250 responses.",
    ),
    "korbench_after_reasoning": (
        "tab:korbench-reasoning-gym-results",
        "KOR-Bench after reasoning adaptation",
        "Accuracy: mean of five categories. "
        "Tokens: mean across all 1,250 responses.",
    ),
}
METHODS = {
    "Pretrained instruct": "Instruct",
    "SFT": "SFT",
    "LoRA": "LoRA",
    r"\methodBT{}": "BT",
    r"\methodBGT{}": "BGT",
    r"\methodLongshot{}": "Longshot",
    r"\methodDiffusionLongshot{}": "Diffusion\nLongshot",
}
COLORS = (
    "#9AA2AB",
    "#56B4E9",
    "#E69F00",
    "#009E73",
    "#0072B2",
    "#CC79A7",
    "#D55E00",
)


def parse_modes(cell: str) -> list[float | None]:
    values = []
    for part in cell.strip().split("/"):
        if part in {r"\textemdash", r"\textemdash{}"}:
            values.append(None)
        else:
            value = float(part)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid table value: {cell}")
            values.append(value)
    if len(values) == 1:
        values.append(None)
    if len(values) != 2:
        raise ValueError(f"Expected thinking off/on: {cell}")
    return values


def read_tables(source: Path) -> list[dict]:
    text = source.read_text(encoding="utf-8")
    blocks = re.findall(r"\\begin\{table\}(.*?)\\end\{table\}", text, re.DOTALL)
    rows = []
    for suite, (label, _, _) in TABLES.items():
        matches = [b for b in blocks if rf"\label{{{label}}}" in b]
        if len(matches) != 1:
            raise ValueError(f"Expected one table with label {label}")
        backbone = None
        columns = None
        seen = set()
        for line in matches[0].splitlines():
            model = re.search(r"\\text(?:bf|it)\{([^{}]+)\}", line)
            if line.startswith(r"\multicolumn") and model:
                backbone = model[1]
                continue
            if "&" not in line or not line.rstrip().endswith(r"\\"):
                continue
            cells = [c.strip() for c in line.rstrip()[:-2].split("&")]
            if cells[0] == "Method":
                columns = cells
                continue
            if backbone is None or columns is None:
                continue
            if cells[0] not in METHODS or len(cells) != len(columns):
                raise ValueError(f"Unrecognised result row: {line}")
            identity = (backbone, cells[0])
            if identity in seen:
                raise ValueError(f"Duplicate result: {suite}, {identity}")
            seen.add(identity)
            scores = parse_modes(cells[columns.index("Avg.")])
            tokens = parse_modes(cells[columns.index("Mean/resp.")])
            for mode, score, count in zip(("off", "on"), scores, tokens):
                if score is not None and score > 100:
                    raise ValueError(f"Accuracy exceeds 100: {line}")
                rows.append(
                    {
                        "suite": suite,
                        "backbone": backbone,
                        "method": METHODS[cells[0]].replace("\n", " "),
                        "thinking": mode,
                        "accuracy_percent": score,
                        "mean_response_tokens": count,
                    }
                )
        if not seen:
            raise ValueError(f"No results found in {label}")
    baselines = {
        (r["suite"], r["backbone"], r["thinking"]): r
        for r in rows
        if r["method"] == "Instruct"
    }
    for row in rows:
        key = (row["suite"], row["backbone"], row["thinking"])
        if key not in baselines:
            raise ValueError(f"Missing pretrained baseline: {key}")
        for field, relative in (
            ("accuracy_percent", "relative_change_percent"),
            ("mean_response_tokens", "relative_tokens_change_percent"),
        ):
            baseline = baselines[key][field]
            value = row[field]
            if value is not None and (baseline is None or baseline == 0):
                raise ValueError(f"Relative {field} undefined: {key}")
            row[f"baseline_{field}"] = baseline
            row[relative] = (
                None if value is None else 100 * (value / baseline - 1)
            )
    return rows


def plot_suite(rows: list[dict], suite: str, option: str, columns: int):
    rows = [r for r in rows if r["suite"] == suite]
    backbones = list(dict.fromkeys(r["backbone"] for r in rows))
    method_labels = list(METHODS.values())
    colors = COLORS
    if option == "relative":
        method_labels = method_labels[1:]
        colors = colors[1:]
        rows = [r for r in rows if r["method"] != "Instruct"]
    method_names = [m.replace("\n", " ") for m in method_labels]
    has_on = any(
        r["thinking"] == "on" and r["accuracy_percent"] is not None
        for r in rows
    )
    field = "accuracy_percent" if option == "raw" else "relative_change_percent"
    if option == "raw":
        limits = (0, 100)
        ylabel = "Average accuracy (%)"
        token_field = "mean_response_tokens"
        token_limits = (0, 4300 if has_on else 2200)
        token_label = "Mean response tokens"
    else:
        token_field = "relative_tokens_change_percent"
        values = [
            r[f] for r in rows for f in (field, token_field) if r[f] is not None
        ]
        lower, upper = min(values + [0]), max(values + [0])
        padding = max(5, (upper - lower) * 0.1)
        limits = (lower - padding, upper + padding)
        ylabel = "Accuracy change vs instruct (%)"
        token_limits = limits
        token_label = "Response-token change vs instruct (%)"
    ncols = min(columns, len(backbones))
    nrows = math.ceil(len(backbones) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.25 * nrows + 1.2),
        squeeze=False,
    )
    x = np.arange(len(method_names))
    for ax, backbone in zip(axes.flat, backbones):
        lookup = {
            (r["method"], r["thinking"]): r
            for r in rows
            if r["backbone"] == backbone
        }
        token_ax = ax.twinx()
        token_ax.set_zorder(ax.get_zorder() + 1)
        token_ax.patch.set_visible(False)
        for mode in ("off", "on"):
            accuracy = np.array(
                [
                    lookup.get((method, mode), {}).get(field)
                    for method in method_names
                ],
                dtype=float,
            )
            tokens = np.array(
                [
                    lookup.get((method, mode), {}).get(token_field)
                    for method in method_names
                ],
                dtype=float,
            )
            for i, value in enumerate(accuracy):
                if not np.isfinite(value):
                    continue
                ax.bar(
                    x[i],
                    value,
                    width=0.68,
                    facecolor=colors[i] if mode == "off" else "none",
                    edgecolor=colors[i] if mode == "off" else "#252A34",
                    alpha=0.78 if mode == "off" else 1,
                    linewidth=0.6 if mode == "off" else 1.35,
                    linestyle="-" if mode == "off" else "--",
                    zorder=2 if mode == "off" else 3,
                )
            token_ax.plot(
                x,
                tokens,
                color="#22252A",
                linewidth=1.15,
                linestyle="-" if mode == "off" else "--",
                marker="o" if mode == "off" else "D",
                markersize=3.5,
                markerfacecolor="#22252A" if mode == "off" else "white",
                markeredgewidth=0.85,
                zorder=5,
            )
        for i, method in enumerate(method_names):
            if all(
                lookup.get((method, mode), {}).get(field) is None
                for mode in ("off", "on")
            ):
                ax.text(
                    i,
                    0.035,
                    "N/A",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#777777",
                )
        panel_on = any(
            r["thinking"] == "on" and r[field] is not None
            for r in lookup.values()
        )
        ax.set_title(
            backbone + ("" if panel_on else " (thinking off only)"),
            fontsize=10,
            pad=9,
        )
        ax.set_xticks(x, method_labels, rotation=38, ha="right", fontsize=8)
        ax.set_xlim(-0.65, len(x) - 0.35)
        ax.set_ylim(*limits)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.axhline(0, color="#555555", linewidth=0.65, zorder=1)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E2E5E9", linewidth=0.6)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        token_ax.set_ylim(*token_limits)
        token_ax.set_ylabel(token_label, fontsize=8)
        token_ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        for axis in (ax, token_ax):
            axis.tick_params(axis="y", labelsize=8, length=2.5)
            axis.spines["top"].set_visible(False)
            for spine in axis.spines.values():
                spine.set_color("#AEB4BC")
    for ax in list(axes.flat)[len(backbones) :]:
        ax.set_visible(False)
    handles = [
        Patch(facecolor="#9AA2AB", alpha=0.78, label="Accuracy: thinking off"),
        Line2D(
            [],
            [],
            color="#22252A",
            marker="o",
            markersize=4,
            label="Tokens: thinking off",
        ),
    ]
    if has_on:
        handles.insert(
            1,
            Patch(
                facecolor="none",
                edgecolor="#252A34",
                linestyle="--",
                label="Accuracy: thinking on",
            ),
        )
        handles.append(
            Line2D(
                [],
                [],
                color="#22252A",
                linestyle="--",
                marker="D",
                markerfacecolor="white",
                markersize=4,
                label="Tokens: thinking on",
            )
        )
    fig.suptitle(TABLES[suite][1], fontsize=13, y=0.98)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
    )
    note = TABLES[suite][2]
    if option == "relative":
        note += "\nBoth axes: 100 × (value / instruct value − 1), "
        note += (
            "matched by backbone and thinking mode. Instruct columns omitted."
        )
    note += (
        "\nMissing evaluations are omitted; N/A means neither mode is reported."
    )
    fig.text(
        0.5, 0.016, note, ha="center", va="bottom", fontsize=8, color="#4C535C"
    )
    fig.tight_layout(rect=(0, 0.105, 1, 0.895), h_pad=1.7, w_pad=1.5)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PAPER / "appendices/D_Extended_Results.tex",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/analysis/paper_results_figures",
    )
    parser.add_argument(
        "--option", choices=("raw", "relative", "both"), default="both"
    )
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--suite", choices=tuple(TABLES), action="append")
    args = parser.parse_args()
    if args.columns < 1:
        parser.error("--columns must be positive")
    rows = read_tables(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
        }
    )
    with (args.output_dir / "plotted_values.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    options = ("raw", "relative") if args.option == "both" else (args.option,)
    suites = args.suite or list(TABLES)
    for option in options:
        for suite in suites:
            fig = plot_suite(rows, suite, option, args.columns)
            path = args.output_dir / f"{suite}_{option}.svg"
            fig.savefig(path, facecolor="white")
            print(path)
            plt.close(fig)
    manifest = {
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "options": list(options),
        "suites": suites,
        "columns": args.columns,
        "format": "svg",
        "relative_statistic": "100 * (value / baseline_value - 1)",
        "relative_quantities": ["accuracy", "mean_response_tokens"],
        "baseline_matching": ["suite", "backbone", "thinking"],
        "source_precision": "Rounded values as reported in the paper tables",
        "matplotlib_version": matplotlib.__version__,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
