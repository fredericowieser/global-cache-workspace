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
from matplotlib.colors import LinearSegmentedColormap
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
    "math_after_reasoning": (
        "tab:paper7-reasoning-gym-results",
        "Mathematical benchmarks after reasoning adaptation",
        "Accuracy: mean of five tasks including LogiQA, excluding TheoremQA. "
        "Tokens: pooled mean across three generation tasks.",
    ),
    "math_after_sequential": (
        "tab:paper7-sequential-adaptation-results",
        "Mathematical benchmarks after sequential adaptation",
        "Accuracy: mean of five tasks including LogiQA, excluding TheoremQA. "
        "Tokens: pooled mean across three generation tasks.",
    ),
    "korbench_after_sequential": (
        "tab:korbench-sequential-adaptation-results",
        "KOR-Bench after sequential adaptation",
        "Accuracy: mean of five tasks. "
        "Tokens: mean across all 1,250 responses.",
    ),
}
ADAPTATIONS = {
    "mathematical": ("math_after_math", "korbench_after_math"),
    "reasoning": ("math_after_reasoning", "korbench_after_reasoning"),
    "sequential": ("math_after_sequential", "korbench_after_sequential"),
}
HEATMAP_COLORS = ["#AC4035", "#E6AB8F", "#FCF6EA", "#A8C8A6", "#236548"]
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
BACKBONE_STYLES = {
    "Llama 3.2-1B": ("#0072B2", "o"),
    "Llama 3.2-3B": ("#E69F00", "s"),
    "Qwen3.5-2B": ("#009E73", "^"),
    "Qwen3.5-4B": ("#CC79A7", "D"),
    "Qwen3.5-27B": ("#D55E00", "p"),
}


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


def read_tables(source: Path, tables: dict | None = None) -> list[dict]:
    text = source.read_text(encoding="utf-8")
    blocks = re.findall(r"\\begin\{table\}(.*?)\\end\{table\}", text, re.DOTALL)
    rows = []
    for suite, (label, _, _) in (TABLES if tables is None else tables).items():
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
        relative_limits = []
        for quantity in (field, token_field):
            values = [0] + [
                r[quantity] for r in rows if r[quantity] is not None
            ]
            lower, upper = min(values), max(values)
            padding = max(5, (upper - lower) * 0.1)
            relative_limits.append((lower - padding, upper + padding))
        zero_fraction = sum(
            -lower / (upper - lower) for lower, upper in relative_limits
        ) / len(relative_limits)
        for i, (lower, upper) in enumerate(relative_limits):
            span = max(-lower / zero_fraction, upper / (1 - zero_fraction))
            relative_limits[i] = (
                -span * zero_fraction,
                span * (1 - zero_fraction),
            )
        limits, token_limits = relative_limits
        ylabel = "Accuracy change vs instruct (%)"
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


def plot_points(rows: list[dict], suite: str):
    rows = [
        r for r in rows if r["suite"] == suite and r["method"] != "Instruct"
    ]
    backbones = list(dict.fromkeys(r["backbone"] for r in rows))
    unknown = set(backbones) - BACKBONE_STYLES.keys()
    if unknown:
        raise ValueError(f"Define point styles for new backbones: {unknown}")
    method_labels = list(METHODS.values())[1:]
    method_names = [label.replace("\n", " ") for label in method_labels]
    offsets = dict(zip(BACKBONE_STYLES, np.linspace(-0.3, 0.3, 5)))
    has_on = any(
        r["thinking"] == "on" and r["accuracy_percent"] is not None
        for r in rows
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.5), sharex=True)
    for ax, field, label in zip(
        axes,
        ("relative_change_percent", "relative_tokens_change_percent"),
        ("Accuracy change (%)", "Token change (%)"),
    ):
        values = [0.0]
        for backbone in backbones:
            color, marker = BACKBONE_STYLES[backbone]
            for mode in ("off", "on"):
                points = [
                    r
                    for r in rows
                    if r["backbone"] == backbone
                    and r["thinking"] == mode
                    and r[field] is not None
                ]
                if not points:
                    continue
                xs = [
                    method_names.index(r["method"]) + offsets[backbone]
                    for r in points
                ]
                ys = [r[field] for r in points]
                values.extend(ys)
                ax.scatter(
                    xs,
                    ys,
                    s=48,
                    marker=marker,
                    linewidths=1.2,
                    facecolors=color if mode == "off" else "none",
                    edgecolors=color,
                    zorder=3,
                )
        padding = max(5, (max(values) - min(values)) * 0.12)
        ax.set_ylim(min(values) - padding, max(values) + padding)
        ax.set_ylabel(label, fontsize=9)
        ax.axhline(0, color="#646A73", linewidth=0.9, zorder=2)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E6E8EB", linewidth=0.6)
        for boundary in np.arange(0.5, len(method_names) - 1, 1):
            ax.axvline(boundary, color="#ECEEF1", linewidth=0.6, zorder=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="both", labelsize=8, length=2.5)
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color("#B6BBC2")
    axes[-1].set_xticks(np.arange(len(method_names)), method_labels)
    axes[-1].set_xlim(-0.55, len(method_names) - 0.45)
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=BACKBONE_STYLES[b][1],
            color=BACKBONE_STYLES[b][0],
            markersize=6.5,
            label=b,
        )
        for b in backbones
    ]
    fig.suptitle(TABLES[suite][1], fontsize=10, y=0.985)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.943),
        ncol=len(handles),
        frameon=False,
        fontsize=7.5,
        handletextpad=0.35,
        columnspacing=1.2,
    )
    mode_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            color="#555555",
            markerfacecolor="#555555" if mode == "off" else "white",
            markersize=6.5,
            label=f"Thinking {mode}",
        )
        for mode in (("off", "on") if has_on else ("off",))
    ]
    fig.legend(
        handles=mode_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.891),
        ncol=len(mode_handles),
        frameon=False,
        fontsize=7.5,
        handletextpad=0.35,
    )
    fig.subplots_adjust(
        left=0.12, right=0.985, bottom=0.12, top=0.79, hspace=0.18
    )
    return fig


def plot_compact(rows: list[dict], suite: str):
    if suite != "korbench_after_reasoning":
        rows = [r for r in rows if r["backbone"] != "Qwen3.5-2B"]
    fig = plot_suite(rows, suite, "relative", 2)
    fig.set_size_inches(7.2, 4.5)
    for text in list(fig.texts):
        if text is not fig._suptitle:
            text.remove()
    fig._suptitle.set_fontsize(10)
    fig._suptitle.set_y(0.985)
    for legend in fig.legends:
        legend.set_bbox_to_anchor((0.5, 0.945))
        for text in legend.get_texts():
            text.set_fontsize(7)
    for i, ax in enumerate(fig.axes[:4]):
        ax.set_title(ax.get_title().split(" (thinking")[0], fontsize=9, pad=5)
        ax.set_ylabel("Accuracy change (%)" if i % 2 == 0 else "", fontsize=8)
        ax.tick_params(axis="y", labelsize=7, labelleft=i % 2 == 0)
        ax.set_xticks(np.arange(6), list(METHODS.values())[1:])
        ax.tick_params(axis="x", labelbottom=i >= 2, labelsize=6.5)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        for text in ax.texts:
            text.set_fontsize(6)
    for i, ax in enumerate(fig.axes[4:]):
        ax.set_ylabel("Token change (%)" if i % 2 == 1 else "", fontsize=8)
        ax.tick_params(
            axis="y", right=i % 2 == 1, labelright=i % 2 == 1, labelsize=7
        )
    fig.subplots_adjust(
        left=0.10,
        right=0.90,
        bottom=0.12,
        top=0.81,
        wspace=0.12,
        hspace=0.30,
    )
    return fig


def plot_tradeoff(rows: list[dict], suite: str):
    rows = [
        r for r in rows if r["suite"] == suite and r["method"] != "Instruct"
    ]
    backbones = list(dict.fromkeys(r["backbone"] for r in rows))
    methods = [label.replace("\n", " ") for label in list(METHODS.values())[1:]]
    has_on = any(
        r["thinking"] == "on" and r["accuracy_percent"] is not None
        for r in rows
    )
    limits = []
    for field in ("relative_tokens_change_percent", "relative_change_percent"):
        values = [0] + [r[field] for r in rows if r[field] is not None]
        padding = max(5, (max(values) - min(values)) * 0.12)
        limits.append((min(values) - padding, max(values) + padding))
    ncols = 3 if len(backbones) > 4 else 2
    fig, axes = plt.subplots(
        2, ncols, figsize=(7.2, 4.8), sharex=True, sharey=True
    )
    for ax, backbone in zip(axes.flat, backbones):
        for method, color in zip(methods, COLORS[1:]):
            for mode in ("off", "on"):
                points = [
                    r
                    for r in rows
                    if r["backbone"] == backbone
                    and r["method"] == method
                    and r["thinking"] == mode
                    and r["relative_change_percent"] is not None
                    and r["relative_tokens_change_percent"] is not None
                ]
                if not points:
                    continue
                ax.scatter(
                    [r["relative_tokens_change_percent"] for r in points],
                    [r["relative_change_percent"] for r in points],
                    s=36,
                    marker="o" if mode == "off" else "^",
                    facecolors=color if mode == "off" else "none",
                    edgecolors=color,
                    linewidths=1.0,
                    zorder=3,
                )
        ax.scatter([0], [0], marker="+", s=45, color="#333333", zorder=4)
        ax.axhline(0, color="#979DA5", linewidth=0.7)
        ax.axvline(0, color="#979DA5", linewidth=0.7)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_title(backbone, fontsize=9, pad=5)
        ax.set_axisbelow(True)
        ax.grid(color="#ECEEF1", linewidth=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(labelsize=7, length=2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(backbones) :]:
        ax.set_visible(False)
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            color=color,
            markersize=5,
            label=method,
        )
        for method, color in zip(methods, COLORS[1:])
    ]
    fig.suptitle(TABLES[suite][1], fontsize=10, y=0.985)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        handletextpad=0.4,
    )
    mode_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            color="#555555",
            label="Thinking off",
            markersize=5,
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="+",
            color="#333333",
            label="Instruct",
            markersize=6,
        ),
    ]
    if has_on:
        mode_handles.insert(
            1,
            Line2D(
                [],
                [],
                linestyle="none",
                marker="^",
                color="#555555",
                markerfacecolor="none",
                label="Thinking on",
                markersize=5,
            ),
        )
    fig.legend(
        handles=mode_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.86),
        ncol=len(mode_handles),
        frameon=False,
        fontsize=7.5,
    )
    fig.supxlabel("Response-token change vs instruct (%)", fontsize=8, y=0.025)
    fig.supylabel("Accuracy change vs instruct (%)", fontsize=8, x=0.015)
    fig.subplots_adjust(
        left=0.10, right=0.985, bottom=0.13, top=0.75, wspace=0.14, hspace=0.32
    )
    return fig


def draw_heatmap_panel(
    ax, values, columns, title, cmap, limit, subtitle="", methods=None
):
    backbones = list(dict.fromkeys(backbone for backbone, _ in columns))
    if methods is None:
        methods = list(METHODS.values())[1:]
    centers = [
        np.mean([i for i, (b, _) in enumerate(columns) if b == backbone])
        for backbone in backbones
    ]
    cmap.set_bad("#E8EBEF")
    mesh = ax.pcolormesh(
        np.arange(len(columns) + 1) - 0.5,
        np.arange(len(methods) + 1) - 0.5,
        np.ma.masked_invalid(values),
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        edgecolors="white",
        linewidth=1.1,
        rasterized=False,
    )
    ax.set_ylim(len(methods) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(columns)), [mode for _, mode in columns])
    ax.set_yticks(np.arange(len(methods)), methods)
    ax.tick_params(
        length=0,
        labelsize=7.5,
        colors="#334155",
    )
    ax.tick_params(axis="x", labelsize=6.5, pad=5)
    ax.tick_params(axis="y", pad=8)
    transform = ax.get_xaxis_transform()
    families = list(dict.fromkeys(b.rsplit("-", 1)[0] for b in backbones))
    for family in families:
        positions = [
            i
            for i, (b, _) in enumerate(columns)
            if b.rsplit("-", 1)[0] == family
        ]
        left, right = min(positions), max(positions)
        ax.text(
            (left + right) / 2,
            1.205,
            family,
            transform=transform,
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="#334155",
        )
        ax.plot(
            [left - 0.38, right + 0.38],
            [1.15, 1.15],
            transform=transform,
            clip_on=False,
            color="#C9D1DA",
            linewidth=0.7,
        )
    for center, backbone in zip(centers, backbones):
        ax.text(
            center,
            1.045,
            backbone.rsplit("-", 1)[1],
            transform=transform,
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#475569",
        )
    if ax.get_subplotspec().is_first_col():
        ax.text(
            -0.035,
            1.045,
            "METHOD",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            fontweight="bold",
            color="#64748B",
        )
        ax.text(
            -0.035,
            -0.060,
            "Thinking",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#64748B",
        )
    for i in range(1, len(columns)):
        if columns[i][0] != columns[i - 1][0]:
            ax.axvline(
                i - 0.5,
                color="white",
                linewidth=2.0,
            )
    for (i, j), value in np.ndenumerate(values):
        text = "N/A" if np.isnan(value) else f"{value:+.0f}"
        if np.isfinite(value) and abs(value) < 0.5:
            text = "0"
        rgb = (
            cmap((value + limit) / (2 * limit))[:3]
            if np.isfinite(value)
            else (1, 1, 1)
        )
        luminance = np.dot(rgb, (0.2126, 0.7152, 0.0722))
        color = "white" if luminance < 0.53 else "#18202B"
        if np.isnan(value):
            color = "#7C8796"
        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=7.5,
            color=color,
            fontweight="normal",
        )
    ax.text(
        0,
        1.445,
        f"({chr(97 + ax.get_subplotspec().colspan.start)})",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        color="#64748B",
        va="bottom",
    )
    ax.text(
        0.09,
        1.445,
        title,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color="#17212E",
        va="bottom",
    )
    ax.text(
        1,
        1.445,
        subtitle,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#64748B",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def plot_heatmap(rows: list[dict], suite: str):
    rows = [r for r in rows if r["suite"] == suite]
    backbones = list(dict.fromkeys(r["backbone"] for r in rows))
    methods = [label.replace("\n", " ") for label in list(METHODS.values())[1:]]
    columns = [
        (backbone, mode)
        for backbone in backbones
        for mode in ("off", "on")
        if any(
            r["backbone"] == backbone
            and r["thinking"] == mode
            and r["accuracy_percent"] is not None
            for r in rows
        )
    ]
    lookup = {(r["method"], r["backbone"], r["thinking"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    fig.subplots_adjust(
        left=0.14, right=0.99, bottom=0.25, top=0.68, wspace=0.12
    )
    improvement_cmap = LinearSegmentedColormap.from_list(
        "terracotta_cream_forest", HEATMAP_COLORS
    )
    for ax, field, title, cmap, endpoints in zip(
        axes,
        ("relative_change_percent", "relative_tokens_change_percent"),
        ("Accuracy", "Response tokens"),
        (improvement_cmap, improvement_cmap.reversed()),
        (
            ("Lower accuracy", "Higher accuracy"),
            ("Fewer tokens", "More tokens"),
        ),
    ):
        values = np.array(
            [
                [
                    lookup.get((method, backbone, mode), {}).get(field)
                    for backbone, mode in columns
                ]
                for method in methods
            ],
            dtype=float,
        )
        limit = max(
            1,
            float(np.max(np.abs(values), where=np.isfinite(values), initial=0)),
        )
        mesh = draw_heatmap_panel(
            ax, values, columns, title, cmap, limit, "relative change"
        )
        position = ax.get_position()
        color_ax = fig.add_axes((position.x0, 0.105, position.width, 0.026))
        colorbar = fig.colorbar(mesh, cax=color_ax, orientation="horizontal")
        colorbar.locator = MaxNLocator(nbins=6, symmetric=True)
        colorbar.update_ticks()
        colorbar.ax.tick_params(
            labelsize=6.5, length=2, pad=2, colors="#475569"
        )
        colorbar.outline.set_visible(False)
        colorbar.solids.set_rasterized(False)
        colorbar.ax.axvline(0, color="#64748B", linewidth=0.7)
        fig.text(position.x0, 0.15, endpoints[0], fontsize=6.5, color=cmap(0.0))
        fig.text(
            position.x1,
            0.15,
            endpoints[1],
            fontsize=6.5,
            color=cmap(1.0),
            ha="right",
        )
    fig.text(
        0.14,
        0.018,
        "CHANGE VS INSTRUCT (%)",
        fontsize=6.2,
        fontweight="bold",
        color="#334155",
    )
    fig.text(
        0.84,
        0.028,
        "N/A",
        fontsize=6.2,
        color="#64748B",
        va="center",
        ha="center",
        bbox={"facecolor": "#E8EBEF", "edgecolor": "none", "pad": 3},
    )
    fig.text(
        0.873, 0.028, "Not reported", fontsize=6.2, color="#64748B", va="center"
    )
    return fig


def plot_paired_heatmap(rows: list[dict], adaptation: str):
    suites = ADAPTATIONS[adaptation]
    rows = [r for r in rows if r["suite"] in suites]
    backbones = list(dict.fromkeys(r["backbone"] for r in rows))
    columns = [
        (backbone, mode)
        for backbone in backbones
        for mode in ("off", "on")
        if any(
            r["backbone"] == backbone
            and r["thinking"] == mode
            and r["relative_change_percent"] is not None
            for r in rows
        )
    ]
    methods = [
        label.replace("\n", " ")
        for label in list(METHODS.values())[1:]
        if label != METHODS[r"\methodDiffusionLongshot{}"]
    ]
    lookup = {
        (r["suite"], r["method"], r["backbone"], r["thinking"]): r for r in rows
    }
    matrices = [
        np.array(
            [
                [
                    lookup.get((suite, method, backbone, mode), {}).get(
                        "relative_change_percent"
                    )
                    for backbone, mode in columns
                ]
                for method in methods
            ],
            dtype=float,
        )
        for suite in suites
    ]
    limit = max(
        1,
        *(
            float(np.max(np.abs(values), where=np.isfinite(values), initial=0))
            for values in matrices
        ),
    )
    cmap = LinearSegmentedColormap.from_list(
        "terracotta_cream_forest",
        HEATMAP_COLORS,
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    task_count = 6 if adaptation == "mathematical" else 5
    for ax, values, title, subtitle in zip(
        axes,
        matrices,
        ("Maths + LogiQA", "KOR-Bench"),
        (f"{task_count} tasks", "5 tasks"),
    ):
        mesh = draw_heatmap_panel(
            ax, values, columns, title, cmap, limit, subtitle, methods=methods
        )
    fig.subplots_adjust(
        left=0.14, right=0.99, bottom=0.25, top=0.68, wspace=0.12
    )
    color_ax = fig.add_axes((0.40, 0.075, 0.39, 0.026))
    colorbar = fig.colorbar(mesh, cax=color_ax, orientation="horizontal")
    colorbar.locator = MaxNLocator(nbins=6, symmetric=True)
    colorbar.update_ticks()
    colorbar.ax.tick_params(labelsize=6.5, length=2, pad=2, colors="#475569")
    colorbar.outline.set_visible(False)
    colorbar.solids.set_rasterized(False)
    colorbar.ax.axvline(0, color="#64748B", linewidth=0.7)
    fig.text(
        0.14,
        0.108,
        "CHANGE VS INSTRUCT",
        fontsize=6.2,
        fontweight="bold",
        color="#334155",
        va="bottom",
    )
    fig.text(0.14, 0.057, "Relative accuracy (%)", fontsize=7, color="#64748B")
    fig.text(0.40, 0.124, "Lower", fontsize=6.5, color=cmap(0.0))
    fig.text(0.79, 0.124, "Higher", fontsize=6.5, color=cmap(1.0), ha="right")
    fig.text(
        0.84,
        0.084,
        "N/A",
        fontsize=6.2,
        color="#64748B",
        va="center",
        ha="center",
        bbox={"facecolor": "#E8EBEF", "edgecolor": "none", "pad": 3},
    )
    fig.text(
        0.873, 0.084, "Not reported", fontsize=6.2, color="#64748B", va="center"
    )
    return fig


def plot_diffusion_heatmap(rows: list[dict]):
    backbones = list(dict.fromkeys(r["backbone"] for r in rows))
    suites = [suite for pair in ADAPTATIONS.values() for suite in pair]
    lookup = {
        (r["suite"], r["backbone"], r["method"]): r["accuracy_percent"]
        for r in rows
        if r["thinking"] == "off"
    }
    values = np.full((len(backbones), len(suites)), np.nan)
    for i, backbone in enumerate(backbones):
        for j, suite in enumerate(suites):
            longshot = lookup.get((suite, backbone, "Longshot"))
            diffusion = lookup.get((suite, backbone, "Diffusion Longshot"))
            if longshot is not None and diffusion is not None:
                values[i, j] = diffusion - longshot
    return draw_campaign_heatmap(
        values, backbones, "DIFFUSION − LONGSHOT", "Accuracy difference (pp)"
    )


def plot_campaign_heatmap(rows: list[dict]):
    methods = [
        method
        for method in METHODS.values()
        if method not in {"Instruct", METHODS[r"\methodDiffusionLongshot{}"]}
    ]
    suites = [suite for pair in ADAPTATIONS.values() for suite in pair]
    values = np.full((len(methods), len(suites)), np.nan)
    stds = np.full(values.shape, np.nan)
    aggregates = []
    for i, method in enumerate(methods):
        for j, suite in enumerate(suites):
            available = [
                r
                for r in rows
                if r["suite"] == suite
                and r["method"] == method
                and r["thinking"] == "off"
                and r["relative_change_percent"] is not None
            ]
            if available:
                changes = [r["relative_change_percent"] for r in available]
                values[i, j] = np.mean(changes)
                stds[i, j] = np.std(changes, ddof=0)
            aggregates.append(
                {
                    "suite": suite,
                    "method": method,
                    "thinking": "off",
                    "mean_relative_change_percent": (
                        float(values[i, j]) if available else None
                    ),
                    "std_relative_change_percent": (
                        float(stds[i, j]) if available else None
                    ),
                    "backbone_count": len(available),
                    "backbones": "; ".join(r["backbone"] for r in available),
                }
            )
    fig = draw_campaign_heatmap(
        values,
        methods,
        "CHANGE VS INSTRUCT",
        "Mean relative accuracy (%)",
        stds=stds,
    )
    return fig, aggregates


def draw_campaign_heatmap(
    values: np.ndarray,
    row_labels: list[str],
    legend_title: str,
    legend_subtitle: str,
    stds: np.ndarray | None = None,
):
    row_count, column_count = values.shape
    limit = max(
        1, float(np.max(np.abs(values), where=np.isfinite(values), initial=0))
    )
    cmap = LinearSegmentedColormap.from_list(
        "terracotta_cream_forest", HEATMAP_COLORS
    )
    cmap.set_bad("#E8EBEF")
    fig, ax = plt.subplots(figsize=(7.2, 2.8 if stds is None else 3.1))
    left_margin = 0.19 if stds is None else 0.095
    fig.subplots_adjust(left=left_margin, right=0.99, bottom=0.25, top=0.77)
    mesh = ax.pcolormesh(
        np.arange(column_count + 1) - 0.5,
        np.arange(row_count + 1) - 0.5,
        np.ma.masked_invalid(values),
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        edgecolors="white",
        linewidth=1.1,
        rasterized=False,
    )
    ax.set_ylim(row_count - 0.5, -0.5)
    ax.set_yticks(range(row_count), row_labels)
    ax.set_xticks(
        range(column_count), ["Maths", "KOR-Bench"] * len(ADAPTATIONS)
    )
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=7.5, colors="#334155", pad=7)
    for group, title in enumerate(
        ("Maths training", "Reasoning Gym", "Sequential")
    ):
        left = 2 * group - 0.5
        ax.text(
            left + 1,
            1.25,
            title,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#17212E",
        )
        ax.plot(
            [left + 0.08, left + 1.92],
            [1.19, 1.19],
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            color="#C9D1DA",
            linewidth=0.7,
        )
        if group:
            ax.axvline(left, color="white", linewidth=3)
    for (i, j), value in np.ndenumerate(values):
        label = "N/A" if not np.isfinite(value) else f"{value:+.1f}"
        if np.isfinite(value) and abs(value) < 0.05:
            label = "0.0"
        color = "#7C8796"
        if np.isfinite(value):
            rgb = cmap((value + limit) / (2 * limit))[:3]
            color = (
                "white"
                if np.dot(rgb, (0.2126, 0.7152, 0.0722)) < 0.53
                else "#18202B"
            )
        ax.text(
            j,
            i if stds is None else i - 0.13,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color=color,
        )
        if stds is not None and np.isfinite(stds[i, j]):
            ax.text(
                j,
                i + 0.23,
                f"± {stds[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=color,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    color_ax = fig.add_axes((0.47, 0.10, 0.38, 0.028))
    colorbar = fig.colorbar(mesh, cax=color_ax, orientation="horizontal")
    colorbar.locator = MaxNLocator(nbins=5, symmetric=True)
    colorbar.update_ticks()
    colorbar.ax.tick_params(labelsize=6.5, length=2, pad=2, colors="#475569")
    colorbar.outline.set_visible(False)
    colorbar.solids.set_rasterized(False)
    colorbar.ax.axvline(0, color="#64748B", linewidth=0.7)
    fig.text(
        left_margin,
        0.12,
        legend_title,
        fontsize=7,
        fontweight="bold",
        color="#334155",
    )
    fig.text(left_margin, 0.065, legend_subtitle, fontsize=7, color="#64748B")
    fig.text(0.47, 0.15, "Lower", fontsize=6.5, color=cmap(0.0))
    fig.text(0.85, 0.15, "Higher", fontsize=6.5, color=cmap(1.0), ha="right")
    fig.text(
        0.90,
        0.11,
        "N/A",
        fontsize=6.5,
        color="#64748B",
        bbox={"facecolor": "#E8EBEF", "edgecolor": "none", "pad": 3},
    )
    if stds is not None:
        fig.text(
            left_margin,
            0.015,
            "Thinking off · Mean ± SD across available backbones (equal weight)",
            fontsize=6.5,
            color="#64748B",
        )
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
        help="Directory for SVG/PDF figures, plotted values, and manifests",
    )
    parser.add_argument(
        "--option",
        choices=("raw", "relative", "both", "difference"),
        default="both",
    )
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--suite", choices=tuple(TABLES), action="append")
    parser.add_argument(
        "--adaptation", choices=tuple(ADAPTATIONS), action="append"
    )
    layouts = {
        "points": plot_points,
        "compact": plot_compact,
        "tradeoff": plot_tradeoff,
        "heatmap": plot_heatmap,
        "paired_heatmap": plot_paired_heatmap,
    }
    parser.add_argument(
        "--layout",
        choices=("bars", *layouts, "diffusion_heatmap", "campaign_heatmap"),
        default="bars",
    )
    args = parser.parse_args()
    if args.columns < 1:
        parser.error("--columns must be positive")
    if args.layout == "diffusion_heatmap":
        if args.option != "difference" or args.suite or args.adaptation:
            parser.error(
                "diffusion_heatmap requires --option difference and all regimes"
            )
    elif args.option == "difference":
        parser.error("--option difference requires --layout diffusion_heatmap")
    elif args.layout != "bars" and args.option != "relative":
        parser.error(f"--layout {args.layout} requires --option relative")
    if args.layout == "campaign_heatmap" and (args.suite or args.adaptation):
        parser.error("campaign_heatmap requires all suites and regimes")
    if args.layout == "paired_heatmap" and args.suite:
        parser.error("Use --adaptation instead of --suite with paired_heatmap")
    if args.adaptation and args.layout != "paired_heatmap":
        parser.error("--adaptation requires --layout paired_heatmap")
    rows = read_tables(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
        }
    )
    csv_name = (
        f"plotted_values_{args.layout}.csv"
        if args.layout
        in {"paired_heatmap", "diffusion_heatmap", "campaign_heatmap"}
        else "plotted_values.csv"
    )
    with (args.output_dir / csv_name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    options = ("raw", "relative") if args.option == "both" else (args.option,)
    suites = args.suite or list(TABLES)
    adaptations = args.adaptation or list(ADAPTATIONS)
    if args.layout == "paired_heatmap":
        suites = [
            s for adaptation in adaptations for s in ADAPTATIONS[adaptation]
        ]
    for option in options:
        for suite in (
            ["overview"]
            if args.layout in {"diffusion_heatmap", "campaign_heatmap"}
            else adaptations if args.layout == "paired_heatmap" else suites
        ):
            if args.layout == "diffusion_heatmap":
                fig = plot_diffusion_heatmap(rows)
                path = args.output_dir / "diffusion_difference_heatmap.svg"
            elif args.layout == "campaign_heatmap":
                fig, aggregates = plot_campaign_heatmap(rows)
                path = args.output_dir / "campaign_relative_heatmap.svg"
                with (args.output_dir / "campaign_heatmap_averages.csv").open(
                    "w", newline=""
                ) as f:
                    writer = csv.DictWriter(f, fieldnames=list(aggregates[0]))
                    writer.writeheader()
                    writer.writerows(aggregates)
            elif args.layout in layouts:
                fig = layouts[args.layout](rows, suite)
                path = args.output_dir / f"{suite}_{option}_{args.layout}.svg"
            else:
                fig = plot_suite(rows, suite, option, args.columns)
                path = args.output_dir / f"{suite}_{option}.svg"
            for extension in ("svg", "pdf"):
                output = path.with_suffix(f".{extension}")
                fig.savefig(output, facecolor="white")
                print(output)
            plt.close(fig)
    manifest = {
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "options": list(options),
        "suites": suites,
        "columns": args.columns,
        "layout": args.layout,
        "palette": (
            "terracotta_cream_forest"
            if args.layout
            in {
                "heatmap",
                "paired_heatmap",
                "diffusion_heatmap",
                "campaign_heatmap",
            }
            else None
        ),
        "adaptations": (
            adaptations
            if args.layout
            in {"paired_heatmap", "diffusion_heatmap", "campaign_heatmap"}
            else []
        ),
        "omitted_backbones": (
            {
                s: ["Qwen3.5-2B"]
                for s in suites
                if s != "korbench_after_reasoning"
            }
            if args.layout == "compact"
            else {}
        ),
        "formats": ["svg", "pdf"],
        "omitted_methods": (
            ["Diffusion Longshot"]
            if args.layout in {"paired_heatmap", "campaign_heatmap"}
            else []
        ),
        "relative_statistic": (
            "diffusion_accuracy_percent - longshot_accuracy_percent"
            if args.layout == "diffusion_heatmap"
            else (
                "mean_backbones(100 * (value / baseline_value - 1))"
                if args.layout == "campaign_heatmap"
                else "100 * (value / baseline_value - 1)"
            )
        ),
        "comparison_baseline": (
            "Longshot" if args.layout == "diffusion_heatmap" else "Instruct"
        ),
        "thinking_modes": (
            ["off"]
            if args.layout in {"diffusion_heatmap", "campaign_heatmap"}
            else ["off", "on"]
        ),
        "relative_quantities": (
            ["accuracy"]
            if args.layout
            in {"paired_heatmap", "diffusion_heatmap", "campaign_heatmap"}
            else ["accuracy", "mean_response_tokens"]
        ),
        "baseline_matching": ["suite", "backbone", "thinking"],
        "source_precision": "Rounded values as reported in the paper tables",
        "matplotlib_version": matplotlib.__version__,
    }
    if args.layout == "campaign_heatmap":
        manifest["aggregation"] = {
            "weighting": "Equal weight per available backbone, separately per cell",
            "missing_values": "Omitted; N/A if no backbones have results",
            "standard_deviation": "Population SD across available backbones (ddof=0)",
            "cell_annotations": "Mean with plus/minus standard deviation",
            "counts_and_backbones": "campaign_heatmap_averages.csv",
        }
    manifest_name = (
        f"manifest_{args.layout}.json"
        if args.layout != "bars"
        else "manifest.json"
    )
    (args.output_dir / manifest_name).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
