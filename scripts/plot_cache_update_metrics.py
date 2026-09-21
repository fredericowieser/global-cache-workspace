#!/usr/bin/env python3
"""Plot episode-averaged diffusion diagnostics from a metrics JSONL sidecar."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

STATISTICS = ("min", "mean", "max", "std")
GROUP_LABELS = {
    "latent_kv": "KV latent change",
    "latent_linear": "Linear latent change",
    "key": "Proposed key change",
    "value": "Proposed value change",
    "linear_key": "Pre-writer linear key change",
    "linear_value": "Pre-writer linear value change",
    "kv": "Proposed KV change",
    "linear_kv": "Pre-writer linear KV change",
    "linear_state": "Proposed linear-state change",
}

DIRECTION_SUFFIXES = (
    "_direction_change",
    "_direction_to_original",
    "_cache_direction_from_original",
)
for _stream in ("key", "value", "linear_key", "linear_value"):
    _label = _stream.replace("_", " ").capitalize()
    GROUP_LABELS[f"{_stream}_direction_change"] = (
        f"{_label} contribution rotation"
    )
    if not _stream.startswith("linear_"):
        GROUP_LABELS[f"{_stream}_direction_to_original"] = (
            f"{_label} contribution vs original"
        )
        GROUP_LABELS[f"{_stream}_cache_direction_from_original"] = (
            f"{_label} proposed cache rotation"
        )


def load_metrics(
    path: Path, tasks: list[str] | None = None, phase: str = "combined"
) -> tuple:
    if phase not in {"combined", "prefill", "decoding"}:
        raise ValueError(f"unknown phase: {phase}")
    samples = defaultdict(list)
    episodes = set()
    task_counts = defaultdict(int)
    empty = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row["schema"] not in {
                    "gcw.cache_update_metrics.v1",
                    "gcw.cache_update_metrics.v2",
                    "gcw.cache_update_metrics.v3",
                }:
                    raise ValueError("unsupported metrics schema")
                task = row["task"]
                if tasks and task not in tasks:
                    continue
                if (
                    phase != "combined"
                    and row["schema"] != "gcw.cache_update_metrics.v3"
                ):
                    raise ValueError(
                        "phase-specific metrics require a v3 sidecar; "
                        "historical sidecars contain only combined metrics"
                    )
                steps = (
                    row["steps"]
                    if phase == "combined"
                    else row["phase_steps"][phase]
                )
                identity = (task, row["example_id"])
                if identity in episodes:
                    raise ValueError(f"duplicate episode {identity}")
                episodes.add(identity)
                task_counts[task] += 1
                empty += not steps
                seen = set()
                for step in steps:
                    iteration = step["external_iteration"]
                    index = step["solver_step"]
                    if any(
                        type(value) is not int or value < 1
                        for value in (iteration, index)
                    ):
                        raise ValueError(
                            "iteration and step must be positive integers"
                        )
                    for group, metrics in step["metrics"].items():
                        key = (group, iteration, index)
                        if key in seen:
                            raise ValueError(f"duplicate step/group {key}")
                        seen.add(key)
                        values = np.asarray(
                            [metrics[stat] for stat in STATISTICS], dtype=float
                        )
                        if not np.isfinite(values).all() or (values < 0).any():
                            raise ValueError(
                                "measurements must be finite and nonnegative"
                            )
                        if metrics["applications"] <= 0:
                            raise ValueError("applications must be positive")
                        samples[key].append(values)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if tasks and (missing := set(tasks) - task_counts.keys()):
        raise ValueError(f"tasks not found: {', '.join(sorted(missing))}")
    if not samples:
        raise ValueError("no update metrics in the selected episodes")
    return dict(samples), dict(task_counts), empty


def plot_metrics(
    samples: dict,
    task_counts: dict,
    empty: int,
    title: str,
    quantity: str = "both",
):
    if quantity not in {"both", "magnitude", "direction"}:
        raise ValueError(f"unknown quantity: {quantity}")
    samples = {
        key: values
        for key, values in samples.items()
        if quantity == "both"
        or key[0].endswith(DIRECTION_SUFFIXES) == (quantity == "direction")
    }
    if not samples:
        raise ValueError(f"no {quantity} metrics in the selected episodes")
    present = {key[0] for key in samples}
    section_definitions = [
        (
            ("latent_kv", "latent_linear"),
            r"KV / Linear latent change — $\|z_s-z_{s-1}\|_2$: internal solver-step change.",
        ),
        (
            ("key", "value", "kv"),
            r"Key / Value contribution magnitude — $\|\Delta u_s\|_2$: this step's additional attention cache update.",
        ),
        (
            ("linear_key", "linear_value", "linear_kv"),
            r"Linear Key / Value contribution magnitude — $\|\Delta f_s\|_2$: this step's factor change before the writer.",
        ),
        (
            ("linear_state",),
            r"Linear-state update magnitude — $\|M_s-M_{s-1}\|_F$: this step's change to the proposed matrix update.",
        ),
        (
            ("key_direction_change", "value_direction_change"),
            r"Key / Value contribution rotation — $D(\Delta u_s,\Delta u_{s-1})$: consecutive update directions ($s\geq2$).",
        ),
        (
            ("key_direction_to_original", "value_direction_to_original"),
            r"Key / Value contribution vs original — $D(\Delta u_s,c_0)$: current update versus original cache.",
        ),
        (
            ("key_cache_direction_from_original", "value_cache_direction_from_original"),
            r"Key / Value proposed cache rotation — $D(c_s,c_0)$: accumulated cache rotation.",
        ),
        (
            ("linear_key_direction_change", "linear_value_direction_change"),
            r"Linear Key / Value contribution rotation — $D(\Delta f_s,\Delta f_{s-1})$: consecutive factor-update directions ($s\geq2$).",
        ),
    ]
    sections = [
        ([group for group in members if group in present], caption)
        for members, caption in section_definitions
        if any(group in present for group in members)
    ]
    known = {group for members, _ in sections for group in members}
    sections.extend(([group], group) for group in sorted(present - known))
    groups = [group for members, _ in sections for group in members]
    iterations = sorted({key[1] for key in samples})
    steps = sorted({key[2] for key in samples})
    counts = [len(values) for values in samples.values()]
    task_summary = ", ".join(
        f"{task}: {count:,}" for task, count in sorted(task_counts.items())
    )
    header_lines = [
        title,
        f"{sum(task_counts.values()):,} examples ({empty:,} empty); "
        f"n per point: {min(counts):,}–{max(counts):,}",
        task_summary,
        r"SHARED NOTATION (separate for each stream/head): $s$ = solver step; $c_0$ = original attention cache.",
        r"$u_s$ = accumulated proposed update; $\Delta u_s=u_s-u_{s-1}$; $c_s=c_0+u_s$; $u_0=0$.",
        r"$f_s$ = proposed linear key/value factor before the writer; $\Delta f_s=f_s-f_{s-1}$; $f_0=0$.",
    ]
    if any(not group.endswith(DIRECTION_SUFFIXES) for group in groups):
        header_lines.extend([
            r"$z_s$ = diffusion latent; $M_s$ = proposed matrix update after the writer's transformation; $M_0=0$.",
            r"$\|\cdot\|_2$ = vector L2 norm; $\|\cdot\|_F$ = matrix Frobenius norm.",
        ])
    if any(group.endswith(DIRECTION_SUFFIXES) for group in groups):
        header_lines.extend([
            r"$D(a,b)=1-\frac{a^\top b}{\|a\|_2\|b\|_2}$: 0 = aligned, 1 = perpendicular, 2 = opposite.",
            "Direction comparisons exclude vectors with norm at or below 1e-12.",
        ])
    header_lines.extend([
        "COLUMNS: head minimum / mean / maximum / std. dev., averaged over applications within each example.",
        "LINE: equal-weight mean across examples. BAND: example 25th–75th percentiles. X AXIS: solver step s.",
    ])
    header_height = 0.27 * len(header_lines) + 0.25
    row_height = 2.8
    caption_height = 0.4
    figure_height = header_height + row_height * len(groups) + caption_height * len(sections) + 0.15
    fig = plt.figure(figsize=(16, figure_height))
    fig.text(
        0.5,
        1 - 0.15 / figure_height,
        "\n".join(header_lines),
        ha="center",
        va="top",
        fontsize=11,
        linespacing=1.65,
    )
    axes = []
    offset = header_height
    for members, caption in sections:
        divider_y = 1 - offset / figure_height
        fig.add_artist(plt.Line2D(
            [0.025, 0.985], [divider_y, divider_y],
            transform=fig.transFigure, color="0.65", linewidth=0.8,
        ))
        fig.text(
            0.5,
            1 - (offset + 0.12) / figure_height,
            caption,
            ha="center",
            va="top",
            fontsize=12,
        )
        offset += caption_height
        for group in members:
            row = []
            for column in range(len(STATISTICS)):
                row.append(fig.add_axes(
                    [
                        0.065 + column * 0.24,
                        1 - (offset + row_height - 0.48) / figure_height,
                        0.185,
                        (row_height - 0.78) / figure_height,
                    ],
                    sharex=axes[0][0] if axes else (row[0] if row else None),
                ))
            axes.append(row)
            offset += row_height
    axes = np.asarray(axes)
    column_titles = ("Minimum", "Mean", "Maximum", "Std. dev.")
    stream_labels = {
        "latent_kv": "KV latent",
        "latent_linear": "Linear latent",
        "kv": "KV",
        "linear_kv": "Linear KV",
        "linear_state": "Linear state",
    }
    for row_index, group in enumerate(groups):
        for column, stat in enumerate(STATISTICS):
            ax = axes[row_index, column]
            for color_index, iteration in enumerate(iterations):
                mean = np.full(len(steps), np.nan)
                low = mean.copy()
                high = mean.copy()
                for position, step in enumerate(steps):
                    values = samples.get((group, iteration, step))
                    if values is not None:
                        values = np.asarray(values)[:, column]
                        mean[position] = values.mean()
                        low[position], high[position] = np.quantile(
                            values, [0.25, 0.75]
                        )
                color = f"C{color_index % 10}"
                ax.plot(
                    steps,
                    mean,
                    "o-",
                    color=color,
                    markersize=4,
                    label=f"External iteration {iteration}",
                )
                ax.fill_between(steps, low, high, color=color, alpha=0.18)
            stream = group
            for suffix in DIRECTION_SUFFIXES:
                stream = stream.removesuffix(suffix)
            label = stream_labels.get(stream, stream.replace("_", " ").capitalize())
            ax.set_title(f"{label} · {column_titles[column]}")
            ax.set_ylabel(
                "Cosine distance (unitless)"
                if group.endswith(DIRECTION_SUFFIXES)
                else (
                    "Magnitude (Frobenius norm)"
                    if group == "linear_state"
                    else "Magnitude (L2 norm)"
                )
            )
            ax.grid(alpha=0.2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
            ax.set_xlabel("Solver step")
    if len(iterations) > 1:
        axes[0, 0].legend(fontsize="small")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Cache-update metrics JSONL")
    parser.add_argument(
        "--out",
        type=Path,
        help="Output image (default: .png, or .direction.png for direction-only)",
    )
    parser.add_argument(
        "--task",
        action="append",
        help="Task to include; repeat for multiple tasks",
    )
    parser.add_argument(
        "--quantity",
        choices=("both", "magnitude", "direction"),
        default="both",
        help="Measurements to plot (default: both)",
    )
    parser.add_argument("--title", default="Diffusion update diagnostics")
    parser.add_argument(
        "--phase",
        choices=("combined", "prefill", "decoding"),
        default="combined",
        help="Application phase to plot (default: combined)",
    )
    args = parser.parse_args()
    suffix = ".direction.png" if args.quantity == "direction" else ".png"
    if args.phase != "combined":
        suffix = f".{args.phase}{suffix}"
    output = args.out or args.results.with_suffix(suffix)
    if output.resolve() == args.results.resolve():
        parser.error("output must not overwrite the input")
    try:
        samples, task_counts, empty = load_metrics(
            args.results, args.task, args.phase
        )
        fig = plot_metrics(
            samples,
            task_counts,
            empty,
            f"{args.title} ({args.phase})",
            args.quantity,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
