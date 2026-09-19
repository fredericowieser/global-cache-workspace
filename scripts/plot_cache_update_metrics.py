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


def load_metrics(path: Path, tasks: list[str] | None = None) -> tuple:
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
                }:
                    raise ValueError("unsupported metrics schema")
                task = row["task"]
                if tasks and task not in tasks:
                    continue
                identity = (task, row["example_id"])
                if identity in episodes:
                    raise ValueError(f"duplicate episode {identity}")
                episodes.add(identity)
                task_counts[task] += 1
                empty += not row["steps"]
                seen = set()
                for step in row["steps"]:
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
    groups = [group for group in GROUP_LABELS if group in present]
    groups += sorted(present - GROUP_LABELS.keys())
    iterations = sorted({key[1] for key in samples})
    steps = sorted({key[2] for key in samples})
    counts = [len(values) for values in samples.values()]
    fig, axes = plt.subplots(
        len(groups),
        len(STATISTICS),
        figsize=(16, 2.8 * len(groups) + 1.5),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    column_titles = ("Head minimum", "Head mean", "Head maximum", "Head std")
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
            ax.set_title(f"{GROUP_LABELS.get(group, group)} · {stat}")
            if row_index == 0:
                ax.set_title(f"{column_titles[column]}\n{ax.get_title()}")
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
            if row_index == len(groups) - 1:
                ax.set_xlabel("Solver step")
    if len(iterations) > 1:
        axes[0, 0].legend(fontsize="small")
    task_summary = ", ".join(
        f"{task}: {count:,}" for task, count in sorted(task_counts.items())
    )
    fig.suptitle(
        f"{title}\n"
        "Line: equal-episode mean; shading: episode 25th–75th percentiles\n"
        f"{sum(task_counts.values()):,} episodes ({empty:,} empty); "
        f"n per point: {min(counts):,}–{max(counts):,}\n{task_summary}",
        fontsize=11,
    )
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
    args = parser.parse_args()
    output = args.out or args.results.with_suffix(
        ".direction.png" if args.quantity == "direction" else ".png"
    )
    if output.resolve() == args.results.resolve():
        parser.error("output must not overwrite the input")
    try:
        samples, task_counts, empty = load_metrics(args.results, args.task)
        fig = plot_metrics(
            samples, task_counts, empty, args.title, args.quantity
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
