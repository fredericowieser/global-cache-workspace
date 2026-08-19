#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TASK_COLUMNS = {
    "gsm8k_openmath": "GSM8K",
    "svamp": "SVAMP",
    "gsm8k_hard": "GSM-Hd",
    "theoremqa": "ThmQA",
    "logiqa": "LogiQA",
    "agieval_gaokao_mathqa": "Gaokao",
}
TASK_ORDER = tuple(TASK_COLUMNS)
TASK_METRICS = {
    "gsm8k_openmath": ("exact_match,flexible",),
    "svamp": ("exact_match,flexible",),
    "gsm8k_hard": ("exact_match,flexible",),
    "theoremqa": ("exact_match,none",),
    "logiqa": ("acc_norm,none",),
    "agieval_gaokao_mathqa": ("acc_norm,none",),
}
TARGET_BGT = {
    "GSM8K": 35.60,
    "SVAMP": 46.70,
    "GSM-Hd": 7.70,
    "ThmQA": 9.20,
    "LogiQA": 29.50,
    "Gaokao": 25.40,
    "AVG": 25.68,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten GCW paper7 eval JSON into the legacy paper table."
    )
    parser.add_argument("--json", required=True, help="GCW eval results JSON")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    parser.add_argument(
        "--md",
        "--out",
        dest="md",
        default=None,
        help="Optional Markdown output path",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.json).read_text())
    row = _extract_row(payload)
    if args.csv:
        _write_csv(row, args.csv)
    text = _markdown(row)
    if args.md:
        Path(args.md).write_text(text, encoding="utf-8")
    print(text)


def _extract_row(payload: dict) -> dict[str, float | None]:
    results = payload.get("results", {})
    row = {
        TASK_COLUMNS[task]: _accuracy(results.get(task, {}), TASK_METRICS[task])
        for task in TASK_ORDER
    }
    values = [value for value in row.values() if value is not None]
    row["AVG"] = sum(values) / len(values) if values else None
    return row


def _accuracy(metrics: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, int | float):
            return 100.0 * float(value)
    return None


def _write_csv(row: dict[str, float | None], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = [*TASK_COLUMNS.values(), "AVG"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow({key: _format(row[key]) for key in columns})


def _markdown(row: dict[str, float | None]) -> str:
    columns = [*TASK_COLUMNS.values(), "AVG"]
    target = _target_for_row(row)
    header = "| Model | " + " | ".join(columns) + " |"
    sep = "|---" * (len(columns) + 1) + "|"
    values = (
        "| GCW BGT | " + " | ".join(_format(row[col]) for col in columns) + " |"
    )
    targets = (
        "| Target BGT | "
        + " | ".join(_format(target[col]) for col in columns)
        + " |"
    )
    deltas = (
        "| Delta | "
        + " | ".join(_delta(row[col], target[col]) for col in columns)
        + " |"
    )
    return "\n".join([header, sep, values, targets, deltas])


def _target_for_row(row: dict[str, float | None]) -> dict[str, float | None]:
    target = {column: TARGET_BGT[column] for column in TASK_COLUMNS.values()}
    present = [
        column
        for column in TASK_COLUMNS.values()
        if row.get(column) is not None
    ]
    target["AVG"] = (
        sum(TARGET_BGT[column] for column in present) / len(present)
        if present
        else None
    )
    return target


def _format(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def _delta(value: float | None, target: float | None) -> str:
    return "" if value is None or target is None else f"{value - target:+.2f}"


if __name__ == "__main__":
    main()
