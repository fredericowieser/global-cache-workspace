from __future__ import annotations

import os

GENERAL_TASKS = [
    "hellaswag",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "mmlu",
]
MATH_TASKS = [
    "gsm8k_openmath",
    "svamp",
    "gsm8k_hard",
    "logiqa",
    "agieval_gaokao_mathqa",
]
GENERAL_RUNS = [
    (["hellaswag", "piqa"], 0),
    (["winogrande"], 5),
    (["arc_easy", "arc_challenge"], 25),
    (["mmlu"], 5),
]
MATH_RUNS = [([task], 0) for task in MATH_TASKS]
TASK_FEWSHOT = {
    task: fewshot
    for tasks, fewshot in [*GENERAL_RUNS, *MATH_RUNS]
    for task in tasks
}


def _filter_skipped_tasks(
    runs: list[tuple[list[str], int]],
) -> list[tuple[list[str], int]]:
    skip = {
        task.strip()
        for task in os.environ.get("GCW_EVAL_SKIP_TASKS", "").split(",")
        if task.strip()
    }
    if not skip:
        return runs
    filtered = []
    for tasks, fewshot in runs:
        kept = [task for task in tasks if task not in skip]
        if kept:
            filtered.append((kept, fewshot))
    return filtered


def _normalize_until(until) -> tuple[str, ...]:
    if until is None:
        return ()
    if isinstance(until, str):
        return (until,)
    return tuple(str(stop) for stop in until if stop is not None)


def _trim_until(text: str, until: tuple[str, ...]) -> str:
    for stop in until:
        if stop and stop in text:
            text = text.split(stop)[0]
    return text


def _run_is_complete(tasks: list[str], merged: dict) -> bool:
    results = merged.get("results", {})
    groups = merged.get("groups", {})
    return all(task in results or task in groups for task in tasks)
