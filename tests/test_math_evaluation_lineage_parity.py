from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_math_evaluation_lineages import compare_lineages
from scripts.run_math_experiments_full_single_seed import (
    CANONICAL_CORE_TASKS,
    CANONICAL_REASONING_TASKS,
    NUMERIC_MATH_TASKS,
)


def _write_lineage(root: Path, arm: str) -> None:
    for group, tasks in (
        ("core", CANONICAL_CORE_TASKS),
        ("reasoning", CANONICAL_REASONING_TASKS),
    ):
        path = (
            root
            / arm
            / "processor-on"
            / "canonical-likelihood"
            / group
            / "result.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"results": {task: {"acc,none": 0.5} for task in tasks}})
        )
    for enable_thinking in (False, True):
        label = "thinking-enabled" if enable_thinking else "thinking-disabled"
        for task in NUMERIC_MATH_TASKS:
            path = (
                root
                / arm
                / "processor-on"
                / "numeric-generation"
                / label
                / "tasks"
                / task
                / "result.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "tasks": {
                            task: {
                                "accuracy": 0.5,
                                "parse_rate": 1.0,
                                "truncation_rate": 0.0,
                            }
                        },
                        "rows": [
                            {
                                "task": task,
                                "source_task": task,
                                "source_index": 0,
                                "target": "1",
                                "prediction": "1",
                            }
                        ],
                    }
                )
            )


def test_identical_evaluation_lineages_pass(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_lineage(reference, "instruct-baseline")
    _write_lineage(candidate, "instruct-baseline")

    result = compare_lineages(
        reference,
        candidate,
        arm="instruct-baseline",
        require_complete_reference=True,
    )

    assert result["passed"] is True
    assert result["violations"] == []
