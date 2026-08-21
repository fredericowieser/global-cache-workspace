from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from scripts.run_math_experiments_full_single_seed import (
    CANONICAL_CORE_TASKS,
    CANONICAL_REASONING_TASKS,
    NUMERIC_MATH_TASKS,
)


def _canonical_metric(payload: dict[str, Any], task: str) -> tuple[str, float]:
    metrics = payload["results"][task]
    key = "acc_norm,none" if "acc_norm,none" in metrics else "acc,none"
    return key, float(metrics[key])


def compare_lineages(
    reference: Path,
    candidate: Path,
    *,
    arm: str,
    require_complete_reference: bool,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    violations: list[str] = []
    for group, tasks in (
        ("core", CANONICAL_CORE_TASKS),
        ("reasoning", CANONICAL_REASONING_TASKS),
    ):
        reference_path = (
            reference
            / arm
            / "processor-on"
            / "canonical-likelihood"
            / group
            / "result.json"
        )
        candidate_path = (
            candidate
            / arm
            / "processor-on"
            / "canonical-likelihood"
            / group
            / "result.json"
        )
        if not reference_path.is_file() or not candidate_path.is_file():
            violations.append(f"missing canonical {group} result")
            continue
        reference_payload = json.loads(reference_path.read_text())
        candidate_payload = json.loads(candidate_path.read_text())
        for task in tasks:
            reference_key, reference_score = _canonical_metric(
                reference_payload, task
            )
            candidate_key, candidate_score = _canonical_metric(
                candidate_payload, task
            )
            delta = candidate_score - reference_score
            passed = reference_key == candidate_key and abs(delta) <= 0.005
            comparisons.append(
                {
                    "protocol": "canonical_likelihood",
                    "task": task,
                    "metric": candidate_key,
                    "reference": reference_score,
                    "candidate": candidate_score,
                    "delta": delta,
                    "tolerance": 0.005,
                    "passed": passed,
                }
            )
            if not passed:
                violations.append(f"canonical parity failed for {task}")
    for thinking in ("thinking_off", "thinking_on"):
        accuracy_tolerance = 0.01 if thinking == "thinking_off" else 0.015
        agreement_tolerance = 0.97 if thinking == "thinking_off" else 0.90
        for task in NUMERIC_MATH_TASKS:
            relative = (
                Path("processor-on")
                / "numeric-generation"
                / thinking
                / "tasks"
                / task
                / "result.json"
            )
            reference_path = reference / arm / relative
            candidate_path = candidate / arm / relative
            if not reference_path.is_file():
                if require_complete_reference:
                    violations.append(
                        f"missing reference numeric result for {thinking}/{task}"
                    )
                continue
            if not candidate_path.is_file():
                violations.append(
                    f"missing candidate numeric result for {thinking}/{task}"
                )
                continue
            reference_payload = json.loads(reference_path.read_text())
            candidate_payload = json.loads(candidate_path.read_text())
            reference_rows = reference_payload["rows"]
            candidate_rows = candidate_payload["rows"]
            reference_identity = [
                (
                    row["task"],
                    row["source_task"],
                    row["source_index"],
                    row["target"],
                )
                for row in reference_rows
            ]
            candidate_identity = [
                (
                    row["task"],
                    row["source_task"],
                    row["source_index"],
                    row["target"],
                )
                for row in candidate_rows
            ]
            identity_match = reference_identity == candidate_identity
            paired = min(len(reference_rows), len(candidate_rows))
            agreement = (
                sum(
                    reference_rows[index].get("prediction")
                    == candidate_rows[index].get("prediction")
                    for index in range(paired)
                )
                / paired
                if paired
                else 0.0
            )
            reference_metrics = reference_payload["tasks"][task]
            candidate_metrics = candidate_payload["tasks"][task]
            deltas = {
                metric: float(candidate_metrics[metric])
                - float(reference_metrics[metric])
                for metric in ("accuracy", "parse_rate", "truncation_rate")
            }
            passed = (
                identity_match
                and abs(deltas["accuracy"]) <= accuracy_tolerance
                and abs(deltas["parse_rate"]) <= 0.02
                and abs(deltas["truncation_rate"]) <= 0.02
                and agreement >= agreement_tolerance
            )
            comparisons.append(
                {
                    "protocol": "numeric_generation",
                    "task": task,
                    "thinking": thinking,
                    "identity_match": identity_match,
                    "prediction_agreement": agreement,
                    "minimum_prediction_agreement": agreement_tolerance,
                    "deltas": deltas,
                    "accuracy_tolerance": accuracy_tolerance,
                    "rate_tolerance": 0.02,
                    "passed": passed,
                }
            )
            if not passed:
                violations.append(
                    f"numeric parity failed for {thinking}/{task}"
                )
    return {
        "schema": "gcw.math_evaluation_lineage_parity.v1",
        "arm": arm,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference": str(reference.resolve()),
        "candidate": str(candidate.resolve()),
        "require_complete_reference": require_complete_reference,
        "passed": not violations,
        "violations": violations,
        "comparisons": comparisons,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare H200 and H100 math-evaluation lineages"
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--allow-incomplete-reference", action="store_true")
    parser.add_argument("--out")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = compare_lineages(
        Path(args.reference).expanduser(),
        Path(args.candidate).expanduser(),
        arm=args.arm,
        require_complete_reference=not args.allow_incomplete_reference,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        output = Path(args.out).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
    else:
        print(encoded, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
