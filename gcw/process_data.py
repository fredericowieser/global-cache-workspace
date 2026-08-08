from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path
import re
from typing import Iterable


_CALCULATOR_RE = re.compile(r"<<([^<>]+)>>")
_FINAL_ANSWER_RE = re.compile(r"^####\s*(.+?)\s*$")


class ProcessDataError(ValueError):
    pass


@dataclass(frozen=True)
class ProcessBranchRecord:
    source_id: str
    prompt: str
    final_answer: str
    reasoning_steps: tuple[str, ...]


def parse_gsm8k_process_record(
    *,
    source_index: int,
    question: str,
    answer: str,
) -> ProcessBranchRecord:
    prompt = question.strip()
    if not prompt:
        raise ProcessDataError("empty_prompt")

    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not lines:
        raise ProcessDataError("empty_answer")
    final_match = _FINAL_ANSWER_RE.fullmatch(lines[-1])
    if final_match is None:
        raise ProcessDataError("missing_final_answer")
    final_value = _parse_number(final_match.group(1), error_code="unsupported_final_answer")

    reasoning_lines = lines[:-1]
    calculations: list[Fraction] = []
    steps = []
    for line in reasoning_lines:
        for annotation in _CALCULATOR_RE.findall(line):
            if "=" not in annotation:
                raise ProcessDataError("malformed_calculator_annotation")
            expression, stated_result = annotation.rsplit("=", 1)
            computed = _evaluate_arithmetic(expression)
            expected = _parse_number(stated_result, error_code="unsupported_calculator_result")
            if computed != expected:
                raise ProcessDataError("calculator_mismatch")
            calculations.append(expected)
        clean = _CALCULATOR_RE.sub("", line).strip()
        if clean:
            steps.append(clean)

    if not calculations:
        raise ProcessDataError("missing_calculator_annotation")
    if calculations[-1] != final_value:
        raise ProcessDataError("final_calculation_mismatch")
    if not steps:
        raise ProcessDataError("missing_reasoning_steps")

    return ProcessBranchRecord(
        source_id=f"gsm8k-train:{source_index}",
        prompt=prompt,
        final_answer=f"\\boxed{{{_format_fraction(final_value)}}}",
        reasoning_steps=tuple(steps),
    )


def prepare_gsm8k_process_records(
    rows: Iterable[dict],
    output: str | Path,
) -> dict[str, object]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_rows = 0
    accepted_rows = 0
    rejection_reasons: Counter[str] = Counter()
    created = False
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            created = True
            for source_index, row in enumerate(rows):
                source_rows += 1
                question = row.get("question") if isinstance(row, dict) else None
                answer = row.get("answer") if isinstance(row, dict) else None
                if not isinstance(question, str) or not isinstance(answer, str):
                    rejection_reasons["invalid_schema"] += 1
                    continue
                try:
                    record = parse_gsm8k_process_record(
                        source_index=source_index,
                        question=question,
                        answer=answer,
                    )
                except ProcessDataError as exc:
                    rejection_reasons[str(exc)] += 1
                    continue
                payload = asdict(record)
                payload["reasoning_steps"] = list(record.reasoning_steps)
                handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
                accepted_rows += 1
    except Exception:
        if created:
            output_path.unlink(missing_ok=True)
        raise
    return {
        "source_rows": source_rows,
        "accepted_rows": accepted_rows,
        "rejected_rows": source_rows - accepted_rows,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }


def read_process_branch_records(
    source: str | Path,
    *,
    skip_rows: int = 0,
    take_rows: int | None = None,
) -> list[ProcessBranchRecord]:
    if skip_rows < 0 or (take_rows is not None and take_rows < 0):
        raise ValueError("process record row bounds must be non-negative")
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[ProcessBranchRecord] = []
    seen_source_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index < skip_rows:
                continue
            if take_rows is not None and len(records) >= take_rows:
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProcessDataError(f"invalid_json_row:{row_index}") from exc
            if not isinstance(payload, dict):
                raise ProcessDataError(f"invalid_schema_row:{row_index}")
            source_id = payload.get("source_id")
            prompt = payload.get("prompt")
            final_answer = payload.get("final_answer")
            reasoning_steps = payload.get("reasoning_steps")
            if (
                not isinstance(source_id, str)
                or not source_id
                or not isinstance(prompt, str)
                or not prompt
                or not isinstance(final_answer, str)
                or not final_answer
                or not isinstance(reasoning_steps, list)
                or not reasoning_steps
                or any(not isinstance(step, str) or not step for step in reasoning_steps)
            ):
                raise ProcessDataError(f"invalid_schema_row:{row_index}")
            if source_id in seen_source_ids:
                raise ProcessDataError(f"duplicate_source_id:{source_id}")
            seen_source_ids.add(source_id)
            records.append(
                ProcessBranchRecord(
                    source_id=source_id,
                    prompt=prompt,
                    final_answer=final_answer,
                    reasoning_steps=tuple(reasoning_steps),
                )
            )
    return records


def _evaluate_arithmetic(expression: str) -> Fraction:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _evaluate_node(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        raise ProcessDataError("unsupported_calculator_expression") from exc
    if abs(value.numerator) > 10**12 or value.denominator > 10**12:
        raise ProcessDataError("unsupported_calculator_expression")
    return value


def _evaluate_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return Fraction(str(node.value))
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
    raise ValueError(f"unsupported arithmetic node: {type(node).__name__}")


def _parse_number(text: str, *, error_code: str) -> Fraction:
    cleaned = text.strip().replace(",", "").removeprefix("$")
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError) as exc:
        raise ProcessDataError(error_code) from exc


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
