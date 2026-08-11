from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
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


@dataclass(frozen=True)
class AnswerOnlyRecord:
    source_id: str
    prompt: str
    final_answer: str


def parse_gsm8k_answer_record(
    *,
    source_split: str,
    source_index: int,
    question: str,
    answer: str,
) -> AnswerOnlyRecord:
    if not re.fullmatch(r"[a-z0-9_-]+", source_split):
        raise ValueError("invalid GSM8K source split")
    prompt = question.strip()
    if not prompt:
        raise ProcessDataError("empty_prompt")
    nonempty_lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not nonempty_lines:
        raise ProcessDataError("empty_answer")
    final_match = _FINAL_ANSWER_RE.fullmatch(nonempty_lines[-1])
    if final_match is None:
        raise ProcessDataError("missing_final_answer")
    final_value = _parse_number(final_match.group(1), error_code="unsupported_final_answer")
    return AnswerOnlyRecord(
        source_id=f"gsm8k-{source_split}:{source_index}",
        prompt=prompt,
        final_answer=f"\\boxed{{{_format_fraction(final_value)}}}",
    )


def parse_gsm8k_eval_record(
    *,
    source_split: str,
    source_index: int,
    question: str,
    answer: str,
) -> ProcessBranchRecord:
    if not re.fullmatch(r"[a-z0-9_-]+", source_split):
        raise ValueError("invalid GSM8K source split")
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
    reasoning_steps = tuple(lines[:-1])
    if not reasoning_steps:
        raise ProcessDataError("missing_reasoning_steps")
    return ProcessBranchRecord(
        source_id=f"gsm8k-{source_split}:{source_index}",
        prompt=prompt,
        final_answer=f"\\boxed{{{_format_fraction(final_value)}}}",
        reasoning_steps=reasoning_steps,
    )


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
        if "<<" in clean or ">>" in clean:
            raise ProcessDataError("malformed_calculator_annotation")
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


def prepare_gsm8k_eval_records(
    rows: Iterable[dict],
    output: str | Path,
    *,
    source_split: str,
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
                    record = parse_gsm8k_eval_record(
                        source_split=source_split,
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


def prepare_gsm8k_answer_records(
    rows: Iterable[dict],
    output: str | Path,
    *,
    source_split: str,
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
                    record = parse_gsm8k_answer_record(
                        source_split=source_split,
                        source_index=source_index,
                        question=question,
                        answer=answer,
                    )
                except ProcessDataError as exc:
                    rejection_reasons[str(exc)] += 1
                    continue
                handle.write(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False) + "\n")
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


def authenticate_answer_only_artifact(
    source_bytes: bytes,
    *,
    expected_artifact_sha256: str,
    expected_source_ids_sha256: str,
) -> list[AnswerOnlyRecord]:
    if not isinstance(source_bytes, bytes):
        raise TypeError("answer-only authentication requires an immutable bytes snapshot")
    if hashlib.sha256(source_bytes).hexdigest() != expected_artifact_sha256:
        raise ProcessDataError("answer-only artifact sha256 mismatch")

    records = read_answer_only_records_bytes(source_bytes)
    encoded_source_ids = json.dumps(
        [record.source_id for record in records],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if hashlib.sha256(encoded_source_ids).hexdigest() != expected_source_ids_sha256:
        raise ProcessDataError("answer-only ordered source-ID sha256 mismatch")
    return records


def _jsonl_payloads(source_bytes: bytes) -> list[tuple[int, object]]:
    if not isinstance(source_bytes, bytes):
        raise TypeError("JSONL snapshot must be bytes")
    try:
        text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProcessDataError("invalid_utf8") from exc
    payloads: list[tuple[int, object]] = []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for row_index, line in enumerate(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProcessDataError(f"invalid_json_row:{row_index}") from exc
        payloads.append((row_index, payload))
    return payloads


def _slice_records(
    records: list[AnswerOnlyRecord] | list[ProcessBranchRecord],
    *,
    skip_rows: int,
    take_rows: int | None,
):
    if skip_rows < 0 or (take_rows is not None and take_rows < 0):
        raise ValueError("record row bounds must be non-negative")
    end = None if take_rows is None else skip_rows + take_rows
    return records[skip_rows:end]


def read_answer_only_records_bytes(
    source_bytes: bytes,
    *,
    skip_rows: int = 0,
    take_rows: int | None = None,
) -> list[AnswerOnlyRecord]:
    records: list[AnswerOnlyRecord] = []
    seen_source_ids: set[str] = set()
    for row_index, payload in _jsonl_payloads(source_bytes):
        if not isinstance(payload, dict) or set(payload) != {"source_id", "prompt", "final_answer"}:
            raise ProcessDataError(f"invalid_schema_row:{row_index}")
        source_id = payload.get("source_id")
        prompt = payload.get("prompt")
        final_answer = payload.get("final_answer")
        if (
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(prompt, str)
            or not prompt
            or not isinstance(final_answer, str)
            or not final_answer
        ):
            raise ProcessDataError(f"invalid_schema_row:{row_index}")
        if source_id in seen_source_ids:
            raise ProcessDataError(f"duplicate_source_id:{source_id}")
        seen_source_ids.add(source_id)
        records.append(
            AnswerOnlyRecord(source_id=source_id, prompt=prompt, final_answer=final_answer)
        )
    return list(_slice_records(records, skip_rows=skip_rows, take_rows=take_rows))


def read_answer_only_records(
    source: str | Path,
    *,
    skip_rows: int = 0,
    take_rows: int | None = None,
) -> list[AnswerOnlyRecord]:
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_answer_only_records_bytes(
        path.read_bytes(), skip_rows=skip_rows, take_rows=take_rows
    )


def read_process_branch_records_bytes(
    source_bytes: bytes,
    *,
    skip_rows: int = 0,
    take_rows: int | None = None,
) -> list[ProcessBranchRecord]:
    records: list[ProcessBranchRecord] = []
    seen_source_ids: set[str] = set()
    for row_index, payload in _jsonl_payloads(source_bytes):
        if not isinstance(payload, dict) or set(payload) != {
            "source_id",
            "prompt",
            "final_answer",
            "reasoning_steps",
        }:
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
    return list(_slice_records(records, skip_rows=skip_rows, take_rows=take_rows))


def read_process_branch_records(
    source: str | Path,
    *,
    skip_rows: int = 0,
    take_rows: int | None = None,
) -> list[ProcessBranchRecord]:
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_process_branch_records_bytes(
        path.read_bytes(), skip_rows=skip_rows, take_rows=take_rows
    )


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
