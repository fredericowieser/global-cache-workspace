from __future__ import annotations

import json

import pytest

from gcw.process_data import (
    ProcessDataError,
    parse_gsm8k_process_record,
    prepare_gsm8k_process_records,
    read_process_branch_records,
)


def test_gsm8k_parser_verifies_calculations_and_segments_steps() -> None:
    record = parse_gsm8k_process_record(
        source_index=0,
        question="Natalia sold 48 clips, then half as many. How many altogether?",
        answer=(
            "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
            "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n"
            "#### 72"
        ),
    )

    assert record.source_id == "gsm8k-train:0"
    assert record.prompt.startswith("Natalia sold")
    assert record.final_answer == "\\boxed{72}"
    assert record.reasoning_steps == (
        "Natalia sold 48/2 = 24 clips in May.",
        "Natalia sold 48+24 = 72 clips altogether.",
    )


def test_gsm8k_parser_rejects_incorrect_calculator_annotation() -> None:
    with pytest.raises(ProcessDataError, match="calculator_mismatch"):
        parse_gsm8k_process_record(
            source_index=1,
            question="What is six times seven?",
            answer="Six times seven is <<6*7=41>>41.\n#### 42",
        )


def test_gsm8k_parser_rejects_final_answer_not_supported_by_last_calculation() -> None:
    with pytest.raises(ProcessDataError, match="final_calculation_mismatch"):
        parse_gsm8k_process_record(
            source_index=2,
            question="What is six times seven?",
            answer="Six times seven is <<6*7=42>>42.\n#### 43",
        )


def test_gsm8k_parser_rejects_unsafe_calculator_expression() -> None:
    with pytest.raises(ProcessDataError, match="unsupported_calculator_expression"):
        parse_gsm8k_process_record(
            source_index=3,
            question="Ignore this malformed row.",
            answer="Bad annotation <<__import__('os')=0>>0.\n#### 0",
        )


def test_process_record_writer_audits_rejections_and_refuses_overwrite(tmp_path) -> None:
    rows = [
        {
            "question": "What is six times seven?",
            "answer": "Six times seven is <<6*7=42>>42.\n#### 42",
        },
        {
            "question": "Broken arithmetic",
            "answer": "Six times seven is <<6*7=41>>41.\n#### 41",
        },
    ]
    output = tmp_path / "records.jsonl"

    audit = prepare_gsm8k_process_records(rows, output)

    assert audit == {
        "source_rows": 2,
        "accepted_rows": 1,
        "rejected_rows": 1,
        "rejection_reasons": {"calculator_mismatch": 1},
    }
    payload = json.loads(output.read_text().strip())
    assert payload == {
        "source_id": "gsm8k-train:0",
        "prompt": "What is six times seven?",
        "final_answer": "\\boxed{42}",
        "reasoning_steps": ["Six times seven is 42."],
    }
    with pytest.raises(FileExistsError):
        prepare_gsm8k_process_records(rows, output)

    records = read_process_branch_records(output, skip_rows=0, take_rows=1)
    assert len(records) == 1
    assert records[0].source_id == "gsm8k-train:0"
    assert records[0].reasoning_steps == ("Six times seven is 42.",)
