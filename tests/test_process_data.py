from __future__ import annotations

import json

import pytest

from gcw.process_data import (
    ProcessDataError,
    parse_gsm8k_eval_record,
    parse_gsm8k_process_record,
    prepare_gsm8k_answer_records,
    prepare_gsm8k_process_records,
    read_answer_only_records,
    read_answer_only_records_bytes,
    read_process_branch_records,
    read_process_branch_records_bytes,
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


def test_gsm8k_eval_parser_keeps_authoritative_target_without_calculator_gate() -> None:
    record = parse_gsm8k_eval_record(
        source_split="test",
        source_index=7,
        question="What is six times seven?",
        answer="A prose solution without annotations.\n#### 42",
    )

    assert record.source_id == "gsm8k-test:7"
    assert record.final_answer == "\\boxed{42}"
    assert record.reasoning_steps == ("A prose solution without annotations.",)


def test_gsm8k_parser_rejects_unmatched_calculator_syntax() -> None:
    with pytest.raises(ProcessDataError, match="malformed_calculator_annotation"):
        parse_gsm8k_process_record(
            source_index=2,
            question="Broken annotations?",
            answer="Bad <<2+2=5 and good <<2+2=4>>4.\n#### 4",
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


def test_answer_only_records_do_not_parse_or_filter_on_reasoning(tmp_path) -> None:
    rows = [
        {
            "question": "What is six times seven?",
            "answer": "Malformed <<6*7=41 reasoning is irrelevant.\n#### 42",
        },
        {
            "question": "What is nine minus four?",
            "answer": "#### 5",
        },
    ]
    output = tmp_path / "answer-only.jsonl"

    audit = prepare_gsm8k_answer_records(rows, output, source_split="train")

    assert audit == {
        "source_rows": 2,
        "accepted_rows": 2,
        "rejected_rows": 0,
        "rejection_reasons": {},
    }
    payloads = [json.loads(line) for line in output.read_text().splitlines()]
    assert payloads == [
        {
            "source_id": "gsm8k-train:0",
            "prompt": "What is six times seven?",
            "final_answer": "\\boxed{42}",
        },
        {
            "source_id": "gsm8k-train:1",
            "prompt": "What is nine minus four?",
            "final_answer": "\\boxed{5}",
        },
    ]
    records = read_answer_only_records(output)
    assert [record.source_id for record in records] == ["gsm8k-train:0", "gsm8k-train:1"]
    assert [record.final_answer for record in records] == ["\\boxed{42}", "\\boxed{5}"]


def test_bytes_readers_parse_complete_snapshot_before_slicing() -> None:
    answer_bytes = b"\n".join(
        [
            b'{"source_id":"a","prompt":"Q0","final_answer":"A0"}',
            b'{"source_id":"b","prompt":"Q1","final_answer":"A1"}',
            b'{"source_id":"a","prompt":"Q2","final_answer":"A2"}',
        ]
    ) + b"\n"
    process_bytes = (
        b'{"source_id":"p","prompt":"Q","final_answer":"A",'
        b'"reasoning_steps":["R"]}\nnot-json\n'
    )

    with pytest.raises(ProcessDataError, match="duplicate_source_id:a"):
        read_answer_only_records_bytes(answer_bytes, skip_rows=0, take_rows=1)
    with pytest.raises(ProcessDataError, match="invalid_json_row:1"):
        read_process_branch_records_bytes(process_bytes, skip_rows=0, take_rows=1)
