from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_module_path = (
    Path(__file__).parents[1] / "scripts" / "prepare_openr1_bgt_mixture.py"
)
_spec = importlib.util.spec_from_file_location(
    "prepare_openr1_bgt_mixture", _module_path
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_interleave_by_token_mass = _module._interleave_by_token_mass
_trace_record = _module._trace_record
_admission_policy = _module._admission_policy


def test_openr1_trace_admission_requires_complete_math_verified_first_generation() -> (
    None
):
    valid = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>2+2=4</think>\\n\\n4"},
        ],
        "is_reasoning_complete": [True],
        "correctness_math_verify": [True],
    }
    answer, reason, raw = _trace_record(valid)
    assert answer == "<think>2+2=4</think>\\n\\n4"
    assert reason == "source_is_reasoning_complete_and_math_verified"
    assert raw == answer

    incomplete = dict(valid)
    incomplete["is_reasoning_complete"] = [False]
    answer, reason, _ = _trace_record(incomplete)
    assert answer is None
    assert reason == "source_reasoning_incomplete"


def test_interleave_by_token_mass_balances_prefix() -> None:
    direct = [
        {"source_arm": "openmath", "target_tokens": 10, "source_id": "d0"},
        {"source_arm": "openmath", "target_tokens": 10, "source_id": "d1"},
        {"source_arm": "openmath", "target_tokens": 10, "source_id": "d2"},
    ]
    cot = [
        {"source_arm": "openr1", "target_tokens": 20, "source_id": "c0"},
        {"source_arm": "openr1", "target_tokens": 20, "source_id": "c1"},
    ]
    rows = _interleave_by_token_mass(direct, cot)
    assert [row["source_arm"] for row in rows] == [
        "openmath",
        "openr1",
        "openmath",
        "openmath",
        "openr1",
    ]


def test_underfilled_openr1_is_explicitly_labelled() -> None:
    with pytest.raises(RuntimeError, match="below direct target"):
        _admission_policy(10, 20, allow_underfilled=False)
    assert (
        _admission_policy(10, 20, allow_underfilled=True)
        == "underfilled_openr1_cap"
    )
    with pytest.raises(RuntimeError, match="below direct target"):
        _admission_policy(0, 20, allow_underfilled=True)
