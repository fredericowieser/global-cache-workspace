from __future__ import annotations

from fractions import Fraction

from gcw.process_eval import extract_numeric_answer, numeric_answers_match


def test_numeric_answer_prefers_balanced_boxed_expression() -> None:
    assert extract_numeric_answer(r"work 99 then \boxed{\frac{3}{2}}") == Fraction(3, 2)


def test_numeric_answer_falls_back_to_last_visible_number() -> None:
    assert extract_numeric_answer("First 7, therefore 42.") == Fraction(42)


def test_numeric_answer_comparison_is_exact() -> None:
    assert numeric_answers_match(r"\boxed{1.5}", r"\boxed{3/2}")
    assert not numeric_answers_match(r"\boxed{4}", r"\boxed{5}")
    assert not numeric_answers_match("no answer", r"\boxed{5}")
