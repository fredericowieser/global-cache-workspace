from __future__ import annotations

from fractions import Fraction
import re


_NUMBER_RE = re.compile(
    r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?"
)
_LATEX_FRACTION_RE = re.compile(
    r"^\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}$"
)


def extract_numeric_answer(text: str) -> Fraction | None:
    boxes = _balanced_box_contents(text)
    if boxes:
        return _parse_numeric_expression(boxes[-1])
    matches = list(_NUMBER_RE.finditer(text))
    if not matches:
        return None
    return _parse_numeric_expression(matches[-1].group(0))


def numeric_answers_match(prediction: str, target: str) -> bool:
    predicted = extract_numeric_answer(prediction)
    expected = extract_numeric_answer(target)
    return predicted is not None and expected is not None and predicted == expected


def _balanced_box_contents(text: str) -> list[str]:
    marker = r"\boxed{"
    results = []
    start = 0
    while True:
        marker_index = text.find(marker, start)
        if marker_index < 0:
            break
        content_start = marker_index + len(marker)
        depth = 1
        index = content_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth:
            return []
        results.append(text[content_start : index - 1])
        start = index
    return results


def _parse_numeric_expression(text: str) -> Fraction | None:
    cleaned = text.strip().replace("$", "").replace(",", "")
    latex_fraction = _LATEX_FRACTION_RE.fullmatch(cleaned)
    if latex_fraction is not None:
        numerator = _parse_numeric_expression(latex_fraction.group(1))
        denominator = _parse_numeric_expression(latex_fraction.group(2))
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator
    cleaned = re.sub(r"\s+", "", cleaned)
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None
