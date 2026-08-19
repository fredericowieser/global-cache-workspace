"""Type-aware TheoremQA scorer for lm-eval-harness."""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List

_PROMPT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within "
    "\\boxed{}. For a True/False question answer with \\boxed{True} or "
    "\\boxed{False}; for a multiple-choice question box the option letter."
)


def doc_to_text(doc: Dict[str, Any]) -> str:
    return doc["Question"] + _PROMPT_SUFFIX


def doc_to_target(doc: Dict[str, Any]) -> str:
    return str(doc["Answer"])


def _last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def _extract_prediction(text: str) -> str:
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    m = re.search(r"(?i)answer\s*(?:is|=|:)\s*(.+?)(?:\.\s*$|\n|$)", text)
    if m:
        return m.group(1).strip()
    lst = re.findall(r"\[[^\[\]]*\]", text)
    if lst:
        return lst[-1].strip()
    nums = re.findall(r"-?\d+\.?\d*", text)
    if nums:
        return nums[-1]
    return text.strip()[-64:]


def _to_float(s: str) -> float | None:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.replace("\\", "").replace("pi", "3.141592653589793")
    try:
        return float(s)
    except ValueError:
        m = re.match(r"^\s*(-?\d+\.?\d*)\s*/\s*(-?\d+\.?\d*)\s*$", s)
        if m:
            try:
                return float(m.group(1)) / float(m.group(2))
            except (ValueError, ZeroDivisionError):
                return None
        return None


def _float_match(pred: str, gold: str) -> bool:
    p, g = _to_float(pred), _to_float(gold)
    if p is None or g is None:
        return False
    if g == 0.0:
        return abs(p) <= 1e-2
    if abs(p - g) <= 1e-2 * abs(g):
        return True
    decimals = len(gold.split(".")[-1]) if "." in gold else 0
    return round(p, decimals) == round(g, decimals)


def _int_match(pred: str, gold: str) -> bool:
    p, g = _to_float(pred), _to_float(gold)
    if p is None or g is None:
        return False
    return abs(p - g) < 1e-6


_TRUE = {"true", "yes", "1", "t"}
_FALSE = {"false", "no", "0", "f"}


def _bool_match(pred: str, gold: str) -> bool:
    pl = pred.strip().lower().strip(".")
    gl = gold.strip().lower().strip(".")
    pbool = None
    for tok in re.findall(r"[a-z]+|\d", pl):
        if tok in _TRUE:
            pbool = True
            break
        if tok in _FALSE:
            pbool = False
            break
    gbool = gl in _TRUE
    return pbool is not None and pbool == gbool


def _option_match(pred: str, gold: str) -> bool:
    pg = re.search(r"[a-eA-E]", gold)
    pp = re.search(r"[a-eA-E]", pred)
    return bool(pg and pp and pg.group(0).lower() == pp.group(0).lower())


def _parse_list(s: str) -> List[float] | None:
    s = s.strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
    except (ValueError, SyntaxError, TypeError):
        pass
    nums = re.findall(r"-?\d+\.?\d*", s)
    return [float(x) for x in nums] if nums else None


def _float_close(p: float, g: float) -> bool:
    if g == 0.0:
        return abs(p) <= 1e-2
    return abs(p - g) <= 1.5e-2 * abs(g)


def _list_match(pred: str, gold: str) -> bool:
    p, g = _parse_list(pred), _parse_list(gold)
    if p is None or g is None or len(p) != len(g):
        return False
    return all(_float_close(a, b) for a, b in zip(p, g))


def _is_correct(pred: str, gold: str, answer_type: str) -> bool:
    at = (answer_type or "").lower()
    try:
        if "bool" in at:
            return _bool_match(pred, gold)
        if "option" in at:
            return _option_match(pred, gold)
        if "list" in at:
            return _list_match(pred, gold)
        if "integer" in at:
            return _int_match(pred, gold)
        if "float" in at:
            return _float_match(pred, gold)
        return _float_match(pred, gold) or pred.strip().lower() == gold.strip().lower()
    except Exception:
        return False


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    completion = results[0]
    pred = _extract_prediction(completion)
    gold = str(doc["Answer"])
    correct = _is_correct(pred, gold, doc.get("Answer_type", ""))
    return {"exact_match": 1.0 if correct else 0.0}
