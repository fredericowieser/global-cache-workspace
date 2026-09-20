"""KOR-Bench single-question grading, adapted under Apache-2.0.

Source: KOR-Bench/KOR-Bench, bb8194dff8e51e2e2247cf62aa9932c3d57d53d7,
eval/eval_utils.py. See licenses/KOR-Bench-NOTICE.md for modifications.
"""

from __future__ import annotations

import ast
import math
import operator
import re

_MATH_IDS = {
    18,
    73,
    74,
    77,
    94,
    115,
    116,
    117,
    121,
    122,
    123,
    125,
    131,
    132,
    134,
    135,
    136,
    141,
    143,
    149,
    145,
    146,
    147,
    151,
    152,
    153,
    154,
    155,
    156,
    160,
    161,
    162,
    164,
    165,
    166,
    170,
    206,
    209,
    211,
    212,
    213,
    214,
    215,
    217,
    218,
}


def extract_answer(text: str, clean_level: str = "basic") -> str | None:
    if clean_level == "puzzle":
        match = re.search(r"\[\[(.*?)\]\]", text, re.DOTALL)
        return match.group(1) if match else None
    matches = re.findall(r"\[\[\s*(.*?)\s*\]\]", text, re.DOTALL)
    if not matches:
        matches = re.findall(r"\$\\boxed\{(.*?)\}\$", text, re.DOTALL)
    if not matches:
        matches = re.findall(r"\[\s*(.*?)\s*\]", text, re.DOTALL)
    if not matches:
        return None
    value = matches[0].strip()
    if clean_level in {"clean", "logic", "math"}:
        value = value.replace('"', "").replace("\n", "")
    if clean_level in {"clean", "logic"}:
        value = value.replace(" ", "")
    if clean_level == "logic":
        value = value.replace(".", "")
    if clean_level in {"clean", "math"}:
        value = value.replace("[", "").replace("]", "")
    if clean_level == "math":
        value = value.replace("$", "")
        if "=" in value:
            value = value.rsplit("=", 1)[1]
        value = re.sub(r"\\[()\[\]]", "", value)
        value = re.sub(r"\\text\{.*?\}", "", value)
        value = re.sub(r"\\(left|right|displaystyle)", "", value)
        return value.replace("\\\\", "\\")
    return f"[[{value}]]"


def arithmetic_value(expression: str) -> float | int:
    if len(expression) > 512:
        raise ValueError("arithmetic expression is too long")
    tree = ast.parse(expression.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("arithmetic expression is too complex")
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            value = operations[type(node.op)](
                evaluate(node.left), evaluate(node.right)
            )
        else:
            raise ValueError("only numeric arithmetic is supported")
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("arithmetic value is out of bounds")
        return value

    return evaluate(tree.body)


def _math_equal(response: str, answer: str) -> bool:
    import sympy
    from sympy.parsing.latex import parse_latex

    predicted = extract_answer(response, "math")
    target = extract_answer(answer, "math")
    if predicted is None or target is None:
        return False
    try:
        if "or" in target:
            return {
                sympy.simplify(parse_latex(part.strip()))
                for part in predicted.split("or")
            } == {
                sympy.simplify(parse_latex(part.strip()))
                for part in target.split("or")
            }
        return sympy.simplify(parse_latex(predicted)) == sympy.simplify(
            parse_latex(target)
        )
    except ImportError:
        raise
    except Exception:
        return predicted == target


def _puzzle_equal(predicted: str, target: str, rule_id: str) -> bool:
    if rule_id == "1":
        return (
            re.sub(r"[^A-Za-z]", "", predicted).lower()
            == re.sub(r"[^A-Za-z]", "", target).lower()
        )
    if rule_id == "2":
        return re.sub(r"[^A-Za-z]", "", predicted).lower() in target.split(",")
    if rule_id == "3":
        return sorted(
            x for x in re.split(r"\W+", predicted.lower()) if x
        ) == sorted(target.split(" "))
    if rule_id == "4":
        return re.sub(r"[^A-Za-z]", "", predicted).lower() in target
    if rule_id == "5":
        return sorted(re.sub(r"\s+", "", predicted).split(",")) == sorted(
            target.split(",")
        )
    if rule_id in {"9", "10"}:
        predicted = predicted.replace("×", "*").replace("−", "-").split("=")[0]
        if rule_id == "9":
            target = target.replace("×", "*").replace("−", "-")
            if re.findall(r"[+\-*/]", predicted) != re.findall(
                r"[+\-*/]", target.split("=")[0]
            ):
                return False
            match = re.search(r"=\s*(-?\d+)", target)
            if match is None:
                return False
            expected = int(match.group(1))
        else:
            target = target.split("\n")[0].split("=")[0]
            if sorted(re.sub(r"\W+", "", predicted)) != sorted(
                re.sub(r"\W+", "", target)
            ):
                return False
            expected = 24
        try:
            return arithmetic_value(predicted) == expected
        except (ValueError, SyntaxError, ArithmeticError, RecursionError):
            return False
    pattern = r"[,\s\[\]]+" if rule_id == "18" else r"\W+"
    return re.sub(pattern, "", predicted) == re.sub(pattern, "", target)


def is_correct(
    response: str, answer: str, category: str, rule_id: str, idx: str
) -> bool:
    level = "puzzle" if category == "puzzle" else "basic"
    extracted = extract_answer(response, level)
    if extracted is None or not extracted.strip("[] \t\r\n"):
        return False
    if category == "logic":
        predicted = extract_answer(response, "logic")
        target = extract_answer(answer, "logic")
        if rule_id == "5":
            return sorted(predicted.split(";")) == sorted(target.split(";"))
        return predicted == target
    if category == "puzzle":
        return _puzzle_equal(
            extracted, extract_answer(answer, "puzzle"), rule_id
        )
    if category == "operation":
        predicted = extract_answer(response, "clean")
        if idx in {"178", "179"}:
            pattern = r"(≥|≤)\s*([-]?\d+\.?\d*)"
            return sorted(re.findall(pattern, predicted)) == sorted(
                re.findall(pattern, answer)
            )
        if rule_id == "18":
            return sorted(re.sub(r"\W+", "", predicted)) == sorted(
                re.sub(r"\W+", "", extract_answer(answer, "puzzle"))
            )
        if rule_id in {"23", "24", "25"}:
            return list(map(int, re.findall(r"\d+", predicted))) == list(
                map(int, re.findall(r"\d+", answer))
            )
        if int(idx) in _MATH_IDS:
            return _math_equal(response, answer)
        if "or" in answer:
            predicted = re.sub(r"\\text\{or\}", "or", predicted)
            target = extract_answer(answer, "clean")
            return sorted(
                x.strip() for x in predicted.strip("[[]]").split("or")
            ) == sorted(x.strip() for x in target.strip("[[]]").split("or"))
    return extract_answer(response, "clean") == answer
