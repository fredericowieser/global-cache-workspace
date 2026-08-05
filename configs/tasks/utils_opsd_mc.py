from __future__ import annotations

import re


_EXPLICIT_FINAL_CHOICE = re.compile(
    r"(?:^|[\n.!?;,:：。！？；，…—]\s*|(?:^|\n)\s*-\s*)"
    r"(?:the\s+)?(?:final\s+answer|answer|最终答案|答案)"
    r"(?:(?:\s*[:：]\s*)|(?:\s+is\s+))"
    r"(?:(?P<plain>[A-Z])|\(\s*(?P<paren>[A-Z])\s*\))"
    r"\s*[.!。！]?\s*$",
    re.I,
)
_EXPLICIT_MARKER = re.compile(
    r"(?:\b(?:the\s+)?(?:final\s+answer|answer)\b|最终答案|答案)"
    r"(?=(?:\s*[:：])|(?:\s+is\s+))",
    re.I,
)
_BOXED_FINAL_CHOICE = re.compile(
    r"\\boxed\s*\{\s*([A-Z])\s*\}\s*[.!。！]?\s*$",
    re.I,
)


def _extract_final_choice(text: str, *, choices: int) -> int | None:
    allowed = {chr(ord("A") + index): index for index in range(choices)}
    normalized = text.strip().upper().replace("**", "").replace("__", "")
    explicit_markers = list(_EXPLICIT_MARKER.finditer(normalized))
    explicit_match = _EXPLICIT_FINAL_CHOICE.search(normalized)
    boxed_matches = list(_BOXED_FINAL_CHOICE.finditer(normalized))
    candidate_count = len(explicit_markers) + len(boxed_matches)
    if candidate_count != 1:
        return None
    if explicit_match is not None and len(explicit_markers) == 1:
        choice = explicit_match.group("plain") or explicit_match.group("paren")
        return allowed.get(choice)
    if len(boxed_matches) == 1 and not explicit_markers:
        return allowed.get(boxed_matches[0].group(1))
    return None


def logiqa_doc_to_text(doc: dict) -> str:
    lines = [
        f"Passage: {doc['context']}",
        f"Question: {doc['question']}",
        "Choices:",
    ]
    lines.extend(
        f"{chr(ord('A') + index)}. {option}"
        for index, option in enumerate(doc["options"])
    )
    lines.append("Give the final answer as 'Answer: X'.")
    return "\n".join(lines)


def logiqa_doc_to_target(doc: dict) -> str:
    return doc["label"].strip().upper()


def process_logiqa(doc: dict, results: list[str]) -> dict[str, float]:
    prediction = _extract_final_choice(results[0], choices=len(doc["options"]))
    gold = ord(logiqa_doc_to_target(doc)) - ord("A")
    return {"acc": float(prediction == gold)}


def gaokao_doc_to_text(doc: dict) -> str:
    return f"{doc['query'].rstrip()}\nGive the final answer as 'Answer: X'."


def gaokao_doc_to_target(doc: dict) -> str:
    return ",".join(chr(ord("A") + int(index)) for index in doc["gold"])


def process_gaokao(doc: dict, results: list[str]) -> dict[str, float]:
    prediction = _extract_final_choice(results[0], choices=len(doc["choices"]))
    gold = {int(index) for index in doc["gold"]}
    return {"acc": float(prediction in gold)}
