from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

UNTIL = (
    "</s>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "\n\nQuestion:",
    "\nQuestion:",
)


@dataclass(frozen=True)
class EvalExample:
    task: str
    index: int
    prompt: str
    target: str
    max_gen_toks: int


def load_examples(
    tasks: Iterable[str], limit: int | None = None
) -> list[EvalExample]:
    from datasets import load_dataset

    examples: list[EvalExample] = []
    for task in tasks:
        if task == "gsm8k_openmath":
            data = load_dataset("openai/gsm8k", "main", split="test")
            for i, row in enumerate(data):
                if limit is not None and i >= limit:
                    break
                examples.append(
                    EvalExample(
                        task, i, str(row["question"]), str(row["answer"]), 512
                    )
                )
        elif task == "svamp":
            data = load_dataset("ChilleD/SVAMP", split="test")
            for i, row in enumerate(data):
                if limit is not None and i >= limit:
                    break
                examples.append(
                    EvalExample(
                        task,
                        i,
                        str(row["question_concat"]),
                        str(row["Answer"]),
                        256,
                    )
                )
        else:
            raise ValueError(f"unsupported task for generation parity: {task}")
    return examples


def request_kwargs(example: EvalExample) -> dict:
    return {
        "until": list(UNTIL),
        "do_sample": False,
        "temperature": 0.0,
        "max_gen_toks": example.max_gen_toks,
    }


def trim_until(text: str) -> str:
    for stop in UNTIL:
        if stop and stop in text:
            text = text.split(stop)[0]
    return text


def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return _normalize(matches[-1]) if matches else None


def extract_flexible(text: str) -> str | None:
    matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    values = [a or b for a, b in matches]
    return _normalize(values[-1]) if values else None


def target_answer(text: str) -> str | None:
    if "####" in text:
        text = text.rsplit("####", 1)[-1]
    return extract_flexible(text)


def _normalize(value: str) -> str:
    value = value.lower()
    for old in (",", "$", ".", " "):
        value = value.replace(old, "")
    return value.strip()
