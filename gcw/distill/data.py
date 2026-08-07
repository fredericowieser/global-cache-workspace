from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from gcw.data.sft import _iter_openmath_rows


@dataclass(frozen=True)
class PromptRecord:
    source_index: int
    prompt: str


def iter_openmath_prompts(
    source: str | Path,
    *,
    skip_rows: int = 0,
    take_rows: int | None = None,
    prompt_suffix: str = "",
) -> Iterator[PromptRecord]:
    """Yield finite prompt-only OpenMath records without reading answer fields."""
    if skip_rows < 0:
        raise ValueError("skip_rows must be non-negative")
    if take_rows is not None and take_rows < 0:
        raise ValueError("take_rows must be non-negative or null")
    if take_rows == 0:
        return

    retained = 0
    for source_index, row in enumerate(_iter_openmath_rows(source)):
        if source_index < skip_rows:
            continue
        prompt = _prompt_only(row)
        if not prompt:
            continue
        if prompt_suffix.strip():
            prompt = f"{prompt}\n\n{prompt_suffix.strip()}"
        yield PromptRecord(source_index=source_index, prompt=prompt)
        retained += 1
        if take_rows is not None and retained >= take_rows:
            return


def _prompt_only(row: dict) -> str:
    for field in ("prompt", "question", "problem", "instruction", "input"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()

    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    prompt_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") == "assistant":
            continue
        content = str(message.get("content", "")).strip()
        if content:
            prompt_parts.append(content)
    return "\n".join(prompt_parts)
