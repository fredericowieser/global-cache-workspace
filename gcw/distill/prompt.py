from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatPrompt:
    base_ids: tuple[int, ...]
    trigger_ids: tuple[int, ...]

    @property
    def full_ids(self) -> tuple[int, ...]:
        return self.base_ids + self.trigger_ids


def format_chat_prompt(tokenizer: Any, prompt: str, *, enable_thinking: bool) -> ChatPrompt:
    """Split a chat prompt before its assistant-generation trigger.

    The split lets BGT process the user/system cache before any assistant token is
    consumed while retaining the tokenizer's exact generation prefix.
    """
    messages = [{"role": "user", "content": prompt}]
    base = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    base_ids = tuple(int(token_id) for token_id in base)
    full_ids = tuple(int(token_id) for token_id in full)
    if not full_ids[: len(base_ids)] == base_ids:
        raise ValueError("chat generation prompt is not an extension of the base prompt")
    trigger_ids = full_ids[len(base_ids) :]
    if not base_ids:
        raise ValueError("chat template produced an empty base prompt")
    if not trigger_ids:
        raise ValueError("chat template produced an empty assistant trigger")
    return ChatPrompt(base_ids=base_ids, trigger_ids=trigger_ids)
