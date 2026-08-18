from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gcw.config.types import SFTLossScope


@dataclass
class DataConfig:
    kind: Literal["random", "fineweb", "sft"] = "fineweb"
    data_dir: str | None = None
    dataset: str | None = None
    validation_dataset: str | None = None
    tokenizer: str = "cosmo2"
    tokenizer_source: str | None = None
    text_field: str = "text"
    max_length: int = 2048
    sft_skip_rows: int = 0
    sft_take_rows: int | None = None
    response_prefix: str = ""
    prompt_suffix: str = ""
    sft_chat_template: bool = False
    sft_enable_thinking: bool = False
    sft_loss_scope: SFTLossScope = "answer_only"
    canonical_execution: bool = False
    expected_artifact_sha256: str | None = None
    expected_source_ids_sha256: str | None = None
    experiment_manifest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sft_chat_template, bool) or not isinstance(self.sft_enable_thinking, bool):
            raise ValueError("SFT chat-template controls must be booleans")
        if self.sft_chat_template and self.response_prefix:
            raise ValueError("data.response_prefix must be empty with sft_chat_template")
        if self.sft_loss_scope not in {"answer_only", "full_sequence"}:
            raise ValueError("data.sft_loss_scope must be 'answer_only' or 'full_sequence'")
        hashes = (self.expected_artifact_sha256, self.expected_source_ids_sha256)
        authenticated = all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            and len(set(value)) > 1
            for value in hashes
        )
        if self.canonical_execution and not authenticated:
            raise ValueError("canonical data requires authenticated artifact and source-ID sha256 hashes")
