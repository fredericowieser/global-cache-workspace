from __future__ import annotations

import functools
from typing import cast

import torch
from tokenizers import Tokenizer as HFTokenizer

COSMO2_REPO = "HuggingFaceTB/cosmo2-tokenizer"
COSMO2_VOCAB_SIZE = 49152
COSMO2_EOT_ID = 0


@functools.lru_cache(maxsize=1)
def _cosmo2() -> HFTokenizer:
    return HFTokenizer.from_pretrained(COSMO2_REPO)


class Tokenizer:
    def __init__(self, tokenizer: HFTokenizer | None = None, vocab_size: int = COSMO2_VOCAB_SIZE, eot_id: int = COSMO2_EOT_ID):
        self.enc = tokenizer or _cosmo2()
        self.vocab_size = vocab_size
        self.eot_id = eot_id

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [item.ids for item in self.enc.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids) -> str:
        decoded = self.enc.decode(list(ids), skip_special_tokens=False)
        if not isinstance(decoded, str):
            raise TypeError("tokenizer decode returned multiple sequences")
        return decoded

    def get_vocab_size(self) -> int:
        return self.vocab_size


class IntTokenizer:
    def __init__(self, vocab_size: int = 256, eot_id: int = 0):
        self.vocab_size = vocab_size
        self.eot_id = eot_id

    def encode(self, text: str) -> list[int]:
        return [b % self.vocab_size for b in text.encode("utf-8")]

    def decode(self, ids) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="ignore")

    def get_vocab_size(self) -> int:
        return self.vocab_size


def get_tokenizer(name: str = "cosmo2"):
    if name.startswith("hf:"):
        return TransformersTokenizer(name.removeprefix("hf:"))
    if name == "int":
        return IntTokenizer()
    if name != "cosmo2":
        raise ValueError(f"unknown tokenizer: {name}")
    return Tokenizer()


class TransformersTokenizer:
    def __init__(self, source: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("HF tokenizers require the convert/eval extra with transformers installed") from exc
        self.enc = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
        self.eot_id = self.enc.eos_token_id or self.enc.pad_token_id or 0
        self.vocab_size = len(self.enc)

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, add_special_tokens=False)

    def decode(self, ids) -> str:
        decoded = self.enc.decode(list(ids), skip_special_tokens=False)
        if not isinstance(decoded, str):
            raise TypeError("tokenizer decode returned multiple sequences")
        return decoded

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        if not tokenize:
            raise ValueError("training chat templates must be tokenised")
        encoded = self.enc.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        if not isinstance(encoded, list) or any(not isinstance(token_id, int) for token_id in encoded):
            raise TypeError("tokenised chat template must return one list of token IDs")
        return cast(list[int], encoded)

    def get_vocab_size(self) -> int:
        return self.vocab_size


def get_token_bytes(device: torch.device | str = "cpu") -> torch.Tensor:
    tok = _cosmo2()
    specials = set(tok.get_added_tokens_decoder().keys())
    out = torch.zeros(COSMO2_VOCAB_SIZE, dtype=torch.long)
    for tid in range(COSMO2_VOCAB_SIZE):
        if tid in specials:
            continue
        s = tok.id_to_token(tid)
        out[tid] = len(s) if s is not None else 0
    return out.to(device)
