from __future__ import annotations

import functools

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
        return self.enc.decode(list(ids), skip_special_tokens=False)

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
    if name == "int":
        return IntTokenizer()
    if name != "cosmo2":
        raise ValueError(f"unknown tokenizer: {name}")
    return Tokenizer()


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
