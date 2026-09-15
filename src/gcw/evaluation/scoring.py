from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import re
from typing import Any, Mapping

import math

import torch
import torch.distributed as dist

from gcw.config import LossBackend
from gcw.data.tokenizers import (
    COSMO2_VOCAB_SIZE,
    IntTokenizer,
    Tokenizer,
    TransformersTokenizer,
    get_tokenizer,
)


def truncate_context_continuation(
    ctx: list[int], cont: list[int], max_length: int
) -> tuple[list[int], list[int]]:
    full = ctx + cont
    if len(full) <= max_length + 1:
        return ctx, cont
    overflow = len(full) - (max_length + 1)
    if overflow < len(ctx):
        return ctx[overflow:], cont
    trimmed = full[overflow:]
    return trimmed[:1], trimmed[1:]


def encode_generate_prompt(
    tokenizer, prompt: str, max_length: int, *, add_special_tokens: bool = False
) -> list[int]:
    try:
        ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    except TypeError:
        ids = tokenizer.encode(prompt)
    return ids[-max_length:]


def generation_config_for_source(
    source: str | Path, *, local_files_only: bool = False
) -> dict[str, Any]:
    from transformers.utils.hub import cached_file

    for filename in ("generation_config.json", "config.json"):
        path = cached_file(
            str(source),
            filename,
            _raise_exceptions_for_missing_entries=False,
            local_files_only=local_files_only,
        )
        if path is not None:
            with open(path) as handle:
                config = json.load(handle)
            if filename == "config.json":
                text_config = config.get("text_config") or {}
                for key in ("eos_token_id", "pad_token_id"):
                    if (
                        config.get(key) is None
                        and text_config.get(key) is not None
                    ):
                        config[key] = text_config[key]
            config["_from_model_config"] = filename == "config.json"
            return config
    return {}


def generation_token_ids(
    tokenizer: Any, generation_config: Mapping[str, Any] | None = None
) -> tuple[tuple[int, ...], int]:
    config = generation_config or {}
    eos = config.get("eos_token_id")
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None and isinstance(
        tokenizer, (IntTokenizer, Tokenizer, TransformersTokenizer)
    ):
        eos = tokenizer.eot_id
    if eos is None:
        raise ValueError(
            "generation requires EOS metadata from the model or tokenizer"
        )
    values = (eos,) if isinstance(eos, int) else eos
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if (
        config.get("_from_model_config")
        and tokenizer_eos is not None
        and isinstance(values, (list, tuple))
    ):
        values = (*values, tokenizer_eos)
    if not isinstance(values, (list, tuple)) or any(
        type(value) is not int or value < 0 for value in values
    ):
        raise ValueError(
            "eos_token_id must be a nonnegative integer or list of integers"
        )
    eos_ids = tuple(dict.fromkeys(values))
    pad = config.get("pad_token_id")
    if pad is None:
        pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None and eos_ids:
        pad = eos_ids[0]
    if type(pad) is not int or pad < 0:
        raise ValueError(
            "generation requires a nonnegative pad_token_id or EOS"
        )
    return eos_ids, pad


def tokenizer_for_checkpoint(
    config, state, *, source=None, local_files_only: bool = False
):
    source = source or state.get("tokenizer_source") or state.get("source")
    if source and config.family in {
        "llama",
        "qwen3",
        "qwen3_5",
        "qwen3_5_moe",
        "gemma4",
    }:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Converted HF checkpoints require transformers for eval tokenization."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        generation_config = generation_config_for_source(
            source, local_files_only=local_files_only
        )
        model_source = state.get("source")
        if (
            state.get("eos_token_id") is None
            and generation_config.get("eos_token_id") is None
            and model_source
            and str(model_source) != str(source)
        ):
            try:
                generation_config = generation_config_for_source(
                    model_source, local_files_only=local_files_only
                )
            except OSError:
                generation_config = {}
        generation_config.update(
            (key, state[key])
            for key in ("eos_token_id", "pad_token_id")
            if state.get(key) is not None
        )
        if state.get("eos_token_id") is not None:
            generation_config["_from_model_config"] = False
        return HFTokenizerAdapter(tokenizer, str(source), generation_config)
    if config.family in {"llama", "qwen3", "qwen3_5", "qwen3_5_moe", "gemma4"}:
        raise ValueError(
            "HF-family checkpoints require train_state.json to include 'source' "
            "or 'tokenizer_source' so eval uses the original tokenizer."
        )
    if config.family == "smollm" and config.vocab_size == COSMO2_VOCAB_SIZE:
        return get_tokenizer("cosmo2")
    return IntTokenizer(vocab_size=config.vocab_size, eot_id=0)


class HFTokenizerAdapter:
    def __init__(
        self,
        tokenizer,
        name_or_path: str | None = None,
        generation_config: Mapping[str, Any] | None = None,
    ):
        self.tokenizer = tokenizer
        self.name_or_path = name_or_path or str(
            getattr(tokenizer, "name_or_path", "")
        )
        self.source = self.name_or_path
        self.generation_config = generation_config
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.eot_id = self.eos_token_id
        if self.eot_id is None:
            self.eot_id = self.pad_token_id or 0
        self.vocab_size = len(tokenizer)
        self.chat_template = getattr(tokenizer, "chat_template", None)
        self.default_chat_template = getattr(
            tokenizer, "default_chat_template", None
        )

    def encode(
        self, text: str, add_special_tokens: bool | None = None
    ) -> list[int]:
        return self.tokenizer.encode(
            text, add_special_tokens=bool(add_special_tokens)
        )

    def decode(self, ids) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def identity_sha256(self) -> str:
        from gcw.data.tokenizers import _tokenizer_identity_sha256

        return _tokenizer_identity_sha256(
            self.tokenizer, self.source, self.vocab_size
        )


def stop_token_sequences(tokenizer, until: tuple[str, ...]) -> list[list[int]]:
    sequences: list[list[int]] = []
    for stop in until:
        if not stop:
            continue
        try:
            ids = tokenizer.encode(stop, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(stop)
        if ids:
            sequences.append(list(ids))
    return sequences


def trim_generated_token_ids(
    tokens: list[int],
    stop_sequences: list[list[int]],
    eos_token_id: int | list[int] | tuple[int, ...] | None,
) -> list[int]:
    end = len(tokens)
    if eos_token_id is not None:
        eos_ids = (
            (eos_token_id,) if isinstance(eos_token_id, int) else eos_token_id
        )
        for index, token in enumerate(tokens):
            if token in eos_ids:
                end = min(end, index)
                break
    for stop in stop_sequences:
        start = find_subsequence(tokens, stop)
        if start is not None:
            end = min(end, start)
    return tokens[:end]


def find_subsequence(tokens: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(tokens):
        return None
    width = len(pattern)
    for start in range(0, len(tokens) - width + 1):
        if tokens[start : start + width] == pattern:
            return start
    return None


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


@torch.no_grad()
def evaluate_nll_bpb(
    model,
    batches,
    steps: int,
    token_bytes: torch.Tensor,
    *,
    loss_backend: LossBackend = "reference",
) -> tuple[float, float]:
    device = model.get_device()
    total_nats = torch.tensor(0.0, device=device)
    total_bytes = torch.tensor(0, dtype=torch.long, device=device)
    total_tokens = torch.tensor(0, dtype=torch.long, device=device)
    it = iter(batches)
    model.eval()
    for _ in range(steps):
        x, y = next(it)[:2]
        loss = model(
            x, y, loss_reduction="none", loss_backend=loss_backend
        ).view(-1)
        y = y.view(-1)
        valid = y >= 0
        safe = torch.where(valid, y, torch.zeros_like(y))
        counts = torch.where(
            valid,
            token_bytes[safe],
            torch.zeros_like(y, dtype=token_bytes.dtype),
        )
        keep = counts > 0
        total_nats += (loss * keep).sum()
        total_bytes += counts.sum()
        total_tokens += keep.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_nats)
        dist.all_reduce(total_bytes)
        dist.all_reduce(total_tokens)
    nats = float(total_nats.item())
    bytes_ = int(total_bytes.item())
    tokens = int(total_tokens.item())
    return nats / tokens if tokens else float("inf"), (
        nats / (math.log(2) * bytes_) if bytes_ else float("inf")
    )
