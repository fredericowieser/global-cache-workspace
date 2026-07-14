from __future__ import annotations

from gcw.data.tokenizers import COSMO2_VOCAB_SIZE, IntTokenizer, get_tokenizer


def truncate_context_continuation(ctx: list[int], cont: list[int], max_length: int) -> tuple[list[int], list[int]]:
    full = ctx + cont
    if len(full) <= max_length + 1:
        return ctx, cont
    overflow = len(full) - (max_length + 1)
    if overflow < len(ctx):
        return ctx[overflow:], cont
    trimmed = full[overflow:]
    return trimmed[:1], trimmed[1:]


def encode_generate_prompt(tokenizer, prompt: str, max_length: int, *, add_special_tokens: bool = False) -> list[int]:
    try:
        ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    except TypeError:
        ids = tokenizer.encode(prompt)
    return ids[-max_length:]


def tokenizer_for_checkpoint(config, state):
    source = state.get("source") or state.get("tokenizer_source")
    if source and config.family in {"llama", "qwen3"}:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Converted HF checkpoints require transformers for eval tokenization.") from exc
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
        return HFTokenizerAdapter(tokenizer)
    if config.family in {"llama", "qwen3"}:
        raise ValueError(
            "HF-family checkpoints require train_state.json to include 'source' "
            "or 'tokenizer_source' so eval uses the original tokenizer."
        )
    if config.family == "smollm" and config.vocab_size == COSMO2_VOCAB_SIZE:
        return get_tokenizer("cosmo2")
    return IntTokenizer(vocab_size=config.vocab_size, eot_id=0)


class HFTokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eot_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
        self.chat_template = getattr(tokenizer, "chat_template", None)
        self.default_chat_template = getattr(tokenizer, "default_chat_template", None)

    def encode(self, text: str, add_special_tokens: bool | None = None) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=bool(add_special_tokens))

    def decode(self, ids) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)


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
    eos_token_id: int | None,
) -> list[int]:
    end = len(tokens)
    if eos_token_id is not None:
        for index, token in enumerate(tokens):
            if token == eos_token_id:
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
