from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch

from gcw.distill.cache import truncate_last_cache_token
from gcw.distill.prompt import ChatPrompt
from gcw.distill.schedule import recurrent_event_due
from gcw.models.bgt import BGT
from gcw.models.cache_state import CacheState, StaticCacheState


@dataclass(frozen=True)
class ReplayResult:
    logits: torch.Tensor
    event_cache_lengths: tuple[int, ...]


@dataclass(frozen=True)
class RolloutResult:
    response_ids: tuple[int, ...]
    event_cache_lengths: tuple[int, ...]


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    valid_vocab_size: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token per batch after temperature, top-k and nucleus filtering."""
    if logits.ndim != 2:
        raise ValueError("sampling logits must have shape [batch, vocabulary]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if valid_vocab_size is not None:
        if not 0 < valid_vocab_size <= logits.shape[-1]:
            raise ValueError("valid_vocab_size must be within the model vocabulary")
        logits = logits[:, :valid_vocab_size]

    filtered = logits / temperature
    if top_k:
        k = min(top_k, filtered.shape[-1])
        threshold = torch.topk(filtered, k, dim=-1).values[:, -1:]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf")).scatter(-1, sorted_indices, sorted_logits)

    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)


class BGTDecodeSession:
    """Prompt-boundary and recurrent BGT answer rollout/replay.

    The session never processes during user/system prefill. It processes once before
    the assistant trigger and then after each complete response block. A periodic
    event refreshes the last consumed token at the same position, so the cache edit
    affects the immediately following distribution without duplicating a token.
    """

    def __init__(
        self,
        model: BGT,
        *,
        process_every: int,
        select_all: bool = False,
        valid_vocab_size: int | None = None,
        processor_enabled: bool = True,
    ) -> None:
        if not isinstance(model, BGT):
            raise TypeError("BGTDecodeSession requires a BGT model")
        if model.config.family != "qwen3":
            raise ValueError("BGTDecodeSession V0 supports only qwen3")
        if process_every <= 0:
            raise ValueError("process_every must be positive")
        self.model = model
        self.process_every = process_every
        self.select_all = select_all
        self.processor_enabled = processor_enabled
        if valid_vocab_size is not None and not 0 < valid_vocab_size <= model.config.vocab_size:
            raise ValueError("valid_vocab_size must be within the model vocabulary")
        self.valid_vocab_size = valid_vocab_size

    def replay(self, prompt: ChatPrompt, response_ids: tuple[int, ...]) -> ReplayResult:
        if not response_ids:
            raise ValueError("replay requires at least one response token")
        self._validate_total_length(prompt, len(response_ids))
        next_logits, cache, event_cache_lengths = self._start(prompt)
        aligned_chunks: list[torch.Tensor] = []
        device = self.model.get_device()
        response_offset = 0

        while response_offset < len(response_ids):
            block_size = min(self.process_every, len(response_ids) - response_offset)
            block_ids = response_ids[response_offset : response_offset + block_size]
            aligned_chunks.append(next_logits[:, None, :])
            event_follows = recurrent_event_due(
                consumed_response_tokens=response_offset + block_size,
                process_every=self.process_every,
                will_continue=response_offset + block_size < len(response_ids),
            )
            feed_count = block_size if event_follows else block_size - 1
            if feed_count:
                tokens = torch.tensor([block_ids[:feed_count]], dtype=torch.long, device=device)
                token_position = len(prompt.full_ids) + response_offset
                block_logits, raw_cache = self.model.backbone.forward_chunk(tokens, token_position, cache)
                cache = self._require_cache_state(raw_cache)
                aligned_chunks.append(block_logits[:, : block_size - 1, :])
                next_logits = block_logits[:, -1, :]

            if event_follows and self.processor_enabled:
                event_cache_lengths.append(cache.seq_len)
                cache = self._process(cache)
                last_token = torch.tensor([[block_ids[-1]]], dtype=torch.long, device=device)
                last_position = len(prompt.full_ids) + response_offset + block_size - 1
                refreshed_logits, raw_cache = self.model.backbone.forward_chunk(
                    last_token,
                    last_position,
                    truncate_last_cache_token(cache),
                )
                cache = self._require_cache_state(raw_cache)
                next_logits = refreshed_logits[:, -1, :]
            response_offset += block_size

        logits = torch.cat(aligned_chunks, dim=1)
        if logits.shape[1] != len(response_ids):
            raise RuntimeError("student replay logits did not align with every response token")
        return ReplayResult(
            logits=logits,
            event_cache_lengths=tuple(event_cache_lengths),
        )

    @torch.no_grad()
    def rollout(
        self,
        prompt: ChatPrompt,
        *,
        max_response_tokens: int,
        eos_token_id: int | None,
        temperature: float,
        top_p: float,
        top_k: int,
        generator: torch.Generator | None = None,
        stop_sequences: Sequence[Sequence[int]] = (),
        greedy: bool = False,
    ) -> RolloutResult:
        if max_response_tokens <= 0:
            raise ValueError("max_response_tokens must be positive")
        normalized_stop_sequences = tuple(tuple(sequence) for sequence in stop_sequences)
        if any(not sequence for sequence in normalized_stop_sequences):
            raise ValueError("stop sequences must be non-empty")
        self._validate_total_length(prompt, max_response_tokens)
        next_logits, cache, event_cache_lengths = self._start_static(
            prompt,
            max_seq_len=len(prompt.full_ids) + max_response_tokens,
        )
        response_ids: list[int] = []

        for response_index in range(max_response_tokens):
            if greedy:
                greedy_logits = next_logits
                if self.valid_vocab_size is not None:
                    greedy_logits = greedy_logits[:, : self.valid_vocab_size]
                sampled = torch.argmax(greedy_logits, dim=-1)
            else:
                sampled = sample_next_token(
                    next_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    valid_vocab_size=self.valid_vocab_size,
                    generator=generator,
                )
            if sampled.numel() != 1:
                raise ValueError("distillation rollout supports batch size one")
            token_id = int(sampled.item())
            response_ids.append(token_id)
            consumed = response_index + 1
            finished = (eos_token_id is not None and token_id == eos_token_id) or any(
                len(response_ids) >= len(sequence)
                and tuple(response_ids[-len(sequence) :]) == sequence
                for sequence in normalized_stop_sequences
            )
            will_continue = not finished and consumed < max_response_tokens
            if not will_continue:
                break
            next_logits, cache = self._consume_static(
                prompt,
                sampled[:, None],
                response_index,
                consumed,
                cache,
                event_cache_lengths,
                will_continue=True,
            )

        return RolloutResult(
            response_ids=tuple(response_ids),
            event_cache_lengths=tuple(event_cache_lengths),
        )

    def _start_static(
        self,
        prompt: ChatPrompt,
        *,
        max_seq_len: int,
    ) -> tuple[torch.Tensor, StaticCacheState, list[int]]:
        device = self.model.get_device()
        num_kv_heads, head_dim = self.model.config.stored_cache_shape
        cache = StaticCacheState.allocate(
            batch_size=1,
            num_layers=self.model.config.cache_num_layers,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            device=device,
            dtype=self.model.backbone.embed_tokens.weight.dtype,
        )
        base_ids = torch.tensor([prompt.base_ids], dtype=torch.long, device=device)
        trigger_ids = torch.tensor([prompt.trigger_ids], dtype=torch.long, device=device)
        _, cache = self.model.backbone.forward_chunk_static(base_ids, 0, cache, logits=False)
        event_cache_lengths: list[int] = []
        if self.processor_enabled:
            event_cache_lengths.append(cache.seq_len)
            cache = self._process_static(cache)
        trigger_logits, cache = self.model.backbone.forward_chunk_static(
            trigger_ids,
            len(prompt.base_ids),
            cache,
        )
        return trigger_logits[:, -1, :], cache, event_cache_lengths

    def _consume_static(
        self,
        prompt: ChatPrompt,
        token: torch.Tensor,
        response_index: int,
        consumed: int,
        cache: StaticCacheState,
        event_cache_lengths: list[int],
        *,
        will_continue: bool,
    ) -> tuple[torch.Tensor, StaticCacheState]:
        token_position = len(prompt.full_ids) + response_index
        token_logits, cache = self.model.backbone.forward_chunk_static(token, token_position, cache)
        next_logits = token_logits[:, -1, :]
        if self.processor_enabled and recurrent_event_due(
            consumed_response_tokens=consumed,
            process_every=self.process_every,
            will_continue=will_continue,
        ):
            event_cache_lengths.append(cache.seq_len)
            cache = self._process_static(cache)
            refreshed_logits, cache = self.model.backbone.forward_chunk_static(token, token_position, cache)
            next_logits = refreshed_logits[:, -1, :]
        return next_logits, cache

    def _start(self, prompt: ChatPrompt) -> tuple[torch.Tensor, CacheState, list[int]]:
        device = self.model.get_device()
        base_ids = torch.tensor([prompt.base_ids], dtype=torch.long, device=device)
        trigger_ids = torch.tensor([prompt.trigger_ids], dtype=torch.long, device=device)

        _, raw_cache = self.model.backbone.forward_chunk(base_ids, 0, logits=False)
        cache = self._require_cache_state(raw_cache)
        event_cache_lengths: list[int] = []
        if self.processor_enabled:
            event_cache_lengths.append(cache.seq_len)
            cache = self._process(cache)

        trigger_logits, raw_cache = self.model.backbone.forward_chunk(
            trigger_ids,
            len(prompt.base_ids),
            cache,
        )
        cache = self._require_cache_state(raw_cache)
        return trigger_logits[:, -1, :], cache, event_cache_lengths

    def _consume(
        self,
        prompt: ChatPrompt,
        token_id: int,
        response_index: int,
        consumed: int,
        cache: CacheState,
        event_cache_lengths: list[int],
        *,
        will_continue: bool,
    ) -> tuple[torch.Tensor, CacheState]:
        device = self.model.get_device()
        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        token_position = len(prompt.full_ids) + response_index
        token_logits, raw_cache = self.model.backbone.forward_chunk(token, token_position, cache)
        cache = self._require_cache_state(raw_cache)
        next_logits = token_logits[:, -1, :]

        if self.processor_enabled and recurrent_event_due(
            consumed_response_tokens=consumed,
            process_every=self.process_every,
            will_continue=will_continue,
        ):
            event_cache_lengths.append(cache.seq_len)
            cache = self._process(cache)
            truncated = truncate_last_cache_token(cache)
            refreshed_logits, raw_cache = self.model.backbone.forward_chunk(
                token,
                token_position,
                truncated,
            )
            cache = self._require_cache_state(raw_cache)
            next_logits = refreshed_logits[:, -1, :]
        return next_logits, cache

    def _process(self, cache: CacheState) -> CacheState:
        processed = self.model.cache_processor(cache, select_all=self.select_all)
        return self._require_cache_state(processed)

    def _process_static(self, cache: StaticCacheState) -> StaticCacheState:
        processed = self.model.cache_processor(cache, select_all=self.select_all)
        if not isinstance(processed, StaticCacheState):
            raise TypeError("BGT distillation requires StaticCacheState-preserving processing")
        return processed

    def _validate_total_length(self, prompt: ChatPrompt, response_length: int) -> None:
        total_length = len(prompt.full_ids) + response_length
        if total_length > self.model.config.sequence_len:
            raise ValueError(
                f"prompt and response require {total_length} tokens but model sequence_len is "
                f"{self.model.config.sequence_len}"
            )

    @staticmethod
    def _require_cache_state(cache: object) -> CacheState:
        if not isinstance(cache, CacheState):
            raise TypeError("distillation decode requires a dynamic CacheState")
        return cast(CacheState, cache)
