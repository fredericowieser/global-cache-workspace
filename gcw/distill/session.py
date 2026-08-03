from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from gcw.distill.cache import truncate_last_cache_token
from gcw.distill.prompt import ChatPrompt
from gcw.models.bgt import BGT
from gcw.models.cache_state import CacheState


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
    if not bool(torch.isfinite(probabilities).all()) or not bool((probabilities.sum(dim=-1) > 0).all()):
        raise RuntimeError("sampling produced an invalid probability distribution")
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

    def replay(self, prompt: ChatPrompt, response_ids: tuple[int, ...]) -> ReplayResult:
        if not response_ids:
            raise ValueError("replay requires at least one response token")
        self._validate_total_length(prompt, len(response_ids))
        next_logits, cache, event_cache_lengths = self._start(prompt)
        aligned_logits: list[torch.Tensor] = []

        for response_index, token_id in enumerate(response_ids):
            aligned_logits.append(next_logits)
            consumed = response_index + 1
            next_logits, cache = self._consume(
                prompt,
                token_id,
                response_index,
                consumed,
                cache,
                event_cache_lengths,
                will_continue=consumed < len(response_ids),
            )

        return ReplayResult(
            logits=torch.stack(aligned_logits, dim=1),
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
    ) -> RolloutResult:
        if max_response_tokens <= 0:
            raise ValueError("max_response_tokens must be positive")
        self._validate_total_length(prompt, max_response_tokens)
        next_logits, cache, event_cache_lengths = self._start(prompt)
        response_ids: list[int] = []

        for response_index in range(max_response_tokens):
            sampled = sample_next_token(
                next_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generator=generator,
            )
            if sampled.numel() != 1:
                raise ValueError("distillation rollout supports batch size one")
            token_id = int(sampled.item())
            response_ids.append(token_id)
            consumed = response_index + 1
            finished = eos_token_id is not None and token_id == eos_token_id
            will_continue = not finished and consumed < max_response_tokens
            if not will_continue:
                break
            next_logits, cache = self._consume(
                prompt,
                token_id,
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

    def _start(self, prompt: ChatPrompt) -> tuple[torch.Tensor, CacheState, list[int]]:
        device = self.model.get_device()
        base_ids = torch.tensor([prompt.base_ids], dtype=torch.long, device=device)
        trigger_ids = torch.tensor([prompt.trigger_ids], dtype=torch.long, device=device)

        _, raw_cache = self.model.backbone.forward_chunk(base_ids, 0, logits=False)
        cache = self._require_cache_state(raw_cache)
        event_cache_lengths = [cache.seq_len]
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

        if will_continue and consumed % self.process_every == 0:
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
