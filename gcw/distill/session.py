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
    event_relative_l2_displacements: tuple[float, ...] = ()


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
        record_displacements: bool = False,
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
        self.record_displacements = record_displacements
        if valid_vocab_size is not None and not 0 < valid_vocab_size <= model.config.vocab_size:
            raise ValueError("valid_vocab_size must be within the model vocabulary")
        self.valid_vocab_size = valid_vocab_size

    def replay(self, prompt: ChatPrompt, response_ids: tuple[int, ...]) -> ReplayResult:
        if not response_ids:
            raise ValueError("replay requires at least one response token")
        self._validate_total_length(prompt, len(response_ids))
        next_logits, cache, event_cache_lengths, event_displacements = self._start(prompt)
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
                cache = self._process_recording(cache, event_displacements)
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
            event_relative_l2_displacements=tuple(event_displacements),
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

    @torch.no_grad()
    def rollout_many(
        self,
        prompts: Sequence[ChatPrompt],
        *,
        max_response_tokens: int,
        eos_token_id: int | None,
        temperature: float,
        top_p: float,
        top_k: int,
        generators: Sequence[torch.Generator | None],
        stop_sequences: Sequence[Sequence[int]] = (),
        greedy: bool = False,
    ) -> tuple[RolloutResult, ...]:
        prompt_list = tuple(prompts)
        generator_list = tuple(generators)
        if not prompt_list:
            raise ValueError("batched rollout requires at least one prompt")
        if len(generator_list) != len(prompt_list):
            raise ValueError("batched rollout requires one generator per prompt")
        if max_response_tokens <= 0:
            raise ValueError("max_response_tokens must be positive")
        base_length = len(prompt_list[0].base_ids)
        trigger_length = len(prompt_list[0].trigger_ids)
        if any(
            len(prompt.base_ids) != base_length
            or len(prompt.trigger_ids) != trigger_length
            for prompt in prompt_list
        ):
            raise ValueError("batched rollout prompts must have equal base and trigger token lengths")
        normalized_stop_sequences = tuple(tuple(sequence) for sequence in stop_sequences)
        if any(not sequence for sequence in normalized_stop_sequences):
            raise ValueError("stop sequences must be non-empty")
        for prompt in prompt_list:
            self._validate_total_length(prompt, max_response_tokens)

        device = self.model.get_device()
        num_kv_heads, head_dim = self.model.config.stored_cache_shape
        full_length = base_length + trigger_length
        cache = StaticCacheState.allocate(
            batch_size=len(prompt_list),
            num_layers=self.model.config.cache_num_layers,
            num_kv_heads=num_kv_heads,
            max_seq_len=full_length + max_response_tokens,
            head_dim=head_dim,
            device=device,
            dtype=self.model.backbone.embed_tokens.weight.dtype,
        )
        base_ids = torch.tensor(
            [prompt.base_ids for prompt in prompt_list], dtype=torch.long, device=device
        )
        trigger_ids = torch.tensor(
            [prompt.trigger_ids for prompt in prompt_list], dtype=torch.long, device=device
        )
        _, cache = self.model.backbone.forward_chunk_static(
            base_ids, 0, cache, logits=False
        )
        event_cache_lengths: list[list[int]] = [[] for _ in prompt_list]
        if self.processor_enabled:
            for events in event_cache_lengths:
                events.append(cache.seq_len)
            cache = self._process_static(cache)
        trigger_logits, cache = self.model.backbone.forward_chunk_static(
            trigger_ids, base_length, cache
        )
        next_logits = trigger_logits[:, -1, :]
        response_ids: list[list[int]] = [[] for _ in prompt_list]
        finished = [False] * len(prompt_list)
        fill_token_id = eos_token_id if eos_token_id is not None else 0

        for response_index in range(max_response_tokens):
            sampled_rows: list[torch.Tensor] = []
            for row, generator in enumerate(generator_list):
                if finished[row]:
                    sampled_rows.append(
                        torch.tensor(fill_token_id, dtype=torch.long, device=device)
                    )
                elif greedy:
                    logits = next_logits[row]
                    if self.valid_vocab_size is not None:
                        logits = logits[: self.valid_vocab_size]
                    sampled_rows.append(torch.argmax(logits, dim=-1))
                else:
                    sampled_rows.append(
                        sample_next_token(
                            next_logits[row : row + 1],
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            valid_vocab_size=self.valid_vocab_size,
                            generator=generator,
                        )[0]
                    )
            sampled = torch.stack(sampled_rows)
            consumed = response_index + 1
            for row in range(len(prompt_list)):
                if finished[row]:
                    continue
                token_id = int(sampled[row].item())
                response_ids[row].append(token_id)
                finished[row] = (
                    eos_token_id is not None and token_id == eos_token_id
                ) or any(
                    len(response_ids[row]) >= len(sequence)
                    and tuple(response_ids[row][-len(sequence) :]) == sequence
                    for sequence in normalized_stop_sequences
                )
            will_continue = [
                not finished[row] and consumed < max_response_tokens
                for row in range(len(prompt_list))
            ]
            if not any(will_continue):
                break
            token_position = full_length + response_index
            token_logits, cache = self.model.backbone.forward_chunk_static(
                sampled[:, None], token_position, cache
            )
            next_logits = token_logits[:, -1, :]
            if self.processor_enabled and recurrent_event_due(
                consumed_response_tokens=consumed,
                process_every=self.process_every,
                will_continue=True,
            ):
                for row, active in enumerate(will_continue):
                    if active:
                        event_cache_lengths[row].append(cache.seq_len)
                cache = self._process_static(cache)
                refreshed_logits, cache = self.model.backbone.forward_chunk_static(
                    sampled[:, None], token_position, cache
                )
                next_logits = refreshed_logits[:, -1, :]

        return tuple(
            RolloutResult(tuple(tokens), tuple(events))
            for tokens, events in zip(response_ids, event_cache_lengths, strict=True)
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

    def _start(
        self, prompt: ChatPrompt
    ) -> tuple[torch.Tensor, CacheState, list[int], list[float]]:
        device = self.model.get_device()
        base_ids = torch.tensor([prompt.base_ids], dtype=torch.long, device=device)
        trigger_ids = torch.tensor([prompt.trigger_ids], dtype=torch.long, device=device)

        _, raw_cache = self.model.backbone.forward_chunk(base_ids, 0, logits=False)
        cache = self._require_cache_state(raw_cache)
        event_cache_lengths: list[int] = []
        event_displacements: list[float] = []
        if self.processor_enabled:
            event_cache_lengths.append(cache.seq_len)
            cache = self._process_recording(cache, event_displacements)

        trigger_logits, raw_cache = self.model.backbone.forward_chunk(
            trigger_ids,
            len(prompt.base_ids),
            cache,
        )
        cache = self._require_cache_state(raw_cache)
        return trigger_logits[:, -1, :], cache, event_cache_lengths, event_displacements

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

    def _process_recording(self, cache: CacheState, displacements: list[float]) -> CacheState:
        if not self.record_displacements:
            return self._process(cache)
        before = tuple(cache.keys) + tuple(cache.values)
        processed = self._process(cache)
        after = tuple(processed.keys) + tuple(processed.values)
        if len(before) != len(after):
            raise RuntimeError("cache processor changed the number of cache tensors")
        with torch.no_grad():
            squared_change = torch.zeros((), dtype=torch.float64, device=cache.device)
            squared_input = torch.zeros((), dtype=torch.float64, device=cache.device)
            for source, target in zip(before, after, strict=True):
                source_float = source.detach().float()
                target_float = target.detach().float()
                squared_change += torch.sum((target_float - source_float).square(), dtype=torch.float64)
                squared_input += torch.sum(source_float.square(), dtype=torch.float64)
            denominator = torch.sqrt(squared_input).clamp_min(torch.finfo(torch.float64).eps)
            relative = torch.sqrt(squared_change) / denominator
            displacements.append(float(relative.cpu().item()))
        return processed

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
