from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Iterator, Protocol

import torch

from gcw.distill.session import sample_next_token
from gcw.models.cache_state import StaticCacheState
from gcw.models.transformer import TF


class TeacherBackbone(Protocol):
    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]: ...

    def forward_full(self, input_ids: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class NativeTeacherTrajectory:
    reasoning_ids: tuple[int, ...]
    visible_response_ids: tuple[int, ...]
    full_completion_ids: tuple[int, ...]


def parse_native_teacher_trajectory(
    generated_ids: tuple[int, ...],
    *,
    teacher_trigger_ids: tuple[int, ...],
    open_ids: tuple[int, ...],
    close_ids: tuple[int, ...],
    eos_token_id: int,
) -> NativeTeacherTrajectory:
    if not open_ids or not close_ids:
        raise ValueError("thinking delimiters must not be empty")
    if not generated_ids or generated_ids[-1] != eos_token_id:
        raise ValueError("native teacher completion must end with EOT")
    if eos_token_id in generated_ids[:-1]:
        raise ValueError("native teacher completion contains an embedded EOT")
    if not _ends_with(list(teacher_trigger_ids), open_ids):
        raise ValueError("teacher trigger must end with the thinking-open delimiter")
    if len(_subsequence_starts(teacher_trigger_ids, open_ids)) != 1:
        raise ValueError("teacher trigger requires exactly one thinking-open delimiter")
    if _find_subsequence(generated_ids, open_ids) is not None:
        raise ValueError("native teacher completion repeats the thinking-open delimiter")
    close_starts = _subsequence_starts(generated_ids, close_ids)
    if len(close_starts) != 1:
        raise ValueError("native teacher completion requires exactly one thinking-close delimiter")
    close_start = close_starts[0]
    if close_start == 0:
        raise ValueError("native teacher completion requires non-empty reasoning")
    visible_start = close_start + len(close_ids)
    visible_ids = generated_ids[visible_start:]
    if len(visible_ids) <= 1:
        raise ValueError("native teacher completion requires a visible answer before EOT")
    return NativeTeacherTrajectory(
        reasoning_ids=generated_ids[:visible_start],
        visible_response_ids=visible_ids,
        full_completion_ids=generated_ids,
    )


@torch.no_grad()
def sample_backbone_completions(
    backbone: TF,
    prompt_ids: Sequence[tuple[int, ...]],
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    stop_ids: tuple[int, ...] | None,
    temperature: float,
    top_p: float,
    top_k: int,
    valid_vocab_size: int | None = None,
    generators: Sequence[torch.Generator | None],
) -> tuple[tuple[int, ...], ...]:
    """Sample an equal-prompt-length batch with independent RNG streams."""
    prompts = tuple(prompt_ids)
    generator_list = tuple(generators)
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("teacher generation requires non-empty prompts")
    if len(generator_list) != len(prompts):
        raise ValueError("teacher generation requires one generator per prompt")
    prompt_length = len(prompts[0])
    if any(len(prompt) != prompt_length for prompt in prompts):
        raise ValueError("batched teacher prompts must have equal token lengths")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if prompt_length + max_new_tokens > backbone.config.sequence_len:
        raise ValueError("teacher prompt and completion exceed model sequence_len")

    device = backbone.get_device()
    num_kv_heads, head_dim = backbone.config.stored_cache_shape
    cache = StaticCacheState.allocate(
        batch_size=len(prompts),
        num_layers=backbone.config.cache_num_layers,
        num_kv_heads=num_kv_heads,
        max_seq_len=prompt_length + max_new_tokens,
        head_dim=head_dim,
        device=device,
        dtype=backbone.embed_tokens.weight.dtype,
    )
    prompt_prefix = torch.tensor(
        [prompt[:-1] for prompt in prompts], dtype=torch.long, device=device
    )
    if prompt_prefix.shape[1]:
        _, cache = backbone.forward_chunk_static(prompt_prefix, 0, cache, logits=False)
    prompt_last = torch.tensor(
        [[prompt[-1]] for prompt in prompts], dtype=torch.long, device=device
    )
    logits, cache = backbone.forward_chunk_static(prompt_last, prompt_length - 1, cache)
    next_logits = logits[:, -1, :]
    generated: list[list[int]] = [[] for _ in prompts]
    finished = [False] * len(prompts)
    fill_token_id = eos_token_id if eos_token_id is not None else 0

    for generated_index in range(max_new_tokens):
        sampled_rows: list[torch.Tensor] = []
        for row, generator in enumerate(generator_list):
            if finished[row]:
                sampled_rows.append(torch.tensor(fill_token_id, dtype=torch.long, device=device))
                continue
            sampled_rows.append(
                sample_next_token(
                    next_logits[row : row + 1],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    valid_vocab_size=valid_vocab_size,
                    generator=generator,
                )[0]
            )
        sampled = torch.stack(sampled_rows)
        for row in range(len(prompts)):
            if finished[row]:
                continue
            token_id = int(sampled[row].item())
            generated[row].append(token_id)
            finished[row] = (eos_token_id is not None and token_id == eos_token_id) or (
                stop_ids is not None and _ends_with(generated[row], stop_ids)
            )
        if all(finished) or generated_index + 1 == max_new_tokens:
            break
        token_position = prompt_length + generated_index
        logits, cache = backbone.forward_chunk_static(
            sampled[:, None], token_position, cache
        )
        next_logits = logits[:, -1, :]
    return tuple(tuple(row) for row in generated)


@torch.no_grad()
def sample_backbone_completion(
    backbone: TF,
    prompt_ids: tuple[int, ...],
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    stop_ids: tuple[int, ...] | None,
    temperature: float,
    top_p: float,
    top_k: int,
    valid_vocab_size: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[int, ...]:
    """Sample from a frozen native backbone without any cache processor."""
    if not prompt_ids:
        raise ValueError("teacher generation requires a non-empty prompt")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if len(prompt_ids) + max_new_tokens > backbone.config.sequence_len:
        raise ValueError("teacher prompt and completion exceed model sequence_len")

    device = backbone.get_device()
    num_kv_heads, head_dim = backbone.config.stored_cache_shape
    cache = StaticCacheState.allocate(
        batch_size=1,
        num_layers=backbone.config.cache_num_layers,
        num_kv_heads=num_kv_heads,
        max_seq_len=len(prompt_ids) + max_new_tokens,
        head_dim=head_dim,
        device=device,
        dtype=backbone.embed_tokens.weight.dtype,
    )
    prompt_prefix = prompt_ids[:-1]
    if prompt_prefix:
        prefix = torch.tensor([prompt_prefix], dtype=torch.long, device=device)
        _, cache = backbone.forward_chunk_static(prefix, 0, cache, logits=False)
    prompt_last = torch.tensor([[prompt_ids[-1]]], dtype=torch.long, device=device)
    logits, cache = backbone.forward_chunk_static(prompt_last, len(prompt_prefix), cache)
    next_logits = logits[:, -1, :]
    generated: list[int] = []

    for generated_index in range(max_new_tokens):
        sampled = sample_next_token(
            next_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            valid_vocab_size=valid_vocab_size,
            generator=generator,
        )
        if sampled.numel() != 1:
            raise ValueError("teacher generation supports batch size one")
        token_id = int(sampled.item())
        generated.append(token_id)
        finished = eos_token_id is not None and token_id == eos_token_id
        finished = finished or (stop_ids is not None and _ends_with(generated, stop_ids))
        if finished or generated_index + 1 == max_new_tokens:
            break
        token_position = len(prompt_ids) + generated_index
        logits, cache = backbone.forward_chunk_static(sampled[:, None], token_position, cache)
        next_logits = logits[:, -1, :]
    return tuple(generated)


def split_teacher_reasoning(
    generated_ids: tuple[int, ...],
    *,
    close_ids: tuple[int, ...],
    eos_token_id: int | None = None,
    close_at_eos: bool = False,
) -> tuple[int, ...]:
    """Keep the teacher trajectory through its first thinking-close separator."""
    if not close_ids:
        raise ValueError("thinking close separator must not be empty")
    stop = _find_subsequence(generated_ids, close_ids)
    if stop is None:
        if close_at_eos and eos_token_id is not None and generated_ids and generated_ids[-1] == eos_token_id:
            return generated_ids[:-1] + close_ids
        raise ValueError("teacher generation is missing the thinking close separator")
    return generated_ids[: stop + len(close_ids)]


@torch.no_grad()
def teacher_answer_logits(
    backbone: TeacherBackbone,
    teacher_context_ids: tuple[int, ...],
    response_ids: tuple[int, ...],
) -> torch.Tensor:
    """Score student answer tokens after privileged teacher reasoning context."""
    if not teacher_context_ids:
        raise ValueError("teacher context must not be empty")
    if not response_ids:
        raise ValueError("teacher scoring requires at least one response token")
    if isinstance(backbone, TF):
        return _native_teacher_answer_logits(backbone, teacher_context_ids, response_ids)
    device = _module_device(backbone)
    input_ids = torch.tensor([teacher_context_ids + response_ids], dtype=torch.long, device=device)
    logits = backbone.forward_full(input_ids)
    start = len(teacher_context_ids) - 1
    stop = start + len(response_ids)
    aligned = logits[:, start:stop, :]
    if aligned.shape[1] != len(response_ids):
        raise RuntimeError("teacher logits did not align with every response token")
    return aligned.detach()


def _native_teacher_answer_logits(
    backbone: TF,
    teacher_context_ids: tuple[int, ...],
    response_ids: tuple[int, ...],
) -> torch.Tensor:
    device = backbone.get_device()
    cache = None
    context_prefix = teacher_context_ids[:-1]
    if context_prefix:
        prefix = torch.tensor([context_prefix], dtype=torch.long, device=device)
        _, cache = backbone.forward_chunk(prefix, 0, logits=False)
    context_last = torch.tensor([[teacher_context_ids[-1]]], dtype=torch.long, device=device)
    first_logits, cache = backbone.forward_chunk(context_last, len(context_prefix), cache)
    aligned = [first_logits[:, -1:, :]]
    if len(response_ids) > 1:
        response_prefix = torch.tensor([response_ids[:-1]], dtype=torch.long, device=device)
        remaining_logits, _ = backbone.forward_chunk(response_prefix, len(teacher_context_ids), cache)
        aligned.append(remaining_logits)
    logits = torch.cat(aligned, dim=1)
    if logits.shape[1] != len(response_ids):
        raise RuntimeError("teacher logits did not align with every response token")
    return logits.detach()


def _subsequence_starts(sequence: tuple[int, ...], pattern: tuple[int, ...]) -> list[int]:
    return [
        start
        for start in range(len(sequence) - len(pattern) + 1)
        if sequence[start : start + len(pattern)] == pattern
    ]


def _find_subsequence(sequence: tuple[int, ...], pattern: tuple[int, ...]) -> int | None:
    last_start = len(sequence) - len(pattern)
    for start in range(last_start + 1):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return None


def _ends_with(sequence: list[int], suffix: tuple[int, ...]) -> bool:
    return len(sequence) >= len(suffix) and tuple(sequence[-len(suffix) :]) == suffix


def _module_device(module: TeacherBackbone) -> torch.device:
    try:
        parameter = next(iter(module.parameters()))
    except StopIteration:
        return torch.device("cpu")
    return parameter.device
