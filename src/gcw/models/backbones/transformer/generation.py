"""Native greedy generation and symbolic parameter accounting for transformer backbones.

Generation here is the reference path the evaluation harness exercises: teacher-forced
TF generation (``generate_tf``) and the faster single-pass variant (``generate_fast_tf``),
plus the per-family non-embedding parameter estimators used for FLOPs accounting.
"""

from __future__ import annotations

from typing import Any, Callable, cast
import torch

from gcw.config import ModelConfig
from gcw.models.cache_processors.state import CacheState, StaticCacheState


def _matches_stop_sequence(
    tokens: torch.Tensor, stop_sequences: list[list[int]] | None
) -> torch.Tensor:
    matched = torch.zeros(
        tokens.shape[0], dtype=torch.bool, device=tokens.device
    )
    if not stop_sequences:
        return matched
    for stop in stop_sequences:
        if not stop or len(stop) > tokens.shape[1]:
            continue
        suffix = torch.tensor(stop, dtype=tokens.dtype, device=tokens.device)
        matched = matched | (tokens[:, -len(stop) :] == suffix).all(dim=1)
    return matched


@torch.no_grad()
def generate_tf(
    model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    processor: Callable[..., Any] | None = None,
    process_every: int = 64,
    process_prefill_windows: bool = True,
    process_after_prefill: bool = False,
    relative_generation_cadence: bool = False,
) -> torch.Tensor:
    out = input_ids[:, -model.config.sequence_len :]
    cache: Any = CacheState(
        [], [], input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32)
    )
    logits = None
    if process_prefill_windows:
        for start in range(0, out.shape[1], process_every):
            if start > 0 and processor is not None:
                cache = processor(cache)
            logits, cache = model.forward_chunk(
                out[:, start : start + process_every], start, cache
            )
    else:
        logits, cache = model.forward_chunk(out, 0, cache)
    if logits is None:
        raise ValueError("generate requires at least one input token")
    for generated_idx in range(max_new_tokens):
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)
        if eos_token_id is not None and bool((next_id == eos_token_id).all()):
            break
        pos = out.shape[1] - 1
        if processor is not None:
            if relative_generation_cadence:
                should_process = generated_idx == 0 and process_after_prefill
                should_process = should_process or (
                    generated_idx > 0 and generated_idx % process_every == 0
                )
            else:
                should_process = pos > 0 and pos % process_every == 0
            if should_process:
                cache = processor(
                    cache,
                    select_all=relative_generation_cadence
                    and generated_idx == 0,
                )
        logits, cache = model.forward_chunk(next_id, pos, cache)
    return out


@torch.no_grad()
def generate_fast_tf(
    model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    processor: Callable[..., Any] | None = None,
    process_every: int = 64,
    process_prefill_windows: bool = True,
    process_after_prefill: bool = False,
    relative_generation_cadence: bool = False,
    stop_sequences: list[list[int]] | None = None,
    pad_token_id: int | None = None,
) -> torch.Tensor:
    out = input_ids[:, -model.config.sequence_len :]
    if out.shape[1] == 0:
        raise ValueError("generate requires at least one input token")
    if model.config.head_dim is None:
        raise RuntimeError("transformer has no resolved head_dim")
    cache = StaticCacheState.allocate(
        batch_size=out.shape[0],
        num_layers=model.config.cache_num_layers,
        num_kv_heads=model.config.num_kv_heads,
        max_seq_len=out.shape[1] + max_new_tokens,
        head_dim=model.config.head_dim,
        device=out.device,
        dtype=model.embed_tokens.weight.dtype,
    )
    logits = None
    if process_prefill_windows:
        for start in range(0, out.shape[1], process_every):
            if start > 0 and processor is not None:
                cache = cast(StaticCacheState, processor(cache))
            logits, cache = model.forward_chunk_static(
                out[:, start : start + process_every], start, cache
            )
    else:
        logits, cache = model.forward_chunk_static(out, 0, cache)
    if logits is None:
        raise ValueError("generate requires at least one input token")

    finished = torch.zeros(out.shape[0], dtype=torch.bool, device=out.device)
    fill_id = eos_token_id if pad_token_id is None else pad_token_id
    if fill_id is None:
        fill_id = 0
    fill = torch.full(
        (out.shape[0], 1),
        int(fill_id),
        dtype=input_ids.dtype,
        device=out.device,
    )
    for generated_idx in range(max_new_tokens):
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        if bool(finished.any()):
            next_id = torch.where(finished[:, None], fill, next_id)
        out = torch.cat([out, next_id], dim=1)
        if eos_token_id is not None:
            finished = finished | (next_id.squeeze(1) == eos_token_id)
        finished = finished | _matches_stop_sequence(out, stop_sequences)
        if bool(finished.all()):
            break
        pos = out.shape[1] - 1
        if processor is not None:
            if relative_generation_cadence:
                should_process = generated_idx == 0 and process_after_prefill
                should_process = should_process or (
                    generated_idx > 0 and generated_idx % process_every == 0
                )
            else:
                should_process = pos > 0 and pos % process_every == 0
            if should_process:
                cache = cast(
                    StaticCacheState,
                    processor(
                        cache,
                        select_all=relative_generation_cadence
                        and generated_idx == 0,
                    ),
                )
        logits, cache = model.forward_chunk_static(next_id, pos, cache)
    return out


def estimate_non_embedding_params(config: ModelConfig) -> int:
    if config.family == "qwen3_5":
        return _estimate_qwen35_non_embedding_params(config)
    if config.family == "qwen3_5_moe":
        return _estimate_qwen35_moe_non_embedding_params(config)
    if config.family == "gemma4":
        return _estimate_gemma4_non_embedding_params(config)
    head_dim = int(config.head_dim or config.hidden_size // config.num_heads)
    attn = config.hidden_size * (config.num_heads * head_dim)
    attn += 2 * config.hidden_size * (config.num_kv_heads * head_dim)
    attn += (config.num_heads * head_dim) * config.hidden_size
    mlp = 3 * config.hidden_size * config.intermediate_size
    norms = 2 * config.hidden_size
    qk = 2 * config.num_layers * config.num_kv_heads * 0
    return config.num_layers * (attn + mlp + norms) + config.hidden_size + qk


def _estimate_qwen35_attention_params(
    config: ModelConfig, layer_type: str
) -> int:
    head_dim = int(config.head_dim or config.hidden_size // config.num_heads)
    full_q_dim = config.num_heads * head_dim
    full_kv_dim = config.num_kv_heads * head_dim
    linear_key_dim = int(
        config.linear_num_key_heads or config.num_kv_heads
    ) * int(config.linear_key_head_dim or head_dim)
    linear_value_heads = int(config.linear_num_value_heads or config.num_heads)
    linear_value_dim = linear_value_heads * int(
        config.linear_value_head_dim or head_dim
    )
    linear_conv_dim = 2 * linear_key_dim + linear_value_dim
    if layer_type == "full_attention":
        return (
            config.hidden_size * full_q_dim * 2
            + 2 * config.hidden_size * full_kv_dim
            + full_q_dim * config.hidden_size
            + 2 * head_dim
        )
    return (
        config.hidden_size * linear_conv_dim
        + config.hidden_size * linear_value_dim
        + 2 * config.hidden_size * linear_value_heads
        + linear_conv_dim * config.linear_conv_kernel_dim
        + linear_value_heads * 2
        + int(config.linear_value_head_dim or head_dim)
        + linear_value_dim * config.hidden_size
    )


def _estimate_qwen35_non_embedding_params(config: ModelConfig) -> int:
    total = config.hidden_size
    for layer_type in config.layer_types or []:
        attn = _estimate_qwen35_attention_params(config, layer_type)
        mlp = 3 * config.hidden_size * config.intermediate_size
        norms = 2 * config.hidden_size
        total += attn + mlp + norms
    return total


def _estimate_qwen35_moe_non_embedding_params(config: ModelConfig) -> int:
    moe_hidden = int(config.moe_intermediate_size or config.intermediate_size)
    shared_hidden = int(config.shared_expert_intermediate_size or moe_hidden)
    total = config.hidden_size
    for layer_type in config.layer_types or []:
        attn = _estimate_qwen35_attention_params(config, layer_type)
        moe = config.num_experts * (
            2 * moe_hidden * config.hidden_size
            + config.hidden_size * moe_hidden
        )
        moe += config.hidden_size * config.num_experts
        moe += 3 * config.hidden_size * shared_hidden
        moe += config.hidden_size
        norms = 2 * config.hidden_size
        total += attn + moe + norms
    return total


def _estimate_gemma4_attention_params(
    config: ModelConfig, layer_idx: int
) -> int:
    if config.layer_types is None:
        raise ValueError("Gemma4 parameter accounting requires layer_types")
    head_dim = config.head_dim_for_layer(layer_idx)
    kv_heads = config.kv_heads_for_layer(layer_idx)
    q_dim = config.num_heads * head_dim
    kv_dim = kv_heads * head_dim
    total = config.hidden_size * q_dim + q_dim * config.hidden_size + head_dim
    first_shared = config.num_layers - config.num_kv_shared_layers
    if layer_idx < first_shared:
        total += config.hidden_size * kv_dim
        if not (
            config.attention_k_eq_v
            and config.layer_types[layer_idx] == "full_attention"
        ):
            total += config.hidden_size * kv_dim
        total += head_dim
    return total


def _estimate_gemma4_non_embedding_params(config: ModelConfig) -> int:
    total = config.hidden_size
    ple = int(config.hidden_size_per_layer_input)
    if ple:
        total += (
            int(config.vocab_size_per_layer_input or config.vocab_size)
            * config.num_layers
            * ple
        )
        total += config.hidden_size * config.num_layers * ple
        total += ple
    first_shared = config.num_layers - config.num_kv_shared_layers
    for i in range(config.num_layers):
        total += _estimate_gemma4_attention_params(config, i)
        intermediate = config.intermediate_size * (
            2 if config.use_double_wide_mlp and i >= first_shared > 0 else 1
        )
        total += 3 * config.hidden_size * intermediate
        total += 4 * config.hidden_size
        if ple:
            total += (
                config.hidden_size * ple
                + ple * config.hidden_size
                + config.hidden_size
            )
    return total
