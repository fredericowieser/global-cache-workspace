#!/usr/bin/env python3
"""Throwaway feasibility spike: append processor-emitted embeddings as causal positions.

This file intentionally mirrors only the embedding entry point missing from
``TF.forward_chunk``. It does not modify GCW production code.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcw.config import ModelConfig  # noqa: E402
from gcw.models import build_model  # noqa: E402
from gcw.models.transformer import Block, TF  # noqa: E402
from gcw.models.windowed_cache import WindowedCache  # noqa: E402


@dataclass(frozen=True)
class ThoughtTrace:
    start_positions: tuple[int, ...]
    cache_lengths: tuple[int, ...]
    latent_norms: tuple[float, ...]


@dataclass(frozen=True)
class SpikeResult:
    architecture: str
    backbone_parameters_frozen: bool
    prompt_length: int
    no_op_cache_identity: bool
    no_op_logit_max_abs_delta: float
    old_cache_prefix_unchanged_after_append: bool
    one_step_cache_length: int
    one_step_start_positions: tuple[int, ...]
    one_step_vs_no_thought_logit_max_abs_delta: float
    one_step_vs_zero_latent_logit_max_abs_delta: float
    zero_latent_vs_no_thought_logit_max_abs_delta: float
    repeated_k: int
    repeated_cache_lengths: tuple[int, ...]
    repeated_start_positions: tuple[int, ...]
    repeated_final_cache_length: int
    repeated_vs_one_step_logit_max_abs_delta: float
    emitter_grad_l1: float
    emitter_parameters_with_nonzero_grad: int
    backbone_parameters_with_grad: int
    terminal_loss: float
    verdict: str


class ContinuousThoughtEmitter(nn.Module):
    """Tiny recurrent processor that emits one model-width latent at a time."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.source = nn.Linear(hidden_size, hidden_size, bias=False)
        self.feedback = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, source: torch.Tensor, feedback: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.source(source) + self.feedback(feedback)))


def clone_cache(cache: WindowedCache) -> WindowedCache:
    return WindowedCache(
        keys=cache.keys.clone(),
        values=cache.values.clone(),
        scores=cache.scores.clone(),
        linear_states=cache.linear_states,
    )


def forward_embedding_chunk(
    backbone: TF,
    inputs_embeds: torch.Tensor,
    start_pos: int,
    cache: WindowedCache,
) -> tuple[torch.Tensor, WindowedCache]:
    """The minimal ``inputs_embeds`` analogue of GCW ``TF.forward_chunk``.

    The frozen decoder computes normal per-layer K/V for the supplied embedding.
    Those K/V are concatenated after the old cache, so this creates positions; it
    does not rewrite an old cache slot.
    """

    if inputs_embeds.ndim != 3:
        raise ValueError("inputs_embeds must have shape (batch, steps, hidden)")
    if inputs_embeds.shape[0] != cache.batch_size:
        raise ValueError("embedding and cache batch sizes must match")
    if inputs_embeds.shape[2] != backbone.config.hidden_size:
        raise ValueError("embedding width must match the decoder hidden size")
    if start_pos != cache.seq_len:
        raise ValueError(f"start_pos {start_pos} must equal cache length {cache.seq_len}")
    if inputs_embeds.shape[1] == 0:
        return inputs_embeds, cache

    x = inputs_embeds
    cos_sin = backbone.rotary.get_range(start_pos, x.shape[1], x.device, x.dtype)
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    layer_scores: list[torch.Tensor] = []
    for layer_index, layer in enumerate(backbone.layers):
        x, key, value, score = cast(Block, layer)(x, cos_sin, cache.layer(layer_index))
        keys.append(key)
        values.append(value)
        layer_scores.append(score)
    hidden = backbone.norm(x)
    scores = torch.cat([cache.scores, torch.stack(layer_scores).mean(dim=0)], dim=1)
    return hidden, WindowedCache.from_lists(keys, values, scores)


def append_continuous_thoughts(
    backbone: TF,
    emitter: ContinuousThoughtEmitter,
    source: torch.Tensor,
    cache: WindowedCache,
    *,
    steps: int,
) -> tuple[WindowedCache, ThoughtTrace]:
    """Emit and consume K thoughts sequentially, one new position per thought."""

    if steps < 0:
        raise ValueError("steps must be non-negative")
    feedback = source
    starts: list[int] = []
    lengths: list[int] = []
    norms: list[float] = []
    for _ in range(steps):
        latent = emitter(source, feedback)
        start = cache.seq_len
        hidden, cache = forward_embedding_chunk(backbone, latent[:, None, :], start, cache)
        feedback = hidden[:, -1]
        starts.append(start)
        lengths.append(cache.seq_len)
        norms.append(float(latent.detach().float().norm()))
    return cache, ThoughtTrace(tuple(starts), tuple(lengths), tuple(norms))


def append_fixed_latents(
    backbone: TF,
    latents: torch.Tensor,
    cache: WindowedCache,
) -> WindowedCache:
    for index in range(latents.shape[1]):
        _, cache = forward_embedding_chunk(backbone, latents[:, index : index + 1], cache.seq_len, cache)
    return cache


def terminal_logits(backbone: TF, token_id: torch.Tensor, cache: WindowedCache) -> tuple[torch.Tensor, WindowedCache]:
    hidden, next_cache = backbone.forward_chunk(
        token_id,
        cache.seq_len,
        cache,
        logits=False,
        cache_format="windowed",
    )
    if not isinstance(next_cache, WindowedCache):
        raise TypeError("terminal decode did not return WindowedCache")
    return backbone.logits_from_hidden(hidden[:, -1])[:, : backbone.config.vocab_size], next_cache


def tiny_frozen_qwen3() -> TF:
    """Instantiate the native GCW Qwen3-compatible decoder used by the spike."""

    config = ModelConfig(
        variant="tf",
        family="qwen3",
        sequence_len=32,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        rope_theta=10000.0,
        norm_eps=1e-6,
        tie_embeddings=False,
        qk_norm=True,
        rotary_style="llama",
    )
    model = build_model(config)
    if not isinstance(model, TF):
        raise TypeError("expected native TF Qwen3 backbone")
    model.init_weights()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _cache_equal(left: WindowedCache, right: WindowedCache) -> bool:
    return bool(
        torch.equal(left.keys, right.keys)
        and torch.equal(left.values, right.values)
        and torch.equal(left.scores, right.scores)
    )


def _max_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def run_checks() -> SpikeResult:
    torch.manual_seed(20260808)
    backbone = tiny_frozen_qwen3()
    emitter = ContinuousThoughtEmitter(backbone.config.hidden_size).eval()

    prompt_ids = torch.tensor([[3, 5, 8, 13]])
    terminal_token = torch.tensor([[21]])
    answer_target = torch.tensor([34])
    with torch.no_grad():
        prompt_hidden, prompt_cache = backbone.forward_chunk(
            prompt_ids,
            0,
            None,
            logits=False,
            cache_format="windowed",
        )
    if not isinstance(prompt_cache, WindowedCache):
        raise TypeError("prefill did not return WindowedCache")
    source = prompt_hidden[:, -1].detach()
    prompt_length = prompt_cache.seq_len

    # K=0 must be a true no-op: do not allocate a position at all.
    untouched = clone_cache(prompt_cache)
    no_op_cache, no_op_trace = append_continuous_thoughts(backbone, emitter, source, untouched, steps=0)
    no_op_cache_identity = _cache_equal(no_op_cache, prompt_cache) and no_op_trace.cache_lengths == ()
    baseline_logits, baseline_final_cache = terminal_logits(backbone, terminal_token, clone_cache(prompt_cache))
    no_op_logits, no_op_final_cache = terminal_logits(backbone, terminal_token, no_op_cache)
    no_op_logit_delta = _max_abs_delta(no_op_logits, baseline_logits)
    assert no_op_cache_identity
    assert _cache_equal(no_op_final_cache, baseline_final_cache)
    assert no_op_logit_delta == 0.0

    # One emitted thought must append exactly one causal position.
    one_cache, one_trace = append_continuous_thoughts(backbone, emitter, source, clone_cache(prompt_cache), steps=1)
    one_logits, _ = terminal_logits(backbone, terminal_token, one_cache)
    assert one_trace.start_positions == (prompt_length,)
    assert one_trace.cache_lengths == (prompt_length + 1,)
    old_cache_prefix_unchanged = bool(
        torch.equal(one_cache.keys[:, :, :, :prompt_length], prompt_cache.keys)
        and torch.equal(one_cache.values[:, :, :, :prompt_length], prompt_cache.values)
        and torch.equal(one_cache.scores[:, :prompt_length], prompt_cache.scores)
    )
    assert old_cache_prefix_unchanged

    # Separate content from mere position growth: compare against a zero latent
    # at the same added position and against no added position.
    zeros = source.new_zeros(source.shape[0], 1, source.shape[1])
    zero_cache = append_fixed_latents(backbone, zeros, clone_cache(prompt_cache))
    zero_logits, _ = terminal_logits(backbone, terminal_token, zero_cache)
    one_vs_baseline = _max_abs_delta(one_logits, baseline_logits)
    one_vs_zero = _max_abs_delta(one_logits, zero_logits)
    zero_vs_baseline = _max_abs_delta(zero_logits, baseline_logits)
    assert one_vs_baseline > 0.0
    assert one_vs_zero > 0.0

    # K>1 repeats the same emission -> frozen decoder -> appended-cache loop.
    repeated_k = 3
    repeated_cache, repeated_trace = append_continuous_thoughts(
        backbone, emitter, source, clone_cache(prompt_cache), steps=repeated_k
    )
    repeated_logits, repeated_final_cache = terminal_logits(backbone, terminal_token, repeated_cache)
    assert repeated_trace.start_positions == tuple(range(prompt_length, prompt_length + repeated_k))
    assert repeated_trace.cache_lengths == tuple(range(prompt_length + 1, prompt_length + repeated_k + 1))
    assert repeated_cache.seq_len == prompt_length + repeated_k
    assert repeated_final_cache.seq_len == prompt_length + repeated_k + 1
    repeated_vs_one = _max_abs_delta(repeated_logits, one_logits)
    assert repeated_vs_one > 0.0

    # Terminal answer loss crosses the frozen decoder and all recurrent thoughts
    # but can update only the emitter.
    emitter.zero_grad(set_to_none=True)
    grad_cache, _ = append_continuous_thoughts(backbone, emitter, source, clone_cache(prompt_cache), steps=repeated_k)
    grad_logits, _ = terminal_logits(backbone, terminal_token, grad_cache)
    loss = F.cross_entropy(grad_logits, answer_target)
    loss.backward()
    emitter_grads = [parameter.grad for parameter in emitter.parameters()]
    nonzero_grad_count = sum(grad is not None and bool(torch.count_nonzero(grad)) for grad in emitter_grads)
    emitter_grad_l1 = float(sum(grad.detach().float().abs().sum() for grad in emitter_grads if grad is not None))
    backbone_grad_count = sum(parameter.grad is not None for parameter in backbone.parameters())
    assert nonzero_grad_count > 0
    assert emitter_grad_l1 > 0.0
    assert backbone_grad_count == 0
    assert all(not parameter.requires_grad for parameter in backbone.parameters())

    return SpikeResult(
        architecture="GCW native tiny Qwen3 TF + external recurrent continuous-thought emitter",
        backbone_parameters_frozen=all(not parameter.requires_grad for parameter in backbone.parameters()),
        prompt_length=prompt_length,
        no_op_cache_identity=no_op_cache_identity,
        no_op_logit_max_abs_delta=no_op_logit_delta,
        old_cache_prefix_unchanged_after_append=old_cache_prefix_unchanged,
        one_step_cache_length=one_cache.seq_len,
        one_step_start_positions=one_trace.start_positions,
        one_step_vs_no_thought_logit_max_abs_delta=one_vs_baseline,
        one_step_vs_zero_latent_logit_max_abs_delta=one_vs_zero,
        zero_latent_vs_no_thought_logit_max_abs_delta=zero_vs_baseline,
        repeated_k=repeated_k,
        repeated_cache_lengths=repeated_trace.cache_lengths,
        repeated_start_positions=repeated_trace.start_positions,
        repeated_final_cache_length=repeated_final_cache.seq_len,
        repeated_vs_one_step_logit_max_abs_delta=repeated_vs_one,
        emitter_grad_l1=emitter_grad_l1,
        emitter_parameters_with_nonzero_grad=nonzero_grad_count,
        backbone_parameters_with_grad=backbone_grad_count,
        terminal_loss=float(loss.detach()),
        verdict="FEASIBLE: emitted embeddings can append differentiable causal positions without rewriting old KV.",
    )


def main() -> None:
    print(json.dumps(asdict(run_checks()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
