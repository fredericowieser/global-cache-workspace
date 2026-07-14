from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Any, Callable, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcw.config import ModelConfig
from gcw.models.cache_state import CacheState, StaticCacheState
from gcw.models.windowed_cache import WindowedCache

CacheLike = CacheState | WindowedCache
ProcessorFn = Callable[..., CacheLike | StaticCacheState]

try:
    from torch.nn.attention import SDPBackend as _SDPBackend, sdpa_kernel as _sdpa_kernel
    from torch.nn.attention.bias import causal_lower_right as _causal_lower_right
except ImportError:  # pragma: no cover - older torch fallback
    _SDPBackend = None  # type: ignore[assignment,misc]
    _causal_lower_right = None  # type: ignore[assignment]
    _sdpa_kernel = None  # type: ignore[assignment]

SDPBackend: Any = _SDPBackend
causal_lower_right: Any = _causal_lower_right
sdpa_kernel: Any = _sdpa_kernel


class Linear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _autocast_dtype(x.device) is not None or x.dtype == self.weight.dtype:
            return F.linear(x, self.weight, None)
        return F.linear(x, self.weight.to(dtype=x.dtype), None)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


def _autocast_dtype(device: torch.device) -> torch.dtype | None:
    if device.type == "cuda" and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype(device.type)
    return None


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, style: str = "llama") -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    if style == "ppt":
        return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class RotaryCache(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, max_seq_len: int, head_dim: int, theta: float):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.theta = theta
        cos, sin = self._build(max_seq_len, head_dim, theta, torch.device("cpu"))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _build(seq_len: int, head_dim: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32, device=device), inv)
        return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]

    def maybe_grow(self, seq_len: int, device: torch.device) -> None:
        if seq_len <= self.cos.shape[1] and self.cos.device == device:
            return
        self.max_seq_len = max(seq_len, self.max_seq_len * 2)
        cos, sin = self._build(self.max_seq_len, self.head_dim, self.theta, device)
        self.cos, self.sin = cos, sin

    def get(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        self.maybe_grow(int(positions.max().item()) + 1, positions.device)
        cos = self.cos.index_select(1, positions).to(dtype)
        sin = self.sin.index_select(1, positions).to(dtype)
        return cos, sin

    def get_range(self, start: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        self.maybe_grow(start + seq_len, device)
        end = start + seq_len
        return self.cos[:, start:end].to(dtype), self.sin[:, start:end].to(dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        if config.head_dim is None:
            raise ValueError("attention requires a resolved model head_dim")
        self.head_dim = config.head_dim
        self.rotary_style = config.rotary_style
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = Linear(config.hidden_size, q_dim, bias=False)
        self.k_proj = Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = Linear(q_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        past: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, _ = x.shape
        q = self.q_proj(x).view(batch, steps, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, steps, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, steps, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        rotary = (cos_sin[0].to(q.dtype), cos_sin[1].to(q.dtype))
        q, k = apply_rotary(q, *rotary, style=self.rotary_style), apply_rotary(k, *rotary, style=self.rotary_style)
        q, k = q.to(v.dtype), k.to(v.dtype)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        past_len = 0
        if past is not None:
            pk, pv = past
            past_len = pk.shape[2]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        mask, is_causal = _causal_chunk_bias(steps, past_len, x.device)
        y = flash_sdpa(q, k, v, attn_mask=mask, is_causal=is_causal, enable_gqa=self.num_kv_heads != self.num_heads)
        y = y.transpose(1, 2).contiguous().view(batch, steps, self.num_heads * self.head_dim)
        attn_out = self.o_proj(y)
        return attn_out, k, v, attn_out.float().norm(dim=-1)

    def forward_static(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cache: StaticCacheState,
        start_pos: int,
        end_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = x.shape
        q = self.q_proj(x).view(batch, steps, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, steps, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, steps, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        rotary = (cos_sin[0].to(q.dtype), cos_sin[1].to(q.dtype))
        q, k = apply_rotary(q, *rotary, style=self.rotary_style), apply_rotary(k, *rotary, style=self.rotary_style)
        q, k = q.to(v.dtype), k.to(v.dtype)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        cache.write_layer(self.layer_idx, start_pos, k, v)
        keys, values = cache.layer_view(self.layer_idx, end_pos)
        mask, is_causal = _causal_chunk_bias(steps, start_pos, x.device)
        y = flash_sdpa(q, keys, values, attn_mask=mask, is_causal=is_causal, enable_gqa=self.num_kv_heads != self.num_heads)
        y = y.transpose(1, 2).contiguous().view(batch, steps, self.num_heads * self.head_dim)
        attn_out = self.o_proj(y)
        return attn_out, attn_out.float().norm(dim=-1)


def _causal_chunk_bias(steps: int, past_len: int, device: torch.device) -> tuple[torch.Tensor | None, bool]:
    if past_len == 0:
        return None, True
    if causal_lower_right is not None:
        return causal_lower_right(steps, past_len + steps), False
    return _causal_chunk_mask(steps, past_len, device), False


def _causal_chunk_mask(steps: int, past_len: int, device: torch.device) -> torch.Tensor:
    q = torch.arange(steps, device=device)[:, None]
    kv = torch.arange(past_len + steps, device=device)[None, :]
    return kv <= (past_len + q)


def _matches_stop_sequence(tokens: torch.Tensor, stop_sequences: list[list[int]] | None) -> torch.Tensor:
    matched = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
    if not stop_sequences:
        return matched
    for stop in stop_sequences:
        if not stop or len(stop) > tokens.shape[1]:
            continue
        suffix = torch.tensor(stop, dtype=tokens.dtype, device=tokens.device)
        matched = matched | (tokens[:, -len(stop) :] == suffix).all(dim=1)
    return matched


def flash_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    enable_gqa: bool = False,
) -> torch.Tensor:
    if q.is_cuda and q.dtype in {torch.float16, torch.bfloat16} and sdpa_kernel is not None:
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    enable_gqa=enable_gqa,
                )
        except RuntimeError:
            if os.environ.get("GCW_STRICT_FLASH") == "1":
                raise
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
    )


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.norm_eps)
        self.self_attn = CausalSelfAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        past: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, keys, values, scores = self.self_attn(self.input_layernorm(x), cos_sin, past)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, keys, values, scores

    def forward_static(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cache: StaticCacheState,
        start_pos: int,
        end_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, scores = self.self_attn.forward_static(self.input_layernorm(x), cos_sin, cache, start_pos, end_pos)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, scores


class TF(nn.Module):
    def __init__(self, config: ModelConfig, pad_vocab_size_to: int = 128):
        super().__init__()
        self.config = replace(config, variant="tf")
        vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.padded_vocab_size = vocab
        self.embed_tokens = nn.Embedding(vocab, config.hidden_size)
        self.layers = nn.ModuleList(Block(config, i) for i in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        if config.head_dim is None:
            raise ValueError("transformer requires a resolved model head_dim")
        self.rotary = RotaryCache(config.sequence_len, config.head_dim, config.rope_theta)
        self.lm_head = None if config.tie_embeddings else Linear(config.hidden_size, vocab, bias=False)

    @torch.no_grad()
    def init_weights(self, std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0.0, std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def get_device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_reduction: str = "mean",
        processor: ProcessorFn | None = None,
        process_every: int = 64,
    ) -> torch.Tensor:
        if processor is None:
            return self.forward_full(input_ids, targets, loss_reduction)
        if targets is not None:
            return self.forward_windowed_loss(input_ids, targets, processor=processor, process_every=process_every, loss_reduction=loss_reduction)
        logits, _ = self.forward_windowed(input_ids, processor=processor, process_every=process_every)
        return logits

    def forward_full(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_reduction: str = "mean",
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        cos_sin = self.rotary.get_range(0, input_ids.shape[1], input_ids.device, _autocast_dtype(x.device) or x.dtype)
        for layer in self.layers:
            x, _, _, _ = cast(Block, layer)(x, cos_sin, None)
        x = self.norm(x)
        if targets is not None:
            return self.loss_from_hidden(x, targets, loss_reduction)
        return self.logits_from_hidden(x)

    def logits_from_hidden(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.tie_embeddings:
            logits = F.linear(x, self.embed_tokens.weight.to(x.dtype))
        else:
            if self.lm_head is None:
                raise RuntimeError("untied transformer has no language-model head")
            logits = self.lm_head(x)
        return logits[..., : self.config.vocab_size].float()

    def loss_from_hidden(self, x: torch.Tensor, targets: torch.Tensor, loss_reduction: str) -> torch.Tensor:
        if loss_reduction == "none":
            raise ValueError("chunked loss does not support reduction='none'")
        chunk_size = 256 if x.is_cuda else 64
        loss_sum = x.new_zeros((), dtype=torch.float32)
        token_count = x.new_zeros((), dtype=torch.float32)
        for start in range(0, x.shape[1], chunk_size):
            end = start + chunk_size
            logits = self.logits_from_hidden(x[:, start:end])
            chunk_targets = targets[:, start:end].reshape(-1).clone()
            valid = chunk_targets >= 0
            chunk_targets[~valid] = -100
            loss_sum = loss_sum + F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                chunk_targets,
                ignore_index=-100,
                reduction="sum",
            )
            token_count = token_count + valid.sum().to(torch.float32)
        if loss_reduction == "sum":
            return loss_sum
        return loss_sum / token_count.clamp_min(1.0)

    def forward_windowed_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        *,
        processor: ProcessorFn,
        process_every: int,
        loss_reduction: str = "mean",
    ) -> torch.Tensor:
        cache: CacheLike = CacheState([], [], input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32))
        hidden_chunks = []
        for start in range(0, input_ids.shape[1], process_every):
            if start > 0:
                cache = cast(CacheLike, processor(cache))
            end = start + process_every
            hidden, cache = self.forward_chunk(input_ids[:, start:end], start, cache, logits=False, cache_format="windowed")
            hidden_chunks.append(hidden)
        return self.loss_from_hidden(torch.cat(hidden_chunks, dim=1), targets, loss_reduction)

    def forward_windowed(
        self,
        input_ids: torch.Tensor,
        processor: ProcessorFn | None = None,
        process_every: int = 64,
    ) -> tuple[torch.Tensor, CacheLike]:
        cache: CacheLike = CacheState([], [], input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32))
        logits = []
        for start in range(0, input_ids.shape[1], process_every):
            if start > 0 and processor is not None:
                cache = cast(CacheLike, processor(cache))
            chunk_logits, cache = self.forward_chunk(input_ids[:, start : start + process_every], start, cache)
            logits.append(chunk_logits)
        return torch.cat(logits, dim=1), cache

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        cache: CacheLike | None = None,
        *,
        logits: bool = True,
        cache_format: str = "list",
    ) -> tuple[torch.Tensor, CacheLike]:
        x = self.embed_tokens(input_ids)
        cos_sin = self.rotary.get_range(
            start_pos,
            input_ids.shape[1],
            input_ids.device,
            _autocast_dtype(x.device) or x.dtype,
        )
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        layer_scores: list[torch.Tensor] = []
        has_cache = cache is not None and cache.scores.numel() > 0
        for i, layer in enumerate(self.layers):
            past = None if cache is None or not has_cache else cache.layer(i)
            x, k, v, scores = cast(Block, layer)(x, cos_sin, past)
            keys.append(k)
            values.append(v)
            layer_scores.append(scores)
        x = self.norm(x)
        out = self.logits_from_hidden(x) if logits else x
        chunk_scores = torch.stack(layer_scores).mean(dim=0)
        old_scores = input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32) if cache is None else cache.scores
        new_scores = chunk_scores if old_scores.numel() == 0 else torch.cat([old_scores, chunk_scores], dim=1)
        if cache_format == "windowed" or isinstance(cache, WindowedCache):
            return out, WindowedCache.from_lists(keys, values, new_scores)
        return out, CacheState(keys, values, new_scores)

    def forward_chunk_static(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        cache: StaticCacheState,
        *,
        logits: bool = True,
    ) -> tuple[torch.Tensor, StaticCacheState]:
        x = self.embed_tokens(input_ids)
        end_pos = start_pos + input_ids.shape[1]
        pos = torch.arange(start_pos, end_pos, device=input_ids.device)
        cos_sin = self.rotary.get(pos, x.dtype)
        layer_scores: list[torch.Tensor] = []
        for layer in self.layers:
            x, scores = cast(Block, layer).forward_static(x, cos_sin, cache, start_pos, end_pos)
            layer_scores.append(scores)
        x = self.norm(x)
        out = self.logits_from_hidden(x) if logits else x
        cache.write_scores(start_pos, torch.stack(layer_scores).mean(dim=0))
        return out, cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        processor: ProcessorFn | None = None,
        process_every: int = 64,
        process_prefill_windows: bool = True,
        process_after_prefill: bool = False,
        relative_generation_cadence: bool = False,
    ) -> torch.Tensor:
        out = input_ids[:, -self.config.sequence_len :]
        cache: CacheLike = CacheState([], [], input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32))
        logits = None
        if process_prefill_windows:
            for start in range(0, out.shape[1], process_every):
                if start > 0 and processor is not None:
                    cache = cast(CacheLike, processor(cache))
                logits, cache = self.forward_chunk(out[:, start : start + process_every], start, cache)
        else:
            logits, cache = self.forward_chunk(out, 0, cache)
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
                    should_process = should_process or (generated_idx > 0 and generated_idx % process_every == 0)
                else:
                    should_process = pos > 0 and pos % process_every == 0
                if should_process:
                    cache = cast(CacheLike, processor(cache, select_all=relative_generation_cadence and generated_idx == 0))
            logits, cache = self.forward_chunk(next_id, pos, cache)
        return out

    @torch.no_grad()
    def generate_fast(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        processor: ProcessorFn | None = None,
        process_every: int = 64,
        process_prefill_windows: bool = True,
        process_after_prefill: bool = False,
        relative_generation_cadence: bool = False,
        stop_sequences: list[list[int]] | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        out = input_ids[:, -self.config.sequence_len :]
        if out.shape[1] == 0:
            raise ValueError("generate requires at least one input token")
        if self.config.head_dim is None:
            raise RuntimeError("transformer has no resolved head_dim")
        cache = StaticCacheState.allocate(
            batch_size=out.shape[0],
            num_layers=len(self.layers),
            num_kv_heads=self.config.num_kv_heads,
            max_seq_len=out.shape[1] + max_new_tokens,
            head_dim=self.config.head_dim,
            device=out.device,
            dtype=self.embed_tokens.weight.dtype,
        )
        logits = None
        if process_prefill_windows:
            for start in range(0, out.shape[1], process_every):
                if start > 0 and processor is not None:
                    cache = cast(StaticCacheState, processor(cache))
                logits, cache = self.forward_chunk_static(out[:, start : start + process_every], start, cache)
        else:
            logits, cache = self.forward_chunk_static(out, 0, cache)
        if logits is None:
            raise ValueError("generate requires at least one input token")

        finished = torch.zeros(out.shape[0], dtype=torch.bool, device=out.device)
        fill_id = eos_token_id if pad_token_id is None else pad_token_id
        if fill_id is None:
            fill_id = 0
        fill = torch.full((out.shape[0], 1), int(fill_id), dtype=input_ids.dtype, device=out.device)
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
                    should_process = should_process or (generated_idx > 0 and generated_idx % process_every == 0)
                else:
                    should_process = pos > 0 and pos % process_every == 0
                if should_process:
                    cache = cast(StaticCacheState, processor(cache, select_all=relative_generation_cadence and generated_idx == 0))
            logits, cache = self.forward_chunk_static(next_id, pos, cache)
        return out

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not non_embedding:
            return total
        total -= self.embed_tokens.weight.numel()
        if self.lm_head is not None:
            total -= self.lm_head.weight.numel()
        return total


def estimate_non_embedding_params(config: ModelConfig) -> int:
    head_dim = int(config.head_dim or config.hidden_size // config.num_heads)
    attn = config.hidden_size * (config.num_heads * head_dim)
    attn += 2 * config.hidden_size * (config.num_kv_heads * head_dim)
    attn += (config.num_heads * head_dim) * config.hidden_size
    mlp = 3 * config.hidden_size * config.intermediate_size
    norms = 2 * config.hidden_size
    qk = 2 * config.num_layers * config.num_kv_heads * 0
    return config.num_layers * (attn + mlp + norms) + config.hidden_size + qk


def cosine_lr(step: int, max_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
