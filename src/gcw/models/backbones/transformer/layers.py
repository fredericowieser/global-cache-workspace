from __future__ import annotations

import os
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from gcw.config import ModelConfig

try:
    from torch.nn.attention import (
        SDPBackend as _SDPBackend,
        sdpa_kernel as _sdpa_kernel,
    )
    from torch.nn.attention.bias import (
        causal_lower_right as _causal_lower_right,
    )
except ImportError:  # pragma: no cover - older torch fallback
    _SDPBackend = None  # type: ignore[assignment,misc]
    _causal_lower_right = None  # type: ignore[assignment]
    _sdpa_kernel = None  # type: ignore[assignment]

SDPBackend: Any = _SDPBackend
causal_lower_right: Any = _causal_lower_right
sdpa_kernel: Any = _sdpa_kernel


def _autocast_dtype(device: torch.device) -> torch.dtype | None:
    if device.type == "cuda" and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype(device.type)
    return None


class Linear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            _autocast_dtype(x.device) is not None
            or x.dtype == self.weight.dtype
        ):
            return F.linear(x, self.weight, None)
        return F.linear(x, self.weight.to(dtype=x.dtype), None)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


def apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, style: str = "llama"
) -> torch.Tensor:
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
        cos, sin = self._build(
            max_seq_len, head_dim, theta, torch.device("cpu")
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _build(
        seq_len: int, head_dim: int, theta: float, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        freqs = torch.outer(
            torch.arange(seq_len, dtype=torch.float32, device=device), inv
        )
        return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]

    def maybe_grow(self, seq_len: int, device: torch.device) -> None:
        if seq_len <= self.cos.shape[1] and self.cos.device == device:
            return
        self.max_seq_len = max(seq_len, self.max_seq_len * 2)
        cos, sin = self._build(
            self.max_seq_len, self.head_dim, self.theta, device
        )
        self.cos, self.sin = cos, sin

    def get(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.maybe_grow(int(positions.max().item()) + 1, positions.device)
        cos = self.cos.index_select(1, positions).to(dtype)
        sin = self.sin.index_select(1, positions).to(dtype)
        return cos, sin

    def get_range(
        self, start: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.maybe_grow(start + seq_len, device)
        end = start + seq_len
        return self.cos[:, start:end].to(dtype), self.sin[:, start:end].to(
            dtype
        )


def _causal_chunk_bias(
    steps: int, past_len: int, device: torch.device
) -> tuple[torch.Tensor | None, bool]:
    if past_len == 0:
        return None, True
    if causal_lower_right is not None:
        return causal_lower_right(steps, past_len + steps), False
    return _causal_chunk_mask(steps, past_len, device), False


def _causal_chunk_mask(
    steps: int, past_len: int, device: torch.device
) -> torch.Tensor:
    q = torch.arange(steps, device=device)[:, None]
    kv = torch.arange(past_len + steps, device=device)[None, :]
    return kv <= (past_len + q)


def flash_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    enable_gqa: bool = False,
    scale: float | None = None,
    try_flash: bool = True,
) -> torch.Tensor:
    if (
        try_flash
        and q.is_cuda
        and q.dtype in {torch.float16, torch.bfloat16}
        and sdpa_kernel is not None
    ):
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
                    scale=scale,
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
        scale=scale,
    )


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
