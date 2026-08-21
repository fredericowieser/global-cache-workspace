from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

_HF_ACT2FN: Any = None

try:
    from transformers.activations import ACT2FN as _imported_hf_act2fn
except Exception:
    pass
else:
    _HF_ACT2FN = _imported_hf_act2fn


def _act_fn(name: str):
    if _HF_ACT2FN is not None and name in _HF_ACT2FN:
        return _HF_ACT2FN[name]
    if name == "gelu_pytorch_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "silu":
        return F.silu
    raise ImportError(
        f"Gemma4 activation {name!r} requires transformers.activations.ACT2FN"
    )


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.pow(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps, -0.5
        )
        if self.with_scale:
            y = y * self.weight.float()
        return y.to(x.dtype)


class Gemma4ScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None,
        embed_scale: float,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.embed_scale: torch.Tensor
        self.register_buffer(
            "embed_scale", torch.tensor(embed_scale), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(
            self.weight.dtype
        )
