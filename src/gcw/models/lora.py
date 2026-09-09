from __future__ import annotations

import math
from contextlib import nullcontext
from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcw.config import LoRAConfig

ModuleT = TypeVar("ModuleT", bound=nn.Module)


class LoRALinear(nn.Linear):
    """A linear layer with a zero-initialized low-rank residual."""

    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__(
            base.in_features, base.out_features, bias=base.bias is not None
        )
        self.weight = base.weight
        if base.bias is not None:
            self.bias = base.bias
        self.lora_a = nn.Parameter(torch.empty(config.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, config.rank))
        self.scaling = config.alpha / config.rank
        self.dropout = config.dropout
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        adapted = F.linear(
            F.linear(F.dropout(x, self.dropout, self.training), self.lora_a),
            self.lora_b,
        )
        return base + adapted * self.scaling


class KVLoRALinear(LoRALinear):
    """Independent rank-r K/V adapters for a native fused [Q, K, V] matrix.

    The original weight/bias names and full native projection are retained.
    There are no adapter parameters for Q; its output is copied unchanged.
    Stacked A and B tensors retain the ordinary LoRA checkpoint/policy ABI.
    """

    def __init__(
        self, base: nn.Linear, config: LoRAConfig, key_dim: int, value_dim: int
    ):
        if (
            key_dim <= 0
            or value_dim <= 0
            or base.out_features != 2 * key_dim + value_dim
        ):
            raise ValueError(
                "in_proj_kv requires native [Q, K, V] projection geometry"
            )
        nn.Linear.__init__(
            self,
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.weight = base.weight
        if base.bias is not None:
            self.bias = base.bias
        self.key_dim, self.value_dim = key_dim, value_dim
        self.rank = config.rank
        self.scaling = config.alpha / config.rank
        self.dropout = config.dropout
        self.lora_a = nn.Parameter(
            torch.empty(
                2 * config.rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                key_dim + value_dim,
                config.rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        for a in self.lora_a.chunk(2, dim=0):
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        dropped = F.dropout(x, self.dropout, self.training)
        ka, va = self.lora_a.chunk(2, dim=0)
        kb, vb = self.lora_b.split((self.key_dim, self.value_dim), dim=0)
        key = F.linear(F.linear(dropped, ka), kb) * self.scaling
        value = F.linear(F.linear(dropped, va), vb) * self.scaling
        q, k, v = base.split(
            (self.key_dim, self.key_dim, self.value_dim), dim=-1
        )
        return torch.cat((q, k + key, v + value), dim=-1)


def apply_lora(module: ModuleT, config: LoRAConfig | None) -> ModuleT:
    """Attach named projections; virtual ``in_proj_kv`` excludes fused Q rows.

    This new virtual target opts into isolated adapter-construction RNG, so a
    jointly built processor starts identically to its adapter-free control.
    Existing target configurations preserve their historical initialization.
    """
    if config is None:
        return module
    targets = set(config.targets)
    kv_only = "in_proj_kv" in targets
    if kv_only and "in_proj_qkv" in targets:
        raise ValueError(
            "in_proj_kv and in_proj_qkv targets are mutually exclusive"
        )
    devices = (
        sorted(
            {
                int(p.device.index or 0)
                for p in module.parameters()
                if p.device.type == "cuda"
            }
        )
        if kv_only
        else []
    )
    matched_kv = 0
    with torch.random.fork_rng(devices=devices) if kv_only else nullcontext():
        for parent in list(module.modules()):
            for name, child in list(parent.named_children()):
                if (
                    kv_only
                    and name == "in_proj_qkv"
                    and isinstance(child, KVLoRALinear)
                ):
                    matched_kv += 1
                if not isinstance(child, nn.Linear) or isinstance(
                    child, LoRALinear
                ):
                    continue
                if kv_only and name == "in_proj_qkv":
                    key_dim = getattr(parent, "key_dim", 0)
                    value_dim = getattr(parent, "value_dim", 0)
                    setattr(
                        parent,
                        name,
                        KVLoRALinear(child, config, key_dim, value_dim),
                    )
                    matched_kv += 1
                elif name in targets:
                    adapter = LoRALinear(child, config)
                    if kv_only:
                        adapter.to(
                            device=child.weight.device, dtype=child.weight.dtype
                        )
                    setattr(parent, name, adapter)
    if kv_only and not matched_kv:
        raise ValueError(
            "in_proj_kv target matched no native fused QKV projections"
        )
    return module
