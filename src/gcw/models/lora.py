from __future__ import annotations

import math
from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcw.config import LoRAConfig


ModuleT = TypeVar("ModuleT", bound=nn.Module)


class LoRALinear(nn.Linear):
    """A linear layer with a zero-initialized low-rank residual."""

    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None)
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
        adapted = F.linear(F.linear(F.dropout(x, self.dropout, self.training), self.lora_a), self.lora_b)
        return base + adapted * self.scaling


def apply_lora(module: ModuleT, config: LoRAConfig | None) -> ModuleT:
    """Attach LoRA only to explicitly named projection modules."""
    if config is None:
        return module
    targets = set(config.targets)
    for parent in module.modules():
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                setattr(parent, name, LoRALinear(child, config))
    return module
