from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class CacheProcessorBase(nn.Module, ABC):
    """Contract for modules that read and rewrite an explicit cache state.

    Concrete processors preserve the cache domain and batch size. They may
    mutate an in-place static cache or return a replacement functional cache,
    according to the input cache contract.
    """

    @abstractmethod
    def forward(self, cache: Any, *, select_all: bool = False) -> Any:
        """Process selected cache state and return the resulting cache."""
