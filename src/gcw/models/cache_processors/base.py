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

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._calls += 1
        return super().__call__(*args, **kwargs)

    def pop_call_count(self) -> int:
        calls = self._calls
        self._calls = 0
        return calls

    @abstractmethod
    def forward(self, cache: Any, *, select_all: bool = False) -> Any:
        """Process selected cache state and return the resulting cache."""
