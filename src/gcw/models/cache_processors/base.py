from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any, Callable

import torch.nn as nn

_RECOMPUTATION_ID: ContextVar[object | None] = ContextVar(
    "gcw_cache_processor_recomputation_id", default=None
)
_GRADIENT_CHECKPOINTING_OVERRIDES: ContextVar[dict[str, bool]] = ContextVar(
    "gcw_processor_gradient_checkpointing_overrides", default={}
)


@contextmanager
def processor_gradient_checkpointing_override(
    enabled: bool, *, variant: str = "longshot"
) -> Iterator[None]:
    """Scope a variant's processor-layer checkpoint policy to one forward."""
    token = _GRADIENT_CHECKPOINTING_OVERRIDES.set(
        {**_GRADIENT_CHECKPOINTING_OVERRIDES.get(), variant: enabled}
    )
    try:
        yield
    finally:
        _GRADIENT_CHECKPOINTING_OVERRIDES.reset(token)


def processor_gradient_checkpointing_enabled(
    configured: bool, *, variant: str = "longshot"
) -> bool:
    return _GRADIENT_CHECKPOINTING_OVERRIDES.get().get(variant, configured)


def cache_processor_recomputation():
    """Capture forward checkpoint policy and suppress bookkeeping during replay."""
    overrides = _GRADIENT_CHECKPOINTING_OVERRIDES.get()

    @contextmanager
    def replay():
        token = _RECOMPUTATION_ID.set(object())
        policy_token = _GRADIENT_CHECKPOINTING_OVERRIDES.set(overrides)
        try:
            yield
        finally:
            _GRADIENT_CHECKPOINTING_OVERRIDES.reset(policy_token)
            _RECOMPUTATION_ID.reset(token)

    return replay()


def cache_processor_recomputation_id() -> object | None:
    return _RECOMPUTATION_ID.get()


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
        if cache_processor_recomputation_id() is None:
            self._calls += 1
        return super().__call__(*args, **kwargs)

    def pop_call_count(self) -> int:
        calls = self._calls
        self._calls = 0
        return calls

    @property
    def recurrent_cache_format(self) -> str:
        """Storage requested from the backbone at training window boundaries."""
        return "windowed"

    def prepare_forward_execution(
        self,
    ) -> AbstractContextManager[Callable[..., Any]]:
        """Return the processor callable to use for one language-model forward.

        Most processors need no preparation and use their eager module call.
        Processors with graph-local derived tensors may override this with a
        context that releases those tensors when the forward exits.
        """
        return nullcontext(self)

    @abstractmethod
    def forward(self, cache: Any, *, select_all: bool = False) -> Any:
        """Process selected cache state and return the resulting cache."""
