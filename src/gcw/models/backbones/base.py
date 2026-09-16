from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class Backbone(nn.Module, ABC):
    """Common execution contract for GCW causal language-model backbones.

    A backbone owns token embeddings, transformer layers, cache construction,
    loss computation, and autoregressive generation. Cache-processor models
    compose a backbone rather than reimplementing those operations.
    """

    @abstractmethod
    def init_weights(self, *args: Any, **kwargs: Any) -> None:
        """Initialize all trainable backbone parameters."""

    @abstractmethod
    def get_device(self) -> torch.device:
        """Return the device that owns the backbone token embeddings."""

    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | list[int] | tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens with the reference cache implementation."""

    @abstractmethod
    def generate_fast(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | list[int] | tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens with the preallocated cache implementation."""

    @abstractmethod
    def num_parameters(self, non_embedding: bool = False) -> int:
        """Count model parameters, optionally excluding embeddings and the LM head."""
