"""Shared fail-closed validation for explicit regional Torch compilation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def require_torch_compile(requested_by: str) -> Callable[..., Any]:
    """Return ``torch.compile`` only when fallback suppression is disabled."""
    compiler = getattr(torch, "compile", None)
    if not callable(compiler):
        raise RuntimeError(f"{requested_by} requires torch.compile")
    dynamo = getattr(torch, "_dynamo", None)
    dynamo_config = getattr(dynamo, "config", None)
    if bool(getattr(dynamo_config, "suppress_errors", False)):
        raise RuntimeError(f"{requested_by} requires torch._dynamo.config.suppress_errors=False")
    return compiler


def is_non_reentrant_checkpoint_early_stop(error: BaseException) -> bool:
    """Identify PyTorch checkpoint replay's private early-stop control signal.

    Compiled fail-closed boundaries must let this exact exception escape. Wrapping
    it breaks non-reentrant checkpoint replay; treating any other exception this
    way would weaken backend failure handling.
    """
    error_type = type(error)
    return (
        error_type.__module__ == "torch.utils.checkpoint"
        and error_type.__name__ == "_StopRecomputationError"
    )
