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
