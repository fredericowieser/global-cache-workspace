"""Legacy whole-model compilation policy.

Regional backends are preferred for cache-processor models. This module keeps
whole-model compilation explicit and preserves the existing environment
contract for architectures that do not use regional compilation.
"""

from __future__ import annotations

import os
from typing import cast

import torch


def whole_model_compile_requested() -> bool:
    """Return whether ``GCW_COMPILE`` requests any whole-model compilation."""
    value = os.environ.get("GCW_COMPILE", "0").strip().lower()
    return value not in {"", "0", "false", "no", "none", "off"}


def maybe_compile_whole_model(model: torch.nn.Module) -> torch.nn.Module:
    """Apply the explicitly requested whole-model compiler or return the model."""
    mode = os.environ.get("GCW_COMPILE", "0").strip().lower()
    if mode in {"", "0", "false", "no", "none", "off"}:
        return model
    if mode == "default":
        return cast(torch.nn.Module, torch.compile(model))
    if mode == "reduce-overhead":
        return cast(
            torch.nn.Module, torch.compile(model, mode="reduce-overhead")
        )
    raise ValueError("GCW_COMPILE must be 0, default, or reduce-overhead")
