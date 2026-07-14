from __future__ import annotations

import os
from typing import cast

import torch


def maybe_compile(model: torch.nn.Module) -> torch.nn.Module:
    mode = os.environ.get("GCW_COMPILE", "0").strip().lower()
    if mode in {"", "0", "false", "no", "none", "off"}:
        return model
    if mode == "default":
        return cast(torch.nn.Module, torch.compile(model))
    if mode == "reduce-overhead":
        return cast(torch.nn.Module, torch.compile(model, mode="reduce-overhead"))
    raise ValueError("GCW_COMPILE must be 0, default, or reduce-overhead")
