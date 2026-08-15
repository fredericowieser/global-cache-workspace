"""Read-only runtime telemetry for optimized model regions."""

from __future__ import annotations

from typing import cast

import torch

from gcw.models.cache_processors.base import cache_processor_recomputation_id
from gcw.performance.longshot_cache import pop_longshot_full_history_metrics


def model_runtime_backends(model: torch.nn.Module, device: torch.device) -> dict[str, str]:
    """Collect effective optional runtime kernels from the unwrapped backbone."""
    model = _unwrap(model)
    model = getattr(model, "backbone", model)
    for module in model.modules():
        effective_backends = getattr(module, "effective_backends", None)
        if effective_backends is not None:
            return effective_backends(device)
    return {}


def cache_processor_metrics(model: torch.nn.Module) -> dict[str, float]:
    """Pop processor iteration metrics without making telemetry model state."""
    if cache_processor_recomputation_id() is not None:
        return {}
    model = _unwrap(model)
    model = getattr(model, "_orig_mod", model)
    processor = getattr(model, "cache_processor", None)
    metrics = pop_longshot_full_history_metrics(processor) if processor is not None else {}
    block = getattr(processor, "block", None)
    pop_metrics = getattr(block, "pop_iteration_metrics", None)
    if pop_metrics is not None:
        metrics.update(pop_metrics())
    return metrics


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return cast(torch.nn.Module, model.module) if hasattr(model, "module") else model
