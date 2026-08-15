"""Read-only runtime telemetry for optimized model regions."""

from __future__ import annotations

import torch


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
    model = _unwrap(model)
    model = getattr(model, "_orig_mod", model)
    processor = getattr(model, "cache_processor", None)
    block = getattr(processor, "block", None)
    pop_metrics = getattr(block, "pop_iteration_metrics", None)
    if pop_metrics is None:
        return {}
    return pop_metrics()


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model
