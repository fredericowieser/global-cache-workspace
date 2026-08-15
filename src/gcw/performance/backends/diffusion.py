"""Explicit regional backend for noisy task-directed Diffusion-BGT.

Noise is always sampled by :class:`TaskDirectedDiffusion` before this module is
called. The optional compiled region therefore contains only the deterministic
recurrence and its field evaluations.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from threading import Lock
from typing import Any, Literal, cast

import torch
import torch.nn as nn

from gcw.models.cache_processors.diffusion import DiffusionField, TaskDirectedDiffusion
from gcw.performance._compile import (
    is_non_reentrant_checkpoint_early_stop,
    require_torch_compile,
)
from gcw.performance.config import NoisyDiffusionBackendConfig
from gcw.provenance import canonical_json_sha256

NoisyDiffusionRecurrenceBackend = Literal["eager", "compile"]
BoundDiffusionRecurrenceRunner = Callable[
    [torch.Tensor, torch.Tensor, DiffusionField],
    torch.Tensor,
]

_COMPILE_OPTIONS = {
    "backend": "inductor",
    "dynamic": False,
    "fullgraph": True,
}
_RUNNER_ATTRIBUTE = "_gcw_noisy_diffusion_compiled_recurrence_runner"
_COMPILE_COUNT_ATTRIBUTE = "_gcw_noisy_diffusion_compile_count"
_DISPATCH_COUNT_ATTRIBUTE = "_gcw_noisy_diffusion_dispatch_count"
_compiled_recurrence_lock = Lock()


def configure_noisy_diffusion_recurrence_backend(
    model: nn.Module,
    model_config: object,
    policy: NoisyDiffusionBackendConfig,
) -> NoisyDiffusionRecurrenceBackend:
    """Apply a state-neutral recurrence backend after model construction."""
    selected = policy.recurrence_backend
    modules = [module for module in model.modules() if isinstance(module, TaskDirectedDiffusion)]
    if selected == "compile":
        variant = getattr(model_config, "variant", None)
        diffusion = getattr(model_config, "diffusion_processor", None)
        processor = getattr(model_config, "cache_processor", None)
        noise_scale = getattr(diffusion, "noise_scale", 0.0)
        attention_dropout = getattr(processor, "attention_dropout", 0.0)
        if variant != "bgt" or diffusion is None or noise_scale <= 0.0:
            raise RuntimeError(
                "compiled noisy-Diffusion recurrence requires BGT with diffusion noise_scale>0"
            )
        if attention_dropout != 0.0:
            raise RuntimeError(
                "compiled noisy-Diffusion recurrence requires cache_processor.attention_dropout=0"
            )
        if len(modules) != 1:
            raise RuntimeError(
                "compiled noisy-Diffusion recurrence requires exactly one TaskDirectedDiffusion module"
            )
        require_torch_compile("performance.noisy_diffusion.recurrence_backend=compile")
    for module in modules:
        module.recurrence_backend = selected
        if selected == "eager":
            module.__dict__.pop(_RUNNER_ATTRIBUTE, None)
            module.__dict__.pop(_COMPILE_COUNT_ATTRIBUTE, None)
            module.__dict__.pop(_DISPATCH_COUNT_ATTRIBUTE, None)
    return selected


def noisy_diffusion_recurrence_backend_metadata(
    model_config: object,
    policy: NoisyDiffusionBackendConfig,
) -> dict[str, object]:
    """Return authenticated provenance for the noisy-Diffusion recurrence boundary."""
    selected = policy.recurrence_backend
    diffusion = getattr(model_config, "diffusion_processor", None)
    execution = {
        "schema": "gcw.noisy_diffusion_recurrence_backend.v1",
        "effective_backend": selected,
        "architecture": "noisy_diffusion_bgt" if diffusion is not None else None,
        "region": (
            "gcw.models.cache_processors.diffusion."
            "TaskDirectedDiffusion.deterministic_recurrence"
        ),
        "noise_boundary": "eager_torch_randn_like_before_recurrence_dispatch",
        "runner_ownership": "per_task_directed_diffusion_module",
        "compile_options": dict(_COMPILE_OPTIONS) if selected == "compile" else None,
        "torch_version": str(torch.__version__),
    }
    return {
        **execution,
        "configured_via": "performance.noisy_diffusion.recurrence_backend",
        "implementation_module": "gcw.performance.backends.diffusion",
        "implementation_symbol": (
            "gcw.performance.backends.diffusion.run_noisy_diffusion_recurrence"
        ),
        "fingerprint_sha256": canonical_json_sha256(execution),
    }


def noisy_diffusion_compile_telemetry(diffusion: TaskDirectedDiffusion) -> dict[str, int | bool]:
    """Return state-neutral runner and process graph counters for CUDA smoke evidence."""
    dynamo: Any = getattr(torch, "_dynamo", None)
    utils: Any = getattr(dynamo, "utils", None)
    counters: Any = getattr(utils, "counters", {})

    def counter(group: str, name: str) -> int:
        values = counters.get(group, {}) if hasattr(counters, "get") else {}
        return int(values.get(name, 0)) if hasattr(values, "get") else 0

    return {
        "runner_owned": _RUNNER_ATTRIBUTE in diffusion.__dict__,
        "runner_compile_count": int(diffusion.__dict__.get(_COMPILE_COUNT_ATTRIBUTE, 0)),
        "runner_dispatch_count": int(diffusion.__dict__.get(_DISPATCH_COUNT_ATTRIBUTE, 0)),
        "dynamo_unique_graphs": counter("stats", "unique_graphs"),
        "dynamo_graph_breaks": sum(counters.get("graph_break", {}).values())
        if hasattr(counters, "get")
        else 0,
        "dynamo_recompiles": sum(counters.get("recompile", {}).values())
        if hasattr(counters, "get")
        else 0,
    }


def run_noisy_diffusion_recurrence(
    backend: str,
    diffusion: TaskDirectedDiffusion,
    state: torch.Tensor,
    bottleneck: torch.Tensor,
    field: DiffusionField,
) -> torch.Tensor:
    """Dispatch an already-noised state; compiled failures never fall back."""
    if backend == "eager":
        return diffusion.deterministic_recurrence(state, bottleneck, field)
    if backend != "compile":
        raise ValueError(f"unsupported noisy-Diffusion recurrence backend: {backend}")
    runner = _get_compiled_recurrence_runner(diffusion)
    object.__setattr__(
        diffusion,
        _DISPATCH_COUNT_ATTRIBUTE,
        int(diffusion.__dict__.get(_DISPATCH_COUNT_ATTRIBUTE, 0)) + 1,
    )
    try:
        return runner(state, bottleneck, field)
    except Exception as error:
        if is_non_reentrant_checkpoint_early_stop(error):
            raise
        raise RuntimeError(
            "compiled noisy-Diffusion recurrence failed; eager fallback is disabled"
        ) from error


def _run_bound_deterministic_recurrence(
    diffusion: TaskDirectedDiffusion,
    state: torch.Tensor,
    bottleneck: torch.Tensor,
    field: DiffusionField,
) -> torch.Tensor:
    return diffusion.deterministic_recurrence(state, bottleneck, field)


def _get_compiled_recurrence_runner(
    diffusion: TaskDirectedDiffusion,
) -> BoundDiffusionRecurrenceRunner:
    runner = cast(
        BoundDiffusionRecurrenceRunner | None,
        diffusion.__dict__.get(_RUNNER_ATTRIBUTE),
    )
    if runner is not None:
        return runner
    with _compiled_recurrence_lock:
        runner = cast(
            BoundDiffusionRecurrenceRunner | None,
            diffusion.__dict__.get(_RUNNER_ATTRIBUTE),
        )
        if runner is None:
            compiler = require_torch_compile(
                "performance.noisy_diffusion.recurrence_backend=compile"
            )
            try:
                runner = cast(
                    BoundDiffusionRecurrenceRunner,
                    compiler(
                        partial(_run_bound_deterministic_recurrence, diffusion),
                        **_COMPILE_OPTIONS,
                    ),
                )
            except Exception as error:
                raise RuntimeError(
                    "failed to create compiled noisy-Diffusion recurrence; "
                    "eager fallback is disabled"
                ) from error
            # Bypass nn.Module registration: runners are execution state, never model state.
            object.__setattr__(diffusion, _RUNNER_ATTRIBUTE, runner)
            object.__setattr__(diffusion, _COMPILE_COUNT_ATTRIBUTE, 1)
            object.__setattr__(diffusion, _DISPATCH_COUNT_ATTRIBUTE, 0)
    return runner
