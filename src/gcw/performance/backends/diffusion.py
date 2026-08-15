"""Explicit regional backend for noisy task-directed Diffusion-BGT.

Noise is always sampled by :class:`TaskDirectedDiffusion` before this module is
called. The optional compiled region therefore contains only the deterministic
recurrence and its field evaluations.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Literal, cast

import torch
import torch.nn as nn

from gcw.models.cache_processors.diffusion import DiffusionField, TaskDirectedDiffusion
from gcw.performance._compile import require_torch_compile
from gcw.performance.config import NoisyDiffusionBackendConfig
from gcw.provenance import canonical_json_sha256

NoisyDiffusionRecurrenceBackend = Literal["eager", "compile"]
DiffusionRecurrenceRunner = Callable[
    [TaskDirectedDiffusion, torch.Tensor, torch.Tensor, DiffusionField],
    torch.Tensor,
]

_COMPILE_OPTIONS = {
    "backend": "inductor",
    "dynamic": False,
    "fullgraph": True,
}
_compiled_recurrence_runner: DiffusionRecurrenceRunner | None = None
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
        noise_scale = getattr(diffusion, "noise_scale", 0.0)
        if variant != "bgt" or diffusion is None or noise_scale <= 0.0:
            raise RuntimeError(
                "compiled noisy-Diffusion recurrence requires BGT with diffusion noise_scale>0"
            )
        if len(modules) != 1:
            raise RuntimeError(
                "compiled noisy-Diffusion recurrence requires exactly one TaskDirectedDiffusion module"
            )
        require_torch_compile("performance.noisy_diffusion.recurrence_backend=compile")
    for module in modules:
        module.recurrence_backend = selected
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
    runner = _get_compiled_recurrence_runner()
    try:
        return runner(diffusion, state, bottleneck, field)
    except Exception as error:
        raise RuntimeError(
            "compiled noisy-Diffusion recurrence failed; eager fallback is disabled"
        ) from error


def _run_deterministic_recurrence(
    diffusion: TaskDirectedDiffusion,
    state: torch.Tensor,
    bottleneck: torch.Tensor,
    field: DiffusionField,
) -> torch.Tensor:
    return diffusion.deterministic_recurrence(state, bottleneck, field)


def _get_compiled_recurrence_runner() -> DiffusionRecurrenceRunner:
    global _compiled_recurrence_runner
    if _compiled_recurrence_runner is not None:
        return _compiled_recurrence_runner
    with _compiled_recurrence_lock:
        if _compiled_recurrence_runner is None:
            compiler = require_torch_compile(
                "performance.noisy_diffusion.recurrence_backend=compile"
            )
            try:
                _compiled_recurrence_runner = cast(
                    DiffusionRecurrenceRunner,
                    compiler(_run_deterministic_recurrence, **_COMPILE_OPTIONS),
                )
            except Exception as error:
                raise RuntimeError(
                    "failed to create compiled noisy-Diffusion recurrence; "
                    "eager fallback is disabled"
                ) from error
    assert _compiled_recurrence_runner is not None
    return _compiled_recurrence_runner
