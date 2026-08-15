from __future__ import annotations

from collections.abc import Callable
import os
from threading import Lock
from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.nn as nn

from gcw.performance._compile import require_torch_compile
from gcw.performance.prepared_bt import (
    PreparedBatchedCacheProcessorParameters,
    run_prepared_batched_cache_processors,
)
from gcw.provenance import canonical_json_sha256

if TYPE_CHECKING:
    from gcw.config import ModelConfig


BTProcessorBackend = Literal["eager", "compile"]
CacheProcessorBackend = Literal["eager", "compile"]
PreparedProcessorRunner = Callable[
    [PreparedBatchedCacheProcessorParameters, torch.Tensor, torch.Tensor],
    torch.Tensor,
]
BGTProcessorRunner = Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]

_BT_BACKEND_ENV = "GCW_BT_PROCESSOR_BACKEND"
_CACHE_BACKEND_ENV = "GCW_CACHE_PROCESSOR_BACKEND"
_COMPILE_OPTIONS = {
    "backend": "inductor",
    "dynamic": False,
    "fullgraph": True,
}
_compiled_runner: PreparedProcessorRunner | None = None
_compiled_runner_lock = Lock()
_compiled_bgt_transform_runner: BGTProcessorRunner | None = None
_compiled_bgt_scaled_update_runner: BGTProcessorRunner | None = None
_compiled_bgt_runner_lock = Lock()


def resolve_bt_processor_backend() -> BTProcessorBackend:
    """Resolve the explicit backend for the prepared BT processor region."""
    requested = os.environ.get(_BT_BACKEND_ENV, "eager").strip().lower()
    if requested not in {"eager", "compile"}:
        raise ValueError(f"{_BT_BACKEND_ENV} must be 'eager' or 'compile'")
    if requested == "compile":
        require_torch_compile(f"{_BT_BACKEND_ENV}=compile")
    return requested  # type: ignore[return-value]


def bt_processor_backend_metadata(backend: BTProcessorBackend | None = None) -> dict[str, object]:
    """Return provenance and a stable fingerprint for the selected BT region backend."""
    selected = resolve_bt_processor_backend() if backend is None else backend
    # Keep the v1 region identifier stable so code-only module moves do not invalidate exact resume.
    execution = {
        "schema": "gcw.bt_processor_backend.v1",
        "effective_backend": selected,
        "region": "gcw.models.cache_processors.ops.run_prepared_batched_cache_processors",
        "compile_options": dict(_COMPILE_OPTIONS) if selected == "compile" else None,
        "torch_version": str(torch.__version__),
    }
    return {
        **execution,
        "requested_via": _BT_BACKEND_ENV,
        "requested_value": os.environ.get(_BT_BACKEND_ENV, "eager"),
        "implementation_module": "gcw.performance.backends.cache_processors",
        "implementation_symbols": [
            "gcw.performance.backends.cache_processors.run_prepared_bt_processor_region",
            "gcw.performance.prepared_bt.run_prepared_batched_cache_processors",
        ],
        "fingerprint_sha256": canonical_json_sha256(execution),
    }


def run_prepared_bt_processor_region(
    backend: BTProcessorBackend,
    prepared: PreparedBatchedCacheProcessorParameters,
    x: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Dispatch only the pure prepared BT tensor region; compiled failures are fatal."""
    if backend == "eager":
        return run_prepared_batched_cache_processors(prepared, x, positions)
    if backend != "compile":
        raise ValueError(f"unsupported BT processor backend: {backend}")

    runner = _get_compiled_runner()
    try:
        return runner(prepared, x, positions)
    except Exception as error:
        raise RuntimeError(
            f"compiled BT processor region failed; eager fallback is disabled when {_BT_BACKEND_ENV}=compile"
        ) from error


def _get_compiled_runner() -> PreparedProcessorRunner:
    global _compiled_runner
    if _compiled_runner is not None:
        return _compiled_runner
    with _compiled_runner_lock:
        if _compiled_runner is None:
            compiler = require_torch_compile(f"{_BT_BACKEND_ENV}=compile")
            try:
                _compiled_runner = cast(
                    PreparedProcessorRunner,
                    compiler(run_prepared_batched_cache_processors, **_COMPILE_OPTIONS),
                )
            except Exception as error:
                raise RuntimeError(
                    "failed to create the compiled BT processor region; eager fallback is disabled"
                ) from error
    assert _compiled_runner is not None
    return _compiled_runner


def resolve_cache_processor_backend(model_config: ModelConfig) -> CacheProcessorBackend:
    """Resolve the regional backend for the non-BT cache-processor region."""
    requested = os.environ.get(_CACHE_BACKEND_ENV, "eager").strip().lower()
    if requested not in {"eager", "compile"}:
        raise ValueError(f"{_CACHE_BACKEND_ENV} must be 'eager' or 'compile'")
    if requested == "eager":
        return "eager"
    if model_config.variant == "bt":
        raise RuntimeError(
            f"{_CACHE_BACKEND_ENV}=compile does not support BT; use "
            f"{_BT_BACKEND_ENV}=compile for BT's prepared backend"
        )
    if model_config.variant != "bgt":
        raise RuntimeError(
            f"{_CACHE_BACKEND_ENV}=compile supports BGT, Flow-BGT, and Diffusion-BGT only; "
            f"{model_config.variant} is explicitly unsupported"
        )
    diffusion = model_config.diffusion_processor
    if diffusion is not None and diffusion.noise_scale > 0.0:
        raise RuntimeError(
            f"{_CACHE_BACKEND_ENV}=compile rejects Diffusion-BGT with noise_scale>0; "
            "fullgraph noise equivalence is not established"
        )
    require_torch_compile(f"{_CACHE_BACKEND_ENV}=compile")
    return "compile"


def cache_processor_backend_metadata(
    model_config: ModelConfig,
    backend: CacheProcessorBackend | None = None,
) -> dict[str, object]:
    """Return provenance and a stable fingerprint for the BGT regional backend."""
    selected = resolve_cache_processor_backend(model_config) if backend is None else backend
    diffusion = model_config.diffusion_processor
    family = "diffusion_bgt" if diffusion is not None else ("flow_bgt" if model_config.flow_processor else "bgt")
    # Keep the v1 region identifiers stable so code-only module moves do not invalidate exact resume.
    execution = {
        "schema": "gcw.cache_processor_backend.v1",
        "effective_backend": selected,
        "architecture": family,
        "regions": [
            "gcw.models.cache_processors.backend._run_bgt_transform",
            "gcw.models.cache_processors.backend._run_bgt_scaled_update",
        ],
        "compile_options": dict(_COMPILE_OPTIONS) if selected == "compile" else None,
        "diffusion_noise_policy": (
            "fullgraph_compile_requires_noise_scale_zero" if diffusion is not None else None
        ),
        "torch_version": str(torch.__version__),
    }
    return {
        **execution,
        "requested_via": _CACHE_BACKEND_ENV,
        "requested_value": os.environ.get(_CACHE_BACKEND_ENV, "eager"),
        "implementation_module": "gcw.performance.backends.cache_processors",
        "implementation_symbols": [
            "gcw.performance.backends.cache_processors._run_bgt_transform",
            "gcw.performance.backends.cache_processors._run_bgt_scaled_update",
        ],
        "fingerprint_sha256": canonical_json_sha256(execution),
    }


def run_bgt_processor_region(
    backend: CacheProcessorBackend,
    block: nn.Module,
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    scaled_update: bool = False,
) -> torch.Tensor:
    """Dispatch only the gathered BGT tensor transform; compiled failures are fatal."""
    function = _run_bgt_scaled_update if scaled_update else _run_bgt_transform
    if backend == "eager":
        return function(block, x, positions)
    if backend != "compile":
        raise ValueError(f"unsupported cache processor backend: {backend}")
    runner = _get_compiled_bgt_runner(scaled_update=scaled_update)
    try:
        return runner(block, x, positions)
    except Exception as error:
        raise RuntimeError(
            f"compiled BGT processor region failed; eager fallback is disabled when {_CACHE_BACKEND_ENV}=compile"
        ) from error


def _run_bgt_transform(block: nn.Module, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return block(x, positions)


def _run_bgt_scaled_update(block: nn.Module, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    scaled_update = getattr(block, "scaled_update")
    return scaled_update(x, positions)


def _get_compiled_bgt_runner(*, scaled_update: bool) -> BGTProcessorRunner:
    global _compiled_bgt_scaled_update_runner, _compiled_bgt_transform_runner
    runner = _compiled_bgt_scaled_update_runner if scaled_update else _compiled_bgt_transform_runner
    if runner is not None:
        return runner
    function = _run_bgt_scaled_update if scaled_update else _run_bgt_transform
    with _compiled_bgt_runner_lock:
        runner = _compiled_bgt_scaled_update_runner if scaled_update else _compiled_bgt_transform_runner
        if runner is None:
            compiler = require_torch_compile(f"{_CACHE_BACKEND_ENV}=compile")
            try:
                runner = cast(BGTProcessorRunner, compiler(function, **_COMPILE_OPTIONS))
            except Exception as error:
                raise RuntimeError(
                    "failed to create the compiled BGT processor region; eager fallback is disabled"
                ) from error
            if scaled_update:
                _compiled_bgt_scaled_update_runner = runner
            else:
                _compiled_bgt_transform_runner = runner
    return runner
