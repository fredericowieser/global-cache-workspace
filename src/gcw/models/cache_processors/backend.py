from __future__ import annotations

from collections.abc import Callable
import os
from threading import Lock
from typing import Literal

import torch

from gcw.models.cache_processors.ops import (
    PreparedBatchedCacheProcessorParameters,
    run_prepared_batched_cache_processors,
)
from gcw.provenance import canonical_json_sha256


BTProcessorBackend = Literal["eager", "compile"]
PreparedProcessorRunner = Callable[
    [PreparedBatchedCacheProcessorParameters, torch.Tensor, torch.Tensor],
    torch.Tensor,
]

_BACKEND_ENV = "GCW_BT_PROCESSOR_BACKEND"
_COMPILE_OPTIONS = {
    "backend": "inductor",
    "dynamic": False,
    "fullgraph": True,
}
_compiled_runner: PreparedProcessorRunner | None = None
_compiled_runner_lock = Lock()


def resolve_bt_processor_backend() -> BTProcessorBackend:
    """Resolve the explicit backend for the prepared BT processor region."""
    requested = os.environ.get(_BACKEND_ENV, "eager").strip().lower()
    if requested not in {"eager", "compile"}:
        raise ValueError(f"{_BACKEND_ENV} must be 'eager' or 'compile'")
    if requested == "compile":
        compiler = getattr(torch, "compile", None)
        if not callable(compiler):
            raise RuntimeError(f"{_BACKEND_ENV}=compile requires torch.compile")
        dynamo = getattr(torch, "_dynamo", None)
        dynamo_config = getattr(dynamo, "config", None)
        if bool(getattr(dynamo_config, "suppress_errors", False)):
            raise RuntimeError(f"{_BACKEND_ENV}=compile requires torch._dynamo.config.suppress_errors=False")
    return requested  # type: ignore[return-value]


def bt_processor_backend_metadata(backend: BTProcessorBackend | None = None) -> dict[str, object]:
    """Return provenance and a stable fingerprint for the selected BT region backend."""
    selected = resolve_bt_processor_backend() if backend is None else backend
    execution = {
        "schema": "gcw.bt_processor_backend.v1",
        "effective_backend": selected,
        "region": "gcw.models.cache_processors.ops.run_prepared_batched_cache_processors",
        "compile_options": dict(_COMPILE_OPTIONS) if selected == "compile" else None,
        "torch_version": str(torch.__version__),
    }
    return {
        **execution,
        "requested_via": _BACKEND_ENV,
        "requested_value": os.environ.get(_BACKEND_ENV, "eager"),
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
            f"compiled BT processor region failed; eager fallback is disabled when {_BACKEND_ENV}=compile"
        ) from error


def _get_compiled_runner() -> PreparedProcessorRunner:
    global _compiled_runner
    if _compiled_runner is not None:
        return _compiled_runner
    with _compiled_runner_lock:
        if _compiled_runner is None:
            compiler = getattr(torch, "compile", None)
            if not callable(compiler):
                raise RuntimeError(f"{_BACKEND_ENV}=compile requires torch.compile")
            try:
                _compiled_runner = compiler(run_prepared_batched_cache_processors, **_COMPILE_OPTIONS)
            except Exception as error:
                raise RuntimeError(
                    "failed to create the compiled BT processor region; eager fallback is disabled"
                ) from error
    return _compiled_runner
