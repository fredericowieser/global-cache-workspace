"""Explicit execution backend for the deterministic Icarus tensor region."""

from __future__ import annotations

from collections.abc import Callable
import os
from threading import Lock
from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.nn as nn

from gcw.performance._compile import require_torch_compile
from gcw.provenance import canonical_json_sha256

if TYPE_CHECKING:
    from gcw.config import ModelConfig


IcarusProcessorBackend = Literal["eager", "compile"]
IcarusProcessorRunner = Callable[
    [nn.Module, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]

_ICARUS_BACKEND_ENV = "GCW_ICARUS_PROCESSOR_BACKEND"
_COMPILE_OPTIONS = {
    "backend": "inductor",
    "dynamic": False,
    "fullgraph": True,
}
_compiled_icarus_runner: IcarusProcessorRunner | None = None
_compiled_icarus_runner_lock = Lock()


def resolve_icarus_processor_backend(model_config: ModelConfig) -> IcarusProcessorBackend:
    """Resolve and validate the narrow post-selection Icarus backend."""
    requested = os.environ.get(_ICARUS_BACKEND_ENV, "eager").strip().lower()
    if requested not in {"eager", "compile"}:
        raise ValueError(f"{_ICARUS_BACKEND_ENV} must be 'eager' or 'compile'")
    if requested == "eager":
        return "eager"
    if model_config.variant != "icarus":
        raise RuntimeError(
            f"{_ICARUS_BACKEND_ENV}=compile supports deterministic Icarus only; "
            f"{model_config.variant} is explicitly unsupported"
        )
    processor = model_config.cache_processor
    if processor is None or processor.processor_mode != "standard":
        raise RuntimeError(f"{_ICARUS_BACKEND_ENV}=compile requires Icarus processor_mode='standard'")
    if processor.attention_dropout != 0.0:
        raise RuntimeError(f"{_ICARUS_BACKEND_ENV}=compile requires attention_dropout=0")
    if model_config.flow_processor is not None or model_config.diffusion_processor is not None:
        raise RuntimeError(
            f"{_ICARUS_BACKEND_ENV}=compile supports the ordinary Icarus transform only; "
            "Flow and Diffusion refinements are explicitly unsupported"
        )
    require_torch_compile(f"{_ICARUS_BACKEND_ENV}=compile")
    return "compile"


def icarus_processor_backend_metadata(
    model_config: ModelConfig,
    backend: IcarusProcessorBackend | None = None,
) -> dict[str, object]:
    """Return provenance and an exact-resume fingerprint for Icarus execution."""
    selected = resolve_icarus_processor_backend(model_config) if backend is None else backend
    execution = {
        "schema": "gcw.icarus_processor_backend.v1",
        "effective_backend": selected,
        "architecture": "icarus",
        "region": "gcw.performance.backends.icarus._run_icarus_post_selection",
        "compile_options": dict(_COMPILE_OPTIONS) if selected == "compile" else None,
        "selection_policy": "eager_exact",
        "writeback_policy": "eager_exact",
        "torch_version": str(torch.__version__),
    }
    return {
        **execution,
        "requested_via": _ICARUS_BACKEND_ENV,
        "requested_value": os.environ.get(_ICARUS_BACKEND_ENV, "eager"),
        "implementation_module": "gcw.performance.backends.icarus",
        "implementation_symbols": [
            "gcw.performance.backends.icarus.run_icarus_post_selection_region",
            "gcw.performance.backends.icarus._run_icarus_post_selection",
        ],
        "fingerprint_sha256": canonical_json_sha256(execution),
    }


def run_icarus_post_selection_region(
    backend: IcarusProcessorBackend,
    block: nn.Module,
    x: torch.Tensor,
    positions: torch.Tensor,
    selected_heads: torch.Tensor,
) -> torch.Tensor:
    """Run only the pure transform after eager selection and before eager writeback."""
    if backend == "eager":
        return _run_icarus_post_selection(block, x, positions, selected_heads)
    if backend != "compile":
        raise ValueError(f"unsupported Icarus processor backend: {backend}")
    runner = _get_compiled_icarus_runner()
    try:
        return runner(block, x, positions, selected_heads)
    except Exception as error:
        raise RuntimeError(
            f"compiled Icarus processor region failed; eager fallback is disabled when {_ICARUS_BACKEND_ENV}=compile"
        ) from error


def _run_icarus_post_selection(
    block: nn.Module,
    x: torch.Tensor,
    positions: torch.Tensor,
    selected_heads: torch.Tensor,
) -> torch.Tensor:
    return block(x, positions, selected_heads)


def _get_compiled_icarus_runner() -> IcarusProcessorRunner:
    global _compiled_icarus_runner
    if _compiled_icarus_runner is not None:
        return _compiled_icarus_runner
    with _compiled_icarus_runner_lock:
        if _compiled_icarus_runner is None:
            compiler = require_torch_compile(f"{_ICARUS_BACKEND_ENV}=compile")
            try:
                _compiled_icarus_runner = cast(
                    IcarusProcessorRunner,
                    compiler(_run_icarus_post_selection, **_COMPILE_OPTIONS),
                )
            except Exception as error:
                raise RuntimeError(
                    "failed to create the compiled Icarus processor region; eager fallback is disabled"
                ) from error
    assert _compiled_icarus_runner is not None
    return _compiled_icarus_runner
