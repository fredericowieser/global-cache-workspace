"""Fail-closed lazy admission for handwritten Triton kernels."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from importlib import metadata
from types import ModuleType

import torch

_SUPPORTED_TRITON_VERSION = "3.6.0"


@dataclass(frozen=True)
class TritonRuntimeIdentity:
    """Runtime identity bound to qualification and performance evidence."""

    torch_version: str
    cuda_runtime: str
    triton_version: str
    compute_capability: tuple[int, int]
    device_name: str


def require_triton_runtime(device: torch.device) -> TritonRuntimeIdentity:
    """Require the exact CUDA/Triton runtime supported by the initial kernels."""
    if device.type != "cuda":
        raise RuntimeError("GCW Triton kernels require a CUDA tensor")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GCW Triton kernels were requested but CUDA is unavailable"
        )
    index = (
        device.index
        if device.index is not None
        else torch.cuda.current_device()
    )
    return _require_triton_runtime_cached(index)


@lru_cache(maxsize=None)
def _require_triton_runtime_cached(index: int) -> TritonRuntimeIdentity:
    capability = torch.cuda.get_device_capability(index)
    if capability != (9, 0):
        raise RuntimeError(
            f"GCW Triton kernels currently require NVIDIA compute capability 9.0 (H100/H200); found {capability}"
        )
    try:
        version = metadata.version("triton")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("GCW Triton kernels require triton==3.6.0") from exc
    if version != _SUPPORTED_TRITON_VERSION:
        raise RuntimeError(
            f"GCW Triton kernels require triton=={_SUPPORTED_TRITON_VERSION}; found {version}"
        )
    return TritonRuntimeIdentity(
        torch_version=str(torch.__version__),
        cuda_runtime=str(torch.version.cuda),
        triton_version=version,
        compute_capability=capability,
        device_name=torch.cuda.get_device_name(index),
    )


def load_kernel_module(module: str, device: torch.device) -> ModuleType:
    """Validate the runtime once, then return one cached JIT-kernel module."""
    if device.type != "cuda":
        raise RuntimeError("GCW Triton kernels require a CUDA tensor")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GCW Triton kernels were requested but CUDA is unavailable"
        )
    index = (
        device.index
        if device.index is not None
        else torch.cuda.current_device()
    )
    return _load_kernel_module_cached(module, index)


@lru_cache(maxsize=None)
def _load_kernel_module_cached(module: str, index: int) -> ModuleType:
    _require_triton_runtime_cached(index)
    if not module.startswith("gcw.performance.triton._"):
        raise ValueError(
            "Triton kernel modules must be private gcw.performance.triton modules"
        )
    try:
        return import_module(module)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"failed to import requested Triton kernel module {module!r}"
        ) from exc
