"""Lazy, default-off Triton execution helpers.

Importing this package never imports the external :mod:`triton` package or
initializes CUDA.  Individual backends load their JIT kernels only after an
explicit CUDA dispatch request.
"""

from gcw.performance.triton.cache_movement import (
    gather_per_cache_head_kv,
    gather_shared_kv,
    scatter_shared_kv,
)
from gcw.performance.triton.processor import fused_processor_silu_mul
from gcw.performance.triton.runtime import (
    TritonRuntimeIdentity,
    require_triton_runtime,
)

__all__ = (
    "TritonRuntimeIdentity",
    "gather_per_cache_head_kv",
    "fused_processor_silu_mul",
    "gather_shared_kv",
    "require_triton_runtime",
    "scatter_shared_kv",
)
