"""Explicit, opt-in training performance policies and implementations.

The performance package owns execution choices that may change throughput or
memory use without changing model state ownership or scientific objectives.
Its public surface is intentionally small; architecture-specific compiled
regions remain in dedicated modules rather than a generic backend registry.
"""

from gcw.performance.compile_telemetry import (
    CompileAcceptanceRecorder,
    install_compile_acceptance_recorder,
)
from gcw.performance.config import (
    AdaptiveBatchingConfig,
    PhysicalBatchBandConfig,
    PerformanceConfig,
    ReferenceLossConfig,
)

__all__ = (
    "AdaptiveBatchingConfig",
    "CompileAcceptanceRecorder",
    "PerformanceConfig",
    "PhysicalBatchBandConfig",
    "ReferenceLossConfig",
    "install_compile_acceptance_recorder",
)
