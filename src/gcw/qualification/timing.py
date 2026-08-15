"""Matched complete-operation timing outside qualification instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import torch

from gcw.qualification.resources import AcceleratorClass, capture_resource_identity


class QualificationTimingError(ValueError):
    """Raised when a timing protocol or sample set is invalid."""


@dataclass(frozen=True)
class TimingProtocol:
    expected_accelerator: AcceleratorClass
    warmup_steps: int
    measured_steps: int

    def __post_init__(self) -> None:
        if self.warmup_steps < 0 or self.measured_steps <= 0:
            raise QualificationTimingError("timing requires nonnegative warmup and positive measured steps")


@dataclass(frozen=True)
class TimingSummary:
    raw_seconds: tuple[float, ...]
    median_seconds: float
    p10_seconds: float
    p90_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def time_complete_operations(
    operation: Callable[[int, bool], None],
    *,
    protocol: TimingProtocol,
    device: torch.device,
    instrumentation_active: bool = False,
) -> TimingSummary:
    """Time full operations with warmup and bookkeeping outside samples.

    The callback receives its ordinal and a boolean indicating warmup. It must
    use distinct scientific inputs where the execution policy requires them.
    """
    if instrumentation_active:
        raise QualificationTimingError("instrumented qualification execution cannot be timed")
    capture_resource_identity(
        expected_accelerator=protocol.expected_accelerator,
        require_single_cuda=device.type == "cuda",
        require_gpu_authority=False,
    )
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        device = torch.device("cuda", index)
    for ordinal in range(protocol.warmup_steps):
        operation(ordinal, True)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    for ordinal in range(protocol.measured_steps):
        _synchronize(device)
        started = time.perf_counter()
        operation(ordinal, False)
        _synchronize(device)
        elapsed = time.perf_counter() - started
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise QualificationTimingError("timing operation produced a nonpositive or nonfinite sample")
        samples.append(elapsed)
    summary = summarize_timings(samples)
    return TimingSummary(
        raw_seconds=summary.raw_seconds,
        median_seconds=summary.median_seconds,
        p10_seconds=summary.p10_seconds,
        p90_seconds=summary.p90_seconds,
        peak_allocated_bytes=(int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0),
        peak_reserved_bytes=(int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0),
    )


def summarize_timings(values: list[float] | tuple[float, ...]) -> TimingSummary:
    """Validate and summarize raw timing values with linear interpolation."""
    samples = tuple(values)
    if not samples or any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise QualificationTimingError("timing samples must be finite and positive")
    return TimingSummary(
        raw_seconds=samples,
        median_seconds=_percentile(samples, 0.5),
        p10_seconds=_percentile(samples, 0.1),
        p90_seconds=_percentile(samples, 0.9),
        peak_allocated_bytes=0,
        peak_reserved_bytes=0,
    )


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
