from __future__ import annotations

import pytest
import torch

from gcw.qualification.timing import (
    QualificationTimingError,
    TimingProtocol,
    summarize_timings,
    time_complete_operations,
)


def test_timing_summary_uses_raw_samples_and_interpolation() -> None:
    summary = summarize_timings([4.0, 1.0, 3.0, 2.0])
    assert summary.raw_seconds == (4.0, 1.0, 3.0, 2.0)
    assert summary.median_seconds == 2.5
    assert summary.p10_seconds == 1.3
    assert summary.p90_seconds == 3.7


def test_complete_operation_timing_excludes_warmups() -> None:
    calls: list[tuple[int, bool]] = []

    def operation(ordinal: int, warmup: bool) -> None:
        calls.append((ordinal, warmup))
        sum(value * value for value in range(100))

    result = time_complete_operations(
        operation,
        protocol=TimingProtocol("cpu", warmup_steps=2, measured_steps=3),
        device=torch.device("cpu"),
    )
    assert calls == [(0, True), (1, True), (0, False), (1, False), (2, False)]
    assert len(result.raw_seconds) == 3
    assert result.peak_allocated_bytes == result.peak_reserved_bytes == 0


def test_timing_rejects_instrumentation_and_invalid_samples() -> None:
    with pytest.raises(QualificationTimingError, match="instrumented"):
        time_complete_operations(
            lambda _ordinal, _warmup: None,
            protocol=TimingProtocol("cpu", warmup_steps=0, measured_steps=1),
            device=torch.device("cpu"),
            instrumentation_active=True,
        )
    with pytest.raises(QualificationTimingError, match="finite and positive"):
        summarize_timings([0.0])
    with pytest.raises(QualificationTimingError, match="positive measured"):
        TimingProtocol("cpu", warmup_steps=0, measured_steps=0)
