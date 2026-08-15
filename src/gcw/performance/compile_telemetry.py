"""Opt-in acceptance telemetry for regionally compiled model execution.

The recorder is deliberately absent from normal training. Acceptance tests and
benchmark canaries may attach one to a backbone to retain cold-call timings,
input specializations, and compiler graph counts without changing model state
or exact-resume fingerprints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, TypeVar

import torch
import torch.nn as nn


CompileRegion = Literal["initial_full_window", "full_window_with_past"]
_Result = TypeVar("_Result")
_RECORDER_ATTRIBUTE = "_gcw_compile_acceptance_recorder"


@dataclass
class _RegionEvidence:
    calls: int = 0
    signatures: set[tuple[int, ...]] = field(default_factory=set)
    cold_call_seconds: dict[tuple[int, ...], float] = field(default_factory=dict)
    graph_count: int | None = None
    recompile_count: int | None = None


class CompileAcceptanceRecorder:
    """Collect explicit compile evidence around an otherwise unchanged runner."""

    def __init__(self, *, expected_static_batch_rows: tuple[int, ...] = (16, 8, 4, 2)) -> None:
        if not expected_static_batch_rows or any(rows <= 0 for rows in expected_static_batch_rows):
            raise ValueError("expected static batch rows must contain positive integers")
        if len(set(expected_static_batch_rows)) != len(expected_static_batch_rows):
            raise ValueError("expected static batch rows must be unique")
        self.expected_static_batch_rows = expected_static_batch_rows
        self._regions = {
            "initial_full_window": _RegionEvidence(),
            "full_window_with_past": _RegionEvidence(),
        }

    def record_call(
        self,
        region: CompileRegion,
        signature: tuple[int, ...],
        device: torch.device,
        operation: Callable[[], _Result],
    ) -> _Result:
        """Time one compiled call and retain its first-call cost per shape."""
        evidence = self._regions[region]
        is_cold_shape = signature not in evidence.signatures
        _synchronize(device)
        started = perf_counter()
        result = operation()
        _synchronize(device)
        elapsed = perf_counter() - started
        evidence.calls += 1
        evidence.signatures.add(signature)
        if is_cold_shape:
            evidence.cold_call_seconds[signature] = elapsed
        return result

    def record_graph_counts(
        self,
        region: CompileRegion,
        *,
        graph_count: int,
        recompile_count: int,
    ) -> None:
        """Attach counts from a Dynamo/Inductor acceptance counter."""
        if graph_count < 0 or recompile_count < 0:
            raise ValueError("compile graph counts must be non-negative")
        if recompile_count > max(0, graph_count - 1):
            raise ValueError("recompile count cannot exceed graph_count - 1")
        evidence = self._regions[region]
        evidence.graph_count = graph_count
        evidence.recompile_count = recompile_count

    def snapshot(self) -> dict[str, object]:
        """Return JSON-compatible acceptance evidence without mutating counters."""
        regions: dict[str, object] = {}
        for name, evidence in self._regions.items():
            regions[name] = {
                "calls": evidence.calls,
                "observed_signatures": [list(signature) for signature in sorted(evidence.signatures)],
                "cold_call_seconds": {
                    ",".join(str(value) for value in signature): seconds
                    for signature, seconds in sorted(evidence.cold_call_seconds.items())
                },
                "graph_count": evidence.graph_count,
                "recompile_count": evidence.recompile_count,
            }
        return {
            "schema": "gcw.compile_acceptance_telemetry.v1",
            "expected_static_batch_rows": list(self.expected_static_batch_rows),
            "expected_initial_static_graphs": len(self.expected_static_batch_rows),
            "expected_initial_shape_recompiles": len(self.expected_static_batch_rows) - 1,
            "expected_past_batch_specializations": len(self.expected_static_batch_rows),
            "expected_past_shape_recompiles": len(self.expected_static_batch_rows) - 1,
            "dynamic_past_length_within_batch_specialization": True,
            "regions": regions,
        }


def install_compile_acceptance_recorder(
    model: nn.Module,
    *,
    expected_static_batch_rows: tuple[int, ...] = (16, 8, 4, 2),
) -> CompileAcceptanceRecorder:
    """Attach opt-in telemetry to one unwrapped backbone instance."""
    backbone = getattr(model, "backbone", model)
    recorder = CompileAcceptanceRecorder(expected_static_batch_rows=expected_static_batch_rows)
    object.__setattr__(backbone, _RECORDER_ATTRIBUTE, recorder)
    return recorder


def compile_acceptance_recorder(backbone: nn.Module) -> CompileAcceptanceRecorder | None:
    """Return the recorder attached to this exact backbone, if any."""
    value = backbone.__dict__.get(_RECORDER_ATTRIBUTE)
    return value if isinstance(value, CompileAcceptanceRecorder) else None


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
