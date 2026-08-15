"""Restoring cache-event instrumentation for adaptive parity runs.

The instrumentation lives outside :mod:`gcw` because it observes execution; it
must never become part of model semantics.  Every patched symbol is restored by
an :class:`~contextlib.ExitStack`, including exceptional exits.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence
from unittest.mock import patch

import torch

from gcw.models.cache_processors import icarus as icarus_module
from gcw.models.cache_processors import longshot as longshot_module
from gcw.models.cache_processors import standard as standard_module
from gcw.models.cache_processors.base import cache_processor_recomputation_id
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import WindowedCache


@dataclass(frozen=True)
class TraceScope:
    """Stable row identities for one physical forward/backward call."""

    step: int
    group: int
    row_keys: tuple[str, ...]


@dataclass(frozen=True)
class CacheEvent:
    """One selected-index or cache-write event for one source row."""

    event_key: str
    operation: str
    step: int
    group: int
    pair_key: str
    row_key: str
    ordinal: int
    phase: str
    recomputation_of: str | None
    destinations: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


_ACTIVE_SCOPE: ContextVar[TraceScope | None] = ContextVar("gcw_parity_trace_scope", default=None)


def pair_key(step: int, pair_index: int) -> str:
    return f"step-{step:04d}/pair-{pair_index:02d}"


def row_key(step: int, pair_index: int, row_in_pair: int) -> str:
    if row_in_pair not in (0, 1):
        raise ValueError("row_in_pair must be zero or one")
    return f"{pair_key(step, pair_index)}/row-{row_in_pair}"


def row_keys_for_pairs(step: int, pair_start: int, pair_stop: int) -> tuple[str, ...]:
    if pair_start < 0 or pair_stop <= pair_start:
        raise ValueError("pair range must be non-empty and non-negative")
    return tuple(
        row_key(step, pair_index, row_in_pair)
        for pair_index in range(pair_start, pair_stop)
        for row_in_pair in (0, 1)
    )


@contextmanager
def active_trace_scope(scope: TraceScope) -> Iterator[None]:
    """Bind stable row identities across forward and checkpoint backward."""
    token = _ACTIVE_SCOPE.set(scope)
    try:
        yield
    finally:
        _ACTIVE_SCOPE.reset(token)


class CacheEventRecorder:
    """Collect cache events while separating primary work from recomputation."""

    def __init__(self) -> None:
        self.events: list[CacheEvent] = []
        self._ordinals: defaultdict[tuple[str, str, str], int] = defaultdict(int)

    def record(self, operation: str, destinations: torch.Tensor) -> None:
        scope = _ACTIVE_SCOPE.get()
        if scope is None:
            raise RuntimeError("cache event occurred outside an active parity trace scope")
        if destinations.ndim != 2 or destinations.shape[0] != len(scope.row_keys):
            raise RuntimeError(
                f"{operation} destinations must have shape ({len(scope.row_keys)}, selected); "
                f"got {tuple(destinations.shape)}"
            )
        recomputation = cache_processor_recomputation_id() is not None
        phase = "recomputation" if recomputation else "primary"
        cpu_destinations = destinations.detach().to(device="cpu", dtype=torch.long)
        for batch_index, stable_row_key in enumerate(scope.row_keys):
            counter_key = (phase, operation, stable_row_key)
            ordinal = self._ordinals[counter_key]
            self._ordinals[counter_key] += 1
            primary_key = f"{stable_row_key}/{operation}-{ordinal:04d}"
            event_key = primary_key if not recomputation else f"{primary_key}/recompute"
            stable_pair_key = stable_row_key.rsplit("/", 1)[0]
            self.events.append(
                CacheEvent(
                    event_key=event_key,
                    operation=operation,
                    step=scope.step,
                    group=scope.group,
                    pair_key=stable_pair_key,
                    row_key=stable_row_key,
                    ordinal=ordinal,
                    phase=phase,
                    recomputation_of=primary_key if recomputation else None,
                    destinations=tuple(int(value) for value in cpu_destinations[batch_index].tolist()),
                )
            )

    def primary_semantics(self) -> dict[str, tuple[int, ...]]:
        """Return grouping-independent semantic events keyed by source row."""
        return {
            event.event_key: event.destinations
            for event in self.events
            if event.phase == "primary"
        }

    def recomputation_alignment(self) -> dict[str, bool]:
        """Report whether every replay exactly matches its primary event."""
        primary = self.primary_semantics()
        return {
            event.event_key: primary.get(event.recomputation_of or "") == event.destinations
            for event in self.events
            if event.phase == "recomputation"
        }


@contextmanager
def trace_cache_events(recorder: CacheEventRecorder) -> Iterator[CacheEventRecorder]:
    """Patch maintained selection/write boundaries and restore all symbols.

    Standard BT/BGT paths are fully covered.  Icarus and Longshot latest-cache
    write helpers are also observed.  The wrappers intentionally capture only
    integer destinations; tensor values remain in the normal autograd graph.
    """

    def wrap_state_selection(original):
        def wrapped(cache, *args, **kwargs):
            indices = original(cache, *args, **kwargs)
            recorder.record("select", indices)
            return indices

        return wrapped

    def wrap_indices_result(original):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            indices = result[0] if isinstance(result, tuple) else result
            recorder.record("select", indices)
            return result

        return wrapped

    def wrap_scatter(original, operation: str):
        def wrapped(cache, indices, *args, **kwargs):
            recorder.record(operation, indices)
            return original(cache, indices, *args, **kwargs)

        return wrapped

    with ExitStack() as stack:
        for cache_type in (CacheState, StaticCacheState, WindowedCache):
            stack.enter_context(
                patch.object(cache_type, "select_indices", wrap_state_selection(cache_type.select_indices))
            )
        stack.enter_context(
            patch.object(
                standard_module,
                "select_kv_norm_indices",
                wrap_indices_result(standard_module.select_kv_norm_indices),
            )
        )
        for module in (standard_module,):
            stack.enter_context(
                patch.object(module, "scatter_cache_layers", wrap_scatter(module.scatter_cache_layers, "write_kv"))
            )
            stack.enter_context(
                patch.object(module, "scatter_cache_values", wrap_scatter(module.scatter_cache_values, "write_v"))
            )
        stack.enter_context(
            patch.object(
                icarus_module,
                "_scatter_latest",
                wrap_scatter(icarus_module._scatter_latest, "write_latest_kv"),
            )
        )
        stack.enter_context(
            patch.object(
                icarus_module,
                "_scatter_latest_values",
                wrap_scatter(icarus_module._scatter_latest_values, "write_latest_v"),
            )
        )
        stack.enter_context(
            patch.object(
                longshot_module,
                "_scatter_latest",
                wrap_scatter(longshot_module._scatter_latest, "write_latest_kv"),
            )
        )
        stack.enter_context(
            patch.object(
                longshot_module,
                "_scatter_latest_values",
                wrap_scatter(longshot_module._scatter_latest_values, "write_latest_v"),
            )
        )
        yield recorder


def events_to_json(events: Sequence[CacheEvent]) -> list[dict[str, Any]]:
    return [event.to_json() for event in events]
