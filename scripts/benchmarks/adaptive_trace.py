"""Restoring BT/BGT cache-event instrumentation for adaptive parity runs.

Events are identified by the source row, operation, and the cache length at
which the processor runs.  This is independent of physical batching and of
checkpoint replay order.  Device-to-CPU copies make this instrumentation
unsuitable for timing.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Iterator, Sequence
from unittest.mock import patch

import torch

from gcw.models.cache_processors import standard as standard_module
from gcw.models.cache_processors.base import cache_processor_recomputation_id
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import WindowedCache


@dataclass(frozen=True)
class TraceScope:
    """Source-row metadata for one physical forward/backward call."""

    step: int
    group: int
    row_keys: tuple[str, ...]
    source_padded_widths: tuple[int, ...]
    process_every: int
    checkpoint_expected: bool = False

    def __post_init__(self) -> None:
        if len(self.row_keys) != len(self.source_padded_widths):
            raise ValueError("trace row keys and source padded widths must align")
        if not self.row_keys or any(width <= 0 for width in self.source_padded_widths):
            raise ValueError("trace scopes require rows with positive source padded widths")
        if self.process_every <= 0:
            raise ValueError("trace process_every must be positive")


@dataclass(frozen=True)
class CacheEvent:
    """One selected-index or cache-write event for one source row."""

    event_key: str
    stable_event_id: str
    operation: str
    step: int
    group: int
    pair_key: str
    row_key: str
    source_padded_width: int
    cache_boundary: int
    window_index: int
    semantic: bool
    padding_only: bool
    phase: str
    replay_index: int
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
        row_key(step, pair_index, row_in_pair) for pair_index in range(pair_start, pair_stop) for row_in_pair in (0, 1)
    )


def row_widths_for_pairs(widths: Sequence[int], pair_start: int, pair_stop: int) -> tuple[int, ...]:
    selected = widths[pair_start:pair_stop]
    if len(selected) != pair_stop - pair_start:
        raise ValueError("pair width range is outside the reservation")
    return tuple(width for width in selected for _ in (0, 1))


@contextmanager
def active_trace_scope(scope: TraceScope) -> Iterator[None]:
    """Bind stable source-row metadata across forward and checkpoint replay."""
    token = _ACTIVE_SCOPE.set(scope)
    try:
        yield
    finally:
        _ACTIVE_SCOPE.reset(token)


class CacheEventRecorder:
    """Collect semantic and grouped-padding BT/BGT cache events."""

    def __init__(self, required_operations: Sequence[str]) -> None:
        operations = tuple(required_operations)
        if set(operations) not in ({"select", "write_kv"}, {"select", "write_v"}):
            raise ValueError("BT/BGT trace coverage requires select plus exactly one standard write operation")
        self.required_operations = operations
        self.events: list[CacheEvent] = []
        self._phase_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._expected_checkpoint: dict[str, bool] = {}
        self._checkpoint_expected_scope_count = 0
        self._active_scope: TraceScope | None = None

    @contextmanager
    def bind_scope(self, scope: TraceScope) -> Iterator[None]:
        """Bind recorder-local scope across autograd-engine context boundaries."""
        if self._active_scope is not None:
            raise RuntimeError("parity trace recorder scope is already active")
        self._active_scope = scope
        try:
            yield
        finally:
            self._active_scope = None

    def expect_scope(self, scope: TraceScope) -> None:
        """Register every semantic operation expected from this source scope."""
        if scope.checkpoint_expected:
            self._checkpoint_expected_scope_count += 1
        for stable_row_key, width in zip(scope.row_keys, scope.source_padded_widths, strict=True):
            for boundary in range(scope.process_every, width, scope.process_every):
                for operation in self.required_operations:
                    identity = _stable_event_id(stable_row_key, boundary, operation)
                    previous = self._expected_checkpoint.setdefault(identity, scope.checkpoint_expected)
                    if previous != scope.checkpoint_expected:
                        raise RuntimeError(f"conflicting checkpoint expectation for {identity}")

    def record(self, operation: str, cache: object, destinations: torch.Tensor) -> None:
        scope = self._active_scope or _ACTIVE_SCOPE.get()
        if scope is None:
            raise RuntimeError("cache event occurred outside an active parity trace scope")
        if operation not in self.required_operations:
            raise RuntimeError(f"unexpected traced BT/BGT operation: {operation}")
        if destinations.ndim != 2 or destinations.shape[0] != len(scope.row_keys):
            raise RuntimeError(
                f"{operation} destinations must have shape ({len(scope.row_keys)}, selected); "
                f"got {tuple(destinations.shape)}"
            )
        boundary = _cache_boundary(cache)
        if boundary < 0:
            raise RuntimeError("cache boundary must be non-negative")
        recomputation = cache_processor_recomputation_id() is not None
        phase = "recomputation" if recomputation else "primary"
        # This synchronization is deliberate correctness instrumentation and is
        # why an instrumented execution must never be used for timing.
        cpu_destinations = destinations.detach().to(device="cpu", dtype=torch.long)
        for batch_index, (stable_row_key, source_width) in enumerate(
            zip(scope.row_keys, scope.source_padded_widths, strict=True)
        ):
            semantic = boundary < source_width
            stable_id = _stable_event_id(stable_row_key, boundary, operation)
            count_key = (phase, stable_id)
            replay_index = self._phase_counts[count_key]
            self._phase_counts[count_key] += 1
            raw = (int(value) for value in cpu_destinations[batch_index].tolist())
            trimmed = tuple(value for value in raw if semantic and 0 <= value < source_width)
            suffix = "replay" if recomputation else "primary"
            event_key = f"{stable_id}/{suffix}-{replay_index:02d}"
            self.events.append(
                CacheEvent(
                    event_key=event_key,
                    stable_event_id=stable_id,
                    operation=operation,
                    step=scope.step,
                    group=scope.group,
                    pair_key=stable_row_key.rsplit("/", 1)[0],
                    row_key=stable_row_key,
                    source_padded_width=source_width,
                    cache_boundary=boundary,
                    window_index=boundary // scope.process_every,
                    semantic=semantic,
                    padding_only=not semantic,
                    phase=phase,
                    replay_index=replay_index,
                    recomputation_of=stable_id if recomputation else None,
                    destinations=trimmed,
                )
            )

    def semantic_primary(self) -> dict[str, tuple[int, ...]]:
        grouped: defaultdict[str, list[tuple[int, ...]]] = defaultdict(list)
        for event in self.events:
            if event.semantic and event.phase == "primary":
                grouped[event.stable_event_id].append(event.destinations)
        return {identity: values[0] for identity, values in grouped.items() if len(values) == 1}

    def coverage(self) -> dict[str, Any]:
        semantic = [event for event in self.events if event.semantic]
        padding = [event for event in self.events if event.padding_only]
        primary_counts = Counter(event.stable_event_id for event in semantic if event.phase == "primary")
        observed_ids = set(primary_counts)
        expected_ids = set(self._expected_checkpoint)
        missing = sorted(expected_ids - observed_ids)
        unexpected = sorted(observed_ids - expected_ids)
        duplicate = {identity: count for identity, count in sorted(primary_counts.items()) if count != 1}
        operation_counts = Counter(event.operation for event in semantic if event.phase == "primary")
        padding_valid = all(
            not event.destinations and event.cache_boundary >= event.source_padded_width for event in padding
        )
        complete = bool(expected_ids) and not missing and not unexpected and not duplicate and padding_valid
        return {
            "required_operations": list(self.required_operations),
            "expected_semantic_primary_count": len(expected_ids),
            "actual_semantic_primary_count": sum(primary_counts.values()),
            "semantic_primary_by_operation": dict(sorted(operation_counts.items())),
            "padding_only_count": len(padding),
            "padding_only_primary_count": sum(event.phase == "primary" for event in padding),
            "padding_only_replay_count": sum(event.phase == "recomputation" for event in padding),
            "padding_trimmed": padding_valid,
            "missing_event_ids": missing,
            "unexpected_event_ids": unexpected,
            "duplicate_event_ids": duplicate,
            "complete": complete,
        }

    def recomputation_report(self) -> dict[str, Any]:
        semantic = [event for event in self.events if event.semantic]
        primary: defaultdict[str, list[tuple[int, ...]]] = defaultdict(list)
        replay: defaultdict[str, list[tuple[int, ...]]] = defaultdict(list)
        for event in semantic:
            target = primary if event.phase == "primary" else replay
            target[event.stable_event_id].append(event.destinations)
        identities = sorted(set(self._expected_checkpoint) | set(primary) | set(replay))
        details: dict[str, Any] = {}
        checkpointing_expected = self._checkpoint_expected_scope_count > 0
        passed = not checkpointing_expected or bool(identities)
        expected_total = 0
        actual_total = 0
        for identity in identities:
            checkpoint_expected = self._expected_checkpoint.get(identity, False)
            expected_replays = len(primary[identity]) if checkpoint_expected else 0
            actual_replays = len(replay[identity])
            values_equal = bool(primary[identity]) and all(value == primary[identity][0] for value in replay[identity])
            identity_passed = (
                len(primary[identity]) == 1
                and actual_replays == expected_replays
                and (actual_replays == 0 or values_equal)
            )
            expected_total += expected_replays
            actual_total += actual_replays
            passed = passed and identity_passed
            details[identity] = {
                "checkpoint_expected": checkpoint_expected,
                "primary_count": len(primary[identity]),
                "expected_replay_count": expected_replays,
                "actual_replay_count": actual_replays,
                "destinations_equal": values_equal if actual_replays else None,
                "passed": identity_passed,
            }
        return {
            "checkpointing_expected": checkpointing_expected,
            "checkpoint_expected_scope_count": self._checkpoint_expected_scope_count,
            "expected_replay_count": expected_total,
            "actual_replay_count": actual_total,
            "events": details,
            "passed": passed,
        }


def _stable_event_id(row: str, boundary: int, operation: str) -> str:
    return f"{row}/boundary-{boundary:08d}/{operation}"


def _cache_boundary(cache: object) -> int:
    boundary = getattr(cache, "seq_len", None)
    if not isinstance(boundary, int):
        raise RuntimeError(f"traced cache {type(cache).__name__} does not expose an integer seq_len")
    return boundary


@contextmanager
def trace_cache_events(recorder: CacheEventRecorder) -> Iterator[CacheEventRecorder]:
    """Patch complete standard BT/BGT selection/write boundaries and restore them."""

    def wrap_state_selection(original):
        def wrapped(cache, *args, **kwargs):
            indices = original(cache, *args, **kwargs)
            recorder.record("select", cache, indices)
            return indices

        return wrapped

    def wrap_indices_result(original):
        def wrapped(cache, *args, **kwargs):
            result = original(cache, *args, **kwargs)
            indices = result[0] if isinstance(result, tuple) else result
            recorder.record("select", cache, indices)
            return result

        return wrapped

    def wrap_scatter(original, operation: str):
        def wrapped(cache, indices, *args, **kwargs):
            recorder.record(operation, cache, indices)
            return original(cache, indices, *args, **kwargs)

        return wrapped

    def wrap_triton_scatter(original):
        def wrapped(old_keys, old_values, indices, update, *args, **kwargs):
            boundary = SimpleNamespace(seq_len=old_keys.shape[3])
            recorder.record("write_kv", boundary, indices)
            return original(old_keys, old_values, indices, update, *args, **kwargs)

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
        stack.enter_context(
            patch.object(
                standard_module,
                "scatter_cache_layers",
                wrap_scatter(standard_module.scatter_cache_layers, "write_kv"),
            )
        )
        stack.enter_context(
            patch.object(
                standard_module,
                "scatter_cache_values",
                wrap_scatter(standard_module.scatter_cache_values, "write_v"),
            )
        )
        stack.enter_context(
            patch.object(
                standard_module,
                "triton_scatter_shared_kv",
                wrap_triton_scatter(standard_module.triton_scatter_shared_kv),
            )
        )
        yield recorder


def events_to_json(events: Sequence[CacheEvent]) -> list[dict[str, Any]]:
    return [event.to_json() for event in events]
