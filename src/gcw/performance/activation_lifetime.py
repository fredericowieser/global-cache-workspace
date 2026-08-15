"""Explicit activation liveness plans for optimized executor candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActivationUse:
    """One named value, its production phase, and every later use phase."""

    name: str
    producer: str
    consumers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.producer:
            raise ValueError("activation names and producer phases must be nonempty")
        if len(set(self.consumers)) != len(self.consumers):
            raise ValueError("activation consumers must be unique")


@dataclass(frozen=True)
class ActivationLifetime:
    name: str
    produced_at: int
    last_used_at: int


@dataclass(frozen=True)
class ActivationLifetimePlan:
    """Closed phase order and deterministic post-phase release schedule."""

    phases: tuple[str, ...]
    lifetimes: tuple[ActivationLifetime, ...]
    releases: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not self.phases or self.phases != tuple(dict.fromkeys(self.phases)):
            raise ValueError("activation phases must be nonempty and unique")
        names = tuple(lifetime.name for lifetime in self.lifetimes)
        if names != tuple(sorted(set(names))):
            raise ValueError("activation lifetimes must be unique and sorted by name")
        if len(self.releases) != len(self.phases):
            raise ValueError("activation releases must align with every phase")
        released = tuple(name for group in self.releases for name in group)
        if tuple(sorted(released)) != names or len(set(released)) != len(released):
            raise ValueError("activation release schedule must cover each lifetime exactly once")

    def release_after(self, phase: str) -> tuple[str, ...]:
        try:
            index = self.phases.index(phase)
        except ValueError as exc:
            raise ValueError(f"unknown activation phase: {phase}") from exc
        return self.releases[index]


def plan_activation_lifetimes(
    phases: tuple[str, ...],
    uses: tuple[ActivationUse, ...],
) -> ActivationLifetimePlan:
    """Build a closed release schedule without inspecting mutable runtime values."""
    if not phases or phases != tuple(dict.fromkeys(phases)):
        raise ValueError("activation phases must be nonempty and unique")
    phase_indices = {name: index for index, name in enumerate(phases)}
    if tuple(use.name for use in uses) != tuple(sorted(use.name for use in uses)):
        raise ValueError("activation uses must be sorted by unique name")
    if len({use.name for use in uses}) != len(uses):
        raise ValueError("activation uses must have unique names")
    lifetimes: list[ActivationLifetime] = []
    release_groups: list[list[str]] = [[] for _ in phases]
    for use in uses:
        if use.producer not in phase_indices or any(consumer not in phase_indices for consumer in use.consumers):
            raise ValueError(f"activation {use.name!r} references an unknown phase")
        producer = phase_indices[use.producer]
        consumer_indices = tuple(phase_indices[consumer] for consumer in use.consumers)
        if consumer_indices != tuple(sorted(consumer_indices)):
            raise ValueError(f"activation {use.name!r} consumers must follow phase order")
        if any(consumer < producer for consumer in consumer_indices):
            raise ValueError(f"activation {use.name!r} is consumed before production")
        last_use = max((producer, *consumer_indices))
        lifetimes.append(ActivationLifetime(use.name, producer, last_use))
        release_groups[last_use].append(use.name)
    return ActivationLifetimePlan(
        phases,
        tuple(lifetimes),
        tuple(tuple(sorted(group)) for group in release_groups),
    )


class ActivationLifetimeRuntime:
    """Own references according to one plan and release them at exact phases."""

    def __init__(self, plan: ActivationLifetimePlan) -> None:
        self.plan = plan
        self._values: dict[str, Any] = {}
        self._next_phase = 0
        self._closed = False

    def register(self, name: str, value: Any) -> None:
        if self._closed:
            raise RuntimeError("closed activation runtime cannot register values")
        lifetime = next((item for item in self.plan.lifetimes if item.name == name), None)
        if lifetime is None:
            raise ValueError(f"activation {name!r} is absent from the closed plan")
        if lifetime.produced_at != self._next_phase:
            raise RuntimeError(f"activation {name!r} was registered outside its producer phase")
        if name in self._values:
            raise RuntimeError(f"activation {name!r} was registered twice")
        self._values[name] = value

    def get(self, name: str) -> Any:
        if name not in self._values:
            raise RuntimeError(f"activation {name!r} is unavailable or already released")
        return self._values[name]

    def finish_phase(self, phase: str) -> tuple[str, ...]:
        if self._closed or self._next_phase >= len(self.plan.phases):
            raise RuntimeError("activation runtime has no phase to finish")
        expected = self.plan.phases[self._next_phase]
        if phase != expected:
            raise RuntimeError(f"activation phase order differs: expected {expected!r}, got {phase!r}")
        required = {
            lifetime.name
            for lifetime in self.plan.lifetimes
            if lifetime.produced_at <= self._next_phase <= lifetime.last_used_at
        }
        missing = tuple(sorted(required.difference(self._values)))
        if missing:
            raise RuntimeError(f"activation phase {phase!r} is missing live values: {missing}")
        released = self.plan.releases[self._next_phase]
        for name in released:
            del self._values[name]
        self._next_phase += 1
        if self._next_phase == len(self.plan.phases):
            if self._values:
                raise RuntimeError("activation plan finished with unreleased values")
            self._closed = True
        return released

    @property
    def live_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._values))
