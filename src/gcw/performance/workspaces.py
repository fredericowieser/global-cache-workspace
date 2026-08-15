"""Lane-local static tensor arenas with explicit autograd lifetimes."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Mapping

import torch


@dataclass(frozen=True)
class StaticTensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if not self.shape or any(size <= 0 for size in self.shape):
            raise ValueError("static tensor shapes must be nonempty and positive")


@dataclass(frozen=True)
class WorkspaceSignature:
    name: str
    device: str
    tensors: tuple[tuple[str, StaticTensorSpec], ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _spec in self.tensors)
        if not self.name or not names or names != tuple(sorted(set(names))):
            raise ValueError("workspace tensor names must be nonempty, sorted, and unique")


class WorkspaceLease:
    """Exclusive ownership token spanning staging through backward completion."""

    def __init__(self, arena: StaticWorkspaceArena, generation: int) -> None:
        self._arena = arena
        self._generation = generation
        self._released = False

    @property
    def tensors(self) -> Mapping[str, torch.Tensor]:
        if self._released:
            raise RuntimeError("released workspace lease has no tensors")
        return self._arena._leased_tensors(self._generation)

    def stage(self, values: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        if self._released:
            raise RuntimeError("released workspace lease cannot stage inputs")
        return self._arena._stage(self._generation, values)

    def release_after_backward(self) -> None:
        if self._released:
            raise RuntimeError("workspace lease was released twice")
        self._arena._release(self._generation)
        self._released = True

    def __enter__(self) -> WorkspaceLease:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if not self._released:
            self.release_after_backward()


class StaticWorkspaceArena:
    """One fixed-signature arena that cannot be restaged while leased."""

    def __init__(self, signature: WorkspaceSignature) -> None:
        self.signature = signature
        self.device = torch.device(signature.device)
        self._tensors = {
            name: torch.empty(spec.shape, dtype=spec.dtype, device=self.device)
            for name, spec in signature.tensors
        }
        self._lock = threading.RLock()
        self._leased = False
        self._generation = 0

    def acquire(self) -> WorkspaceLease:
        with self._lock:
            if self._leased:
                raise RuntimeError("static workspace is still owned by an active forward/backward")
            self._leased = True
            self._generation += 1
            return WorkspaceLease(self, self._generation)

    def _leased_tensors(self, generation: int) -> Mapping[str, torch.Tensor]:
        with self._lock:
            self._require_generation(generation)
            return dict(self._tensors)

    def _stage(
        self,
        generation: int,
        values: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        with self._lock:
            self._require_generation(generation)
            if set(values) != set(self._tensors):
                raise ValueError("staged workspace tensor names differ from its signature")
            for name, target in self._tensors.items():
                source = values[name]
                if source.shape != target.shape or source.dtype != target.dtype or source.device != target.device:
                    raise ValueError(f"staged tensor {name!r} differs from its workspace signature")
                target.copy_(source)
            return dict(self._tensors)

    def _release(self, generation: int) -> None:
        with self._lock:
            self._require_generation(generation)
            self._leased = False

    def _require_generation(self, generation: int) -> None:
        if not self._leased or generation != self._generation:
            raise RuntimeError("workspace lease is stale or inactive")


class StaticWorkspacePool:
    """Closed signature/lane arena collection."""

    def __init__(self, signatures: tuple[WorkspaceSignature, ...], lanes: int) -> None:
        if lanes <= 0:
            raise ValueError("workspace pool requires positive lanes")
        if not signatures or len(set(signatures)) != len(signatures):
            raise ValueError("workspace signatures must be nonempty and unique")
        self._arenas = {
            (signature, lane): StaticWorkspaceArena(signature)
            for signature in signatures
            for lane in range(lanes)
        }
        self.lanes = lanes

    def acquire(self, signature: WorkspaceSignature, lane: int) -> WorkspaceLease:
        if not 0 <= lane < self.lanes:
            raise ValueError("workspace lane is outside the configured pool")
        try:
            arena = self._arenas[(signature, lane)]
        except KeyError as exc:
            raise RuntimeError("unseen workspace signature is not admitted") from exc
        return arena.acquire()
