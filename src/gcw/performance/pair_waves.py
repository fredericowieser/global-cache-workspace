"""Exact fixed-batch-two forward overlap with source-ordered backward.

The executor never collates pairs or changes their tensors.  It overlaps only
independent forwards within one contiguous wave, then launches ordinary
``Tensor.backward`` calls in original pair order with explicit CUDA-event
serialization.  It is default-off and has no eager fallback.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from gcw.data.training import TrainingBatch
from gcw.performance.adjacent_pairs import PAIRS_PER_STEP

PairLoss = Callable[[int, TrainingBatch], torch.Tensor]
PairContext = Callable[[int, TrainingBatch], AbstractContextManager[Any]]


@dataclass(frozen=True)
class PairWave:
    """One contiguous source-order wave."""

    pair_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.pair_indices:
            raise ValueError("pair wave must not be empty")
        expected = tuple(range(self.pair_indices[0], self.pair_indices[-1] + 1))
        if self.pair_indices != expected:
            raise ValueError("pair wave indices must be contiguous and ordered")


@dataclass(frozen=True)
class PairWavePlan:
    """Closed plan covering one 32-pair optimizer step."""

    wave_size: int
    waves: tuple[PairWave, ...]

    def __post_init__(self) -> None:
        if self.wave_size not in {1, 2, 4}:
            raise ValueError("pair wave_size must be 1, 2, or 4")
        flattened = tuple(index for wave in self.waves for index in wave.pair_indices)
        if flattened != tuple(range(PAIRS_PER_STEP)):
            raise ValueError(f"pair waves must cover exactly {PAIRS_PER_STEP} source pairs")
        if any(len(wave.pair_indices) > self.wave_size for wave in self.waves):
            raise ValueError("pair wave exceeds configured wave_size")


def plan_pair_waves(wave_size: int) -> PairWavePlan:
    """Return contiguous waves over exactly 32 pair indices."""
    if wave_size not in {1, 2, 4}:
        raise ValueError("pair wave_size must be 1, 2, or 4")
    return PairWavePlan(
        wave_size,
        tuple(
            PairWave(tuple(range(start, min(start + wave_size, PAIRS_PER_STEP))))
            for start in range(0, PAIRS_PER_STEP, wave_size)
        ),
    )


@dataclass(frozen=True)
class PairWaveResult:
    """Differentiable execution outputs and physical scheduling counters."""

    pair_losses: tuple[torch.Tensor, ...]
    plan: PairWavePlan
    forward_replays: int
    backward_replays: int


class PairWaveRuntime:
    """Reusable CUDA streams and events for one fixed wave width."""

    def __init__(self, device: torch.device, wave_size: int) -> None:
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            device = torch.device("cuda", index)
        self.device = device
        self.plan = plan_pair_waves(wave_size)
        if wave_size > 1 and device.type != "cuda":
            raise ValueError("overlapped pair waves require CUDA")
        self._closed = False
        self._streams = (
            tuple(torch.cuda.Stream(device=device) for _ in range(wave_size))
            if device.type == "cuda"
            else ()
        )
        self._ready = torch.cuda.Event() if device.type == "cuda" else None
        self._forward_done = (
            tuple(torch.cuda.Event() for _ in range(wave_size)) if device.type == "cuda" else ()
        )
        self._backward_done = (
            tuple(torch.cuda.Event() for _ in range(wave_size)) if device.type == "cuda" else ()
        )

    def execute(
        self,
        pairs: Sequence[TrainingBatch],
        pair_loss: PairLoss,
        parameters: Sequence[torch.nn.Parameter],
        *,
        normalizer: int = PAIRS_PER_STEP,
        pair_context: PairContext | None = None,
    ) -> PairWaveResult:
        """Execute one complete step's pair losses and ordinary backwards."""
        if self._closed:
            raise RuntimeError("closed pair-wave runtime cannot execute")
        pair_tuple = tuple(pairs)
        if len(pair_tuple) != PAIRS_PER_STEP:
            raise ValueError(f"pair-wave execution requires exactly {PAIRS_PER_STEP} pairs")
        if normalizer != PAIRS_PER_STEP:
            raise ValueError(f"pair-wave normalization must remain exactly 1/{PAIRS_PER_STEP}")
        parameter_tuple = tuple(parameters)
        if not parameter_tuple:
            raise ValueError("pair-wave execution requires trainable parameters")
        if any(parameter.grad is not None for parameter in parameter_tuple):
            raise ValueError("pair-wave execution requires unset parameter gradients")
        _validate_pair_devices(pair_tuple, parameter_tuple, self.device)
        context_factory = pair_context or (lambda _index, _pair: nullcontext())
        losses: list[torch.Tensor | None] = [None] * PAIRS_PER_STEP
        forward_replays = 0
        backward_replays = 0
        try:
            for wave in self.plan.waves:
                if self.device.type == "cuda":
                    self._execute_cuda_wave(
                        wave,
                        pair_tuple,
                        pair_loss,
                        losses,
                        normalizer,
                        context_factory,
                    )
                else:
                    pair_index = wave.pair_indices[0]
                    with context_factory(pair_index, pair_tuple[pair_index]):
                        loss = _scalar_loss(pair_loss(pair_index, pair_tuple[pair_index]), pair_index)
                    losses[pair_index] = loss.detach()
                    with context_factory(pair_index, pair_tuple[pair_index]):
                        (loss / normalizer).backward()
                forward_replays += len(wave.pair_indices)
                backward_replays += len(wave.pair_indices)
        except BaseException:
            self._quiesce()
            for parameter in parameter_tuple:
                parameter.grad = None
            raise
        if any(loss is None for loss in losses):
            raise RuntimeError("pair-wave execution did not produce every pair loss")
        return PairWaveResult(
            pair_losses=tuple(loss for loss in losses if loss is not None),
            plan=self.plan,
            forward_replays=forward_replays,
            backward_replays=backward_replays,
        )

    def _execute_cuda_wave(
        self,
        wave: PairWave,
        pairs: tuple[TrainingBatch, ...],
        pair_loss: PairLoss,
        losses: list[torch.Tensor | None],
        normalizer: int,
        context_factory: PairContext,
    ) -> None:
        authority = torch.cuda.current_stream(self.device)
        assert self._ready is not None
        self._ready.record(authority)
        live_losses: list[torch.Tensor] = []
        for lane, pair_index in enumerate(wave.pair_indices):
            stream = self._streams[lane]
            stream.wait_event(self._ready)
            with torch.cuda.stream(stream), context_factory(pair_index, pairs[pair_index]):
                loss = _scalar_loss(pair_loss(pair_index, pairs[pair_index]), pair_index)
                self._forward_done[lane].record(stream)
            live_losses.append(loss)
            losses[pair_index] = loss.detach()
        for lane in range(len(wave.pair_indices)):
            authority.wait_event(self._forward_done[lane])

        previous: torch.cuda.Event | None = None
        for lane, loss in enumerate(live_losses):
            stream = self._streams[lane]
            if previous is not None:
                stream.wait_event(previous)
            pair_index = wave.pair_indices[lane]
            with torch.cuda.stream(stream), context_factory(pair_index, pairs[pair_index]):
                (loss / normalizer).backward()
                self._backward_done[lane].record(stream)
            previous = self._backward_done[lane]
        assert previous is not None
        previous.synchronize()

    def _quiesce(self) -> None:
        if self.device.type != "cuda":
            return
        for stream in self._streams:
            stream.synchronize()

    def close(self) -> None:
        """Quiesce work and make the runtime permanently unusable."""
        if self._closed:
            return
        self._quiesce()
        self._closed = True

    def __enter__(self) -> PairWaveRuntime:
        if self._closed:
            raise RuntimeError("closed pair-wave runtime cannot be re-entered")
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        self.close()


def _scalar_loss(loss: torch.Tensor, pair_index: int) -> torch.Tensor:
    if loss.ndim != 0 or not loss.is_floating_point():
        raise ValueError(f"pair {pair_index} loss must be one floating scalar")
    return loss


def _validate_pair_devices(
    pairs: tuple[TrainingBatch, ...],
    parameters: tuple[torch.nn.Parameter, ...],
    device: torch.device,
) -> None:
    for pair_index, pair in enumerate(pairs):
        tensors = (pair.x, pair.y, pair.attention_mask)
        if any(tensor is not None and tensor.device != device for tensor in tensors):
            raise ValueError(f"pair {pair_index} tensors are not on the runtime device")
        if pair.x.shape[0] != 2 or pair.y.shape != pair.x.shape:
            raise ValueError(f"pair {pair_index} must remain one physical batch-two pair")
    if any(parameter.device != device for parameter in parameters):
        raise ValueError("pair-wave parameters are not on the runtime device")
