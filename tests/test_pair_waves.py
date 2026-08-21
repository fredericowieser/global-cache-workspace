from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from gcw.data.training import TrainingBatch
from gcw.performance.adjacent_pairs import PAIRS_PER_STEP
from gcw.performance.pair_waves import PairWaveRuntime, plan_pair_waves


def _pairs(
    device: torch.device = torch.device("cpu"),
) -> tuple[TrainingBatch, ...]:
    return tuple(
        TrainingBatch(
            x=torch.arange(8, device=device, dtype=torch.float32).reshape(2, 4)
            + index,
            y=torch.full((2, 4), float(index), device=device),
            attention_mask=torch.ones(2, 4, device=device, dtype=torch.bool),
        )
        for index in range(PAIRS_PER_STEP)
    )


def _pair_loss(
    weight: torch.nn.Parameter, _index: int, pair: TrainingBatch
) -> torch.Tensor:
    prediction = pair.x * weight
    return (prediction - pair.y).square().mean()


def test_pair_wave_plan_is_contiguous_and_complete() -> None:
    for width, expected_waves in ((1, 32), (2, 16), (4, 8)):
        plan = plan_pair_waves(width)
        assert len(plan.waves) == expected_waves
        assert tuple(
            index for wave in plan.waves for index in wave.pair_indices
        ) == tuple(range(PAIRS_PER_STEP))
    with pytest.raises(ValueError, match="1, 2, or 4"):
        plan_pair_waves(8)


def test_serial_pair_wave_uses_ordinary_source_ordered_backward() -> None:
    pairs = _pairs()
    reference = torch.nn.Parameter(torch.linspace(-0.2, 0.3, 4))
    candidate = torch.nn.Parameter(reference.detach().clone())
    reference_losses = []
    for index, pair in enumerate(pairs):
        loss = _pair_loss(reference, index, pair)
        reference_losses.append(loss.detach())
        (loss / PAIRS_PER_STEP).backward()

    contexts: list[tuple[str, int]] = []

    @contextmanager
    def pair_context(index: int, _pair: TrainingBatch):
        contexts.append(("enter", index))
        try:
            yield
        finally:
            contexts.append(("exit", index))

    with PairWaveRuntime(torch.device("cpu"), 1) as runtime:
        result = runtime.execute(
            pairs,
            lambda index, pair: _pair_loss(candidate, index, pair),
            (candidate,),
            pair_context=pair_context,
        )

    assert result.forward_replays == result.backward_replays == PAIRS_PER_STEP
    assert torch.equal(
        torch.stack(result.pair_losses), torch.stack(reference_losses)
    )
    assert candidate.grad is not None and reference.grad is not None
    assert torch.equal(candidate.grad, reference.grad)
    assert [entry for entry in contexts if entry[0] == "enter"] == [
        ("enter", index) for index in range(PAIRS_PER_STEP) for _ in range(2)
    ]


def test_pair_wave_failure_clears_partial_gradients_and_closes() -> None:
    pairs = _pairs()
    parameter = torch.nn.Parameter(torch.ones(4))

    def fail_at_pair(index: int, pair: TrainingBatch) -> torch.Tensor:
        if index == 3:
            raise RuntimeError("injected")
        return _pair_loss(parameter, index, pair)

    runtime = PairWaveRuntime(torch.device("cpu"), 1)
    with pytest.raises(RuntimeError, match="injected"):
        runtime.execute(pairs, fail_at_pair, (parameter,))
    assert parameter.grad is None
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.execute(pairs, fail_at_pair, (parameter,))


def test_overlapped_pair_waves_reject_cpu() -> None:
    with pytest.raises(ValueError, match="require CUDA"):
        PairWaveRuntime(torch.device("cpu"), 2)
