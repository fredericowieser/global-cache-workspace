from __future__ import annotations

import os

import pytest
import torch

from gcw.data.training import TrainingBatch
from gcw.performance.adjacent_pairs import PAIRS_PER_STEP
from gcw.performance.pair_waves import PairWaveRuntime

pytestmark = [pytest.mark.gpu, pytest.mark.performance]


def _require_cuda_gate() -> None:
    if os.environ.get("GCW_RUN_PAIR_WAVE_CUDA_TESTS") != "1":
        pytest.skip(
            "set GCW_RUN_PAIR_WAVE_CUDA_TESTS=1 for pair-wave CUDA tests"
        )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("pair-wave CUDA acceptance currently targets H100/H200")


def _pairs() -> tuple[TrainingBatch, ...]:
    generator = torch.Generator(device="cuda").manual_seed(321)
    return tuple(
        TrainingBatch(
            x=torch.randn(2, 64, device="cuda", generator=generator),
            y=torch.randn(2, 64, device="cuda", generator=generator),
            attention_mask=torch.ones(2, 64, device="cuda", dtype=torch.bool),
        )
        for _ in range(PAIRS_PER_STEP)
    )


def _loss(weight: torch.nn.Parameter, pair: TrainingBatch) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = pair.x @ weight
        return (prediction.float() - pair.y).square().mean()


@pytest.mark.parametrize("wave_size", (2, 4))
def test_overlapped_forward_source_ordered_backward_matches_serial(
    wave_size: int,
) -> None:
    _require_cuda_gate()
    torch.manual_seed(24)
    pairs = _pairs()
    reference = torch.nn.Parameter(torch.randn(64, 64, device="cuda"))
    candidate = torch.nn.Parameter(reference.detach().clone())
    reference_losses = []
    for pair in pairs:
        loss = _loss(reference, pair)
        reference_losses.append(loss.detach())
        (loss / PAIRS_PER_STEP).backward()
    torch.cuda.synchronize()

    with PairWaveRuntime(torch.device("cuda"), wave_size) as runtime:
        result = runtime.execute(
            pairs,
            lambda _index, pair: _loss(candidate, pair),
            (candidate,),
        )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        torch.stack(result.pair_losses), torch.stack(reference_losses)
    )
    assert reference.grad is not None and candidate.grad is not None
    torch.testing.assert_close(
        candidate.grad, reference.grad, atol=0.0, rtol=0.0
    )
