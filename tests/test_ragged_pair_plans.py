from __future__ import annotations

import pytest
import torch

from gcw.performance.ragged import plan_ragged_pair


def test_ragged_pair_plan_preserves_independent_valid_lengths_and_boundaries() -> None:
    mask = torch.arange(160)[None, :] < torch.tensor([[130], [65]])
    plan = plan_ragged_pair(mask, process_every=64)
    assert plan.valid_lengths == (130, 65)
    assert [(window.start, window.stop, window.active_rows) for window in plan.windows] == [
        (0, 64, (0, 1)),
        (64, 128, (0, 1)),
        (128, 130, (0,)),
    ]
    assert plan.semantic_boundaries(0) == (64, 128)
    assert plan.semantic_boundaries(1) == (64,)


def test_ragged_pair_plan_rejects_holes_empty_rows_and_nonpairs() -> None:
    hole = torch.tensor([[True, False, True], [True, True, False]])
    with pytest.raises(ValueError, match="prefix-shaped"):
        plan_ragged_pair(hole, 2)
    empty = torch.tensor([[True, False], [False, False]])
    with pytest.raises(ValueError, match="at least one"):
        plan_ragged_pair(empty, 2)
    with pytest.raises(ValueError, match="two-row"):
        plan_ragged_pair(torch.ones(3, 4, dtype=torch.bool), 2)
