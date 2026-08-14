from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F

from gcw.training.core import _accumulate_step_metrics, _new_step_metric_totals, _read_step_metric_totals


def test_step_metric_totals_match_per_microbatch_host_calculation() -> None:
    expected_loss = 0.0
    expected_non_padding = 0
    expected_supervised = 0
    totals = _new_step_metric_totals(torch.device("cpu"))

    for micro_step in range(32):
        loss = torch.tensor((micro_step - 7.25) / 13.0, dtype=torch.float32, requires_grad=True)
        x = torch.arange(24).reshape(3, 8)
        y = x.clone()
        y[:, : micro_step % 5] = -100
        attention_mask = None if micro_step % 3 == 0 else (x % (micro_step % 4 + 2) != 0)

        expected_loss += float(loss.detach().cpu())
        expected_non_padding += int(x.numel() if attention_mask is None else attention_mask.sum().item())
        expected_supervised += int((y != -100).sum().item())
        _accumulate_step_metrics(totals, loss=loss, x=x, y=y, attention_mask=attention_mask)

    assert totals.grad_fn is None
    assert _read_step_metric_totals(totals) == (expected_loss, expected_non_padding, expected_supervised)


def test_metric_accumulation_preserves_gradients_and_optimizer_state() -> None:
    torch.manual_seed(1234)
    baseline = _TinyTokenModel()
    candidate = deepcopy(baseline)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=3e-3, weight_decay=0.1)
    candidate_optimizer = torch.optim.AdamW(candidate.parameters(), lr=3e-3, weight_decay=0.1)
    accumulation_steps = 4
    batches = _training_batches(steps=3, accumulation_steps=accumulation_steps)

    for step in range(3):
        baseline_optimizer.zero_grad(set_to_none=True)
        candidate_optimizer.zero_grad(set_to_none=True)
        totals = _new_step_metric_totals(torch.device("cpu"))
        expected_loss = 0.0
        expected_non_padding = 0
        expected_supervised = 0

        for x, y, attention_mask in batches[step]:
            baseline_loss = baseline(x, y)
            candidate_loss = candidate(x, y)
            expected_loss += float(baseline_loss.detach().cpu())
            expected_non_padding += int(x.numel() if attention_mask is None else attention_mask.sum().item())
            expected_supervised += int((y != -100).sum().item())
            _accumulate_step_metrics(
                totals,
                loss=candidate_loss,
                x=x,
                y=y,
                attention_mask=attention_mask,
            )
            (baseline_loss / accumulation_steps).backward()
            (candidate_loss / accumulation_steps).backward()

        assert _read_step_metric_totals(totals) == (expected_loss, expected_non_padding, expected_supervised)
        for baseline_parameter, candidate_parameter in zip(
            baseline.parameters(), candidate.parameters(), strict=True
        ):
            assert baseline_parameter.grad is not None
            assert candidate_parameter.grad is not None
            assert torch.equal(baseline_parameter.grad, candidate_parameter.grad)

        baseline_optimizer.step()
        candidate_optimizer.step()

        for baseline_parameter, candidate_parameter in zip(
            baseline.parameters(), candidate.parameters(), strict=True
        ):
            assert torch.equal(baseline_parameter, candidate_parameter)
        _assert_nested_equal(baseline_optimizer.state_dict(), candidate_optimizer.state_dict())


class _TinyTokenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 6)
        self.projection = torch.nn.Linear(6, 17)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.projection(self.embedding(x))
        return F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)


def _training_batches(
    *, steps: int, accumulation_steps: int
) -> list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]]:
    generator = torch.Generator().manual_seed(5678)
    batches = []
    for step in range(steps):
        step_batches = []
        for micro_step in range(accumulation_steps):
            x = torch.randint(0, 17, (2, 5), generator=generator)
            y = torch.roll(x, shifts=-1, dims=1)
            y[0, (step + micro_step) % y.shape[1]] = -100
            attention_mask = None if micro_step % 2 == 0 else x % 3 != 0
            step_batches.append((x, y, attention_mask))
        batches.append(step_batches)
    return batches


def _assert_nested_equal(expected: Any, actual: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, list | tuple):
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_nested_equal(expected_item, actual_item)
    else:
        assert expected == actual
