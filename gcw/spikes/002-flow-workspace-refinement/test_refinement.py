from __future__ import annotations

import pytest
import torch

from refinement import (
    ConditionalVelocityField,
    RefinementConfig,
    ResidualFlowRefiner,
    ZeroField,
    parameter_count,
)
from run_experiment import run


@pytest.fixture
def tensors() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    return torch.randn(2, 3, 5), torch.randn(2, 3, 4, 7)


@pytest.mark.parametrize("solver", ["euler", "midpoint"])
@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_zero_field_is_exact_identity(
    tensors: tuple[torch.Tensor, torch.Tensor], solver: str, steps: int
) -> None:
    workspace, source = tensors
    refiner = ResidualFlowRefiner(ZeroField(), RefinementConfig(steps=steps, solver=solver))
    assert torch.equal(refiner(workspace, source), workspace)


@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_shape_gradient_bound_and_tied_parameter_count(
    tensors: tuple[torch.Tensor, torch.Tensor], steps: int
) -> None:
    workspace, source = tensors
    workspace.requires_grad_(True)
    source.requires_grad_(True)
    field = ConditionalVelocityField(5, 7, hidden_dim=24, max_velocity=0.4)
    refiner = ResidualFlowRefiner(field, RefinementConfig(steps=steps, solver="midpoint"))
    output = refiner(workspace, source)
    output.square().mean().backward()

    assert output.shape == workspace.shape
    assert (output - workspace).abs().max().item() <= 0.4 + 1e-6
    assert workspace.grad is not None and workspace.grad.norm().item() > 0
    assert source.grad is not None and source.grad.norm().item() > 0
    assert parameter_count(refiner) == parameter_count(field)


def test_shape_validation(tensors: tuple[torch.Tensor, torch.Tensor]) -> None:
    workspace, source = tensors
    field = ConditionalVelocityField(5, 7)
    with pytest.raises(ValueError, match="batch/slot"):
        field(workspace, 0.5, source[:, :2])
    with pytest.raises(ValueError, match="workspace width"):
        field(workspace[..., :4], 0.5, source)


def test_executed_feasibility_contract() -> None:
    result = run()
    assert result["all_pass"]
    assert all(result["checks"].values())
