#!/usr/bin/env python3
"""Execute the GCW flow-workspace-refinement feasibility spike."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from refinement import (
    ConditionalVelocityField,
    DirectWorkspaceRecurrence,
    RefinementConfig,
    ResidualFlowRefiner,
    ZeroField,
    parameter_count,
)

BATCH = 64
SLOTS = 4
SOURCE_TIMES = 5
WORKSPACE_DIM = 6
SOURCE_DIM = 8
HIDDEN_DIM = 48
MAX_VELOCITY = 0.75


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_toy_data(seed: int = 2026) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create persistent slots and a source-conditioned residual target."""
    generator = torch.Generator().manual_seed(seed)
    workspace = 0.10 * (2 * torch.rand(BATCH, SLOTS, WORKSPACE_DIM, generator=generator) - 1)
    source = torch.randn(BATCH, SLOTS, SOURCE_TIMES, SOURCE_DIM, generator=generator)
    teacher = torch.randn(SOURCE_DIM, WORKSPACE_DIM, generator=generator) / SOURCE_DIM**0.5
    local = F.layer_norm(source, (SOURCE_DIM,)).mean(dim=2)
    global_summary = local.mean(dim=1, keepdim=True)
    residual = 0.30 * torch.tanh(local @ teacher + 0.35 * (global_summary @ teacher))
    target = workspace + residual
    return workspace, source, target


def fresh_field() -> ConditionalVelocityField:
    return ConditionalVelocityField(
        WORKSPACE_DIM,
        SOURCE_DIM,
        hidden_dim=HIDDEN_DIM,
        max_velocity=MAX_VELOCITY,
    )


def mse(module: torch.nn.Module, workspace: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        return F.mse_loss(module(workspace, source), target).item()


def train_terminal_objective(
    module: torch.nn.Module,
    workspace: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    iterations: int = 500,
) -> tuple[float, float]:
    initial = mse(module, workspace, source, target)
    optimizer = torch.optim.Adam(module.parameters(), lr=8e-3)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(module(workspace, source), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
    return initial, mse(module, workspace, source, target)


def train_conditional_flow_matching(
    field: ConditionalVelocityField,
    workspace: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    iterations: int = 600,
) -> tuple[float, float, list[float]]:
    """Fit v_theta(W_t,t,source) to dW_t/dt on a straight bridge."""
    target_velocity = target - workspace
    optimizer = torch.optim.Adam(field.parameters(), lr=8e-3)

    def objective() -> torch.Tensor:
        times = torch.rand(workspace.shape[0])
        bridge = workspace + times[:, None, None] * target_velocity
        prediction = field(bridge, times, source)
        return F.mse_loss(prediction, target_velocity)

    with torch.no_grad():
        initial = objective().item()
    checkpoints: list[float] = []
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), 1.0)
        optimizer.step()
        if iteration in {0, iterations // 4, iterations // 2, iterations - 1}:
            checkpoints.append(loss.detach().item())
    with torch.no_grad():
        final = objective().item()
    return initial, final, checkpoints


def dimension_and_gradient_probe(workspace: torch.Tensor, source: torch.Tensor) -> dict[str, object]:
    field = fresh_field()
    refiner = ResidualFlowRefiner(field, RefinementConfig(steps=4, solver="midpoint"))
    workspace_probe = workspace[:3].clone().requires_grad_(True)
    source_probe = source[:3].clone().requires_grad_(True)
    output = refiner(workspace_probe, source_probe)
    weights = torch.linspace(0.5, 1.5, output.numel()).reshape_as(output)
    (output * weights).sum().backward()
    parameter_gradients = [parameter.grad for parameter in field.parameters()]
    finite_nonzero = [
        gradient is not None and torch.isfinite(gradient).all().item() and gradient.abs().sum().item() > 0
        for gradient in parameter_gradients
    ]
    return {
        "workspace_shape": list(workspace_probe.shape),
        "source_shape": list(source_probe.shape),
        "output_shape": list(output.shape),
        "workspace_grad_norm": workspace_probe.grad.norm().item(),
        "source_grad_norm": source_probe.grad.norm().item(),
        "parameter_tensors_with_finite_nonzero_grad": sum(finite_nonzero),
        "parameter_tensors": len(parameter_gradients),
        "pass": output.shape == workspace_probe.shape and all(finite_nonzero),
    }


def zero_field_probe(workspace: torch.Tensor, source: torch.Tensor) -> dict[str, object]:
    cases: dict[str, bool] = {}
    for solver in ("euler", "midpoint"):
        for steps in (1, 2, 4, 8):
            output = ResidualFlowRefiner(ZeroField(), RefinementConfig(steps=steps, solver=solver))(
                workspace, source
            )
            cases[f"{solver}_{steps}"] = torch.equal(output, workspace)
    return {"cases": cases, "pass": all(cases.values())}


def stability_probe(
    state_dict: dict[str, torch.Tensor],
    workspace: torch.Tensor,
    source: torch.Tensor,
) -> dict[str, object]:
    field = fresh_field()
    field.load_state_dict(state_dict)
    reference = ResidualFlowRefiner(field, RefinementConfig(steps=64, solver="midpoint"))(workspace, source)
    rows: dict[str, dict[str, float | bool]] = {}
    with torch.no_grad():
        for steps in (1, 2, 4, 8):
            output = ResidualFlowRefiner(field, RefinementConfig(steps=steps, solver="midpoint"))(workspace, source)
            max_update = (output - workspace).abs().max().item()
            rows[str(steps)] = {
                "max_abs_update": max_update,
                "rmse_to_64_step": F.mse_loss(output, reference).sqrt().item(),
                "within_hard_bound": max_update <= MAX_VELOCITY + 1e-6,
            }
    return {
        "solver": "midpoint",
        "hard_coordinate_bound": MAX_VELOCITY,
        "steps": rows,
        "pass": all(bool(row["within_hard_bound"]) for row in rows.values()),
    }


def tied_parameter_probe() -> dict[str, object]:
    field = fresh_field()
    counts = {}
    registrations = []
    for steps in (1, 2, 4, 8):
        model = ResidualFlowRefiner(field, RefinementConfig(steps=steps, solver="midpoint"))
        counts[str(steps)] = parameter_count(model)
        registrations.append(model.field is field)
    return {
        "parameter_count_by_steps": counts,
        "same_field_object_reused": all(registrations),
        "pass": len(set(counts.values())) == 1 and all(registrations),
    }


def endpoint_comparison(
    workspace: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, object]:
    """Train direct recurrence and residual Euler flow on exactly the same MSE."""
    seed_everything(31415)
    initial_field = fresh_field()
    direct_field = copy.deepcopy(initial_field)
    flow_field = copy.deepcopy(initial_field)
    direct = DirectWorkspaceRecurrence(direct_field, steps=4)
    flow = ResidualFlowRefiner(flow_field, RefinementConfig(steps=4, solver="euler"))
    direct_initial, direct_final = train_terminal_objective(direct, workspace, source, target)
    flow_initial, flow_final = train_terminal_objective(flow, workspace, source, target)
    return {
        "objective": "full-batch endpoint MSE",
        "optimizer": "Adam(lr=0.008), gradient clipping 1.0",
        "updates": 500,
        "field_evaluations_per_forward": 4,
        "same_initial_parameters": True,
        "parameter_count_direct": parameter_count(direct),
        "parameter_count_flow": parameter_count(flow),
        "direct_recurrence": {"initial_mse": direct_initial, "final_mse": direct_final},
        "residual_flow": {"initial_mse": flow_initial, "final_mse": flow_final},
        "flow_to_direct_final_mse_ratio": flow_final / direct_final,
        "pass": direct_final < 2e-3 and flow_final < 2e-3,
    }


def run() -> dict[str, object]:
    seed_everything(7)
    workspace, source, target = make_toy_data()
    dimensions = dimension_and_gradient_probe(workspace, source)
    zero_identity = zero_field_probe(workspace[:2], source[:2])
    tied = tied_parameter_probe()

    seed_everything(2718)
    matching_field = fresh_field()
    matching_initial, matching_final, checkpoints = train_conditional_flow_matching(
        matching_field, workspace, source, target
    )
    matching_refiner = ResidualFlowRefiner(
        matching_field,
        RefinementConfig(steps=8, solver="midpoint"),
    )
    matching_endpoint_mse = mse(matching_refiner, workspace, source, target)
    matching = {
        "bridge": "W_t = W_0 + t * (W_target - W_0)",
        "target_velocity": "W_target - W_0",
        "updates": 600,
        "initial_velocity_mse": matching_initial,
        "checkpoint_velocity_mse": checkpoints,
        "final_velocity_mse": matching_final,
        "eight_step_endpoint_mse": matching_endpoint_mse,
        "pass": matching_final < 2e-3 and matching_endpoint_mse < 2e-3,
    }
    stability = stability_probe(matching_field.state_dict(), workspace, source)
    comparison = endpoint_comparison(workspace, source, target)

    checks = {
        "dimensions_and_gradients": bool(dimensions["pass"]),
        "zero_field_identity": bool(zero_identity["pass"]),
        "tied_parameters": bool(tied["pass"]),
        "bounded_updates": bool(stability["pass"]),
        "conditional_flow_matching": bool(matching["pass"]),
        "same_objective_comparison": bool(comparison["pass"]),
    }
    return {
        "environment": {"torch": torch.__version__, "device": "cpu", "seeded_deterministic": True},
        "contract": {
            "workspace": "[batch, persistent_slots, workspace_dim]",
            "source": "[batch, persistent_slots, selected_time, source_dim]",
            "toy_shapes": {
                "workspace": list(workspace.shape),
                "source": list(source.shape),
                "target": list(target.shape),
            },
        },
        "dimensions_and_gradients": dimensions,
        "zero_field_identity": zero_identity,
        "tied_parameters": tied,
        "conditional_flow_matching": matching,
        "step_stability": stability,
        "direct_vs_flow": comparison,
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run()
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
