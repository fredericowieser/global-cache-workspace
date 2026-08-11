"""Standalone prototype for conditional flow refinement of persistent GCW workspace slots.

The production Longshot workspace has shape [batch, cache_heads, output_dim] and
its encoded source has shape [batch, cache_heads, selected_time, input_dim].  The
prototype deliberately keeps that contract while depending only on PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


class WorkspaceField(Protocol):
    def __call__(self, workspace: torch.Tensor, time: float | torch.Tensor, source: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class RefinementConfig:
    steps: int = 4
    solver: str = "midpoint"

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.solver not in {"euler", "midpoint"}:
            raise ValueError("solver must be 'euler' or 'midpoint'")


def time_features(
    time: float | torch.Tensor,
    *,
    batch: int,
    slots: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return [B, S, 4] continuous-time features."""
    value = torch.as_tensor(time, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        value = value.expand(batch)
    if value.shape != (batch,):
        raise ValueError(f"time must be scalar or [batch], got {tuple(value.shape)}")
    value = value[:, None, None].expand(batch, slots, 1)
    return torch.cat(
        [value, value.square(), torch.sin(math.pi * value), torch.cos(math.pi * value)],
        dim=-1,
    )


class ConditionalVelocityField(nn.Module):
    """A tied bounded v_theta(W, t, source) over persistent workspace slots.

    `source` is per-slot history [B, S, T, D_source].  The field sees each
    slot's pooled history plus a global summary across slots.  The tanh output
    gives a hard coordinate-wise speed bound, which also bounds total flow
    displacement over t in [0, 1].
    """

    def __init__(
        self,
        workspace_dim: int,
        source_dim: int,
        hidden_dim: int = 64,
        max_velocity: float = 0.75,
    ) -> None:
        super().__init__()
        if min(workspace_dim, source_dim, hidden_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        if max_velocity <= 0:
            raise ValueError("max_velocity must be positive")
        self.workspace_dim = workspace_dim
        self.source_dim = source_dim
        self.max_velocity = float(max_velocity)
        self.workspace_norm = nn.LayerNorm(workspace_dim)
        self.source_norm = nn.LayerNorm(source_dim)
        input_dim = workspace_dim + 2 * source_dim + 4
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, workspace_dim),
        )

    def forward(
        self,
        workspace: torch.Tensor,
        time: float | torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_shapes(workspace, source)
        batch, slots, _ = workspace.shape
        normalized_source = self.source_norm(source)
        local_source = normalized_source.mean(dim=2)
        global_source = local_source.mean(dim=1, keepdim=True).expand(-1, slots, -1)
        features = torch.cat(
            [
                self.workspace_norm(workspace),
                local_source,
                global_source,
                time_features(time, batch=batch, slots=slots, reference=workspace),
            ],
            dim=-1,
        )
        return self.max_velocity * torch.tanh(self.network(features))

    def _validate_shapes(self, workspace: torch.Tensor, source: torch.Tensor) -> None:
        if workspace.ndim != 3:
            raise ValueError(f"workspace must have shape [B, S, D], got {tuple(workspace.shape)}")
        if source.ndim != 4:
            raise ValueError(f"source must have shape [B, S, T, D_source], got {tuple(source.shape)}")
        if workspace.shape[:2] != source.shape[:2]:
            raise ValueError("workspace and source batch/slot dimensions must match")
        if workspace.shape[-1] != self.workspace_dim:
            raise ValueError(f"expected workspace width {self.workspace_dim}")
        if source.shape[-1] != self.source_dim:
            raise ValueError(f"expected source width {self.source_dim}")
        if source.shape[2] == 0:
            raise ValueError("source must contain at least one time position")


class ResidualFlowRefiner(nn.Module):
    """Integrate one shared field from t=0 to t=1 using Euler or midpoint."""

    def __init__(self, field: WorkspaceField, config: RefinementConfig) -> None:
        super().__init__()
        # A single registered field is reused at every solver evaluation.
        self.field = field  # type: ignore[assignment]
        self.config = config

    def forward(self, workspace: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        state = workspace
        step_size = 1.0 / self.config.steps
        for step in range(self.config.steps):
            time = step * step_size
            if self.config.solver == "euler":
                state = state + step_size * self.field(state, time, source)
            else:
                first_velocity = self.field(state, time, source)
                midpoint = state + 0.5 * step_size * first_velocity
                midpoint_velocity = self.field(midpoint, time + 0.5 * step_size, source)
                state = state + step_size * midpoint_velocity
        return state


class DirectWorkspaceRecurrence(nn.Module):
    """Tied direct-state baseline: W_{k+1} = F_theta(W_k, t_k, source).

    It uses exactly the same conditional network and parameter count as the
    residual flow, but replaces state rather than integrating a displacement.
    """

    def __init__(self, transition: WorkspaceField, steps: int) -> None:
        super().__init__()
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.transition = transition  # type: ignore[assignment]
        self.steps = steps

    def forward(self, workspace: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        state = workspace
        for step in range(self.steps):
            state = self.transition(state, step / self.steps, source)
        return state


class ZeroField(nn.Module):
    def forward(
        self,
        workspace: torch.Tensor,
        time: float | torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        del time, source
        return torch.zeros_like(workspace)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
