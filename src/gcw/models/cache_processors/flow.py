from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

from gcw.config import FlowProcessorConfig

FlowField = Callable[[torch.Tensor, float], torch.Tensor]


class LatentFlow(nn.Module):
    """Integrate a shared latent vector field and return its displacement."""

    def __init__(self, hidden_size: int, config: FlowProcessorConfig):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("latent flow hidden_size must be positive")
        self.hidden_size = hidden_size
        self.config = config
        self.flow_gate = nn.Parameter(torch.full((1,), config.flow_gate_init))

    def forward(self, initial: torch.Tensor, field: FlowField) -> torch.Tensor:
        if initial.shape[-1] != self.hidden_size:
            raise ValueError(
                f"latent flow expected hidden width {self.hidden_size}, got {initial.shape[-1]}"
            )
        state = initial
        steps = self.config.num_steps
        step_size = self.scale(state.dtype) / steps
        for step in range(steps):
            time = step / steps
            field_input = state
            if self.config.time_conditioning:
                field_input = field_input + sinusoidal_time_embedding(
                    time, self.hidden_size, state
                )
            velocity = field(field_input, time)
            if velocity.shape != state.shape:
                raise ValueError(
                    f"latent flow field changed state shape from {tuple(state.shape)} to {tuple(velocity.shape)}"
                )
            state = state + step_size * velocity
        return state - initial

    def scale(self, dtype: torch.dtype) -> torch.Tensor:
        return (self.config.eta_max * torch.sigmoid(self.flow_gate)).to(dtype)

    def set_num_steps(self, num_steps: int) -> None:
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps <= 0
        ):
            raise ValueError("flow solver steps must be a positive integer")
        self.config.num_steps = num_steps


def sinusoidal_time_embedding(
    time: float,
    width: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    half = (width + 1) // 2
    exponent = torch.arange(
        half, device=reference.device, dtype=torch.float32
    ) / max(half - 1, 1)
    frequencies = torch.exp(-math.log(10_000.0) * exponent)
    angles = frequencies * time
    embedding = torch.cat((angles.sin(), angles.cos()))[:width].to(
        reference.dtype
    )
    shape = (1,) * (reference.ndim - 1) + (width,)
    return embedding.view(shape)


def set_flow_solver_steps(model: nn.Module, num_steps: int) -> None:
    flows = [
        module for module in model.modules() if isinstance(module, LatentFlow)
    ]
    if not flows:
        raise ValueError(
            "flow solver steps require a flow-enabled BGT-family model"
        )
    for flow in flows:
        flow.set_num_steps(num_steps)
