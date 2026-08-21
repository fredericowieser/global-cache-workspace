from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

from gcw.config import DiffusionProcessorConfig

DiffusionField = Callable[[torch.Tensor, float], torch.Tensor]


class TaskDirectedDiffusion(nn.Module):
    """Refine a noisy bottleneck state toward a task-useful cache update.

    The endpoint has no cache-reconstruction target. Training gradients arrive
    through the frozen language model's future-token objective.
    """

    def __init__(self, hidden_size: int, config: DiffusionProcessorConfig):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("diffusion hidden_size must be positive")
        self.hidden_size = hidden_size
        self.config = config
        # Execution-only policy. It is intentionally absent from state_dict.
        self.recurrence_backend: str = "eager"
        self.refinement_gate = nn.Parameter(
            torch.full((1,), config.refinement_gate_init)
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        field: DiffusionField,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bottleneck.shape[-1] != self.hidden_size:
            raise ValueError(
                f"diffusion expected hidden width {self.hidden_size}, got {bottleneck.shape[-1]}"
            )
        state = bottleneck
        noisy_inference = (
            not self.training and self.config.inference_mode == "stochastic"
        )
        if (self.training or noisy_inference) and self.config.noise_scale > 0.0:
            sampled = torch.randn_like(state) if noise is None else noise
            if sampled.shape != state.shape:
                raise ValueError(
                    "diffusion noise must match the bottleneck state shape"
                )
            state = state + self.config.noise_scale * sampled
        elif noise is not None:
            raise ValueError(
                "explicit diffusion noise requires noisy training or stochastic inference"
            )

        # Noise sampling remains eager so compiled recurrence cannot alter RNG order.
        from gcw.performance.backends.diffusion import (
            run_noisy_diffusion_recurrence,
        )

        return run_noisy_diffusion_recurrence(
            self.recurrence_backend,
            self,
            state,
            bottleneck,
            field,
        )

    def deterministic_recurrence(
        self,
        state: torch.Tensor,
        bottleneck: torch.Tensor,
        field: DiffusionField,
    ) -> torch.Tensor:
        """Run the task-directed recurrence from an already-noised state."""
        initial = state
        step_size = self.scale(state.dtype) / self.config.num_steps
        for step in range(self.config.num_steps):
            time = step / self.config.num_steps
            field_input = state + self.config.conditioning_scale * bottleneck
            if self.config.time_conditioning:
                field_input = field_input + sinusoidal_time_embedding(
                    time, self.hidden_size, state
                )
            velocity = field(field_input, time)
            if velocity.shape != state.shape:
                raise ValueError(
                    f"diffusion field changed state shape from {tuple(state.shape)} "
                    f"to {tuple(velocity.shape)}"
                )
            state = state + step_size * velocity
        return state - initial

    def scale(self, dtype: torch.dtype) -> torch.Tensor:
        return (self.config.eta_max * torch.sigmoid(self.refinement_gate)).to(
            dtype
        )


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
