from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

from gcw.config import DiffusionProcessorConfig
from gcw.models.cache_processors.base import cache_processor_recomputation_id
from gcw.models.cache_processors.update_metrics import UPDATE_OBSERVER

DiffusionField = Callable[[torch.Tensor, float], torch.Tensor]


class TaskDirectedDiffusion(nn.Module):
    """Refine a bottleneck toward a task-useful cache update, without a target.

    Fixed mode preserves the original learned-gate recurrence. Scheduled mode
    samples a noise level and integrates an ungated trajectory toward zero.
    """

    def __init__(self, hidden_size: int, config: DiffusionProcessorConfig):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("diffusion hidden_size must be positive")
        self.hidden_size = hidden_size
        self.config = config
        # Execution-only policy and counters are absent from state_dict.
        self.recurrence_backend: str = "eager"
        if config.training_method == "fixed":
            self.refinement_gate: nn.Parameter | None = nn.Parameter(
                torch.full((1,), config.refinement_gate_init)
            )
        else:
            self.register_parameter("refinement_gate", None)
        self._scheduled_level_counts = [0] * config.scheduled_horizon
        self._scheduled_level_total = 0
        self.eval_solver_steps_override: int | None = None
        self._field_evaluation_count = 0

    def forward(
        self,
        bottleneck: torch.Tensor,
        field: DiffusionField,
        *,
        noise: torch.Tensor | None = None,
        start_level: int | None = None,
    ) -> torch.Tensor:
        if bottleneck.shape[-1] != self.hidden_size:
            raise ValueError(
                f"diffusion expected hidden width {self.hidden_size}, got {bottleneck.shape[-1]}"
            )
        if self.config.training_method == "scheduled":
            return self._scheduled_forward(
                bottleneck, field, noise=noise, start_level=start_level
            )
        if start_level is not None:
            raise ValueError(
                "explicit diffusion start_level requires training_method='scheduled'"
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
            if self.config.perturbation_enabled:
                state = state + self.config.noise_scale * sampled
        elif noise is not None:
            raise ValueError(
                "explicit diffusion noise requires noisy training or stochastic inference"
            )

        # Noise stays eager so compiled recurrence cannot alter RNG order.
        from gcw.performance.backends.diffusion import (
            run_noisy_diffusion_recurrence,
        )

        return run_noisy_diffusion_recurrence(
            self.recurrence_backend, self, state, bottleneck, field
        )

    def _scheduled_forward(
        self,
        bottleneck: torch.Tensor,
        field: DiffusionField,
        *,
        noise: torch.Tensor | None,
        start_level: int | None,
    ) -> torch.Tensor:
        if self.recurrence_backend != "eager":
            raise RuntimeError(
                "scheduled diffusion currently requires the eager recurrence backend"
            )
        level = self._resolve_start_level(start_level)
        sigma = self.scheduled_sigma(level)
        state = bottleneck
        noisy_inference = (
            not self.training and self.config.inference_mode == "stochastic"
        )
        if self.training or noisy_inference:
            sampled = torch.randn_like(state) if noise is None else noise
            if sampled.shape != state.shape:
                raise ValueError(
                    "diffusion noise must match the bottleneck state shape"
                )
            if self.config.perturbation_enabled:
                state = state + sigma * sampled
        elif noise is not None:
            raise ValueError(
                "explicit diffusion noise requires noisy training or stochastic inference"
            )
        if self.training and cache_processor_recomputation_id() is None:
            self._scheduled_level_counts[level - 1] += 1
            self._scheduled_level_total += level
        solver_steps = None
        if self.config.decoupled_schedule:
            solver_steps = (
                self.config.scheduled_training_solver_steps
                if self.training
                else self.eval_solver_steps_override
                or self.config.scheduled_inference_solver_steps
            )
        return self.scheduled_recurrence(
            state, bottleneck, field, level, solver_steps=solver_steps
        )

    def _resolve_start_level(self, start_level: int | None) -> int:
        horizon = self.config.scheduled_horizon
        if start_level is None:
            if not self.training:
                return horizon
            return int(torch.randint(1, horizon + 1, ()).item())
        if isinstance(start_level, bool) or not isinstance(start_level, int):
            raise ValueError("diffusion start_level must be an integer")
        if not 1 <= start_level <= horizon:
            raise ValueError(f"diffusion start_level must be in [1, {horizon}]")
        return start_level

    def _validate_eval_solver_steps(self, steps: int | None) -> None:
        if steps is None:
            return
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError(
                "diffusion evaluation solver steps must be positive"
            )
        if self.config.training_method != "scheduled":
            raise ValueError(
                "diffusion evaluation solver steps require scheduled diffusion"
            )
        if not self.config.decoupled_schedule:
            raise ValueError(
                "diffusion evaluation solver steps require a decoupled schedule"
            )
        if self.recurrence_backend != "eager":
            raise ValueError(
                "diffusion evaluation solver steps require eager recurrence"
            )

    def set_eval_solver_steps(self, steps: int | None) -> None:
        """Set an execution-only solver depth for scheduled evaluation."""
        self._validate_eval_solver_steps(steps)
        self.eval_solver_steps_override = steps

    def pop_field_evaluation_count(self) -> int:
        calls = self._field_evaluation_count
        self._field_evaluation_count = 0
        return calls

    def scheduled_sigma(self, level: float) -> float:
        horizon = self.config.scheduled_horizon
        if not 0 <= level <= horizon:
            raise ValueError(
                f"diffusion schedule level must be in [0, {horizon}]"
            )
        return self.config.noise_scale * level / horizon

    def scheduled_recurrence(
        self,
        state: torch.Tensor,
        bottleneck: torch.Tensor,
        field: DiffusionField,
        start_level: int,
        *,
        solver_steps: int | None = None,
    ) -> torch.Tensor:
        """Integrate the normalized sampled t -> 0 interval.

        Unset solver depth retains integer-level stepping; explicit K uses
        exactly K right-endpoint Euler evaluations, regardless of the horizon.
        """
        initial = state
        if solver_steps is None:
            step_size = self.config.eta_max / self.config.num_steps
            levels = [float(level) for level in range(start_level, 0, -1)]
        else:
            if (
                isinstance(solver_steps, bool)
                or not isinstance(solver_steps, int)
                or solver_steps <= 0
            ):
                raise ValueError("scheduled solver_steps must be positive")
            normalized_start = start_level / self.config.scheduled_horizon
            step_size = self.config.eta_max * normalized_start / solver_steps
            # Evaluate strictly above zero, where log(sigma) is defined.
            levels = [
                start_level * (1.0 - step / solver_steps)
                for step in range(solver_steps)
            ]
        observer = (
            UPDATE_OBSERVER.get()
            if not self.training and not torch.compiler.is_compiling()
            else None
        )
        for step, level in enumerate(levels):
            log_sigma = math.log(self.scheduled_sigma(level))
            field_input = state + self.config.conditioning_scale * bottleneck
            if self.config.time_conditioning:
                field_input = field_input + sinusoidal_time_embedding(
                    log_sigma, self.hidden_size, state
                )
            velocity = field(field_input, log_sigma)
            self._field_evaluation_count += 1
            if velocity.shape != state.shape:
                raise ValueError(
                    f"diffusion field changed state shape from {tuple(state.shape)} "
                    f"to {tuple(velocity.shape)}"
                )
            before = state
            state = state + step_size * velocity
            if observer is not None:
                observer.step(
                    before, state, initial, final=step + 1 == len(levels)
                )
        return state - initial

    def deterministic_recurrence(
        self,
        state: torch.Tensor,
        bottleneck: torch.Tensor,
        field: DiffusionField,
    ) -> torch.Tensor:
        """Run the legacy task-directed recurrence from an already-noised state."""
        initial = state
        step_size = self.scale(state.dtype) / self.config.num_steps
        observer = (
            UPDATE_OBSERVER.get()
            if not self.training and not torch.compiler.is_compiling()
            else None
        )
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
            before = state
            state = state + step_size * velocity
            if observer is not None:
                observer.step(
                    before,
                    state,
                    initial,
                    final=step + 1 == self.config.num_steps,
                )
        return state - initial

    def scale(self, dtype: torch.dtype) -> torch.Tensor:
        if self.refinement_gate is None:
            raise RuntimeError(
                "scheduled diffusion uses fixed steps and has no refinement gate"
            )
        return (self.config.eta_max * torch.sigmoid(self.refinement_gate)).to(
            dtype
        )

    def pop_scheduled_metrics(self) -> dict[str, float]:
        calls = sum(self._scheduled_level_counts)
        if calls == 0:
            return {}
        horizon = self.config.scheduled_horizon
        metrics = {
            "cache_processor/diffusion_start_level_mean": (
                self._scheduled_level_total / calls
            ),
            "cache_processor/diffusion_start_time_mean": (
                self._scheduled_level_total / (horizon * calls)
            ),
            "cache_processor/diffusion_sigma_mean": (
                self.config.noise_scale
                * self._scheduled_level_total
                / (horizon * calls)
            ),
            **{
                f"cache_processor/diffusion_start_level_{level}_fraction": count
                / calls
                for level, count in enumerate(
                    self._scheduled_level_counts, start=1
                )
            },
        }
        self._scheduled_level_counts = [0] * horizon
        self._scheduled_level_total = 0
        return metrics


def set_diffusion_eval_solver_steps(model: nn.Module, steps: int | None) -> int:
    """Apply a runtime-only solver-depth override atomically to all modules."""
    modules = [
        module
        for module in model.modules()
        if isinstance(module, TaskDirectedDiffusion)
    ]
    if not modules:
        raise ValueError(
            "diffusion evaluation solver steps require a diffusion processor"
        )
    for module in modules:
        module._validate_eval_solver_steps(steps)
    for module in modules:
        module.set_eval_solver_steps(steps)
    return len(modules)


def pop_diffusion_field_evaluation_count(model: nn.Module) -> int:
    """Return and reset completed scheduled field calls across modules.

    This is exact for inference without checkpoint replay, not a training
    compute counter. Backward replay can repeat field calls or stop inside
    a field before its counter increment.
    """
    return sum(
        module.pop_field_evaluation_count()
        for module in model.modules()
        if isinstance(module, TaskDirectedDiffusion)
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
