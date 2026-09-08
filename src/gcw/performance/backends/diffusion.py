"""Explicit regional backend for noisy task-directed Diffusion-BGT.

Noise is always sampled by :class:`TaskDirectedDiffusion` before this module is
called. The optional compiled region therefore contains only the deterministic
recurrence and its field evaluations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field as dataclass_field
from _thread import LockType
from threading import Lock
from typing import Any, Literal, cast
from weakref import WeakKeyDictionary

import torch
import torch.nn as nn

from gcw.models.cache_processors.diffusion import (
    DiffusionField,
    TaskDirectedDiffusion,
)
from gcw.performance._compile import (
    is_non_reentrant_checkpoint_early_stop,
    require_torch_compile,
)
from gcw.performance.config import NoisyDiffusionBackendConfig
from gcw.provenance import canonical_json_sha256

NoisyDiffusionRecurrenceBackend = Literal["eager", "compile"]
UnboundDiffusionRecurrenceRunner = Callable[
    [TaskDirectedDiffusion, torch.Tensor, torch.Tensor, DiffusionField],
    torch.Tensor,
]

_COMPILE_OPTIONS = {
    "backend": "inductor",
    "dynamic": False,
    "fullgraph": True,
}
_LEGACY_RUNNER_ATTRIBUTES = (
    "_gcw_noisy_diffusion_compiled_recurrence_runner",
    "_gcw_noisy_diffusion_compile_count",
    "_gcw_noisy_diffusion_dispatch_count",
)
_compiled_recurrence_lock = Lock()


@dataclass(frozen=True)
class _DynamoCounters:
    unique_graphs: int
    graph_breaks: int
    recompiles: int


@dataclass
class _CompiledRunnerState:
    runner: UnboundDiffusionRecurrenceRunner | None = None
    process_counter_baseline: _DynamoCounters | None = None
    dispatch_count: int = 0
    execution_lock: LockType = dataclass_field(default_factory=Lock, repr=False)


_compiled_runner_states: WeakKeyDictionary[
    TaskDirectedDiffusion, _CompiledRunnerState
] = WeakKeyDictionary()


def configure_noisy_diffusion_recurrence_backend(
    model: nn.Module,
    model_config: object,
    policy: NoisyDiffusionBackendConfig,
) -> NoisyDiffusionRecurrenceBackend:
    """Apply a state-neutral recurrence backend after model construction."""
    selected = policy.recurrence_backend
    modules = [
        module
        for module in model.modules()
        if isinstance(module, TaskDirectedDiffusion)
    ]
    if selected == "compile":
        variant = getattr(model_config, "variant", None)
        diffusion = getattr(model_config, "diffusion_processor", None)
        processor = getattr(model_config, "cache_processor", None)
        noise_scale = getattr(diffusion, "noise_scale", 0.0)
        attention_dropout = getattr(processor, "attention_dropout", 0.0)
        if getattr(diffusion, "training_method", "fixed") != "fixed":
            raise RuntimeError(
                "compiled noisy-Diffusion recurrence does not support scheduled diffusion"
            )
        if variant != "bgt" or diffusion is None or noise_scale <= 0.0:
            raise RuntimeError(
                "compiled noisy-Diffusion recurrence requires BGT with diffusion noise_scale>0"
            )
        if attention_dropout != 0.0:
            raise RuntimeError(
                "compiled noisy-Diffusion recurrence requires cache_processor.attention_dropout=0"
            )
        if len(modules) != 1:
            raise RuntimeError(
                "compiled noisy-Diffusion recurrence requires exactly one TaskDirectedDiffusion module"
            )
        require_torch_compile(
            "performance.noisy_diffusion.recurrence_backend=compile"
        )
    for module in modules:
        module.recurrence_backend = selected
        with _compiled_recurrence_lock:
            _compiled_runner_states.pop(module, None)
        # Clean live modules created by the superseded bound-runner implementation.
        for attribute in _LEGACY_RUNNER_ATTRIBUTES:
            module.__dict__.pop(attribute, None)
    return selected


def noisy_diffusion_recurrence_backend_metadata(
    model_config: object,
    policy: NoisyDiffusionBackendConfig,
) -> dict[str, object]:
    """Return authenticated provenance for the noisy-Diffusion recurrence boundary."""
    selected = policy.recurrence_backend
    diffusion = getattr(model_config, "diffusion_processor", None)
    execution = {
        "schema": "gcw.noisy_diffusion_recurrence_backend.v1",
        "effective_backend": selected,
        "architecture": (
            "noisy_diffusion_bgt" if diffusion is not None else None
        ),
        "region": (
            "gcw.models.cache_processors.diffusion."
            "TaskDirectedDiffusion.deterministic_recurrence"
        ),
        "noise_boundary": "eager_torch_randn_like_before_recurrence_dispatch",
        "runner_ownership": "per_task_directed_diffusion_module",
        "compile_options": (
            dict(_COMPILE_OPTIONS) if selected == "compile" else None
        ),
        "torch_version": str(torch.__version__),
    }
    if (
        diffusion is not None
        and getattr(diffusion, "training_method", "fixed") == "scheduled"
    ):
        execution.update(
            architecture=f"scheduled_diffusion_{getattr(model_config, 'variant', 'bgt')}",
            region=(
                "gcw.models.cache_processors.diffusion."
                "TaskDirectedDiffusion.scheduled_recurrence"
            ),
            schedule_horizon=diffusion.scheduled_horizon,
            training_solver_steps=(
                diffusion.scheduled_training_solver_steps
                if diffusion.decoupled_schedule
                else None
            ),
            inference_solver_steps=(
                diffusion.scheduled_inference_solver_steps
                if diffusion.decoupled_schedule
                else diffusion.num_steps
            ),
        )
    return {
        **execution,
        "configured_via": "performance.noisy_diffusion.recurrence_backend",
        "implementation_module": "gcw.performance.backends.diffusion",
        "implementation_symbol": (
            "gcw.performance.backends.diffusion.run_noisy_diffusion_recurrence"
        ),
        "fingerprint_sha256": canonical_json_sha256(execution),
    }


def noisy_diffusion_compile_telemetry(
    diffusion: TaskDirectedDiffusion,
) -> dict[str, int | bool | str]:
    """Return module-local dispatch evidence and process-wide Dynamo diagnostics.

    Runner ownership, creation, and dispatch counts belong to ``diffusion``. Dynamo
    exposes only process-global counters, so both their absolute values and deltas
    since this runner materialized may include unrelated compilations. Those fields
    are diagnostics only and must not participate in acceptance decisions.
    """
    with _compiled_recurrence_lock:
        runner_state = _compiled_runner_states.get(diffusion)
    if runner_state is None:
        current = _dynamo_counters()
        baseline = current
        dispatch_count = 0
        runner_owned = False
        runner_creation_count = 0
    else:
        with runner_state.execution_lock:
            # Sample after any in-flight specialization finishes. The counters remain
            # process-wide and may still include unrelated compilation activity.
            current = _dynamo_counters()
            baseline = runner_state.process_counter_baseline or current
            dispatch_count = runner_state.dispatch_count
            runner_owned = runner_state.runner is not None
            runner_creation_count = int(runner_owned)
    unique_graph_delta = max(0, current.unique_graphs - baseline.unique_graphs)
    graph_break_delta = max(0, current.graph_breaks - baseline.graph_breaks)
    recompile_delta = max(0, current.recompiles - baseline.recompiles)
    return {
        "telemetry_schema": "gcw.noisy_diffusion_compile_telemetry.v2",
        "runner_owned": runner_owned,
        "runner_creation_count": runner_creation_count,
        "runner_dispatch_count": dispatch_count,
        "dynamo_process_diagnostics_acceptance_eligible": False,
        "dynamo_process_delta_unique_graphs": unique_graph_delta,
        "dynamo_process_delta_graph_breaks": graph_break_delta,
        "dynamo_process_delta_recompiles": recompile_delta,
        "dynamo_process_unique_graphs": current.unique_graphs,
        "dynamo_process_graph_breaks": current.graph_breaks,
        "dynamo_process_recompiles": current.recompiles,
    }


def _dynamo_counters() -> _DynamoCounters:
    dynamo: Any = getattr(torch, "_dynamo", None)
    utils: Any = getattr(dynamo, "utils", None)
    counters: Any = getattr(utils, "counters", {})

    def value(group: str, name: str) -> int:
        values = counters.get(group, {}) if hasattr(counters, "get") else {}
        return int(values.get(name, 0)) if hasattr(values, "get") else 0

    def total(group: str) -> int:
        values = counters.get(group, {}) if hasattr(counters, "get") else {}
        return int(sum(values.values())) if hasattr(values, "values") else 0

    return _DynamoCounters(
        unique_graphs=value("stats", "unique_graphs"),
        graph_breaks=total("graph_break"),
        recompiles=total("recompile"),
    )


def run_noisy_diffusion_recurrence(
    backend: str,
    diffusion: TaskDirectedDiffusion,
    state: torch.Tensor,
    bottleneck: torch.Tensor,
    field: DiffusionField,
) -> torch.Tensor:
    """Dispatch an already-noised state; compiled failures never fall back."""
    if backend == "eager":
        return diffusion.deterministic_recurrence(state, bottleneck, field)
    if backend != "compile":
        raise ValueError(
            f"unsupported noisy-Diffusion recurrence backend: {backend}"
        )
    runner_state = _get_or_create_compiled_recurrence_state(diffusion)
    # Dynamo/Inductor lazy specialization is not safe under concurrent first calls.
    # Serialize materialization and dispatch per diffusion module, while unrelated
    # modules retain independent execution locks.
    with runner_state.execution_lock:
        runner = _materialize_compiled_recurrence_runner(runner_state)
        runner_state.dispatch_count += 1
        try:
            return runner(diffusion, state, bottleneck, field)
        except Exception as error:
            if is_non_reentrant_checkpoint_early_stop(error):
                raise
            raise RuntimeError(
                "compiled noisy-Diffusion recurrence failed; eager fallback is disabled"
            ) from error


def _run_unbound_deterministic_recurrence(
    diffusion: TaskDirectedDiffusion,
    state: torch.Tensor,
    bottleneck: torch.Tensor,
    field: DiffusionField,
) -> torch.Tensor:
    """Run recurrence using only state supplied at dispatch time."""
    return diffusion.deterministic_recurrence(state, bottleneck, field)


def _get_or_create_compiled_recurrence_state(
    diffusion: TaskDirectedDiffusion,
) -> _CompiledRunnerState:
    """Return weak execution state without binding it to model serialization."""
    with _compiled_recurrence_lock:
        runner_state = _compiled_runner_states.get(diffusion)
        if runner_state is None:
            runner_state = _CompiledRunnerState()
            _compiled_runner_states[diffusion] = runner_state
        return runner_state


def _materialize_compiled_recurrence_runner(
    runner_state: _CompiledRunnerState,
) -> UnboundDiffusionRecurrenceRunner:
    """Create one lazy compile wrapper while holding the state's execution lock."""
    if runner_state.runner is not None:
        return runner_state.runner
    compiler = require_torch_compile(
        "performance.noisy_diffusion.recurrence_backend=compile"
    )
    try:
        runner = cast(
            UnboundDiffusionRecurrenceRunner,
            compiler(_run_unbound_deterministic_recurrence, **_COMPILE_OPTIONS),
        )
    except Exception as error:
        raise RuntimeError(
            "failed to create compiled noisy-Diffusion recurrence; "
            "eager fallback is disabled"
        ) from error
    # torch.compile returns a lazy wrapper. Process-global diagnostics begin after
    # wrapper creation, while the execution lock also protects first specialization.
    runner_state.runner = runner
    runner_state.process_counter_baseline = _dynamo_counters()
    return runner
