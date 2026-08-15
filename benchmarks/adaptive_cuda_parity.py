"""Formal fixed/adaptive CUDA parity harness.

This module never times instrumented execution. It aligns semantic cache
operations by stable source-row and actual cache-boundary identities, then
compares losses, gradients, optimizer moments, RNG, and a short parameter
trajectory. Reservation identity is not presented as production resume evidence.
Cache values are not captured. Production training remains untouched.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
import math
from pathlib import Path
import random
from typing import Any, Literal

import numpy as np
import torch

from benchmarks.adaptive_trace import (
    CacheEvent,
    CacheEventRecorder,
    TraceScope,
    active_trace_scope,
    events_to_json,
    row_keys_for_pairs,
    row_widths_for_pairs,
    trace_cache_events,
)
from benchmarks.harness import device_identity, write_authenticated_json
from gcw.checkpoint import load_backbone_checkpoint
from gcw.config import load_train_config, to_dict
from gcw.data.training import TrainingBatch, TrainingBatchStream, build_training_stream, cursor_to_dict
from gcw.models import build_model
from gcw.performance.adjacent_pairs import (
    PAIRS_PER_STEP,
    ReservedAdjacentPairs,
    collate_pair_group,
    plan_contiguous_pair_groups,
    reserve_adjacent_pairs,
)
from gcw.performance.backends.qwen3 import (
    Qwen3BackboneChunkBackend,
    configure_qwen3_backbone_chunk_backend,
)
from gcw.performance.config import AdaptiveBatchingConfig
from gcw.provenance import (
    canonical_json_sha256,
    file_sha256,
    find_repository_root,
    repository_source_identity,
)
from gcw.training.core import (
    _apply_trainable_policy,
    _build_optimizer,
    _clip_grad_norm,
    _load_initial_model,
    _weight_decay,
    _wsd_lr,
)
from gcw.training.objectives import row_loss_sums_and_counts

ExecutionKind = Literal["fixed", "adaptive"]
PairLossFunction = Callable[[torch.nn.Module, TrainingBatch], torch.Tensor]
ModelFactory = Callable[[], torch.nn.Module]
OptimizerFactory = Callable[[torch.nn.Module], torch.optim.Optimizer]
ForwardContext = Callable[[torch.nn.Module, TrainingBatch], AbstractContextManager[Any]]
CheckpointExpectation = Callable[[TrainingBatch], bool]
PrepareOptimizerStep = Callable[
    [torch.nn.Module, Sequence[torch.nn.Parameter], torch.optim.Optimizer, int],
    None,
]


@dataclass(frozen=True)
class ParityTolerance:
    """Elementwise tolerance retained for losses and parameter trajectories."""

    atol: float
    rtol: float


@dataclass(frozen=True)
class VectorTolerance:
    """Scale-aware gates for gradients and optimizer moments."""

    relative_l2: float
    minimum_cosine: float
    near_zero_norm: float
    near_zero_atol: float


@dataclass(frozen=True)
class ParityTolerances:
    pair_losses: ParityTolerance = ParityTolerance(2e-5, 2e-4)
    gradients: VectorTolerance = VectorTolerance(5e-4, 0.99999, 1e-10, 1e-9)
    parameters: ParityTolerance = ParityTolerance(2e-5, 2e-4)
    adam_moments: VectorTolerance = VectorTolerance(1e-3, 0.9999, 1e-12, 1e-12)


@dataclass(frozen=True)
class ParityVariant:
    """One execution policy initialized from the common model state."""

    name: str
    execution: ExecutionKind
    model_factory: ModelFactory


@dataclass(frozen=True)
class ReservedParityStep:
    """One immutable 32-pair reservation and its one-time reservation identity."""

    reservation: ReservedAdjacentPairs
    cursor_before: Mapping[str, Any]
    cursor_after: Mapping[str, Any]


@dataclass(frozen=True)
class RngSnapshot:
    python: tuple[Any, ...]
    numpy: Any
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...]

    def digest(self) -> str:
        payload = {
            "python": repr(self.python),
            "numpy": repr(self.numpy),
            "torch_cpu": self.torch_cpu.tolist(),
            "torch_cuda": [state.tolist() for state in self.torch_cuda],
        }
        return canonical_json_sha256(payload)


@dataclass
class VariantStepResult:
    pair_losses: torch.Tensor
    gradients: dict[str, torch.Tensor]
    missing_gradients: tuple[str, ...]
    parameters: dict[str, torch.Tensor]
    adam_moments: dict[str, dict[str, torch.Tensor]]
    optimizer_steps: dict[str, float]
    rng_sha256: str
    events: list[CacheEvent]
    event_coverage: dict[str, Any]
    recomputation: dict[str, Any]


@dataclass
class VariantResult:
    name: str
    execution: ExecutionKind
    initial_rng_sha256: str
    steps: list[VariantStepResult]


def capture_rng() -> RngSnapshot:
    """Capture every RNG family used by maintained single-process training."""
    return RngSnapshot(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=tuple(state.clone() for state in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
    )


def restore_rng(snapshot: RngSnapshot) -> None:
    random.setstate(snapshot.python)
    np.random.set_state(snapshot.numpy)
    torch.set_rng_state(snapshot.torch_cpu)
    if torch.cuda.is_available() and snapshot.torch_cuda:
        torch.cuda.set_rng_state_all(list(snapshot.torch_cuda))


def reserve_parity_steps(stream: TrainingBatchStream, steps: int) -> tuple[ReservedParityStep, ...]:
    """Reserve complete optimizer steps once, before any compared execution."""
    if steps <= 0:
        raise ValueError("parity trajectory must contain at least one step")
    reserved: list[ReservedParityStep] = []
    for _ in range(steps):
        before = cursor_to_dict(stream.cursor)
        reservation = reserve_adjacent_pairs(stream)
        after = cursor_to_dict(stream.cursor)
        reserved.append(ReservedParityStep(reservation, before, after))
    return tuple(reserved)


def run_adaptive_parity(
    initial_model: torch.nn.Module,
    variants: Sequence[ParityVariant],
    reserved_steps: Sequence[ReservedParityStep],
    pair_loss: PairLossFunction,
    optimizer_factory: OptimizerFactory,
    *,
    adaptive_policy: AdaptiveBatchingConfig,
    process_every: int,
    required_event_operations: Sequence[str],
    checkpoint_expected: CheckpointExpectation | None = None,
    forward_context: ForwardContext | None = None,
    prepare_optimizer_step: PrepareOptimizerStep | None = None,
    tolerances: ParityTolerances = ParityTolerances(),
    resume_evidence_requested: bool = False,
) -> dict[str, Any]:
    """Run and compare fixed batch-2, adaptive eager, and adaptive compiled paths."""
    if len(variants) < 2:
        raise ValueError("formal parity requires at least two execution variants")
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise ValueError("parity variant names must be unique")
    if not reserved_steps:
        raise ValueError("formal parity requires reserved steps")
    if process_every <= 0:
        raise ValueError("formal parity requires a positive cache process interval")
    if resume_evidence_requested:
        raise ValueError(
            "production resume evidence requested, but this benchmark does not execute production checkpoint save/reload"
        )
    initial_state = _clone_named_tensors(initial_model.state_dict())
    initial_rng = capture_rng()
    context_factory = forward_context or (lambda _model, _batch: nullcontext())
    checkpoint_factory = checkpoint_expected or (lambda _batch: False)
    results: list[VariantResult] = []
    for variant in variants:
        model = variant.model_factory()
        model.load_state_dict(initial_state, strict=True)
        model.train()
        optimizer = optimizer_factory(model)
        restore_rng(initial_rng)
        results.append(
            _run_variant(
                variant,
                model,
                optimizer,
                reserved_steps,
                pair_loss,
                adaptive_policy=adaptive_policy,
                process_every=process_every,
                required_event_operations=required_event_operations,
                checkpoint_expected=checkpoint_factory,
                forward_context=context_factory,
                prepare_optimizer_step=prepare_optimizer_step,
            )
        )
    reference = results[0]
    comparisons = [
        _compare_variant(reference, candidate, tolerances)
        for candidate in results[1:]
    ]
    passed = all(comparison["passed"] for comparison in comparisons)
    payload = {
        "schema": "gcw.adaptive_cuda_parity.v2",
        "instrumented": True,
        "timing_policy": "forbidden_in_instrumented_runs",
        "initial_model_sha256": _tensor_mapping_sha256(initial_state),
        "initial_rng_sha256": initial_rng.digest(),
        "reservation_identity": {
            "kind": "single_preexecution_reservation_not_per_variant_resume_evidence",
            "cursor_before": dict(reserved_steps[0].cursor_before),
            "cursor_after": dict(reserved_steps[-1].cursor_after),
            "steps": [
                {
                    "step": step_index,
                    "padded_lengths": list(step.reservation.padded_lengths),
                }
                for step_index, step in enumerate(reserved_steps)
            ],
        },
        "production_resume_gate": {
            "requested": False,
            "available": False,
            "passed": True,
            "reason": "production checkpoint save/reload is outside this reservation-only harness",
        },
        "variants": [_variant_to_json(result) for result in results],
        "comparisons": comparisons,
        "passed": passed,
    }
    return payload


def _run_variant(
    variant: ParityVariant,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    reserved_steps: Sequence[ReservedParityStep],
    pair_loss: PairLossFunction,
    *,
    adaptive_policy: AdaptiveBatchingConfig,
    process_every: int,
    required_event_operations: Sequence[str],
    checkpoint_expected: CheckpointExpectation,
    forward_context: ForwardContext,
    prepare_optimizer_step: PrepareOptimizerStep | None,
) -> VariantResult:
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError(f"parity variant {variant.name!r} has no trainable parameters")
    initial_rng_sha256 = capture_rng().digest()
    step_results: list[VariantStepResult] = []
    for step_index, reserved in enumerate(reserved_steps):
        optimizer.zero_grad(set_to_none=True)
        pair_values: list[torch.Tensor | None] = [None] * PAIRS_PER_STEP
        recorder = CacheEventRecorder(required_event_operations)
        with trace_cache_events(recorder):
            if variant.execution == "fixed":
                groups: Iterator[tuple[int, int, TrainingBatch]] = (
                    (pair_index, pair_index + 1, batch)
                    for pair_index, batch in enumerate(reserved.reservation.pairs)
                )
            else:
                plan = plan_contiguous_pair_groups(reserved.reservation, adaptive_policy)
                groups = (
                    (group.pair_start, group.pair_stop, collate_pair_group(plan, group).batch)
                    for group in plan.groups
                )
            for group_index, (pair_start, pair_stop, batch) in enumerate(groups):
                scope = TraceScope(
                    step=step_index,
                    group=group_index,
                    row_keys=row_keys_for_pairs(step_index, pair_start, pair_stop),
                    source_padded_widths=row_widths_for_pairs(
                        reserved.reservation.padded_lengths, pair_start, pair_stop
                    ),
                    process_every=process_every,
                    checkpoint_expected=checkpoint_expected(batch),
                )
                recorder.expect_scope(scope)
                with active_trace_scope(scope), forward_context(model, batch):
                    losses = pair_loss(model, batch)
                    if losses.shape != (pair_stop - pair_start,):
                        raise RuntimeError(
                            f"pair loss returned {tuple(losses.shape)} for {pair_stop - pair_start} pairs"
                        )
                    for offset, value in enumerate(losses.detach().to(device="cpu", dtype=torch.float64)):
                        pair_values[pair_start + offset] = value
                    (losses.sum() / PAIRS_PER_STEP).backward()
        if any(value is None for value in pair_values):
            raise RuntimeError("parity execution did not produce every reserved pair loss")
        gradients = {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in parameters
            if parameter.grad is not None
        }
        missing_gradients = tuple(sorted(name for name, parameter in parameters if parameter.grad is None))
        if prepare_optimizer_step is not None:
            prepare_optimizer_step(model, [parameter for _, parameter in parameters], optimizer, step_index)
        optimizer.step()
        moments, optimizer_steps = _optimizer_snapshot(optimizer, parameters)
        step_results.append(
            VariantStepResult(
                pair_losses=torch.stack([value for value in pair_values if value is not None]),
                gradients=gradients,
                missing_gradients=missing_gradients,
                parameters={name: parameter.detach().cpu().clone() for name, parameter in parameters},
                adam_moments=moments,
                optimizer_steps=optimizer_steps,
                rng_sha256=capture_rng().digest(),
                events=list(recorder.events),
                event_coverage=recorder.coverage(),
                recomputation=recorder.recomputation_report(),
            )
        )
    return VariantResult(
        name=variant.name,
        execution=variant.execution,
        initial_rng_sha256=initial_rng_sha256,
        steps=step_results,
    )


def _optimizer_snapshot(
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
    moments: dict[str, dict[str, torch.Tensor]] = {}
    steps: dict[str, float] = {}
    for name, parameter in parameters:
        state = optimizer.state.get(parameter, {})
        tensor_moments = {
            key: value.detach().cpu().clone()
            for key, value in state.items()
            if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"} and isinstance(value, torch.Tensor)
        }
        moments[name] = tensor_moments
        raw_step = state.get("step", 0)
        steps[name] = float(raw_step.detach().cpu()) if isinstance(raw_step, torch.Tensor) else float(raw_step)
    return moments, steps


def _compare_variant(
    reference: VariantResult,
    candidate: VariantResult,
    tolerances: ParityTolerances,
) -> dict[str, Any]:
    if len(reference.steps) != len(candidate.steps):
        raise ValueError("parity variants have different trajectory lengths")
    step_comparisons: list[dict[str, Any]] = []
    for step_index, (expected, actual) in enumerate(zip(reference.steps, candidate.steps, strict=True)):
        pair_comparison = _compare_tensor(expected.pair_losses, actual.pair_losses, tolerances.pair_losses)
        gradient_comparison = _compare_vector_family(expected.gradients, actual.gradients, tolerances.gradients)
        missing_gradients_equal = expected.missing_gradients == actual.missing_gradients
        parameter_comparison = _compare_named_tensors(expected.parameters, actual.parameters, tolerances.parameters)
        expected_moments = _flatten_nested_tensors(expected.adam_moments)
        actual_moments = _flatten_nested_tensors(actual.adam_moments)
        moment_comparison = _compare_vector_family(expected_moments, actual_moments, tolerances.adam_moments)
        event_comparison = _compare_events(expected, actual)
        optimizer_steps_finite = all(
            math.isfinite(value) for value in (*expected.optimizer_steps.values(), *actual.optimizer_steps.values())
        )
        optimizer_steps_equal = optimizer_steps_finite and expected.optimizer_steps == actual.optimizer_steps
        rng_equal = expected.rng_sha256 == actual.rng_sha256
        passed = all(
            (
                pair_comparison["passed"],
                gradient_comparison["passed"],
                missing_gradients_equal,
                parameter_comparison["passed"],
                moment_comparison["passed"],
                event_comparison["passed"],
                optimizer_steps_equal,
                rng_equal,
            )
        )
        step_comparisons.append(
            {
                "step": step_index,
                "pair_losses": pair_comparison,
                "gradients": gradient_comparison,
                "missing_gradients_equal": missing_gradients_equal,
                "reference_missing_gradients": list(expected.missing_gradients),
                "candidate_missing_gradients": list(actual.missing_gradients),
                "parameters": parameter_comparison,
                "adam_moments": moment_comparison,
                "optimizer_steps_finite": optimizer_steps_finite,
                "optimizer_steps_equal": optimizer_steps_equal,
                "rng_equal": rng_equal,
                "cache_events": event_comparison,
                "passed": passed,
            }
        )
    initial_rng_equal = reference.initial_rng_sha256 == candidate.initial_rng_sha256
    return {
        "reference": reference.name,
        "candidate": candidate.name,
        "initial_rng_equal": initial_rng_equal,
        "steps": step_comparisons,
        "passed": initial_rng_equal and all(step["passed"] for step in step_comparisons),
    }


def _compare_events(expected: VariantStepResult, actual: VariantStepResult) -> dict[str, Any]:
    expected_primary = _semantic_primary_events(expected.events)
    actual_primary = _semantic_primary_events(actual.events)
    primary_equal = expected_primary == actual_primary
    return {
        "semantic_primary_equal": primary_equal,
        "reference_semantic_primary_count": len(expected_primary),
        "candidate_semantic_primary_count": len(actual_primary),
        "reference_coverage": expected.event_coverage,
        "candidate_coverage": actual.event_coverage,
        "reference_recomputation": expected.recomputation,
        "candidate_recomputation": actual.recomputation,
        "passed": (
            primary_equal
            and expected.event_coverage["complete"]
            and actual.event_coverage["complete"]
            and expected.recomputation["passed"]
            and actual.recomputation["passed"]
        ),
    }


def _semantic_primary_events(events: Sequence[CacheEvent]) -> dict[str, tuple[int, ...]]:
    selected = [event for event in events if event.semantic and event.phase == "primary"]
    identities = [event.stable_event_id for event in selected]
    if len(identities) != len(set(identities)):
        return {}
    return {event.stable_event_id: event.destinations for event in selected}


def _compare_named_tensors(
    expected: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
    tolerance: ParityTolerance,
) -> dict[str, Any]:
    names_equal = expected.keys() == actual.keys()
    comparisons = {
        name: _compare_tensor(expected[name], actual[name], tolerance)
        for name in expected.keys() & actual.keys()
    }
    return {
        "names_equal": names_equal,
        "tensors": comparisons,
        "passed": names_equal and all(result["passed"] for result in comparisons.values()),
    }


def _flatten_nested_tensors(
    values: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {
        f"{parameter}/{moment}": tensor
        for parameter, moments in values.items()
        for moment, tensor in moments.items()
    }


def _compare_vector_family(
    expected: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
    tolerance: VectorTolerance,
) -> dict[str, Any]:
    names_equal = expected.keys() == actual.keys()
    common = sorted(expected.keys() & actual.keys())
    tensors = {name: _compare_vector(expected[name], actual[name], tolerance) for name in common}
    shape_compatible = all(expected[name].shape == actual[name].shape for name in common)
    if common and shape_compatible:
        expected_global = torch.cat([expected[name].detach().reshape(-1).to(torch.float64) for name in common])
        actual_global = torch.cat([actual[name].detach().reshape(-1).to(torch.float64) for name in common])
        global_comparison = _compare_vector(expected_global, actual_global, tolerance)
    else:
        global_comparison = {"available": False, "passed": False}
    return {
        "names_equal": names_equal,
        "tolerance": {
            "relative_l2": tolerance.relative_l2,
            "minimum_cosine": tolerance.minimum_cosine,
            "near_zero_norm": tolerance.near_zero_norm,
            "near_zero_atol": tolerance.near_zero_atol,
        },
        "global": global_comparison,
        "tensors": tensors,
        "passed": (
            names_equal
            and bool(common)
            and global_comparison["passed"]
            and all(result["passed"] for result in tensors.values())
        ),
    }


def _compare_vector(
    expected: torch.Tensor,
    actual: torch.Tensor,
    tolerance: VectorTolerance,
) -> dict[str, Any]:
    if expected.shape != actual.shape:
        return {"shape_equal": False, "passed": False}
    expected_float = expected.detach().to(torch.float64)
    actual_float = actual.detach().to(torch.float64)
    expected_nonfinite = int((~torch.isfinite(expected_float)).sum())
    actual_nonfinite = int((~torch.isfinite(actual_float)).sum())
    finite = expected_nonfinite == 0 and actual_nonfinite == 0
    base: dict[str, Any] = {
        "shape_equal": True,
        "shape": list(expected.shape),
        "expected_nonfinite_count": expected_nonfinite,
        "actual_nonfinite_count": actual_nonfinite,
        "finite": finite,
    }
    if not finite:
        return {
            **base,
            "expected_l2": None,
            "actual_l2": None,
            "difference_l2": None,
            "relative_l2": None,
            "cosine": None,
            "near_zero": None,
            "max_abs": None,
            "passed": False,
        }
    difference = actual_float - expected_float
    expected_norm = _finite_metric(torch.linalg.vector_norm(expected_float))
    actual_norm = _finite_metric(torch.linalg.vector_norm(actual_float))
    difference_norm = _finite_metric(torch.linalg.vector_norm(difference))
    max_abs = _finite_metric(difference.abs().max()) if difference.numel() else 0.0
    metrics_finite = all(value is not None for value in (expected_norm, actual_norm, difference_norm, max_abs))
    if not metrics_finite:
        return {
            **base,
            "metric_overflow": True,
            "expected_l2": expected_norm,
            "actual_l2": actual_norm,
            "difference_l2": difference_norm,
            "relative_l2": None,
            "cosine": None,
            "near_zero": None,
            "max_abs": max_abs,
            "passed": False,
        }
    assert expected_norm is not None and actual_norm is not None
    assert difference_norm is not None and max_abs is not None
    near_zero = expected_norm <= tolerance.near_zero_norm
    raw_relative_l2 = difference_norm / expected_norm if expected_norm > 0 else None
    relative_l2 = raw_relative_l2 if raw_relative_l2 is None or math.isfinite(raw_relative_l2) else None
    cosine = (
        _finite_metric(
            torch.dot(expected_float.reshape(-1), actual_float.reshape(-1)) / (expected_norm * actual_norm)
        )
        if expected_norm > 0 and actual_norm > 0
        else None
    )
    if near_zero:
        passed = max_abs <= tolerance.near_zero_atol
    else:
        passed = (
            relative_l2 is not None
            and relative_l2 <= tolerance.relative_l2
            and cosine is not None
            and cosine >= tolerance.minimum_cosine
        )
    return {
        **base,
        "metric_overflow": False,
        "expected_l2": expected_norm,
        "actual_l2": actual_norm,
        "difference_l2": difference_norm,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "near_zero": near_zero,
        "max_abs": max_abs,
        "passed": passed,
    }


def _compare_tensor(
    expected: torch.Tensor,
    actual: torch.Tensor,
    tolerance: ParityTolerance,
) -> dict[str, Any]:
    shape_equal = expected.shape == actual.shape
    if not shape_equal:
        return {"shape_equal": False, "passed": False}
    expected_float = expected.detach().to(dtype=torch.float64)
    actual_float = actual.detach().to(dtype=torch.float64)
    expected_nonfinite = int((~torch.isfinite(expected_float)).sum())
    actual_nonfinite = int((~torch.isfinite(actual_float)).sum())
    finite = expected_nonfinite == 0 and actual_nonfinite == 0
    if not finite:
        return {
            "shape_equal": True,
            "shape": list(expected.shape),
            "expected_nonfinite_count": expected_nonfinite,
            "actual_nonfinite_count": actual_nonfinite,
            "max_abs": None,
            "max_rel": None,
            "close_fraction": None,
            "atol": tolerance.atol,
            "rtol": tolerance.rtol,
            "passed": False,
        }
    difference = (actual_float - expected_float).abs()
    denominator = expected_float.abs().clamp_min(torch.finfo(torch.float64).eps)
    close = torch.isclose(actual_float, expected_float, atol=tolerance.atol, rtol=tolerance.rtol)
    max_abs = _finite_metric(difference.max()) if difference.numel() else 0.0
    max_rel = _finite_metric((difference / denominator).max()) if difference.numel() else 0.0
    metrics_finite = max_abs is not None and max_rel is not None
    return {
        "shape_equal": True,
        "shape": list(expected.shape),
        "expected_nonfinite_count": 0,
        "actual_nonfinite_count": 0,
        "metric_overflow": not metrics_finite,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "close_fraction": float(close.float().mean()) if close.numel() else 1.0,
        "atol": tolerance.atol,
        "rtol": tolerance.rtol,
        "passed": metrics_finite and bool(close.all()),
    }


def _finite_metric(value: torch.Tensor) -> float | None:
    return float(value) if bool(torch.isfinite(value)) else None


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().cpu().contiguous()
    float_value = cpu.float()
    nonfinite_count = int((~torch.isfinite(float_value)).sum())
    norm = _finite_metric(torch.linalg.vector_norm(float_value)) if cpu.numel() and not nonfinite_count else None
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256": _tensor_mapping_sha256({"value": cpu}),
        "nonfinite_count": nonfinite_count,
        "l2_norm": norm,
    }


def _variant_to_json(result: VariantResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "execution": result.execution,
        "initial_rng_sha256": result.initial_rng_sha256,
        "steps": [
            {
                "step": step_index,
                "pair_losses": [_json_scalar(value) for value in step.pair_losses],
                "pair_losses_by_key": {
                    f"step-{step_index:04d}/pair-{pair_index:02d}": _json_scalar(value)
                    for pair_index, value in enumerate(step.pair_losses)
                },
                "gradients": {name: _tensor_summary(value) for name, value in step.gradients.items()},
                "missing_gradients": list(step.missing_gradients),
                "parameters": {name: _tensor_summary(value) for name, value in step.parameters.items()},
                "adam_moments": {
                    name: {moment: _tensor_summary(value) for moment, value in moments.items()}
                    for name, moments in step.adam_moments.items()
                },
                "optimizer_steps": {name: _json_float(value) for name, value in step.optimizer_steps.items()},
                "rng_sha256": step.rng_sha256,
                "event_coverage": step.event_coverage,
                "recomputation": step.recomputation,
                "events": events_to_json(step.events),
            }
            for step_index, step in enumerate(result.steps)
        ],
    }


def _json_float(number: float) -> float | dict[str, str]:
    if math.isfinite(number):
        return number
    if math.isnan(number):
        classification = "nan"
    elif number > 0:
        classification = "positive_infinity"
    else:
        classification = "negative_infinity"
    return {"nonfinite": classification}


def _json_scalar(value: torch.Tensor) -> float | dict[str, str]:
    return _json_float(float(value))


def _clone_named_tensors(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in state.items()}


def _tensor_mapping_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash exact tensor metadata and bytes, including scalar tensors."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        byte_view = value.reshape(1).view(torch.uint8) if value.ndim == 0 else value.view(torch.uint8)
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(byte_view.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def _temporary_environment(updates: Mapping[str, str | None]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in updates}
    try:
        for name, value in updates.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _real_model_cli(args: argparse.Namespace) -> dict[str, Any]:
    config = load_train_config(args.train_config)
    repository = find_repository_root(Path(__file__))
    source = repository_source_identity(
        repository,
        executed_files=(Path(__file__), Path(__file__).with_name("adaptive_trace.py")),
        require_git=True,
    )
    if source["git_dirty"]:
        raise ValueError("CUDA parity evidence requires a clean source tree")
    if config.resume is not None:
        raise ValueError("CUDA parity CLI rejects training resume configs")
    if getattr(args, "require_resume_evidence", False):
        raise ValueError(
            "production resume evidence is unavailable: this benchmark does not save/reload production checkpoints"
        )
    if args.steps <= 0 or args.steps > config.max_steps:
        raise ValueError("CUDA parity steps must be positive and no greater than train max_steps")
    variant_set = getattr(args, "variant_set", "adaptive")
    if variant_set not in {"adaptive", "fixed_regional"}:
        raise ValueError(f"unsupported CUDA parity variant set: {variant_set!r}")
    policy = config.performance.adaptive_batching
    if variant_set == "adaptive" and not policy.enabled:
        raise ValueError("adaptive CUDA parity requires performance.adaptive_batching.enabled=true")
    if config.distributed != "none":
        raise ValueError("CUDA parity CLI requires distributed='none'")
    diffusion = config.model.diffusion_processor
    if diffusion is not None and diffusion.noise_scale > 0:
        raise ValueError("CUDA parity CLI rejects noisy Diffusion before model allocation")
    if config.model.family != "qwen3" or config.model.variant not in {"bt", "bgt"}:
        raise ValueError("compiled CUDA parity CLI currently supports deterministic Qwen3 BT/BGT only")
    processor = config.model.cache_processor
    if processor is not None and processor.attention_dropout != 0:
        raise ValueError("CUDA parity CLI rejects processor dropout")
    if config.model.lora is not None and config.model.lora.dropout != 0:
        raise ValueError("CUDA parity CLI rejects LoRA dropout")
    if processor is None:
        raise ValueError("CUDA parity CLI requires a standard BT/BGT cache processor")
    required_operations = ("select", "write_v" if processor.update_stream == "v" else "write_kv")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA parity requested but CUDA is unavailable")

    def build(backbone_backend: Qwen3BackboneChunkBackend | None) -> torch.nn.Module:
        compile_regions = backbone_backend is not None
        environment = {
            "GCW_BT_PROCESSOR_BACKEND": (
                "compile" if compile_regions and config.model.variant == "bt" else "eager"
            ),
            "GCW_CACHE_PROCESSOR_BACKEND": (
                "compile" if compile_regions and config.model.variant == "bgt" else "eager"
            ),
        }
        with _temporary_environment(environment):
            model = build_model(config.model).to(device)
            model.init_weights()
            if config.init_checkpoint:
                _load_initial_model(model, config.init_checkpoint)
            elif config.backbone_checkpoint:
                load_backbone_checkpoint(model, config.backbone_checkpoint)
            _apply_trainable_policy(model, config.trainable)
            if backbone_backend is not None:
                configure_qwen3_backbone_chunk_backend(model, backbone_backend)
            return model

    initial_model = build(None)
    stream = build_training_stream(config, device, rank=0, world_size=1)
    reserved = reserve_parity_steps(stream, args.steps)

    def pair_losses(model: torch.nn.Module, batch: TrainingBatch) -> torch.Tensor:
        row_sums, row_counts = row_loss_sums_and_counts(
            model,
            batch.x,
            batch.y,
            loss_backend=config.loss_backend,
            attention_mask=batch.attention_mask,
            activation_checkpoint_windows=config.activation_checkpoint_windows,
            activation_checkpoint_min_padded_sequence_length=(
                config.activation_checkpoint_min_padded_sequence_length
            ),
        )
        return row_sums.reshape(-1, 2).sum(dim=1) / row_counts.reshape(-1, 2).sum(dim=1).clamp_min(1)

    def optimizer_factory(model: torch.nn.Module) -> torch.optim.Optimizer:
        return _build_optimizer([parameter for parameter in model.parameters() if parameter.requires_grad], config)

    def autocast_context(_model: torch.nn.Module, _batch: TrainingBatch) -> AbstractContextManager[Any]:
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=config.bf16 and device.type == "cuda",
        )

    def expects_checkpoint(batch: TrainingBatch) -> bool:
        threshold = config.activation_checkpoint_min_padded_sequence_length
        return config.activation_checkpoint_windows > 0 and (threshold == 0 or batch.x.shape[1] >= threshold)

    def prepare_optimizer_step(
        model: torch.nn.Module,
        parameters: Sequence[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        step: int,
    ) -> None:
        if config.grad_clip:
            _clip_grad_norm(model, list(parameters), config.grad_clip)
        lr_scale = _wsd_lr(
            step, config.max_steps, config.warmup_steps, config.warmdown_ratio, config.final_lr_frac
        )
        weight_decay = _weight_decay(step, config.max_steps, config.weight_decay)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_scale
            if group.get("decay", False):
                group["weight_decay"] = weight_decay

    if variant_set == "adaptive":
        variants = (
            ParityVariant("fixed_batch2_eager", "fixed", lambda: build(None)),
            ParityVariant("adaptive_eager", "adaptive", lambda: build(None)),
            ParityVariant("adaptive_compiled", "adaptive", lambda: build("compile")),
        )
    elif variant_set == "fixed_regional":
        variants = (
            ParityVariant("fixed_batch2_eager", "fixed", lambda: build(None)),
            ParityVariant("fixed_batch2_compiled", "fixed", lambda: build("compile")),
            ParityVariant(
                "fixed_batch2_compile_full_windows",
                "fixed",
                lambda: build("compile_full_windows"),
            ),
        )
    else:
        raise ValueError(f"unsupported CUDA parity variant set: {variant_set!r}")
    payload = run_adaptive_parity(
        initial_model,
        variants,
        reserved,
        pair_losses,
        optimizer_factory,
        adaptive_policy=policy,
        process_every=processor.process_every,
        required_event_operations=required_operations,
        checkpoint_expected=expects_checkpoint,
        forward_context=autocast_context,
        prepare_optimizer_step=prepare_optimizer_step,
    )
    payload.update(
        {
            "source_commit": source["git_commit"],
            "source_dirty": source["git_dirty"],
            "source_tree_sha256": source["source_tree_sha256"],
            "train_config": str(args.train_config.resolve()),
            "train_config_file_sha256": file_sha256(args.train_config),
            "train_config_sha256": canonical_json_sha256(to_dict(config)),
            "environment": device_identity(device),
            "device": str(device),
            "variant": config.model.variant,
            "trajectory_steps": args.steps,
            "variant_set": variant_set,
        }
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare fixed batch-2, adaptive eager, and adaptive compiled training without timing"
    )
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--variant-set",
        choices=("adaptive", "fixed_regional"),
        default="adaptive",
        help="compare adaptive grouping or exact-shape fixed regional compilers against fixed eager",
    )
    parser.add_argument(
        "--require-resume-evidence",
        action="store_true",
        help="fail closed because production checkpoint save/reload evidence is not implemented here",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = _real_model_cli(args)
    document = write_authenticated_json(args.output, payload)
    print(json.dumps(document, sort_keys=True, allow_nan=False))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
