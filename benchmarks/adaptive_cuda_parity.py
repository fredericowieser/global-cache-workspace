"""Formal fixed/adaptive CUDA parity harness.

This module never times instrumented execution.  It aligns semantic cache
operations by stable source-pair and row identities, then compares losses,
gradients, optimizer moments, RNG, cursor state, and a short parameter
trajectory.  Production training remains untouched.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
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
from gcw.performance.backends.qwen3 import configure_qwen3_backbone_chunk_backend
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
PrepareOptimizerStep = Callable[
    [torch.nn.Module, Sequence[torch.nn.Parameter], torch.optim.Optimizer, int],
    None,
]


@dataclass(frozen=True)
class ParityTolerance:
    """Numerical tolerances for one aligned tensor family."""

    atol: float
    rtol: float


@dataclass(frozen=True)
class ParityTolerances:
    pair_losses: ParityTolerance = ParityTolerance(2e-5, 2e-4)
    gradients: ParityTolerance = ParityTolerance(2e-5, 2e-4)
    parameters: ParityTolerance = ParityTolerance(2e-5, 2e-4)
    adam_moments: ParityTolerance = ParityTolerance(2e-6, 2e-4)


@dataclass(frozen=True)
class ParityVariant:
    """One execution policy initialized from the common model state."""

    name: str
    execution: ExecutionKind
    model_factory: ModelFactory


@dataclass(frozen=True)
class ReservedParityStep:
    """One immutable 32-pair reservation plus cursor evidence."""

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
    parameters: dict[str, torch.Tensor]
    adam_moments: dict[str, dict[str, torch.Tensor]]
    optimizer_steps: dict[str, float]
    rng_sha256: str
    events: list[CacheEvent]


@dataclass
class VariantResult:
    name: str
    execution: ExecutionKind
    cursor_before: Mapping[str, Any]
    cursor_after: Mapping[str, Any]
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
    forward_context: ForwardContext | None = None,
    prepare_optimizer_step: PrepareOptimizerStep | None = None,
    tolerances: ParityTolerances = ParityTolerances(),
) -> dict[str, Any]:
    """Run and compare fixed batch-2, adaptive eager, and adaptive compiled paths."""
    if len(variants) < 2:
        raise ValueError("formal parity requires at least two execution variants")
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise ValueError("parity variant names must be unique")
    if not reserved_steps:
        raise ValueError("formal parity requires reserved steps")
    initial_state = _clone_named_tensors(initial_model.state_dict())
    initial_rng = capture_rng()
    context_factory = forward_context or (lambda _model, _batch: nullcontext())
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
        "schema": "gcw.adaptive_cuda_parity.v1",
        "instrumented": True,
        "timing_policy": "forbidden_in_instrumented_runs",
        "initial_model_sha256": _tensor_mapping_sha256(initial_state),
        "initial_rng_sha256": initial_rng.digest(),
        "reserved_steps": [
            {
                "step": step_index,
                "cursor_before": dict(step.cursor_before),
                "cursor_after": dict(step.cursor_after),
                "padded_lengths": list(step.reservation.padded_lengths),
            }
            for step_index, step in enumerate(reserved_steps)
        ],
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
        recorder = CacheEventRecorder()
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
                )
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
        if len(gradients) != len(parameters):
            missing = sorted(name for name, parameter in parameters if parameter.grad is None)
            raise RuntimeError(f"parity execution has missing named gradients: {missing}")
        if prepare_optimizer_step is not None:
            prepare_optimizer_step(model, [parameter for _, parameter in parameters], optimizer, step_index)
        optimizer.step()
        moments, optimizer_steps = _optimizer_snapshot(optimizer, parameters)
        step_results.append(
            VariantStepResult(
                pair_losses=torch.stack([value for value in pair_values if value is not None]),
                gradients=gradients,
                parameters={name: parameter.detach().cpu().clone() for name, parameter in parameters},
                adam_moments=moments,
                optimizer_steps=optimizer_steps,
                rng_sha256=capture_rng().digest(),
                events=list(recorder.events),
            )
        )
    return VariantResult(
        name=variant.name,
        execution=variant.execution,
        cursor_before=reserved_steps[0].cursor_before,
        cursor_after=reserved_steps[-1].cursor_after,
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
        gradient_comparison = _compare_named_tensors(expected.gradients, actual.gradients, tolerances.gradients)
        parameter_comparison = _compare_named_tensors(expected.parameters, actual.parameters, tolerances.parameters)
        moment_comparison = _compare_nested_tensors(expected.adam_moments, actual.adam_moments, tolerances.adam_moments)
        event_comparison = _compare_events(expected.events, actual.events)
        optimizer_steps_equal = expected.optimizer_steps == actual.optimizer_steps
        rng_equal = expected.rng_sha256 == actual.rng_sha256
        passed = all(
            (
                pair_comparison["passed"],
                gradient_comparison["passed"],
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
                "parameters": parameter_comparison,
                "adam_moments": moment_comparison,
                "optimizer_steps_equal": optimizer_steps_equal,
                "rng_equal": rng_equal,
                "cache_events": event_comparison,
                "passed": passed,
            }
        )
    cursor_equal = (
        dict(reference.cursor_before) == dict(candidate.cursor_before)
        and dict(reference.cursor_after) == dict(candidate.cursor_after)
    )
    initial_rng_equal = reference.initial_rng_sha256 == candidate.initial_rng_sha256
    return {
        "reference": reference.name,
        "candidate": candidate.name,
        "cursor_equal": cursor_equal,
        "initial_rng_equal": initial_rng_equal,
        "steps": step_comparisons,
        "passed": cursor_equal and initial_rng_equal and all(step["passed"] for step in step_comparisons),
    }


def _compare_events(expected: Sequence[CacheEvent], actual: Sequence[CacheEvent]) -> dict[str, Any]:
    expected_primary = {
        event.event_key: event.destinations for event in expected if event.phase == "primary"
    }
    actual_primary = {
        event.event_key: event.destinations for event in actual if event.phase == "primary"
    }
    expected_recompute = _recomputation_status(expected)
    actual_recompute = _recomputation_status(actual)
    expected_coverage = _trace_coverage(expected)
    actual_coverage = _trace_coverage(actual)
    return {
        "primary_equal": expected_primary == actual_primary,
        "reference_primary_count": len(expected_primary),
        "candidate_primary_count": len(actual_primary),
        "reference_recomputation": expected_recompute,
        "candidate_recomputation": actual_recompute,
        "reference_coverage": expected_coverage,
        "candidate_coverage": actual_coverage,
        "passed": (
            expected_primary == actual_primary
            and expected_coverage["complete"]
            and actual_coverage["complete"]
            and all(expected_recompute.values())
            and all(actual_recompute.values())
        ),
    }


def _trace_coverage(events: Sequence[CacheEvent]) -> dict[str, Any]:
    primary_operations = {event.operation for event in events if event.phase == "primary"}
    has_selection = "select" in primary_operations
    has_write = any(operation.startswith("write") for operation in primary_operations)
    return {
        "primary_operations": sorted(primary_operations),
        "has_selection": has_selection,
        "has_write": has_write,
        "complete": has_selection and has_write,
    }


def _recomputation_status(events: Sequence[CacheEvent]) -> dict[str, bool]:
    primary = {event.event_key: event.destinations for event in events if event.phase == "primary"}
    return {
        event.event_key: primary.get(event.recomputation_of or "") == event.destinations
        for event in events
        if event.phase == "recomputation"
    }


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


def _compare_nested_tensors(
    expected: Mapping[str, Mapping[str, torch.Tensor]],
    actual: Mapping[str, Mapping[str, torch.Tensor]],
    tolerance: ParityTolerance,
) -> dict[str, Any]:
    names_equal = expected.keys() == actual.keys()
    comparisons = {
        name: _compare_named_tensors(expected[name], actual[name], tolerance)
        for name in expected.keys() & actual.keys()
    }
    return {
        "names_equal": names_equal,
        "parameters": comparisons,
        "passed": names_equal and all(result["passed"] for result in comparisons.values()),
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
    difference = (actual_float - expected_float).abs()
    denominator = expected_float.abs().clamp_min(torch.finfo(torch.float64).eps)
    close = torch.isclose(actual_float, expected_float, atol=tolerance.atol, rtol=tolerance.rtol)
    return {
        "shape_equal": True,
        "shape": list(expected.shape),
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "max_rel": float((difference / denominator).max()) if difference.numel() else 0.0,
        "close_fraction": float(close.float().mean()) if close.numel() else 1.0,
        "atol": tolerance.atol,
        "rtol": tolerance.rtol,
        "passed": bool(close.all()),
    }


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().cpu().contiguous()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256": _tensor_mapping_sha256({"value": cpu}),
        "l2_norm": float(torch.linalg.vector_norm(cpu.float())) if cpu.numel() else 0.0,
    }


def _variant_to_json(result: VariantResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "execution": result.execution,
        "cursor_before": dict(result.cursor_before),
        "cursor_after": dict(result.cursor_after),
        "initial_rng_sha256": result.initial_rng_sha256,
        "steps": [
            {
                "step": step_index,
                "pair_losses": step.pair_losses.tolist(),
                "pair_losses_by_key": {
                    f"step-{step_index:04d}/pair-{pair_index:02d}": float(value)
                    for pair_index, value in enumerate(step.pair_losses)
                },
                "gradients": {name: _tensor_summary(value) for name, value in step.gradients.items()},
                "parameters": {name: _tensor_summary(value) for name, value in step.parameters.items()},
                "adam_moments": {
                    name: {moment: _tensor_summary(value) for moment, value in moments.items()}
                    for name, moments in step.adam_moments.items()
                },
                "optimizer_steps": step.optimizer_steps,
                "rng_sha256": step.rng_sha256,
                "events": events_to_json(step.events),
            }
            for step_index, step in enumerate(result.steps)
        ],
    }


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
    policy = config.performance.adaptive_batching
    if not policy.enabled:
        raise ValueError("CUDA parity CLI requires performance.adaptive_batching.enabled=true")
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
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA parity requested but CUDA is unavailable")

    def build(compile_regions: bool) -> torch.nn.Module:
        environment = {
            "GCW_BT_PROCESSOR_BACKEND": "compile" if compile_regions and config.model.variant == "bt" else "eager",
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
            if compile_regions:
                configure_qwen3_backbone_chunk_backend(model, "compile")
            return model

    initial_model = build(False)
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

    variants = (
        ParityVariant("fixed_batch2_eager", "fixed", lambda: build(False)),
        ParityVariant("adaptive_eager", "adaptive", lambda: build(False)),
        ParityVariant("adaptive_compiled", "adaptive", lambda: build(True)),
    )
    payload = run_adaptive_parity(
        initial_model,
        variants,
        reserved,
        pair_losses,
        optimizer_factory,
        adaptive_policy=policy,
        forward_context=autocast_context,
        prepare_optimizer_step=prepare_optimizer_step,
    )
    repository = find_repository_root(Path(__file__))
    source = repository_source_identity(
        repository,
        executed_files=(Path(__file__), Path(__file__).with_name("adaptive_trace.py")),
        require_git=True,
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
