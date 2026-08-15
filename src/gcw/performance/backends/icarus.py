"""Explicit training-only execution backend for deterministic Icarus tensors."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
import os
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import torch.nn as nn

from gcw.performance._compile import require_torch_compile
from gcw.provenance import canonical_json_sha256

if TYPE_CHECKING:
    from gcw.config import ModelConfig


IcarusProcessorBackend = Literal["eager", "compile"]
IcarusProcessorRunner = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

_ICARUS_BACKEND_ENV = "GCW_ICARUS_PROCESSOR_BACKEND"
_COMPILE_OPTIONS = {
    "backend": "inductor",
    "dynamic": True,
    "fullgraph": True,
}
_RUNNER_ATTRIBUTE = "_gcw_icarus_compiled_runner"
_ADMITTED_BATCHES_ATTRIBUTE = "_gcw_icarus_admitted_training_batches"
_ACTIVE_BATCH_ATTRIBUTE = "_gcw_icarus_active_training_batch"


def resolve_icarus_processor_backend(model_config: ModelConfig) -> IcarusProcessorBackend:
    """Resolve and validate the narrow post-selection Icarus backend."""
    requested = os.environ.get(_ICARUS_BACKEND_ENV, "eager").strip().lower()
    if requested not in {"eager", "compile"}:
        raise ValueError(f"{_ICARUS_BACKEND_ENV} must be 'eager' or 'compile'")
    if requested == "eager":
        return "eager"
    if model_config.variant != "icarus":
        raise RuntimeError(
            f"{_ICARUS_BACKEND_ENV}=compile supports deterministic Icarus only; "
            f"{model_config.variant} is explicitly unsupported"
        )
    processor = model_config.cache_processor
    if processor is None or processor.processor_mode != "standard":
        raise RuntimeError(f"{_ICARUS_BACKEND_ENV}=compile requires Icarus processor_mode='standard'")
    if processor.attention_dropout != 0.0:
        raise RuntimeError(f"{_ICARUS_BACKEND_ENV}=compile requires attention_dropout=0")
    if model_config.flow_processor is not None or model_config.diffusion_processor is not None:
        raise RuntimeError(
            f"{_ICARUS_BACKEND_ENV}=compile supports the ordinary Icarus transform only; "
            "Flow and Diffusion refinements are explicitly unsupported"
        )
    require_torch_compile(f"{_ICARUS_BACKEND_ENV}=compile")
    return "compile"


def validate_icarus_training_admission(
    backend: IcarusProcessorBackend,
    *,
    distributed: str,
    world_size: int,
    distributed_initialized: bool,
    adaptive_batch_sizes: Sequence[int] | None,
    device: torch.device,
) -> None:
    """Fail before model execution for topologies and shapes lacking CUDA evidence."""
    if backend == "eager":
        return
    if backend != "compile":
        raise ValueError(f"unsupported Icarus processor backend: {backend}")
    if distributed != "none" or world_size != 1 or distributed_initialized:
        raise RuntimeError(
            "GCW_ICARUS_PROCESSOR_BACKEND=compile is not supported with DDP or FSDP; "
            "distributed execution has not been CUDA-qualified"
        )
    if adaptive_batch_sizes is not None and (
        device.type != "cuda" or set(adaptive_batch_sizes) != {2, 4, 8, 16}
    ):
        raise RuntimeError(
            "compiled Icarus adaptive batching is qualified only on CUDA with the 16/8/4/2 row bands"
        )


def icarus_processor_backend_metadata(
    model_config: ModelConfig,
    backend: IcarusProcessorBackend | None = None,
) -> dict[str, object]:
    """Return provenance and an exact-resume fingerprint for Icarus execution."""
    selected = resolve_icarus_processor_backend(model_config) if backend is None else backend
    execution = {
        "schema": "gcw.icarus_processor_backend.v2",
        "effective_backend": selected,
        "architecture": "icarus",
        "region": "gcw.performance.backends.icarus._run_icarus_post_selection",
        "compile_options": dict(_COMPILE_OPTIONS) if selected == "compile" else None,
        "scope": "admitted_training_forward_and_backward_only" if selected == "compile" else None,
        "runner_ownership": "processor_module" if selected == "compile" else None,
        "selection_policy": "eager_fp32_exact_topk",
        "writeback_policy": "eager_exact",
        "direct_processor_calls": "eager",
        "generation": "eager",
        "torch_version": str(torch.__version__),
    }
    return {
        **execution,
        "requested_via": _ICARUS_BACKEND_ENV,
        "requested_value": os.environ.get(_ICARUS_BACKEND_ENV, "eager"),
        "implementation_module": "gcw.performance.backends.icarus",
        "implementation_symbols": [
            "gcw.performance.backends.icarus.run_icarus_post_selection_region",
            "gcw.performance.backends.icarus._run_icarus_post_selection",
        ],
        "fingerprint_sha256": canonical_json_sha256(execution),
    }


def configure_icarus_training_execution(model: nn.Module, admitted_batch_sizes: Sequence[int]) -> None:
    """Attach admitted training shapes to each Icarus block after model allocation.

    This is an allocation-time preflight, not a claim that Inductor kernels have
    been generated. Kernels are generated only by :func:`warmup_icarus_training_execution`.
    """
    batches = tuple(sorted(set(int(size) for size in admitted_batch_sizes)))
    if not batches or any(size <= 0 for size in batches):
        raise ValueError("compiled Icarus training requires positive admitted batch sizes")
    processors = _compiled_icarus_processors(model)
    if not processors:
        raise RuntimeError("compiled Icarus allocation preflight found no allocated Icarus processor")
    for processor in processors:
        object.__setattr__(processor.block, _ADMITTED_BATCHES_ATTRIBUTE, batches)


@contextmanager
def icarus_training_execution(model: nn.Module, batch_size: int) -> Iterator[None]:
    """Activate compilation only for one admitted training forward/backward."""
    active: list[tuple[nn.Module, object]] = []
    for processor in _compiled_icarus_processors(model):
        block = processor.block
        admitted = block.__dict__.get(_ADMITTED_BATCHES_ATTRIBUTE)
        if admitted is None:
            raise RuntimeError(
                "compiled Icarus execution was entered before allocation preflight; "
                "configure_icarus_training_execution must run first"
            )
        if batch_size not in admitted:
            raise RuntimeError(
                f"compiled Icarus training batch {batch_size} was not admitted; admitted batches are {admitted}"
            )
        previous = block.__dict__.get(_ACTIVE_BATCH_ATTRIBUTE)
        if previous is not None and previous != batch_size:
            raise RuntimeError("nested compiled Icarus scopes must use the same batch size")
        active.append((block, previous))
        object.__setattr__(block, _ACTIVE_BATCH_ATTRIBUTE, batch_size)
    try:
        yield
    finally:
        for block, previous in reversed(active):
            if previous is None:
                block.__dict__.pop(_ACTIVE_BATCH_ATTRIBUTE, None)
            else:
                object.__setattr__(block, _ACTIVE_BATCH_ATTRIBUTE, previous)


def warmup_icarus_training_execution(
    model: nn.Module,
    admitted_batch_sizes: Sequence[int],
    *,
    sequence_length: int,
    device: torch.device,
    bf16: bool,
) -> None:
    """Generate admitted forward/backward kernels without data or RNG consumption."""
    processors = _compiled_icarus_processors(model)
    if not processors:
        return
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        for processor in processors:
            block = processor.block
            config: Any = processor.config
            cache_heads = int(processor.num_cache_heads)
            selected_lengths = _training_selected_lengths(
                sequence_length, int(config.process_every), int(config.topk)
            )
            selected_count = min(int(config.topk_heads), cache_heads)
            parameter = next(block.parameters())
            input_dtype = torch.bfloat16 if bf16 and device.type == "cuda" else parameter.dtype
            for batch_size in sorted(set(int(size) for size in admitted_batch_sizes), reverse=True):
                for selected_length in sorted(selected_lengths, reverse=True):
                    x = torch.zeros(
                        batch_size,
                        cache_heads,
                        selected_length,
                        2 * int(processor.head_dim),
                        device=device,
                        dtype=input_dtype,
                        requires_grad=True,
                    )
                    positions = torch.arange(selected_length, device=device).view(1, 1, -1).expand(
                        batch_size, cache_heads, -1
                    )
                    selected_heads = torch.arange(
                        selected_count, device=device, dtype=torch.long
                    ).view(1, -1).expand(batch_size, -1)
                    with icarus_training_execution(processor, batch_size):
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16,
                            enabled=bf16 and device.type == "cuda",
                        ):
                            output = run_icarus_post_selection_region(
                                "compile", block, x, positions, selected_heads
                            )
                        trainable = tuple(parameter for parameter in block.parameters() if parameter.requires_grad)
                        torch.autograd.grad(output.float().sum(), (x, *trainable), allow_unused=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def _training_selected_lengths(sequence_length: int, process_every: int, topk: int) -> tuple[int, ...]:
    if sequence_length <= 0 or process_every <= 0:
        raise ValueError("Icarus warmup sequence_length and process_every must be positive")
    lengths = {
        process_every + min(topk, cached_length - process_every)
        for cached_length in range(process_every, sequence_length, process_every)
    }
    return tuple(sorted(lengths))


def run_icarus_post_selection_region(
    backend: IcarusProcessorBackend,
    block: nn.Module,
    x: torch.Tensor,
    positions: torch.Tensor,
    selected_heads: torch.Tensor,
) -> torch.Tensor:
    """Run the pure transform; compilation requires an explicit admitted scope."""
    active_batch = block.__dict__.get(_ACTIVE_BATCH_ATTRIBUTE)
    if backend == "eager" or active_batch is None:
        return _run_icarus_post_selection(block, x, positions, selected_heads)
    if backend != "compile":
        raise ValueError(f"unsupported Icarus processor backend: {backend}")
    if x.shape[0] != active_batch:
        raise RuntimeError(
            f"compiled Icarus scope admitted batch {active_batch}, but region received {x.shape[0]} rows"
        )
    runner = _get_compiled_icarus_runner(block)
    try:
        return runner(x, positions, selected_heads)
    except Exception as error:
        if _is_non_reentrant_checkpoint_early_stop(error):
            raise
        raise RuntimeError(
            "compiled Icarus processor region failed with "
            f"shape={tuple(x.shape)}, positions={tuple(positions.shape)}, heads={tuple(selected_heads.shape)}, "
            f"dtype={x.dtype}, device={x.device}; eager fallback is disabled when "
            f"{_ICARUS_BACKEND_ENV}=compile"
        ) from error


def _run_icarus_post_selection(
    block: nn.Module,
    x: torch.Tensor,
    positions: torch.Tensor,
    selected_heads: torch.Tensor,
) -> torch.Tensor:
    return block(x, positions, selected_heads)


def _get_compiled_icarus_runner(block: nn.Module) -> IcarusProcessorRunner:
    runner: IcarusProcessorRunner | None = block.__dict__.get(_RUNNER_ATTRIBUTE)
    if runner is not None:
        return runner
    compiler = require_torch_compile(f"{_ICARUS_BACKEND_ENV}=compile")
    try:
        runner = cast(
            IcarusProcessorRunner,
            compiler(partial(_run_icarus_post_selection, block), **_COMPILE_OPTIONS),
        )
    except Exception as error:
        raise RuntimeError(
            "failed to create the module-owned compiled Icarus processor region after allocation; "
            "eager fallback is disabled"
        ) from error
    object.__setattr__(block, _RUNNER_ATTRIBUTE, runner)
    return runner


def _compiled_icarus_processors(model: nn.Module) -> list[Any]:
    processors: list[Any] = []
    for module in model.modules():
        if (
            getattr(module, "execution_backend", None) == "compile"
            and getattr(getattr(module, "_model_config", None), "variant", None) == "icarus"
            and hasattr(module, "block")
        ):
            processors.append(module)
    return processors


def _is_non_reentrant_checkpoint_early_stop(error: Exception) -> bool:
    """Preserve checkpoint's private control flow without weakening fail-closed execution."""
    error_type = type(error)
    return error_type.__module__ == "torch.utils.checkpoint" and error_type.__name__ == "_StopRecomputationError"
