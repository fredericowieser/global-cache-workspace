"""Default-off CUDA Graph callables with external runtime ownership."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import warnings
from typing import Any, Callable
from weakref import WeakKeyDictionary

import torch
import torch.nn as nn

TensorOperation = Callable[..., Any]


@dataclass(frozen=True)
class TensorGraphSignature:
    region: str
    tensors: tuple[tuple[tuple[int, ...], tuple[int, ...], str, str, bool], ...]


@dataclass(frozen=True)
class CUDAGraphCaptureStats:
    signature: TensorGraphSignature
    capture_ordinal: int
    allocated_growth_bytes: int
    reserved_growth_bytes: int


@dataclass
class _GraphEntry:
    runner: TensorOperation
    operation: TensorOperation
    stats: CUDAGraphCaptureStats
    replay_count: int = 0


class _OwnerState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.entries: dict[TensorGraphSignature, _GraphEntry] = {}
        self.capture_count = 0


_STATES: WeakKeyDictionary[nn.Module, _OwnerState] = WeakKeyDictionary()
_STATES_LOCK = threading.Lock()


def tensor_graph_signature(
    region: str, tensors: tuple[torch.Tensor, ...]
) -> TensorGraphSignature:
    """Return a closed shape/layout identity for one graph region."""
    if not region:
        raise ValueError("CUDA Graph region must be nonempty")
    if not tensors:
        raise ValueError("CUDA Graph signature requires tensor inputs")
    return TensorGraphSignature(
        region,
        tuple(
            (
                tuple(tensor.shape),
                tuple(tensor.stride()),
                str(tensor.dtype),
                str(tensor.device),
                tensor.requires_grad,
            )
            for tensor in tensors
        ),
    )


def run_graphed_callable(
    owner: nn.Module,
    *,
    region: str,
    operation: TensorOperation,
    args: tuple[torch.Tensor, ...],
    warmup_iterations: int = 3,
) -> Any:
    """Capture or replay one exact callable; failures never retry eagerly."""
    if not args or any(tensor.device.type != "cuda" for tensor in args):
        raise RuntimeError("CUDA Graph execution requires CUDA tensor inputs")
    if warmup_iterations <= 0:
        raise ValueError("CUDA Graph warmup_iterations must be positive")
    devices = {tensor.device for tensor in args}
    if len(devices) != 1:
        raise ValueError("CUDA Graph inputs must share one device")
    signature = tensor_graph_signature(region, args)
    state = _owner_state(owner)
    with state.lock:
        entry = state.entries.get(signature)
        if entry is None:
            entry = _capture_entry(
                owner,
                operation,
                args,
                signature,
                state.capture_count,
                warmup_iterations,
            )
            state.entries[signature] = entry
            state.capture_count += 1
        elif entry.operation is not operation:
            # Bound methods and partial objects are commonly recreated per call.
            # Region+owner+tensor signature remains authoritative, but changing
            # the underlying Python code object is not allowed.
            if _operation_identity(entry.operation) != _operation_identity(
                operation
            ):
                raise RuntimeError(
                    "CUDA Graph region was rebound to a different operation"
                )
        try:
            result = entry.runner(*args)
        except BaseException as exc:
            raise RuntimeError(
                f"CUDA Graph replay failed for region {region!r}; eager fallback is disabled"
            ) from exc
        entry.replay_count += 1
        return result


def cuda_graph_capture_stats(
    owner: nn.Module,
) -> tuple[CUDAGraphCaptureStats, ...]:
    """Return observed capture records without creating owner state."""
    with _STATES_LOCK:
        state = _STATES.get(owner)
    if state is None:
        return ()
    with state.lock:
        return tuple(entry.stats for entry in state.entries.values())


def cuda_graph_replay_counts(
    owner: nn.Module,
) -> dict[TensorGraphSignature, int]:
    """Return replay counts for qualification only."""
    with _STATES_LOCK:
        state = _STATES.get(owner)
    if state is None:
        return {}
    with state.lock:
        return {
            signature: entry.replay_count
            for signature, entry in state.entries.items()
        }


def clear_cuda_graph_runtime(owner: nn.Module) -> None:
    """Release the external owner mapping after work has quiesced."""
    if any(parameter.device.type == "cuda" for parameter in owner.parameters()):
        torch.cuda.synchronize(next(owner.parameters()).device)
    with _STATES_LOCK:
        _STATES.pop(owner, None)


def _owner_state(owner: nn.Module) -> _OwnerState:
    with _STATES_LOCK:
        state = _STATES.get(owner)
        if state is None:
            state = _OwnerState()
            _STATES[owner] = state
        return state


def _capture_entry(
    owner: nn.Module,
    operation: TensorOperation,
    args: tuple[torch.Tensor, ...],
    signature: TensorGraphSignature,
    ordinal: int,
    warmup_iterations: int,
) -> _GraphEntry:
    device = args[0].device
    index = (
        device.index
        if device.index is not None
        else torch.cuda.current_device()
    )
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(index)
    gradients = tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in owner.parameters()
    )
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    # Capture from isolated leaf tensors. Passing live nonleaf cache tensors
    # would make capture-time autograd traverse into the preceding processor on
    # the legacy stream, invalidating CUDA stream capture. The graphed callable
    # later reconnects its input gradients to each real nonleaf argument.
    sample_args = tuple(
        tensor.detach()
        .clone(memory_format=torch.preserve_format)
        .requires_grad_(tensor.requires_grad)
        for tensor in args
    )
    autocast_enabled = torch.is_autocast_enabled("cuda")
    autocast_dtype = torch.get_autocast_dtype("cuda")
    try:
        # make_graphed_callables deliberately captures on a side stream. Its
        # internal autograd probe can trigger PyTorch's generic cross-stream
        # AccumulateGrad warning even though this API owns the required waits.
        # Suppress only that capture-local diagnostic, never runtime warnings.
        with (
            warnings.catch_warnings(),
            torch.autocast(
                "cuda",
                enabled=autocast_enabled,
                dtype=autocast_dtype,
                cache_enabled=False,
            ),
        ):
            warnings.filterwarnings(
                "ignore",
                message="The AccumulateGrad node's stream does not match.*",
                category=UserWarning,
            )
            runner = torch.cuda.make_graphed_callables(
                operation,
                sample_args,
                num_warmup_iters=warmup_iterations,
                allow_unused_input=True,
            )
    except BaseException as exc:
        raise RuntimeError(
            f"CUDA Graph capture failed for region {signature.region!r}; eager fallback is disabled"
        ) from exc
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, index)
        for parameter, gradient in zip(
            owner.parameters(), gradients, strict=True
        ):
            parameter.grad = gradient
    if not callable(runner):
        raise RuntimeError("CUDA Graph capture returned a non-callable runner")
    stats = CUDAGraphCaptureStats(
        signature=signature,
        capture_ordinal=ordinal,
        allocated_growth_bytes=int(
            torch.cuda.memory_allocated(device) - allocated_before
        ),
        reserved_growth_bytes=int(
            torch.cuda.memory_reserved(device) - reserved_before
        ),
    )
    return _GraphEntry(runner=runner, operation=operation, stats=stats)


def _operation_identity(
    operation: TensorOperation,
) -> tuple[object, object | None]:
    function = getattr(operation, "func", operation)
    return getattr(function, "__code__", function), getattr(
        operation, "args", None
    )
