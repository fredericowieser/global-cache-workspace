from __future__ import annotations

from collections.abc import Callable
from functools import partial
from threading import Lock
from typing import Any, Literal

import torch
import torch.nn as nn

from gcw.performance._compile import (
    is_non_reentrant_checkpoint_early_stop,
    require_torch_compile,
)
from gcw.performance.compile_telemetry import compile_acceptance_recorder
from gcw.performance.cuda_graphs import run_graphed_callable
from gcw.provenance import canonical_json_sha256


Qwen3BackboneChunkBackend = Literal["eager", "compile", "compile_full_windows", "cuda_graph"]
Qwen3ChunkRunner = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]
Qwen3InitialChunkRunner = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]

_QWEN3_CHUNK_STEPS = 64
_QWEN3_COMPILE_OPTIONS = {
    "backend": "inductor",
    "dynamic": True,
    "fullgraph": True,
}
_QWEN3_INITIAL_COMPILE_OPTIONS = {
    "backend": "inductor",
    "dynamic": False,
    "fullgraph": True,
}
_QWEN3_RUNNER_ATTRIBUTE = "_gcw_qwen3_compiled_chunk_runner"
_QWEN3_INITIAL_RUNNER_ATTRIBUTE = "_gcw_qwen3_compiled_initial_chunk_runner"
_qwen3_compiled_runners_lock = Lock()


def validate_qwen3_backbone_chunk_backend(backend: Qwen3BackboneChunkBackend) -> None:
    """Reject unavailable or fallback-enabled compile execution before dispatch."""
    if backend == "eager":
        return
    if backend == "cuda_graph":
        if not torch.cuda.is_available():
            raise RuntimeError("backbone_chunk_backend=cuda_graph requires CUDA")
        return
    if backend not in {"compile", "compile_full_windows"}:
        raise ValueError(f"unsupported Qwen3 backbone chunk backend: {backend}")
    require_torch_compile(f"backbone_chunk_backend={backend}")


def configure_qwen3_backbone_chunk_backend(
    model: nn.Module,
    backend: Qwen3BackboneChunkBackend,
) -> None:
    """Configure a state-neutral backend only after the Qwen3 backbone is frozen."""
    validate_qwen3_backbone_chunk_backend(backend)
    backbone = getattr(model, "backbone", model)
    config: Any = getattr(backbone, "config", None)
    if backend != "eager":
        if config is None or getattr(config, "family", None) != "qwen3":
            raise ValueError(f"backbone_chunk_backend={backend} requires a Qwen3 backbone")
        if any(parameter.requires_grad for parameter in backbone.parameters()):
            raise ValueError(f"backbone_chunk_backend={backend} requires a frozen backbone")
    if not hasattr(backbone, "backbone_chunk_backend"):
        if backend != "eager":
            raise ValueError(f"backbone_chunk_backend={backend} requires the standard Qwen3 transformer")
        return
    object.__setattr__(backbone, "backbone_chunk_backend", backend)


def qwen3_backbone_chunk_backend_metadata(
    backend: Qwen3BackboneChunkBackend,
) -> dict[str, object]:
    """Return stable provenance for the explicitly selected Qwen3 chunk region."""
    if backend in {"eager", "compile"}:
        # Keep the complete v1 payload stable so old exact-resume fingerprints remain valid.
        execution: dict[str, object] = {
            "schema": "gcw.qwen3_backbone_chunk_backend.v1",
            "effective_backend": backend,
            "region": "gcw.models.backbones.backend._qwen3_backbone_chunk_tensor_region",
            "chunk_steps": _QWEN3_CHUNK_STEPS,
            "dynamic_past_length": backend == "compile",
            "compile_options": dict(_QWEN3_COMPILE_OPTIONS) if backend == "compile" else None,
            "attention_bias_boundary": "causal_lower_right",
            "initial_chunk": "eager",
            "partial_chunks": "eager",
            "torch_version": str(torch.__version__),
        }
        return {
            **execution,
            "requested_via": "TrainConfig.backbone_chunk_backend",
            "requested_value": backend,
            "implementation_module": "gcw.performance.backends.qwen3",
            "implementation_symbol": ("gcw.performance.backends.qwen3._qwen3_backbone_chunk_tensor_region"),
            "fingerprint_sha256": canonical_json_sha256(execution),
        }

    if backend == "cuda_graph":
        execution = {
            "schema": "gcw.qwen3_backbone_chunk_backend.v3",
            "effective_backend": backend,
            "regions": ["initial_full_window", "full_window_with_exact_past"],
            "chunk_steps": _QWEN3_CHUNK_STEPS,
            "shape_policy": "closed_static_tensor_signature",
            "initial_and_partial_policy": "initial_full_graph_partial_eager",
            "backward": "torch_cuda_make_graphed_callables",
            "runtime_ownership": "external_weak_model_identity",
            "capture_rng": "saved_and_restored",
            "fallback": "disabled",
            "campaign_admitted": False,
            "torch_version": str(torch.__version__),
        }
        return {
            **execution,
            "requested_via": "TrainConfig.backbone_chunk_backend",
            "requested_value": backend,
            "implementation_module": "gcw.performance.cuda_graphs",
            "fingerprint_sha256": canonical_json_sha256(execution),
        }
    if backend != "compile_full_windows":
        raise ValueError(f"unsupported Qwen3 backbone chunk backend: {backend}")
    execution = {
        "schema": "gcw.qwen3_backbone_chunk_backend.v2",
        "effective_backend": backend,
        "regions": {
            "initial_full_window": "gcw.performance.backends.qwen3._qwen3_initial_chunk_tensor_region",
            "full_window_with_past": "gcw.performance.backends.qwen3._qwen3_backbone_chunk_tensor_region",
        },
        "chunk_steps": _QWEN3_CHUNK_STEPS,
        "dynamic_past_length": True,
        "compile_options": {
            "initial_full_window": dict(_QWEN3_INITIAL_COMPILE_OPTIONS),
            "full_window_with_past": dict(_QWEN3_COMPILE_OPTIONS),
        },
        "attention_bias_boundaries": {
            "initial_full_window": "flash_is_causal",
            "full_window_with_past": "causal_lower_right",
        },
        "initial_chunk": "compile_fullgraph",
        "partial_chunks": "eager",
        "torch_version": str(torch.__version__),
    }
    return {
        **execution,
        "requested_via": "TrainConfig.backbone_chunk_backend",
        "requested_value": backend,
        "implementation_module": "gcw.performance.backends.qwen3",
        "implementation_symbols": (
            "gcw.performance.backends.qwen3._qwen3_initial_chunk_tensor_region",
            "gcw.performance.backends.qwen3._qwen3_backbone_chunk_tensor_region",
        ),
        "fingerprint_sha256": canonical_json_sha256(execution),
    }


def run_qwen3_backbone_chunk_region(
    backbone: nn.Module,
    backend: Qwen3BackboneChunkBackend,
    inputs_embeds: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    past_keys: torch.Tensor,
    past_values: torch.Tensor,
    attention_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one complete 64-token backbone chunk; compiled failures are fatal."""
    if backend == "eager":
        return _qwen3_backbone_chunk_tensor_region(
            backbone, inputs_embeds, cos, sin, past_keys, past_values, attention_bias
        )
    if backend == "cuda_graph":
        operation = partial(_qwen3_backbone_chunk_tensor_region, backbone)
        return run_graphed_callable(
            backbone,
            region="qwen3_full_window_with_past",
            operation=operation,
            args=(inputs_embeds, cos, sin, past_keys, past_values, attention_bias),
        )
    if backend not in {"compile", "compile_full_windows"}:
        raise ValueError(f"unsupported Qwen3 backbone chunk backend: {backend}")
    runner = _get_qwen3_compiled_runner(backbone)
    recorder = compile_acceptance_recorder(backbone)
    try:
        if recorder is None:
            return runner(inputs_embeds, cos, sin, past_keys, past_values, attention_bias)
        operation = partial(runner, inputs_embeds, cos, sin, past_keys, past_values, attention_bias)
        return recorder.record_call(
            "full_window_with_past",
            (inputs_embeds.shape[0], inputs_embeds.shape[1], past_keys.shape[3]),
            inputs_embeds.device,
            operation,
        )
    except Exception as error:
        if _is_non_reentrant_checkpoint_early_stop(error):
            raise
        raise RuntimeError(
            f"compiled Qwen3 backbone chunk failed; eager fallback is disabled when backbone_chunk_backend={backend}"
        ) from error


def run_qwen3_initial_backbone_chunk_region(
    backbone: nn.Module,
    backend: Qwen3BackboneChunkBackend,
    inputs_embeds: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the initial full causal window; compilation is explicit and fail closed."""
    if backend == "eager":
        return _qwen3_initial_chunk_tensor_region(backbone, inputs_embeds, cos, sin)
    if backend == "cuda_graph":
        operation = partial(_qwen3_initial_chunk_tensor_region, backbone)
        return run_graphed_callable(
            backbone,
            region="qwen3_initial_full_window",
            operation=operation,
            args=(inputs_embeds, cos, sin),
        )
    if backend != "compile_full_windows":
        raise ValueError(f"backend {backend!r} does not compile the initial Qwen3 window")
    runner = _get_qwen3_compiled_initial_runner(backbone)
    recorder = compile_acceptance_recorder(backbone)
    try:
        if recorder is None:
            return runner(inputs_embeds, cos, sin)
        operation = partial(runner, inputs_embeds, cos, sin)
        return recorder.record_call(
            "initial_full_window",
            (inputs_embeds.shape[0], inputs_embeds.shape[1]),
            inputs_embeds.device,
            operation,
        )
    except Exception as error:
        if _is_non_reentrant_checkpoint_early_stop(error):
            raise
        raise RuntimeError(
            "compiled initial Qwen3 backbone chunk failed; eager fallback is disabled when "
            "backbone_chunk_backend=compile_full_windows"
        ) from error


def _is_non_reentrant_checkpoint_early_stop(error: BaseException) -> bool:
    """Preserve checkpoint control flow through the shared exact predicate."""
    return is_non_reentrant_checkpoint_early_stop(error)


def _qwen3_initial_chunk_tensor_region(
    backbone: nn.Module,
    inputs_embeds: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure tensor boundary for one initial 64-token causal Qwen3 window."""
    x = inputs_embeds
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    layer_scores: list[torch.Tensor] = []
    layers: Any = getattr(backbone, "layers")
    for layer in layers:
        x, keys_i, values_i, scores_i = layer(x, (cos, sin), None)
        keys.append(keys_i)
        values.append(values_i)
        layer_scores.append(scores_i)
    norm: Any = getattr(backbone, "norm")
    return x if norm is None else norm(x), torch.stack(keys), torch.stack(values), torch.stack(layer_scores).mean(dim=0)


def _qwen3_backbone_chunk_tensor_region(
    backbone: nn.Module,
    inputs_embeds: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    past_keys: torch.Tensor,
    past_values: torch.Tensor,
    attention_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensor cache boundary spanning every layer; CausalBias preserves Flash SDPA."""
    x = inputs_embeds
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    layer_scores: list[torch.Tensor] = []
    layers: Any = getattr(backbone, "layers")
    for layer_idx, layer in enumerate(layers):
        x, keys_i, values_i, scores_i = layer.forward_tensor_past(
            x,
            (cos, sin),
            past_keys[layer_idx],
            past_values[layer_idx],
            attention_bias,
        )
        keys.append(keys_i)
        values.append(values_i)
        layer_scores.append(scores_i)
    norm: Any = getattr(backbone, "norm")
    return x if norm is None else norm(x), torch.stack(keys), torch.stack(values), torch.stack(layer_scores).mean(dim=0)


def _get_qwen3_compiled_runner(backbone: nn.Module) -> Qwen3ChunkRunner:
    runner: Qwen3ChunkRunner | None = backbone.__dict__.get(_QWEN3_RUNNER_ATTRIBUTE)
    if runner is not None:
        return runner
    with _qwen3_compiled_runners_lock:
        runner = backbone.__dict__.get(_QWEN3_RUNNER_ATTRIBUTE)
        if runner is None:
            compiler = require_torch_compile("backbone_chunk_backend=compile")
            try:
                runner = compiler(partial(_qwen3_backbone_chunk_tensor_region, backbone), **_QWEN3_COMPILE_OPTIONS)
            except Exception as error:
                raise RuntimeError(
                    "failed to create the compiled Qwen3 backbone chunk; eager fallback is disabled"
                ) from error
            object.__setattr__(backbone, _QWEN3_RUNNER_ATTRIBUTE, runner)
    return runner


def _get_qwen3_compiled_initial_runner(backbone: nn.Module) -> Qwen3InitialChunkRunner:
    runner: Qwen3InitialChunkRunner | None = backbone.__dict__.get(_QWEN3_INITIAL_RUNNER_ATTRIBUTE)
    if runner is not None:
        return runner
    with _qwen3_compiled_runners_lock:
        runner = backbone.__dict__.get(_QWEN3_INITIAL_RUNNER_ATTRIBUTE)
        if runner is None:
            compiler = require_torch_compile("backbone_chunk_backend=compile_full_windows")
            try:
                runner = compiler(
                    partial(_qwen3_initial_chunk_tensor_region, backbone),
                    **_QWEN3_INITIAL_COMPILE_OPTIONS,
                )
            except Exception as error:
                raise RuntimeError(
                    "failed to create the compiled initial Qwen3 backbone chunk; eager fallback is disabled"
                ) from error
            object.__setattr__(backbone, _QWEN3_INITIAL_RUNNER_ATTRIBUTE, runner)
    return runner
