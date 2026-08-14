from __future__ import annotations

from collections.abc import Callable
from functools import partial
from threading import Lock
from typing import Any, Literal

import torch
import torch.nn as nn

from gcw.provenance import canonical_json_sha256


Qwen3BackboneChunkBackend = Literal["eager", "compile"]
Qwen3ChunkRunner = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]

_QWEN3_CHUNK_STEPS = 64
_QWEN3_COMPILE_OPTIONS = {
    "backend": "inductor",
    "dynamic": True,
    "fullgraph": True,
}
_QWEN3_RUNNER_ATTRIBUTE = "_gcw_qwen3_compiled_chunk_runner"
_qwen3_compiled_runners_lock = Lock()


def validate_qwen3_backbone_chunk_backend(backend: Qwen3BackboneChunkBackend) -> None:
    """Reject unavailable or fallback-enabled compile execution before dispatch."""
    if backend == "eager":
        return
    if backend != "compile":
        raise ValueError(f"unsupported Qwen3 backbone chunk backend: {backend}")
    compiler = getattr(torch, "compile", None)
    if not callable(compiler):
        raise RuntimeError("backbone_chunk_backend=compile requires torch.compile")
    dynamo = getattr(torch, "_dynamo", None)
    dynamo_config = getattr(dynamo, "config", None)
    if bool(getattr(dynamo_config, "suppress_errors", False)):
        raise RuntimeError("backbone_chunk_backend=compile requires torch._dynamo.config.suppress_errors=False")


def configure_qwen3_backbone_chunk_backend(
    model: nn.Module,
    backend: Qwen3BackboneChunkBackend,
) -> None:
    """Configure a state-neutral backend only after the Qwen3 backbone is frozen."""
    validate_qwen3_backbone_chunk_backend(backend)
    backbone = getattr(model, "backbone", model)
    config: Any = getattr(backbone, "config", None)
    if backend == "compile":
        if config is None or getattr(config, "family", None) != "qwen3":
            raise ValueError("backbone_chunk_backend=compile requires a Qwen3 backbone")
        if any(parameter.requires_grad for parameter in backbone.parameters()):
            raise ValueError("backbone_chunk_backend=compile requires a frozen backbone")
    if not hasattr(backbone, "backbone_chunk_backend"):
        if backend == "compile":
            raise ValueError("backbone_chunk_backend=compile requires the standard Qwen3 transformer")
        return
    object.__setattr__(backbone, "backbone_chunk_backend", backend)


def qwen3_backbone_chunk_backend_metadata(
    backend: Qwen3BackboneChunkBackend,
) -> dict[str, object]:
    """Return stable provenance for the explicitly selected Qwen3 chunk region."""
    execution = {
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
    if backend != "compile":
        raise ValueError(f"unsupported Qwen3 backbone chunk backend: {backend}")
    runner = _get_qwen3_compiled_runner(backbone)
    try:
        return runner(inputs_embeds, cos, sin, past_keys, past_values, attention_bias)
    except Exception as error:
        raise RuntimeError(
            "compiled Qwen3 backbone chunk failed; eager fallback is disabled when backbone_chunk_backend=compile"
        ) from error


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
            compiler = getattr(torch, "compile", None)
            if not callable(compiler):
                raise RuntimeError("backbone_chunk_backend=compile requires torch.compile")
            try:
                runner = compiler(partial(_qwen3_backbone_chunk_tensor_region, backbone), **_QWEN3_COMPILE_OPTIONS)
            except Exception as error:
                raise RuntimeError(
                    "failed to create the compiled Qwen3 backbone chunk; eager fallback is disabled"
                ) from error
            object.__setattr__(backbone, _QWEN3_RUNNER_ATTRIBUTE, runner)
    return runner
