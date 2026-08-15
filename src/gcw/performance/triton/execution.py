"""State-neutral admission for the benchmark-only Triton cache backend."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch

from gcw.performance.config import TritonCacheMovementConfig
from gcw.performance.triton.runtime import require_triton_runtime
from gcw.provenance import canonical_json_sha256

_POLICY_ATTRIBUTE = "_gcw_triton_cache_movement"
_ALLOW_ENV = "GCW_ALLOW_UNQUALIFIED_TRITON_CACHE"


@dataclass(frozen=True)
class TritonCacheExecutionContext:
    model_variant: str
    model_family: str
    physical_batch_size: int
    distributed: str
    trainable: str
    bf16: bool
    backbone_chunk_backend: str
    adaptive_batching_enabled: bool
    whole_model_compile_requested: bool


def validate_triton_cache_movement_request(
    config: TritonCacheMovementConfig,
    device: torch.device,
) -> None:
    """Reject unavailable benchmark requests before model allocation."""
    if not config.enabled:
        return
    if os.environ.get(_ALLOW_ENV) != "1":
        raise RuntimeError(
            "Triton cache movement remains benchmark-only pending full H100/H200 optimizer-step "
            f"qualification; set {_ALLOW_ENV}=1 only in the explicit qualification harness"
        )
    require_triton_runtime(device)


def configure_triton_cache_movement(
    model: torch.nn.Module,
    config: TritonCacheMovementConfig,
    *,
    context: TritonCacheExecutionContext,
) -> None:
    """Attach an admitted execution flag without changing model state ownership."""
    processor: Any = getattr(model, "cache_processor", None)
    if not config.enabled:
        if processor is not None and hasattr(processor, _POLICY_ATTRIBUTE):
            object.__delattr__(processor, _POLICY_ATTRIBUTE)
        return
    if processor is None:
        raise ValueError("Triton cache movement requires a cache processor")
    validate_triton_cache_movement_request(config, next(processor.parameters()).device)
    if context.model_variant not in {"bt", "bgt", "longshot", "icarus"}:
        raise ValueError("Triton cache movement found an unsupported model variant")
    if context.model_family != "qwen3":
        raise ValueError("Triton cache movement currently admits only Qwen3")
    if context.physical_batch_size != 2 or context.distributed != "none":
        raise ValueError("Triton cache movement requires fixed batch two and one process")
    if context.trainable != "cache_processor" or not context.bf16:
        raise ValueError("Triton cache movement requires BF16 cache-processor-only training")
    if context.backbone_chunk_backend != "eager" or context.whole_model_compile_requested:
        raise ValueError("Triton cache movement does not compose with compiled execution")
    if context.adaptive_batching_enabled:
        raise ValueError("Triton cache movement does not compose with adaptive batching")
    processor_config = getattr(processor, "config", None)
    if processor_config is None or getattr(processor_config, "attention_dropout", None) != 0.0:
        raise ValueError("Triton cache movement requires zero processor attention dropout")
    object.__setattr__(processor, _POLICY_ATTRIBUTE, True)


def triton_cache_movement_enabled(processor: torch.nn.Module) -> bool:
    """Return the explicitly attached movement policy."""
    value = getattr(processor, _POLICY_ATTRIBUTE, False)
    if not isinstance(value, bool):
        raise TypeError("Triton cache movement policy has an invalid type")
    return value


def triton_cache_movement_metadata(config: TritonCacheMovementConfig) -> dict[str, object]:
    """Return stable benchmark provenance for the exact movement boundary."""
    execution = {
        "schema": "gcw.triton_cache_movement.v2",
        "enabled": config.enabled,
        "status": "benchmark_only_not_campaign_admitted",
        "physical_batch": 2,
        "dtype": "bfloat16_raw_int16_payload",
        "layouts": ["bt_layer", "bgt_global", "per_cache_head"],
        "forward": "fused_gather_layout_and_bt_bgt_split_functional_scatter",
        "backward": "exact_unique_index_gather_scatter",
        "selection": "unchanged_eager",
        "writeback": "triton_bt_bgt_eager_longshot_icarus",
        "runtime": "triton_3.6.0_sm90",
        "fallback": "disabled",
    }
    return {
        **execution,
        "requested_via": "TrainConfig.performance.triton_cache_movement",
        "fingerprint_sha256": canonical_json_sha256(execution),
    }
