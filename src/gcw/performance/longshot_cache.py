"""Opt-in, state-neutral Longshot full-history cache execution."""

from __future__ import annotations

from typing import Any

import torch

from gcw.models.cache_processors.base import cache_processor_recomputation_id
from gcw.performance.config import LongshotFullHistoryConfig
from gcw.performance.longshot_scores import LONGSHOT_SCORE_FP32_CHUNK_BYTES
from gcw.provenance import canonical_json_sha256

_POLICY_ATTRIBUTE = "_gcw_longshot_full_history_execution"
_METRICS_ATTRIBUTE = "_gcw_longshot_full_history_metrics"
_SCHEMA = "gcw.longshot_full_history_execution.v1"


def configure_longshot_full_history_execution(
    model: torch.nn.Module,
    config: LongshotFullHistoryConfig,
) -> None:
    """Attach an enabled execution policy without changing module state."""
    if not config.enabled:
        return
    processor: Any = getattr(model, "cache_processor", model)
    if type(processor).__module__ != "gcw.models.cache_processors.longshot":
        raise ValueError("Longshot full-history execution found no Longshot cache processor")
    processor_config = processor.config
    if processor_config.score != "kv_norm":
        raise ValueError("Longshot full-history execution requires score='kv_norm'")
    if processor_config.attention_dropout != 0.0:
        raise ValueError("Longshot full-history execution requires attention_dropout=0")
    if processor_config.linear_state_cache:
        raise ValueError("Longshot full-history execution rejects linear recurrent cache state")
    if processor_config.unrope_keys:
        raise ValueError(
            "Longshot full-history execution rejects unrope_keys=True because rotating the full "
            "history would violate its bounded-memory contract"
        )
    if (
        processor_config.iterations > 1
        and processor_config.iteration_mode == "internal"
        and processor_config.internal_iteration_score != "kv_norm"
    ):
        raise ValueError("Longshot full-history execution rejects internal non-KV recurrence")
    minimum_temp_bytes = 2 * processor.head_dim * 4
    if config.max_temp_bytes < minimum_temp_bytes:
        raise ValueError(
            f"Longshot full-history max_temp_bytes must be at least one FP32 K/V row ({minimum_temp_bytes})"
        )
    object.__setattr__(processor, _POLICY_ATTRIBUTE, config)


def longshot_full_history_execution(processor: torch.nn.Module) -> LongshotFullHistoryConfig:
    """Return the attached policy; absent configuration means exact eager execution."""
    value = getattr(processor, _POLICY_ATTRIBUTE, None)
    if value is None:
        return LongshotFullHistoryConfig()
    if not isinstance(value, LongshotFullHistoryConfig):
        raise TypeError("configured Longshot full-history execution policy has an invalid type")
    return value


def longshot_full_history_execution_metadata(config: LongshotFullHistoryConfig) -> dict[str, Any]:
    """Describe and authenticate the non-scientific selected-history execution."""
    execution = {
        "schema": _SCHEMA,
        "enabled": config.enabled,
        "max_temp_bytes": config.max_temp_bytes,
        "score_oracle": "cat_kv_float_pow2_sum_last_per_row",
        "unrope_keys": "rejected_to_preserve_bounded_full_history_scoring",
        "selection_oracle": "stable_existing_fp32_scores_and_torch_topk",
        "gather": "selected_layer_head_positions_only",
        "gradient": "selected_history_only_with_chunk_recomputation",
        "checkpoint_recomputation": "supported_non_reentrant_and_segmented",
        "recurrence": "internal_or_external_exact_reselection",
        "distributed": "single_process_only",
        "backbone_chunk_backend": "eager_only",
        "adaptive_batching": "rejected",
        "dropout": "zero_only",
        "linear_state_cache": "rejected",
        "implementation": "gcw.performance.longshot_cache.gather_windowed_per_cache_head_kv",
    }
    return {
        **execution,
        "requested_via": "TrainConfig.performance.longshot_full_history",
        "fingerprint_sha256": canonical_json_sha256(execution),
    }


def gather_windowed_per_cache_head_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather selected layer/head histories without flattening the full cache.

    Args:
        keys: Layer-major keys with shape ``[layers, batch, heads, seq, dim]``.
        values: Values with the same shape as ``keys``.
        indices: Per-cache-head positions with shape
            ``[batch, layers * heads, selected]``.

    Returns:
        Selected keys and values with shape
        ``[batch, layers * heads, selected, dim]``.
    """
    _validate_inputs(keys, values, indices)
    layers, batch, heads, _, head_dim = keys.shape
    selected = indices.shape[-1]
    by_layer = indices.reshape(batch, layers, heads, selected).permute(1, 0, 2, 3)
    gather_index = by_layer[..., None].expand(layers, batch, heads, selected, head_dim)
    selected_keys = torch.gather(keys, dim=3, index=gather_index)
    selected_values = torch.gather(values, dim=3, index=gather_index)
    output_shape = (batch, layers * heads, selected, head_dim)
    return (
        selected_keys.permute(1, 0, 2, 3, 4).reshape(output_shape),
        selected_values.permute(1, 0, 2, 3, 4).reshape(output_shape),
    )


def record_longshot_full_history_event(
    processor: torch.nn.Module,
    *,
    keys: torch.Tensor,
    selected_positions: int,
    max_temp_bytes: int,
) -> None:
    """Record logical payload estimates once per original forward.

    These counters describe tensor payload and analytical score working sets. They
    are not allocator measurements; CUDA peak-memory telemetry remains authoritative.
    """
    if cache_processor_recomputation_id() is not None:
        return
    layers, batch, heads, sequence, head_dim = keys.shape
    element_bytes = keys.element_size()
    full_elements = layers * batch * heads * sequence * 2 * head_dim
    selected_elements = layers * batch * heads * selected_positions * 2 * head_dim
    score_elements = batch * layers * heads * sequence * 2 * head_dim
    event = {
        "events": 1,
        "reference_full_history_logical_payload_bytes": full_elements * element_bytes,
        "selected_history_logical_payload_bytes": selected_elements * element_bytes,
        "logical_payload_bytes_avoided": max(0, full_elements - selected_elements) * element_bytes,
        "reference_score_estimated_fp32_working_set_bound_bytes": 2
        * min(score_elements * 4, LONGSHOT_SCORE_FP32_CHUNK_BYTES),
        "optimized_score_estimated_fp32_working_set_bound_bytes": 2
        * min(score_elements * 4, max_temp_bytes),
    }
    current = getattr(processor, _METRICS_ATTRIBUTE, None)
    if current is None:
        current = {key: 0 for key in event}
        object.__setattr__(processor, _METRICS_ATTRIBUTE, current)
    for key, value in event.items():
        if key == "events":
            current[key] += value
        else:
            current[key] = max(current[key], value)


def pop_longshot_full_history_metrics(processor: torch.nn.Module) -> dict[str, float]:
    """Pop profiling counters while keeping them outside parameters and buffers."""
    if cache_processor_recomputation_id() is not None:
        return {}
    values = getattr(processor, _METRICS_ATTRIBUTE, None)
    if values is None:
        return {}
    object.__setattr__(processor, _METRICS_ATTRIBUTE, None)
    prefix = "performance/longshot_full_history_"
    return {f"{prefix}{key}": float(value) for key, value in values.items()}


def _validate_inputs(keys: torch.Tensor, values: torch.Tensor, indices: torch.Tensor) -> None:
    if keys.ndim != 5:
        raise ValueError("Longshot windowed cache tensors must have shape (layers, batch, heads, seq, dim)")
    if keys.shape != values.shape:
        raise ValueError("Longshot cache keys and values must have identical shapes")
    if keys.dtype != values.dtype or keys.device != values.device:
        raise ValueError("Longshot cache keys and values must share dtype and device")
    if not keys.is_contiguous() or not values.is_contiguous():
        raise ValueError("Longshot selected-history execution requires contiguous layer-major cache tensors")
    if indices.ndim != 3:
        raise ValueError("Longshot cache indices must have shape (batch, cache_heads, selected)")
    layers, batch, heads, _, _ = keys.shape
    if indices.shape[:2] != (batch, layers * heads):
        raise ValueError("Longshot cache indices do not match the layer-major cache shape")
    if indices.dtype != torch.long or indices.device != keys.device:
        raise ValueError("Longshot cache indices must be long tensors on the cache device")
