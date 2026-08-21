"""Opt-in, state-neutral Longshot full-history cache execution."""

from __future__ import annotations

from typing import Any

import torch

from gcw.models.cache_processors.base import cache_processor_recomputation_id
from gcw.performance.config import LongshotFullHistoryConfig
from gcw.performance.longshot_scores import (
    LONGSHOT_SCORE_FP32_CHUNK_BYTES,
    unroped_score_estimated_temp_bytes_per_row,
    unrope_rotation_estimated_fixed_temp_bytes,
    unrope_rotation_estimated_inner_allocation_bytes,
    unrope_rotation_estimated_temp_bytes_per_row,
)
from gcw.provenance import canonical_json_sha256

_POLICY_ATTRIBUTE = "_gcw_longshot_full_history_execution"
_METRICS_ATTRIBUTE = "_gcw_longshot_full_history_metrics"
_SCHEMA = "gcw.longshot_full_history_execution.v1"
_UNROPE_SCHEMA = "gcw.longshot_full_history_execution.v2"
_UNROPE_STRATEGY = "chunk_local_inverse_rope_v1"


def configure_longshot_full_history_execution(
    model: torch.nn.Module,
    config: LongshotFullHistoryConfig,
) -> None:
    """Attach an enabled execution policy without changing module state."""
    if not config.enabled:
        return
    processor: Any = getattr(model, "cache_processor", model)
    if type(processor).__module__ != "gcw.models.cache_processors.longshot":
        raise ValueError(
            "Longshot full-history execution found no Longshot cache processor"
        )
    processor_config = processor.config
    if processor_config.score != "kv_norm":
        raise ValueError(
            "Longshot full-history execution requires score='kv_norm'"
        )
    if processor_config.attention_dropout != 0.0:
        raise ValueError(
            "Longshot full-history execution requires attention_dropout=0"
        )
    if processor_config.linear_state_cache:
        raise ValueError(
            "Longshot full-history execution rejects linear recurrent cache state"
        )
    if (
        processor_config.iterations > 1
        and processor_config.iteration_mode == "internal"
        and processor_config.internal_iteration_score != "kv_norm"
    ):
        raise ValueError(
            "Longshot full-history execution rejects internal non-KV recurrence"
        )
    minimum_temp_bytes = (
        unroped_score_estimated_temp_bytes_per_row(
            dtype=torch.float32,
            head_dim=processor.head_dim,
            rotary_dim=processor.rotary_dim,
        )
        if processor_config.unrope_keys
        else 2 * processor.head_dim * 4
    )
    if config.max_temp_bytes < minimum_temp_bytes:
        strategy = (
            "un-RoPE rotation and score"
            if processor_config.unrope_keys
            else "FP32 K/V score"
        )
        raise ValueError(
            f"Longshot full-history max_temp_bytes must fit one complete {strategy} row "
            f"({minimum_temp_bytes})"
        )
    object.__setattr__(processor, _POLICY_ATTRIBUTE, config)


def longshot_full_history_execution(
    processor: torch.nn.Module,
) -> LongshotFullHistoryConfig:
    """Return the attached policy; absent configuration means exact eager execution."""
    value = getattr(processor, _POLICY_ATTRIBUTE, None)
    if value is None:
        return LongshotFullHistoryConfig()
    if not isinstance(value, LongshotFullHistoryConfig):
        raise TypeError(
            "configured Longshot full-history execution policy has an invalid type"
        )
    return value


def longshot_full_history_execution_metadata(
    config: LongshotFullHistoryConfig,
    *,
    unrope_keys: bool = False,
) -> dict[str, Any]:
    """Describe and authenticate the non-scientific selected-history execution."""
    execution = {
        "schema": _UNROPE_SCHEMA if unrope_keys else _SCHEMA,
        "enabled": config.enabled,
        "max_temp_bytes": config.max_temp_bytes,
        "score_oracle": (
            "chunk_local_inverse_rope_then_cat_kv_float_pow2_sum_last_per_row"
            if unrope_keys
            else "cat_kv_float_pow2_sum_last_per_row"
        ),
        "unrope_keys": (
            _UNROPE_STRATEGY
            if unrope_keys
            else "rejected_to_preserve_bounded_full_history_scoring"
        ),
        "selection_oracle": "stable_existing_fp32_scores_and_torch_topk",
        "gather": (
            "selected_layer_head_positions_then_chunk_local_inverse_rope"
            if unrope_keys
            else "selected_layer_head_positions_only"
        ),
        "gradient": "selected_history_only_with_chunk_recomputation",
        "checkpoint_recomputation": "supported_non_reentrant_and_segmented",
        "recurrence": "internal_or_external_exact_reselection",
        "distributed": "single_process_only",
        "backbone_chunk_backend": "eager_only",
        "adaptive_batching": "rejected",
        "dropout": "zero_only",
        "linear_state_cache": "rejected",
        **(
            {
                "unrope_position_mapping": "layer_major_row_mod_sequence_v1",
                "unrope_temporary_bound": "rotation_trig_pointwise_merge_fp32_v1",
            }
            if unrope_keys
            else {}
        ),
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
    *,
    unrope_keys: bool = False,
    rope_theta: float = 10000.0,
    rotary_style: str = "llama",
    rotary_dim: int | None = None,
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
    by_layer = indices.reshape(batch, layers, heads, selected).permute(
        1, 0, 2, 3
    )
    gather_index = by_layer[..., None].expand(
        layers, batch, heads, selected, head_dim
    )
    selected_keys = torch.gather(keys, dim=3, index=gather_index)
    selected_values = torch.gather(values, dim=3, index=gather_index)
    output_shape = (batch, layers * heads, selected, head_dim)
    selected_keys = selected_keys.permute(1, 0, 2, 3, 4).reshape(output_shape)
    selected_values = selected_values.permute(1, 0, 2, 3, 4).reshape(
        output_shape
    )
    if unrope_keys:
        from gcw.models.cache_processors.ops import rotate_cache_keys

        selected_keys = rotate_cache_keys(
            selected_keys,
            indices,
            theta=rope_theta,
            rotary_style=rotary_style,
            inverse=True,
            rotary_dim=rotary_dim,
        )
    return selected_keys, selected_values


def record_longshot_full_history_event(
    processor: torch.nn.Module,
    *,
    keys: torch.Tensor,
    selected_positions: int,
    max_temp_bytes: int,
    unrope_keys: bool = False,
    rotary_dim: int | None = None,
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
    selected_elements = (
        layers * batch * heads * selected_positions * 2 * head_dim
    )
    score_elements = batch * layers * heads * sequence * 2 * head_dim
    event = {
        "events": 1,
        "reference_full_history_logical_payload_bytes": full_elements
        * element_bytes,
        "selected_history_logical_payload_bytes": selected_elements
        * element_bytes,
        "logical_payload_bytes_avoided": max(
            0, full_elements - selected_elements
        )
        * element_bytes,
        "reference_score_estimated_fp32_working_set_bound_bytes": 2
        * min(score_elements * 4, LONGSHOT_SCORE_FP32_CHUNK_BYTES),
        "optimized_score_estimated_fp32_working_set_bound_bytes": 2
        * min(score_elements * 4, max_temp_bytes),
    }
    if unrope_keys:
        if rotary_dim is None:
            raise ValueError("Longshot un-RoPE telemetry requires rotary_dim")
        full_rows = layers * batch * heads * sequence
        selected_rows = layers * batch * heads * selected_positions
        score_row_bound = unroped_score_estimated_temp_bytes_per_row(
            dtype=keys.dtype,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
        )
        rotation_row_bound = unrope_rotation_estimated_temp_bytes_per_row(
            dtype=keys.dtype,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
        )
        rotation_fixed_bound = unrope_rotation_estimated_fixed_temp_bytes(
            rotary_dim=rotary_dim
        )
        score_rows_per_chunk = max(1, max_temp_bytes // score_row_bound)
        score_rotation_calls = (
            full_rows + score_rows_per_chunk - 1
        ) // score_rows_per_chunk
        selected_rotation_calls = int(selected_rows > 0)
        unrope_live_bytes_per_row = (
            rotation_row_bound + 4 * head_dim * element_bytes
        )
        score_working_set_bound = (
            min(full_rows * score_row_bound, max_temp_bytes)
            + rotation_fixed_bound
        )
        event["optimized_score_estimated_fp32_working_set_bound_bytes"] = (
            score_working_set_bound
        )
        event.update(
            {
                "reference_unrope_estimated_live_bytes": full_rows
                * unrope_live_bytes_per_row
                + rotation_fixed_bound,
                "optimized_selected_unrope_estimated_live_bytes": selected_rows
                * unrope_live_bytes_per_row
                + selected_rotation_calls * rotation_fixed_bound,
                "optimized_unrope_score_estimated_working_set_bound_bytes": (
                    score_working_set_bound
                ),
                "optimized_score_unrope_rotate_cache_keys_calls": score_rotation_calls,
                "optimized_score_unrope_rotate_cache_keys_rows": full_rows,
                "optimized_score_unrope_rotate_cache_keys_estimated_inner_allocation_bytes": (
                    unrope_rotation_estimated_inner_allocation_bytes(
                        full_rows,
                        dtype=keys.dtype,
                        head_dim=head_dim,
                        rotary_dim=rotary_dim,
                        calls=score_rotation_calls,
                    )
                ),
                "optimized_selected_unrope_rotate_cache_keys_calls": (
                    selected_rotation_calls
                ),
                "optimized_selected_unrope_rotate_cache_keys_rows": selected_rows,
                "optimized_selected_unrope_rotate_cache_keys_estimated_inner_allocation_bytes": (
                    unrope_rotation_estimated_inner_allocation_bytes(
                        selected_rows,
                        dtype=keys.dtype,
                        head_dim=head_dim,
                        rotary_dim=rotary_dim,
                        calls=selected_rotation_calls,
                    )
                ),
                "reference_unrope_rotate_cache_keys_calls": int(full_rows > 0),
                "reference_unrope_rotate_cache_keys_rows": full_rows,
                "reference_unrope_rotate_cache_keys_estimated_inner_allocation_bytes": (
                    unrope_rotation_estimated_inner_allocation_bytes(
                        full_rows,
                        dtype=keys.dtype,
                        head_dim=head_dim,
                        rotary_dim=rotary_dim,
                        calls=int(full_rows > 0),
                    )
                ),
            }
        )
    current = getattr(processor, _METRICS_ATTRIBUTE, None)
    if current is None:
        current = {key: 0 for key in event}
        object.__setattr__(processor, _METRICS_ATTRIBUTE, current)
    for key, value in event.items():
        if key == "events":
            current[key] += value
        else:
            current[key] = max(current[key], value)


def pop_longshot_full_history_metrics(
    processor: torch.nn.Module,
) -> dict[str, float]:
    """Pop profiling counters while keeping them outside parameters and buffers."""
    if cache_processor_recomputation_id() is not None:
        return {}
    values = getattr(processor, _METRICS_ATTRIBUTE, None)
    if values is None:
        return {}
    object.__setattr__(processor, _METRICS_ATTRIBUTE, None)
    prefix = "performance/longshot_full_history_"
    return {f"{prefix}{key}": float(value) for key, value in values.items()}


def _validate_inputs(
    keys: torch.Tensor, values: torch.Tensor, indices: torch.Tensor
) -> None:
    if keys.ndim != 5:
        raise ValueError(
            "Longshot windowed cache tensors must have shape (layers, batch, heads, seq, dim)"
        )
    if keys.shape != values.shape:
        raise ValueError(
            "Longshot cache keys and values must have identical shapes"
        )
    if keys.dtype != values.dtype or keys.device != values.device:
        raise ValueError(
            "Longshot cache keys and values must share dtype and device"
        )
    if not keys.is_contiguous() or not values.is_contiguous():
        raise ValueError(
            "Longshot selected-history execution requires contiguous layer-major cache tensors"
        )
    if indices.ndim != 3:
        raise ValueError(
            "Longshot cache indices must have shape (batch, cache_heads, selected)"
        )
    layers, batch, heads, _, _ = keys.shape
    if indices.shape[:2] != (batch, layers * heads):
        raise ValueError(
            "Longshot cache indices do not match the layer-major cache shape"
        )
    if indices.dtype != torch.long or indices.device != keys.device:
        raise ValueError(
            "Longshot cache indices must be long tensors on the cache device"
        )
