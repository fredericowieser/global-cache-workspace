"""Exact, detached Longshot KV-score sidecar lifecycle.

The sidecar is forward-local cache state.  It is deliberately not a module
buffer or checkpointed model state, and it never replaces the live K/V tensors
used by the cache processor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from gcw.models.cache_processors.ops import rotate_cache_keys
from gcw.performance.config import KVScoreSidecarConfig
from gcw.performance.longshot_scores import chunked_fp32_squared_l2
from gcw.provenance import canonical_json_sha256

_POLICY_ATTRIBUTE = "_gcw_kv_score_sidecar_execution"
_SCHEMA = "gcw.kv_score_sidecar_state.v1"
_EXECUTION_SCHEMA = "gcw.kv_score_sidecar_execution.v1"


@dataclass(frozen=True)
class KVScoreSidecarSpec:
    """Immutable identity of one exact Longshot score table."""

    schema: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    unrope_keys: bool
    rope_theta: float
    rotary_style: str
    rotary_dim: int


@dataclass(frozen=True)
class KVScoreSidecarState:
    """Detached FP32 scores with shape ``[batch, layers * heads, sequence]``."""

    spec: KVScoreSidecarSpec
    scores: torch.Tensor


def kv_score_sidecar_spec(processor: Any) -> KVScoreSidecarSpec:
    """Build the state identity from a configured Longshot processor."""
    return KVScoreSidecarSpec(
        schema=_SCHEMA,
        num_layers=int(processor.num_layers),
        num_kv_heads=int(processor.num_kv_heads),
        head_dim=int(processor.head_dim),
        unrope_keys=bool(processor.config.unrope_keys),
        rope_theta=float(processor.rope_theta),
        rotary_style=str(processor.rotary_style),
        rotary_dim=int(processor.rotary_dim),
    )


def configure_kv_score_sidecar_execution(model: torch.nn.Module, config: KVScoreSidecarConfig) -> None:
    """Attach an enabled policy as state-neutral execution configuration."""
    if not config.enabled:
        return
    processor: Any = getattr(model, "cache_processor", model)
    if type(processor).__module__ != "gcw.models.cache_processors.longshot":
        raise ValueError("KV-score sidecar found no Longshot cache processor")
    processor_config = processor.config
    if processor_config.score != "kv_norm":
        raise ValueError("KV-score sidecar requires cache_processor.score='kv_norm'")
    if processor_config.attention_dropout != 0.0:
        raise ValueError("KV-score sidecar requires cache_processor.attention_dropout=0")
    if processor_config.linear_state_cache:
        raise ValueError("KV-score sidecar rejects linear recurrent cache state")
    object.__setattr__(processor, _POLICY_ATTRIBUTE, config)


def kv_score_sidecar_execution(processor: torch.nn.Module) -> KVScoreSidecarConfig:
    """Return the attached policy; absence preserves the eager oracle path."""
    value = getattr(processor, _POLICY_ATTRIBUTE, None)
    if value is None:
        return KVScoreSidecarConfig()
    if not isinstance(value, KVScoreSidecarConfig):
        raise TypeError("configured KV-score sidecar policy has an invalid type")
    return value


def kv_score_sidecar_execution_metadata(config: KVScoreSidecarConfig, *, unrope_keys: bool) -> dict[str, Any]:
    """Describe and authenticate the conservative selector-only execution."""
    execution = {
        "schema": _EXECUTION_SCHEMA,
        "enabled": config.enabled,
        "state_schema": _SCHEMA,
        "score_oracle": (
            "inverse_rope_key_then_cat_kv_float_pow2_sum_last_per_row"
            if unrope_keys
            else "cat_kv_float_pow2_sum_last_per_row"
        ),
        "score_dtype": "float32",
        "score_state": "forward_local_detached_cache_state",
        "bootstrap": "first_kv_norm_selection",
        "append": "new_backbone_window_rows_before_selection",
        "write_refresh": "all_processor_written_rows_after_final_kv_storage",
        "selection": "existing_longshot_topk_sort_and_head_reduction",
        "kv_gather": "authoritative_full_history_live_cache_with_autograd",
        "checkpoint_replay": "native_non_reentrant_and_segmented_flattened_state",
        "fresh_process_resume": "reconstruct_on_first_forward",
        "physical_batch": 2,
        "distributed": "single_process_only",
        "backbone_backend": "eager_only",
        "processor_backend": "eager_only",
        "campaign_admitted": False,
        "implementation": "gcw.performance.kv_score_sidecar",
    }
    return {
        **execution,
        "requested_via": "TrainConfig.performance.kv_score_sidecar",
        "fingerprint_sha256": canonical_json_sha256(execution),
    }


@torch.no_grad()
def synchronize_kv_score_sidecar(
    state: KVScoreSidecarState | None,
    keys: torch.Tensor,
    values: torch.Tensor,
    spec: KVScoreSidecarSpec,
) -> KVScoreSidecarState:
    """Bootstrap or append exact scores; malformed carried state fails closed."""
    _validate_cache_tensors(keys, values, spec)
    sequence = keys.shape[3]
    if state is None:
        prefix = 0
        scores = torch.empty(
            keys.shape[1],
            spec.num_layers * spec.num_kv_heads,
            0,
            dtype=torch.float32,
            device=keys.device,
        )
    else:
        _validate_state(state, keys, spec)
        prefix = state.scores.shape[2]
        if prefix > sequence:
            raise RuntimeError("KV-score sidecar sequence is longer than its live cache")
        scores = state.scores
    if prefix == sequence:
        return KVScoreSidecarState(spec, scores.detach())
    positions = torch.arange(prefix, sequence, dtype=torch.long, device=keys.device)
    positions = positions.unsqueeze(0).expand(keys.shape[1], -1)
    appended = score_kv_positions(keys, values, positions, spec)
    complete = appended if prefix == 0 else torch.cat([scores, appended], dim=2)
    return KVScoreSidecarState(spec, complete.detach())


@torch.no_grad()
def refresh_kv_score_sidecar(
    state: KVScoreSidecarState | None,
    keys: torch.Tensor,
    values: torch.Tensor,
    positions: torch.Tensor,
    spec: KVScoreSidecarSpec,
) -> KVScoreSidecarState:
    """Functionally replace every score row whose live K/V was written."""
    current = synchronize_kv_score_sidecar(state, keys, values, spec)
    _validate_positions(positions, keys)
    if positions.shape[1] == 0:
        return current
    refreshed = score_kv_positions(keys, values, positions, spec)
    index = positions[:, None, :].expand(-1, current.scores.shape[1], -1)
    scores = current.scores.scatter(2, index, refreshed)
    return KVScoreSidecarState(spec, scores.detach())


@torch.no_grad()
def score_kv_positions(
    keys: torch.Tensor,
    values: torch.Tensor,
    positions: torch.Tensor,
    spec: KVScoreSidecarSpec,
) -> torch.Tensor:
    """Evaluate complete concatenated rows with the authoritative FP32 expression."""
    _validate_cache_tensors(keys, values, spec)
    _validate_positions(positions, keys)
    layers, batch, heads, _, head_dim = keys.shape
    selected = positions.shape[1]
    if selected == 0:
        return torch.empty(batch, layers * heads, 0, dtype=torch.float32, device=keys.device)
    index = positions[None, :, None, :, None].expand(layers, batch, heads, selected, head_dim)
    selected_keys = torch.gather(keys, dim=3, index=index)
    selected_values = torch.gather(values, dim=3, index=index)
    by_cache_head_shape = (batch, layers * heads, selected, head_dim)
    selected_keys = selected_keys.permute(1, 0, 2, 3, 4).reshape(by_cache_head_shape)
    selected_values = selected_values.permute(1, 0, 2, 3, 4).reshape(by_cache_head_shape)
    expanded_positions = positions[:, None, :].expand(batch, layers * heads, selected)
    if spec.unrope_keys:
        selected_keys = rotate_cache_keys(
            selected_keys,
            expanded_positions,
            theta=spec.rope_theta,
            rotary_style=spec.rotary_style,
            inverse=True,
            rotary_dim=spec.rotary_dim,
        )
    # Do not split the K/V reductions: this expression defines bitwise parity.
    merged = torch.cat([selected_keys, selected_values], dim=-1)
    return chunked_fp32_squared_l2(merged.contiguous()).detach()


def _validate_cache_tensors(keys: torch.Tensor, values: torch.Tensor, spec: KVScoreSidecarSpec) -> None:
    if spec.schema != _SCHEMA:
        raise RuntimeError("KV-score sidecar spec has an unsupported schema")
    expected_prefix = (spec.num_layers, keys.shape[1], spec.num_kv_heads)
    if keys.ndim != 5 or keys.shape[:3] != expected_prefix or keys.shape[-1] != spec.head_dim:
        raise RuntimeError("KV-score sidecar spec does not match the live cache shape")
    if keys.shape != values.shape:
        raise RuntimeError("KV-score sidecar requires identical key and value shapes")
    if keys.dtype != values.dtype or keys.device != values.device or not keys.is_floating_point():
        raise RuntimeError("KV-score sidecar requires floating K/V with matching dtype and device")
    if keys.shape[1] != 2:
        raise RuntimeError("KV-score sidecar requires fixed physical batch size 2")
    if keys.dtype not in {torch.float32, torch.bfloat16}:
        raise RuntimeError("KV-score sidecar supports only FP32 or BF16 live cache tensors")
    if not keys.is_contiguous() or not values.is_contiguous():
        raise RuntimeError("KV-score sidecar requires contiguous dense WindowedCache storage")


def _validate_state(state: KVScoreSidecarState, keys: torch.Tensor, spec: KVScoreSidecarSpec) -> None:
    if not isinstance(state, KVScoreSidecarState) or state.spec != spec:
        raise RuntimeError("KV-score sidecar state does not match the requested execution spec")
    expected_prefix = (keys.shape[1], spec.num_layers * spec.num_kv_heads)
    if state.scores.ndim != 3 or state.scores.shape[:2] != expected_prefix:
        raise RuntimeError("KV-score sidecar score shape does not match the live cache")
    if state.scores.dtype != torch.float32 or state.scores.device != keys.device:
        raise RuntimeError("KV-score sidecar scores must be FP32 on the live cache device")
    if state.scores.requires_grad or state.scores.grad_fn is not None:
        raise RuntimeError("KV-score sidecar scores must be detached selector-only data")


def _validate_positions(positions: torch.Tensor, keys: torch.Tensor) -> None:
    if positions.ndim != 2 or positions.shape[0] != keys.shape[1]:
        raise RuntimeError("KV-score sidecar positions must have shape (batch, selected)")
    if positions.dtype != torch.long or positions.device != keys.device:
        raise RuntimeError("KV-score sidecar positions must be long tensors on the cache device")
    if positions.numel() and (bool((positions < 0).any()) or bool((positions >= keys.shape[3]).any())):
        raise RuntimeError("KV-score sidecar positions are outside the live cache")
