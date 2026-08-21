from __future__ import annotations

from typing import Any, cast
import torch

from gcw.config import WorkspaceFlowRefinementConfig
from gcw.models.cache_processors.flow import sinusoidal_time_embedding
from gcw.models.cache_processors.ops import (
    scatter_latest_values as _scatter_latest_values,
    split_merged_cache_values as _split_merged_values,
)
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import WindowedCache

CacheLike = CacheState | StaticCacheState | WindowedCache


def validate_workspace_shape(
    workspace: torch.Tensor,
    source: Any,
    output_dim: int,
) -> None:
    expected_shape = source.merged[:, :, -1, -output_dim:].shape
    if workspace.shape != expected_shape:
        raise ValueError(
            f"Longshot workspace shape must be {tuple(expected_shape)}, got {tuple(workspace.shape)}"
        )


def workspace_cycle_input(
    workspace: torch.Tensor,
    source: Any,
    output_dim: int,
) -> torch.Tensor:
    frontier = torch.cat(
        [source.merged[:, :, -1, :-output_dim], workspace],
        dim=-1,
    )
    return torch.cat([source.merged[:, :, :-1], frontier[:, :, None]], dim=2)


def deliberate_workspace(
    processor: Any,
    workspace: torch.Tensor,
    source: Any,
) -> torch.Tensor:
    validate_workspace_shape(workspace, source, processor.block.output_dim)
    cycle_input = workspace_cycle_input(
        workspace, source, processor.block.output_dim
    )
    return processor.block(cycle_input, source.selected_heads)


def refine_workspace(
    processor: Any,
    workspace: torch.Tensor,
    source: Any,
) -> torch.Tensor:
    """Integrate the opt-in tied workspace field without changing source state."""
    validate_workspace_shape(workspace, source, processor.block.output_dim)
    config = processor.workspace_flow_refinement
    if config is None:
        return workspace
    state = workspace
    step_size = 1.0 / config.solver_steps
    for step in range(config.solver_steps):
        time = step * step_size
        if config.solver == "euler":
            state = state + step_size * workspace_velocity(
                processor, state, time, source, config
            )
        else:
            first_velocity = workspace_velocity(
                processor, state, time, source, config
            )
            midpoint = state + 0.5 * step_size * first_velocity
            midpoint_velocity = workspace_velocity(
                processor,
                midpoint,
                time + 0.5 * step_size,
                source,
                config,
            )
            state = state + step_size * midpoint_velocity
    return state


def workspace_velocity(
    processor: Any,
    workspace: torch.Tensor,
    time: float,
    source: Any,
    config: WorkspaceFlowRefinementConfig,
) -> torch.Tensor:
    time_condition = sinusoidal_time_embedding(
        time, workspace.shape[-1], workspace
    )
    cycle_input = workspace_cycle_input(
        workspace + time_condition, source, processor.block.output_dim
    )
    raw_velocity = processor.block.compute_update(
        cycle_input, source.selected_heads
    )
    return config.max_velocity * torch.tanh(raw_velocity)


def write_workspace_for_readout(
    processor: Any,
    cache: CacheLike,
    source: Any,
    workspace: torch.Tensor,
) -> WindowedCache:
    if processor.config.update_stream != "v":
        raise ValueError(
            "Longshot workspace readout initially supports value-only writes"
        )
    if not isinstance(cache, WindowedCache):
        raise TypeError("Longshot workspace readout requires a WindowedCache")
    if cache.linear_states is not None:
        raise ValueError(
            "Longshot workspace readout does not support shared linear cache state"
        )
    validate_workspace_shape(workspace, source, processor.block.output_dim)
    workspace = refine_workspace(processor, workspace, source)
    branch = WindowedCache(
        keys=cache.keys.clone(),
        values=cache.values.clone(),
        scores=cache.scores.clone(),
        linear_states=cache.linear_states,
        kv_score_sidecar=cache.kv_score_sidecar,
    )
    values = _split_merged_values(
        workspace[:, :, None],
        processor.num_layers,
        processor.num_kv_heads,
        processor.head_dim,
    )
    branch = cast(
        WindowedCache,
        _scatter_latest_values(branch, source.latest_indices, values),
    )
    return cast(
        WindowedCache,
        processor._refresh_kv_score_sidecar(branch, source.latest_indices),
    )
