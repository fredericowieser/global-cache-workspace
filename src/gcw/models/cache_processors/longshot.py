from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

# Re-exported so ``longshot_processor`` can look it up via ``getattr(ls_pkg, "checkpoint")``;
# tests monkeypatch ``longshot.checkpoint`` to assert checkpointing call counts.
from torch.utils.checkpoint import checkpoint  # noqa: F401

from gcw.config import (
    ModelConfig,
    WorkspaceFlowRefinementConfig,
    linear_attention_following_cache_indices,
)
from gcw.models.cache_processors.base import CacheProcessorBase
from gcw.models.cache_processors.ops import (
    expand_cache_head_indices,
    gather_per_cache_head_kv,
    gather_per_cache_head_scores,
    latest_shared_indices,
    rotate_cache_keys,
    scatter_latest as _scatter_latest,
    scatter_latest_values as _scatter_latest_values,
    split_merged_cache as _split_merged,
    split_merged_cache_values as _split_merged_values,
)
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.models.cache_processors.longshot_workspace import (
    deliberate_workspace,
    refine_workspace,
    validate_workspace_shape,
    workspace_cycle_input,
    workspace_velocity,
    write_workspace_for_readout,
)
from gcw.models.cache_processors.longshot_selection import (
    cache_movement_backend,
    refresh_sidecar,
    select_kv_norm_history,
    synchronize_sidecar,
)
from gcw.models.cache_processors.longshot_layers import (
    EncoderMethod,
    LongshotWorkspaceSource,
    _select_heads_from_head_scores,
)
from gcw.models.cache_processors.longshot_processor import CacheProcessor
from gcw.models.cache_processors.longshot_linear_state import (
    LinearStateInputs,
    LinearStateSpec,
    _topk_linear_heads,
)
from gcw.performance.longshot_scores import (
    chunked_fp32_l2_norm,
    # Re-exported: ``longshot_selection`` reaches it via ``ls_pkg`` and tests
    # monkeypatch it to compare eager vs optimized scoring.
    chunked_fp32_squared_l2,  # noqa: F401
)
from gcw.performance.kv_score_sidecar import (
    kv_score_sidecar_execution,
    # Re-exported: ``longshot_selection`` reaches these via ``getattr(ls_pkg, ...)``
    # and tests monkeypatch them to assert sidecar lifecycle behavior.
    refresh_kv_score_sidecar,  # noqa: F401
    synchronize_kv_score_sidecar,  # noqa: F401
)

# Re-exported for backward compatibility: downstream modules and tests import
# these layer primitives from the historical monolithic ``longshot`` module
# location rather than from ``longshot_layers``.
from gcw.models.cache_processors.longshot_layers import (  # noqa: F401
    ProcessorAttention,
    QueryAttentionTimeEncoder,
    _gradient_checkpointing_enabled,
    longshot_gradient_checkpointing_override,
)

# The cache object any processor path accepts: static, mutable, or windowed.
CacheLike = CacheState | StaticCacheState | WindowedCache


class GlobalCacheProcessor(CacheProcessorBase):
    def __init__(
        self,
        model_config: ModelConfig,
        *,
        encoder_method: EncoderMethod | None = None,
        encoder_seq_len: int | None = None,
        linear_state_dynamics: dict[int, Any] | None = None,
    ):
        super().__init__()
        self.linear_state_dynamics = linear_state_dynamics or {}
        model_config.ensure_uniform_stored_cache_shape("Longshot")
        if model_config.cache_processor is None:
            raise ValueError("Longshot requires cache_processor config")
        self.config = model_config.cache_processor
        self.workspace_flow_refinement = model_config.workspace_flow_refinement
        self._gcw_longshot_model_composition = (
            model_config.variant,
            model_config.family,
            model_config.flow_processor is not None,
            model_config.diffusion_processor is not None,
            model_config.workspace_flow_refinement is not None,
        )
        if self.config.processor_mode != "standard":
            raise ValueError(
                "Longshot supports only cache_processor.processor_mode='standard'"
            )
        if self.config.update_stream not in {"kv", "v"}:
            raise ValueError(
                "Longshot supports only cache_processor.update_stream='kv' or 'v'"
            )
        if self.config.score not in {"attn_l2", "kv_norm"}:
            raise ValueError(
                "Longshot supports only cache_processor.score='attn_l2' or 'kv_norm'"
            )
        encoder_method = encoder_method or self.config.encoder_method
        self.num_layers = model_config.cache_num_layers
        self.num_kv_heads, self.head_dim = model_config.stored_cache_shape
        self.rope_theta = model_config.rope_theta
        self.rotary_style = model_config.rotary_style
        self.rotary_dim = int(
            self.head_dim * model_config.partial_rotary_factor
        )
        self.num_kv_cache_heads = self.num_layers * self.num_kv_heads
        self.linear_state_specs = self._build_linear_state_specs(model_config)
        if (
            self.config.linear_state_update_mode == "gated_delta"
            and self.linear_state_specs
        ):
            missing = sorted(
                spec.layer_idx
                for spec in self.linear_state_specs
                if spec.layer_idx not in self.linear_state_dynamics
            )
            if missing:
                raise ValueError(
                    "gated_delta linear-state updates require matching Qwen3.5 linear-attention "
                    f"modules for layers {missing}"
                )
        self.num_linear_state_heads = sum(
            spec.head_count for spec in self.linear_state_specs
        )
        self.num_cache_heads = (
            self.num_kv_cache_heads + self.num_linear_state_heads
        )
        output_dim = (
            self.head_dim
            if self.config.update_stream == "v"
            else 2 * self.head_dim
        )
        self.block = CacheProcessor(
            2 * self.head_dim,
            self.num_cache_heads,
            self.config,
            encoder_method=encoder_method,
            encoder_seq_len=encoder_seq_len or self.config.encoder_seq_len,
            output_dim=output_dim,
            flow_config=model_config.flow_processor,
            kv_cache_heads=self.num_kv_cache_heads,
            model_config=model_config,
            linear_state_specs=self.linear_state_specs,
        )

    def _build_linear_state_specs(
        self, model_config: ModelConfig
    ) -> list[LinearStateSpec]:
        if not self.config.linear_state_cache:
            return []
        linear_value_heads = int(
            model_config.linear_num_value_heads or model_config.num_heads
        )
        return [
            LinearStateSpec(
                layer_idx=layer_idx,
                following_cache_idx=following_cache_idx,
                head_count=linear_value_heads,
            )
            for layer_idx, following_cache_idx in linear_attention_following_cache_indices(
                model_config,
                require_layer_types=True,
            )
        ]

    def forward(
        self, cache: CacheLike, *, select_all: bool = False
    ) -> CacheLike:
        if self.config.iteration_mode == "external":
            previous_selected_heads = None
            for iteration in range(1, self.config.iterations + 1):
                cache, selected_heads = self._forward_once(
                    cache, select_all=select_all
                )
                if (
                    previous_selected_heads is not None
                    and selected_heads is not None
                ):
                    self.block.record_selected_head_overlap(
                        iteration - 1, previous_selected_heads, selected_heads
                    )
                previous_selected_heads = selected_heads
            return cache
        cache, _ = self._forward_once(cache, select_all=select_all)
        return cache

    def encode_source(
        self, cache: CacheLike, *, select_all: bool = False
    ) -> LongshotWorkspaceSource:
        if self.config.score == "kv_norm" and not select_all:
            merged, scores, indices = self._select_kv_norm_history(cache)
            if indices.shape[-1] == 0:
                raise ValueError(
                    "Longshot workspace source requires at least one selected position"
                )
            head_scores = (
                gather_per_cache_head_scores(scores, indices)
                .float()
                .sum(dim=-1)
            )
        else:
            if select_all:
                indices = expand_cache_head_indices(
                    cache.all_indices(), self.num_kv_cache_heads
                )
            else:
                indices = expand_cache_head_indices(
                    cache.select_indices(
                        self.config.process_every, self.config.topk
                    ),
                    self.num_kv_cache_heads,
                )
            if indices.shape[-1] == 0:
                raise ValueError(
                    "Longshot workspace source requires at least one selected position"
                )
            merged = gather_per_cache_head_kv(
                cache,
                indices,
                unrope_keys=self.config.unrope_keys,
                rope_theta=self.rope_theta,
                rotary_style=self.rotary_style,
                rotary_dim=self.rotary_dim,
                backend=self._cache_movement_backend(),
            )
            head_scores = chunked_fp32_l2_norm(merged)
        if select_all:
            selected_heads = (
                torch.arange(
                    self.num_kv_cache_heads,
                    dtype=torch.long,
                    device=merged.device,
                )
                .unsqueeze(0)
                .expand(merged.shape[0], -1)
            )
        else:
            selected_heads = _select_heads_from_head_scores(
                head_scores, self.config.topk_heads
            )
        return LongshotWorkspaceSource(
            merged=merged,
            indices=indices,
            selected_heads=selected_heads,
            latest_indices=latest_shared_indices(indices),
        )

    def initialise_workspace(
        self, source: LongshotWorkspaceSource
    ) -> torch.Tensor:
        return source.merged[:, :, -1, -self.block.output_dim :].clone()

    def _validate_workspace_shape(
        self,
        workspace: torch.Tensor,
        source: LongshotWorkspaceSource,
    ) -> None:
        validate_workspace_shape(workspace, source, self.block.output_dim)

    def deliberate(
        self, workspace: torch.Tensor, source: LongshotWorkspaceSource
    ) -> torch.Tensor:
        return deliberate_workspace(self, workspace, source)

    def _workspace_cycle_input(
        self,
        workspace: torch.Tensor,
        source: LongshotWorkspaceSource,
    ) -> torch.Tensor:
        return workspace_cycle_input(workspace, source, self.block.output_dim)

    def refine_workspace(
        self,
        workspace: torch.Tensor,
        source: LongshotWorkspaceSource,
    ) -> torch.Tensor:
        """Integrate the opt-in tied workspace field without changing source state."""
        return refine_workspace(self, workspace, source)

    def _workspace_velocity(
        self,
        workspace: torch.Tensor,
        time: float,
        source: LongshotWorkspaceSource,
        config: WorkspaceFlowRefinementConfig,
    ) -> torch.Tensor:
        return workspace_velocity(self, workspace, time, source, config)

    def write_for_readout(
        self,
        cache: CacheLike,
        source: LongshotWorkspaceSource,
        workspace: torch.Tensor,
    ) -> WindowedCache:
        return write_workspace_for_readout(self, cache, source, workspace)

    def _select_kv_norm_history(
        self,
        cache: CacheLike,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return select_kv_norm_history(self, cache)

    def _forward_once(
        self, cache: CacheLike, *, select_all: bool = False
    ) -> tuple[CacheLike, torch.Tensor | None]:
        if kv_score_sidecar_execution(self).enabled:
            if isinstance(cache, CacheState):
                cache = WindowedCache.from_cache_state(cache)
            if not isinstance(cache, WindowedCache):
                raise TypeError(
                    "enabled KV-score sidecar requires contiguous WindowedCache storage"
                )
            self._synchronize_kv_score_sidecar(cache)
        if self.config.score == "kv_norm" and not select_all:
            merged, scores, indices = self._select_kv_norm_history(cache)
            if indices.shape[-1] == 0:
                return cache, None
            kv_head_scores = (
                gather_per_cache_head_scores(scores, indices)
                .float()
                .sum(dim=-1)
            )
        else:
            if select_all:
                indices = expand_cache_head_indices(
                    cache.all_indices(), self.num_kv_cache_heads
                )
            else:
                indices = expand_cache_head_indices(
                    cache.select_indices(
                        self.config.process_every, self.config.topk
                    ),
                    self.num_kv_cache_heads,
                )
            if indices.shape[-1] == 0:
                return cache, None
            merged = gather_per_cache_head_kv(
                cache,
                indices,
                unrope_keys=self.config.unrope_keys,
                rope_theta=self.rope_theta,
                rotary_style=self.rotary_style,
                rotary_dim=self.rotary_dim,
                backend=self._cache_movement_backend(),
            )
            kv_head_scores = chunked_fp32_l2_norm(merged)
        latest = latest_shared_indices(indices)
        kv_hidden = (
            self.block.encode_input(merged)
            if self.block.has_linear_state_adapter() and cache.linear_states
            else None
        )
        linear_inputs = self._linear_state_inputs(cache, kv_hidden)
        selected_heads = self._select_combined_heads(
            batch=merged.shape[0],
            kv_head_scores=kv_head_scores,
            linear_inputs=linear_inputs,
            select_all=select_all,
            device=merged.device,
        )
        if linear_inputs is None:
            updated = (
                self.block(merged, selected_heads)
                if kv_hidden is None
                else self.block.forward_from_hidden(
                    merged, kv_hidden, selected_heads
                )
            )
        else:
            if kv_hidden is None:
                raise RuntimeError(
                    "linear state inputs require encoded KV cache features"
                )
            updated, linear_delta = (
                self.block.forward_with_linear_state_features(
                    merged,
                    kv_hidden,
                    linear_inputs.features,
                    selected_heads,
                )
            )
            cache = self._scatter_linear_state_updates(
                cache, linear_delta, linear_inputs.specs
            )
        if self.config.update_stream == "v":
            values = _split_merged_values(
                updated[:, :, None, :],
                self.num_layers,
                self.num_kv_heads,
                self.head_dim,
            )
            cache = _scatter_latest_values(cache, latest, values)
            return self._refresh_kv_score_sidecar(cache, latest), selected_heads
        keys, values = _split_merged(
            updated[:, :, None, :],
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
        )
        if self.config.unrope_keys:
            keys = rotate_cache_keys(
                keys,
                latest,
                theta=self.rope_theta,
                rotary_style=self.rotary_style,
                rotary_dim=self.rotary_dim,
            )
        cache = _scatter_latest(cache, latest, keys, values)
        return self._refresh_kv_score_sidecar(cache, latest), selected_heads

    def _synchronize_kv_score_sidecar(self, cache: WindowedCache) -> None:
        synchronize_sidecar(self, cache)

    def _refresh_kv_score_sidecar(
        self, cache: CacheLike, positions: torch.Tensor
    ) -> CacheLike:
        return refresh_sidecar(self, cache, positions)

    def _cache_movement_backend(self) -> str:
        return cache_movement_backend(self)

    def _select_combined_heads(
        self,
        *,
        batch: int,
        kv_head_scores: torch.Tensor,
        linear_inputs: LinearStateInputs | None,
        select_all: bool,
        device: torch.device,
    ) -> torch.Tensor:
        if select_all:
            head_count = self.num_kv_cache_heads + (
                0 if linear_inputs is None else linear_inputs.features.shape[1]
            )
            return (
                torch.arange(head_count, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(batch, -1)
            )
        topk_heads = int(self.config.topk_heads or 0)
        if linear_inputs is None:
            return _select_heads_from_head_scores(kv_head_scores, topk_heads)
        selected = [_select_heads_from_head_scores(kv_head_scores, topk_heads)]
        topk_linear = int(_topk_linear_heads(self.config) or 0)
        if topk_linear > 0:
            key_dim = self.block.linear_state_update_dims()[0]
            linear_head_scores = (
                linear_inputs.features[..., key_dim:].float().pow(2).sum(dim=-1)
            )
            selected.append(
                _select_heads_from_head_scores(linear_head_scores, topk_linear)
                + self.num_kv_cache_heads
            )
        return torch.cat(selected, dim=-1)

    def _linear_state_inputs(
        self, cache: CacheLike, kv_hidden: torch.Tensor | None
    ) -> LinearStateInputs | None:
        if kv_hidden is None or not cache.linear_states:
            return None
        feature_parts: list[torch.Tensor] = []
        active_specs: list[LinearStateSpec] = []
        context_cache: dict[int, torch.Tensor] = {}
        query_cache: dict[tuple[str, int], torch.Tensor] = {}
        for spec in self.linear_state_specs:
            state = cache.linear_states.get(spec.layer_idx)
            if state is None or state[1] is None:
                continue
            recurrent_state = state[1]
            if spec.following_cache_idx not in context_cache:
                context_cache[spec.following_cache_idx] = (
                    self._following_kv_hidden_context(
                        kv_hidden, spec.following_cache_idx
                    )
                )
            query_key = self.block.linear_state_query_cache_key(
                spec.layer_idx, spec.following_cache_idx
            )
            cache_key = (query_key, spec.following_cache_idx)
            if cache_key not in query_cache:
                query_cache[cache_key] = self.block.linear_state_query(
                    spec.layer_idx,
                    spec.following_cache_idx,
                    context_cache[spec.following_cache_idx],
                )
            query = query_cache[cache_key].to(recurrent_state.dtype)
            readout = torch.einsum("bhk,bhkv->bhv", query, recurrent_state)
            features = torch.cat([query.to(readout.dtype), readout], dim=-1)
            feature_parts.append(features)
            active_specs.append(spec)
        if not feature_parts:
            return None
        return LinearStateInputs(
            features=torch.cat(feature_parts, dim=1),
            specs=active_specs,
        )

    def _following_kv_hidden_context(
        self, kv_hidden: torch.Tensor, cache_idx: int
    ) -> torch.Tensor:
        start = cache_idx * self.num_kv_heads
        end = start + self.num_kv_heads
        return kv_hidden[:, start:end].mean(dim=1)

    def _linear_state_forward(
        self,
        layer_idx: int,
        recurrent_state: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        from gcw.models.backbones.qwen3 import qwen35_gated_delta_rule

        dynamics = self.linear_state_dynamics.get(layer_idx)
        if dynamics is None:
            raise RuntimeError(
                f"missing linear state dynamics for layer {layer_idx}"
            )
        key, value, g, beta = dynamics(query)
        _, final_state = qwen35_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
        )
        return final_state.to(recurrent_state.dtype)

    def _scatter_linear_state_updates(
        self,
        cache: CacheLike,
        linear_delta: torch.Tensor,
        specs: list[LinearStateSpec],
    ) -> CacheLike:
        if not self.block.has_linear_state_adapter() or not specs:
            return cache
        states = (
            {} if cache.linear_states is None else dict(cache.linear_states)
        )
        offset = 0
        key_dim, value_dim = self.block.linear_state_update_dims()
        for spec in specs:
            update = linear_delta[:, offset : offset + spec.head_count]
            offset += spec.head_count
            if spec.layer_idx not in states:
                continue
            conv_state, recurrent_state = states[spec.layer_idx]
            if recurrent_state is None:
                continue
            if self.config.linear_state_update_mode == "gated_delta":
                new_recurrent = self._gated_delta_linear_state_update(
                    spec.layer_idx,
                    update,
                    recurrent_state,
                )
                states[spec.layer_idx] = (conv_state, new_recurrent)
                continue
            delta_key, delta_value = torch.split(
                update, [key_dim, value_dim], dim=-1
            )
            delta_key = F.normalize(delta_key.float(), p=2, dim=-1).to(
                delta_value.dtype
            )
            delta_state = torch.einsum(
                "bhk,bhv->bhkv", delta_key.float(), delta_value.float()
            )
            states[spec.layer_idx] = (
                conv_state,
                recurrent_state + delta_state.to(recurrent_state.dtype),
            )
        cache.linear_states = states
        return cache

    def _gated_delta_linear_state_update(
        self,
        layer_idx: int,
        update: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> torch.Tensor:
        from gcw.models.backbones.qwen3 import qwen35_gated_delta_rule

        dynamics = self.linear_state_dynamics.get(layer_idx)
        if dynamics is None:
            raise ValueError(
                f"missing Qwen3.5 linear-state dynamics for layer {layer_idx}"
            )
        key_dim, value_dim = self.block.linear_state_update_dims()
        synthetic_key, synthetic_value, beta_logit, decay_input = torch.split(
            update, [key_dim, value_dim, 1, 1], dim=-1
        )
        batch, heads, _ = synthetic_key.shape
        dynamics_heads = int(dynamics.A_log.shape[0])
        if heads != dynamics_heads:
            raise ValueError(
                f"linear state head count {heads} does not match layer {layer_idx} value heads {dynamics_heads}"
            )
        query = synthetic_key.reshape(batch, 1, heads, key_dim)
        key = query
        value = synthetic_value.reshape(batch, 1, heads, value_dim)
        beta = torch.sigmoid(beta_logit.squeeze(-1)).unsqueeze(1)
        g = (
            -dynamics.A_log.float().exp()
            * F.softplus(decay_input.squeeze(-1).float() + dynamics.dt_bias)
        ).unsqueeze(1)
        _, final_state = qwen35_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
        )
        return final_state.to(recurrent_state.dtype)
