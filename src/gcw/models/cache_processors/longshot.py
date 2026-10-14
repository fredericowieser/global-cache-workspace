from __future__ import annotations

from typing import Any
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from gcw.config import (
    ModelConfig,
    linear_attention_following_cache_indices,
)
from gcw.models.cache_processors.base import CacheProcessorBase
from gcw.models.cache_processors.update_metrics import UPDATE_BATCH
from gcw.models.cache_processors.ops import (
    expand_cache_head_indices,
    gather_per_cache_head_kv,
    gather_per_cache_head_scores,
    select_per_cache_head_kv_norm_indices_from_scores,
    latest_shared_indices,
    rotate_cache_keys,
    scatter_latest as _scatter_latest,
    scatter_latest_values as _scatter_latest_values,
    split_merged_cache as _split_merged,
    split_merged_cache_values as _split_merged_values,
)
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import (
    GroupedWindowedCache,
    WindowedCache,
)
from gcw.models.cache_processors.layout import cache_shape_groups
from gcw.models.cache_processors.longshot_selection import (
    _select_heads_from_head_scores,
    cache_movement_backend,
    refresh_sidecar,
    select_kv_norm_history,
    synchronize_sidecar,
)
from gcw.models.cache_processors.longshot_layers import (
    EncoderMethod,
)
from gcw.models.cache_processors.longshot_processor import CacheProcessor
from gcw.models.cache_processors.longshot_linear_state import (
    LinearStateInputs,
    LinearStateSpec,
    _topk_linear_heads,
)
from gcw.performance.longshot_scores import (
    chunked_fp32_l2_norm,
    native_cache_squared_l2,
)
from gcw.performance.kv_score_sidecar import kv_score_sidecar_execution
from gcw.performance.longshot_cache import longshot_full_history_execution

CacheLike = CacheState | StaticCacheState | WindowedCache | GroupedWindowedCache


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
        groups = cache_shape_groups(model_config.stored_cache_shapes)
        self.shape_groups = groups if len(groups) > 1 else ()
        if model_config.cache_processor is None:
            raise ValueError("Longshot requires cache_processor config")
        self.config = model_config.cache_processor
        self._gcw_longshot_model_composition = (
            model_config.variant,
            model_config.family,
            model_config.flow_processor is not None,
            model_config.diffusion_processor is not None,
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
        self.num_kv_heads, self.head_dim = (
            (0, 0) if self.shape_groups else model_config.stored_cache_shape
        )
        self.rope_theta = model_config.rope_theta
        self.rotary_style = model_config.rotary_style
        self.rotary_dim = int(
            self.head_dim * model_config.partial_rotary_factor
        )
        self.num_kv_cache_heads = model_config.cache_num_heads
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
            diffusion_config=model_config.diffusion_processor,
            kv_cache_heads=self.num_kv_cache_heads,
            model_config=model_config,
            linear_state_specs=self.linear_state_specs,
            shape_groups=self.shape_groups,
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
                batch_metrics = (
                    UPDATE_BATCH.get() if not self.training else None
                )
                if batch_metrics is not None:
                    batch_metrics.external_iteration = iteration
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

    def _forward_once(
        self, cache: CacheLike, *, select_all: bool = False
    ) -> tuple[CacheLike, torch.Tensor | None]:
        if self.shape_groups:
            return self._forward_grouped(cache, select_all=select_all)
        if kv_score_sidecar_execution(self).enabled:
            if isinstance(cache, CacheState):
                cache = WindowedCache.from_cache_state(cache)
            if not isinstance(cache, WindowedCache):
                raise TypeError(
                    "enabled KV-score sidecar requires contiguous WindowedCache storage"
                )
            synchronize_sidecar(self, cache)
        routed_initial_selection = self.config.head_selection_method in {
            "learned_contextual",
            "random",
        }
        if self.config.score == "kv_norm" and not select_all:
            merged, scores, indices = select_kv_norm_history(self, cache)
            if indices.shape[-1] == 0:
                return cache, None
            kv_head_scores = None
            if not routed_initial_selection:
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
                backend=cache_movement_backend(self),
            )
            kv_head_scores = (
                None
                if routed_initial_selection or select_all
                else chunked_fp32_l2_norm(merged)
            )
        if (
            kv_head_scores is not None
            and self.config.head_score_normalization == "rms"
        ):
            divisor = 2 * self.head_dim
            kv_head_scores = kv_head_scores / (
                divisor
                if self.config.score == "kv_norm" and not select_all
                else divisor**0.5
            )
        latest = latest_shared_indices(indices)
        kv_hidden = (
            self.block.encode(merged)
            if self.block.linear_state_adapter is not None
            and cache.linear_states
            else None
        )
        linear_inputs = self._linear_state_inputs(cache, kv_hidden)
        if routed_initial_selection or select_all:
            selected_heads = None
        else:
            if kv_head_scores is None:
                raise RuntimeError("magnitude head scores were not computed")
            selected_heads = self._select_combined_heads(
                batch=merged.shape[0],
                kv_head_scores=kv_head_scores,
                linear_inputs=linear_inputs,
                select_all=select_all,
                device=merged.device,
            )
        linear_features: torch.Tensor | None = None
        topk_heads = self.config.topk_heads
        if linear_inputs is not None:
            linear_features = linear_inputs.features
            topk_heads += _topk_linear_heads(self.config)
        batch_metrics = UPDATE_BATCH.get() if not self.training else None
        observation = (
            batch_metrics.application(
                self,
                cache,
                linear_inputs.specs if linear_inputs else [],
                merged[:, :, -1, :],
            )
            if batch_metrics is not None
            else nullcontext()
        )
        with observation:
            updated, linear_delta, selected_heads = self.block(
                merged,
                selected_heads,
                kv_hidden=kv_hidden,
                linear_features=linear_features,
                topk_heads=topk_heads,
                select_all=select_all,
            )
        if linear_inputs is not None:
            if linear_delta is None:
                raise RuntimeError(
                    "linear state processor did not produce an update"
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
            return refresh_sidecar(self, cache, latest), selected_heads
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
        return refresh_sidecar(self, cache, latest), selected_heads

    @property
    def recurrent_cache_format(self) -> str:
        if (
            self._gcw_longshot_model_composition[1] == "gemma4"
            and not kv_score_sidecar_execution(self).enabled
            and not longshot_full_history_execution(self).enabled
            and cache_movement_backend(self) == "eager"
        ):
            return "list"
        return "windowed"

    def _forward_grouped(
        self, cache: CacheLike, *, select_all: bool
    ) -> tuple[CacheLike, torch.Tensor | None]:
        if (
            kv_score_sidecar_execution(self).enabled
            or longshot_full_history_execution(self).enabled
            or cache_movement_backend(self) != "eager"
        ):
            raise ValueError(
                "mixed-shape Longshot requires reference cache movement"
            )
        if isinstance(cache, GroupedWindowedCache):
            if cache.layer_groups != tuple(
                group.layers for group in self.shape_groups
            ):
                raise ValueError(
                    "cache groups do not match the processor layer layout"
                )
            groups = cache.groups
        elif isinstance(cache, (CacheState, StaticCacheState)):
            groups = tuple(
                (
                    StaticCacheState(
                        [cache.keys[i] for i in spec.layers],
                        [cache.values[i] for i in spec.layers],
                        cache.score_storage,
                        cache.length,
                    )
                    if isinstance(cache, StaticCacheState)
                    else CacheState(
                        [cache.keys[i] for i in spec.layers],
                        [cache.values[i] for i in spec.layers],
                        cache.scores,
                    )
                )
                for spec in self.shape_groups
            )
        else:
            raise TypeError(
                "mixed-shape Longshot requires a grouped, list, or static cache"
            )
        if cache.seq_len == 0:
            return cache, None
        magnitude_selection = (
            self.config.head_selection_method == "magnitude" and not select_all
        )
        histories, head_scores, frontiers = [], [], []
        for spec, group in zip(self.shape_groups, groups, strict=True):
            magnitude = None
            if self.config.score == "kv_norm" and not select_all:
                scores = native_cache_squared_l2(group)
                indices = select_per_cache_head_kv_norm_indices_from_scores(
                    scores, self.config.process_every, self.config.topk
                )
                if magnitude_selection:
                    magnitude = (
                        gather_per_cache_head_scores(scores, indices)
                        .float()
                        .sum(-1)
                    )
                divisor = 2 * spec.head_dim
            else:
                indices = expand_cache_head_indices(
                    (
                        cache.all_indices()
                        if select_all
                        else cache.select_indices(
                            self.config.process_every, self.config.topk
                        )
                    ),
                    len(spec.head_indices),
                )
                divisor = (2 * spec.head_dim) ** 0.5
            merged = gather_per_cache_head_kv(group, indices, unrope_keys=False)
            if magnitude_selection:
                if magnitude is None:
                    magnitude = chunked_fp32_l2_norm(merged)
                if self.config.head_score_normalization == "rms":
                    magnitude = magnitude / divisor
                head_scores.append(magnitude)
            histories.append(merged)
            frontiers.append(latest_shared_indices(indices))
        selected = None
        if magnitude_selection:
            scores = torch.cat(head_scores, dim=1).index_select(
                1, self.block.canonical_order
            )
            selected = self._select_combined_heads(
                batch=cache.batch_size,
                kv_head_scores=scores,
                linear_inputs=None,
                select_all=select_all,
                device=scores.device,
            )
        metrics = UPDATE_BATCH.get() if not self.training else None
        observation = (
            metrics.application(
                self, cache, [], tuple(values[:, :, -1] for values in histories)
            )
            if metrics is not None
            else nullcontext()
        )
        with observation:
            updates, _, selected = self.block(
                tuple(histories), selected, select_all=select_all
            )
        for spec, group, latest, updated in zip(
            self.shape_groups, groups, frontiers, updates, strict=True
        ):
            if self.config.update_stream == "v":
                values = _split_merged_values(
                    updated[:, :, None],
                    len(spec.layers),
                    spec.num_heads,
                    spec.head_dim,
                )
                _scatter_latest_values(group, latest, values)
            else:
                keys, values = _split_merged(
                    updated[:, :, None],
                    len(spec.layers),
                    spec.num_heads,
                    spec.head_dim,
                )
                _scatter_latest(group, latest, keys, values)
            if isinstance(cache, CacheState):
                for local, layer_idx in enumerate(spec.layers):
                    cache.keys[layer_idx], cache.values[layer_idx] = (
                        group.layer(local)
                    )
        return cache, selected

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
        topk_linear = _topk_linear_heads(self.config)
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
            if state is None or state[0] is None:
                raise RuntimeError(
                    f"linear state layer {spec.layer_idx} has no recurrent "
                    "state; cache heads are packed positionally and cannot be "
                    "processed with a missing group"
                )
            recurrent_state = state[0]
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
            if features.shape[1] != spec.head_count:
                raise RuntimeError(
                    f"linear state layer {spec.layer_idx} produced "
                    f"{features.shape[1]} heads but its spec declares "
                    f"{spec.head_count}; head identities would misalign"
                )
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

    def _scatter_linear_state_updates(
        self,
        cache: CacheLike,
        linear_delta: torch.Tensor,
        specs: list[LinearStateSpec],
    ) -> CacheLike:
        if self.block.linear_state_adapter is None or not specs:
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
            recurrent_state, conv_state = states[spec.layer_idx]
            if recurrent_state is None:
                continue
            if self.config.linear_state_update_mode == "gated_delta":
                new_recurrent = self._gated_delta_linear_state_update(
                    spec.layer_idx,
                    update,
                    recurrent_state,
                )
                states[spec.layer_idx] = (new_recurrent, conv_state)
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
                recurrent_state + delta_state.to(recurrent_state.dtype),
                conv_state,
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
        if heads != dynamics.num_heads:
            raise ValueError(
                f"linear state head count {heads} does not match layer {layer_idx} "
                f"value heads {dynamics.num_heads}"
            )
        query = synthetic_key.reshape(batch, 1, heads, key_dim)
        key = query
        value = synthetic_value.reshape(batch, 1, heads, value_dim)
        beta = torch.sigmoid(beta_logit.squeeze(-1)).unsqueeze(1)
        g = dynamics(decay_input.squeeze(-1)).unsqueeze(1)
        _, final_state = qwen35_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
        )
        return final_state.to(recurrent_state.dtype)
