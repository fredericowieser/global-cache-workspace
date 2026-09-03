from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from gcw.config import (
    CacheProcessorConfig,
    FlowProcessorConfig,
    ModelConfig,
)
from gcw.models.backbones.transformer import RMSNorm
from gcw.models.cache_processors.base import cache_processor_recomputation_id
from gcw.models.cache_processors.flow import LatentFlow
from gcw.models.cache_processors.longshot_layers import (
    EncoderMethod,
    PerCacheHeadLinear,
    ProcessorLayer,
    QueryAttentionTimeEncoder,
    _fit_time_length,
    _gradient_checkpointing_enabled,
    _zero_projection,
)
from gcw.models.cache_processors.longshot_linear_state import (
    LinearStateAdapter,
    LinearStateSpec,
    _topk_linear_heads,
)
from gcw.models.cache_processors.longshot_selection import (
    ContextualHeadSelector,
    _select_heads_from_head_scores,
)


class CacheProcessor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_cache_heads: int,
        config: CacheProcessorConfig,
        *,
        encoder_method: EncoderMethod = "linear",
        encoder_seq_len: int | None = None,
        output_dim: int | None = None,
        flow_config: FlowProcessorConfig | None = None,
        kv_cache_heads: int | None = None,
        model_config: ModelConfig | None = None,
        linear_state_specs: list[LinearStateSpec] | None = None,
    ):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.encoder_method = encoder_method
        self.encoder_query_source = config.encoder_query_source
        self.encoder_seq_len = (
            encoder_seq_len
            or config.encoder_seq_len
            or config.process_every + max(config.topk, 0)
        )
        self.kv_cache_heads = kv_cache_heads or num_cache_heads
        self.in_norm = RMSNorm(input_dim, config.norm_eps)
        if encoder_method == "linear":
            in_features = self.encoder_seq_len * input_dim
        elif encoder_method == "query_attention":
            in_features = input_dim
            self.time_encoder = QueryAttentionTimeEncoder(
                input_dim, self.kv_cache_heads, config
            )
        else:
            raise ValueError(f"unknown encoder_method: {encoder_method}")
        self.in_proj: nn.Module
        if config.per_cache_head_projections:
            self.in_proj = PerCacheHeadLinear(
                self.kv_cache_heads, in_features, config.hidden_size
            )
        else:
            self.in_proj = nn.Linear(in_features, config.hidden_size)
        self.layers = nn.ModuleList(
            ProcessorLayer(config) for _ in range(config.num_layers)
        )
        self.out_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.out_proj: nn.Module
        if config.per_cache_head_projections:
            self.out_proj = PerCacheHeadLinear(
                self.kv_cache_heads, config.hidden_size, self.output_dim
            )
        else:
            self.out_proj = nn.Linear(config.hidden_size, self.output_dim)
        self.gate = nn.Parameter(torch.full((1,), config.gate_init))
        self.flow = (
            LatentFlow(config.hidden_size, flow_config)
            if flow_config is not None
            else None
        )
        self._selected_head_overlap_totals: dict[int, torch.Tensor] = {}
        self._selected_head_overlap_counts: dict[int, int] = {}
        self.head_selector: ContextualHeadSelector | None
        if (
            config.head_selection_method == "learned_contextual"
            or config.internal_iteration_score == "learned_contextual"
        ):
            self.head_selector = ContextualHeadSelector(
                config.hidden_size,
                config.head_selector_hidden_size,
                num_cache_heads,
                norm_eps=config.norm_eps,
                use_identity_bias=config.head_selector_use_identity_bias,
            )
        else:
            self.head_selector = None
        self.linear_state_adapter: LinearStateAdapter | None
        self.init_weights()
        if linear_state_specs:
            if model_config is None:
                raise ValueError("linear_state_specs require model_config")
            self.linear_state_adapter = LinearStateAdapter(
                model_config, config, linear_state_specs
            )
        else:
            self.linear_state_adapter = None

    def init_weights(self, std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, PerCacheHeadLinear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, QueryAttentionTimeEncoder) and hasattr(
                module, "query"
            ):
                nn.init.normal_(module.query, mean=0.0, std=std)
        nn.init.constant_(self.gate, self.config.gate_init)
        if self.config.out_proj_init == "zero":
            _zero_projection(self.out_proj)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x)
        if self.encoder_method == "linear":
            x = _fit_time_length(x, self.encoder_seq_len)
            return self.in_proj(x.flatten(start_dim=2))
        if self.encoder_method == "query_attention":
            return self.in_proj(self.time_encoder(x))
        raise ValueError(f"unknown encoder_method: {self.encoder_method}")

    def select_heads(
        self, hidden: torch.Tensor, topk_heads: int
    ) -> torch.Tensor:
        if self.head_selector is None:
            raise RuntimeError(
                "learned contextual head selection was not configured"
            )
        return _select_heads_from_head_scores(
            self.head_selector(hidden), topk_heads
        )

    def forward(
        self,
        x: torch.Tensor,
        selected_heads: torch.Tensor | None = None,
        *,
        kv_hidden: torch.Tensor | None = None,
        linear_features: torch.Tensor | None = None,
        topk_heads: int | None = None,
        select_all: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if x.shape[2] == 0:
            raise ValueError(
                "Longshot cache processor requires at least one selected time position"
            )
        kv_hidden = self.encode(x) if kv_hidden is None else kv_hidden
        hidden = kv_hidden
        linear_adapter = self.linear_state_adapter
        if linear_features is not None:
            if linear_adapter is None:
                raise ValueError("linear state adapter is not configured")
            hidden = torch.cat(
                [hidden, linear_adapter.encode(linear_features)],
                dim=1,
            )
        if topk_heads is None:
            topk_heads = self.config.topk_heads + (
                _topk_linear_heads(self.config)
                if linear_features is not None
                else 0
            )
        if select_all:
            selected_heads = torch.arange(
                hidden.shape[1], dtype=torch.long, device=hidden.device
            ).expand(hidden.shape[0], -1)
        elif selected_heads is None:
            selected_heads = self.select_heads(hidden, topk_heads)
        processed = self._iterate_layers(hidden, selected_heads, topk_heads)
        gate = self._gate_scale(processed.dtype)
        primary_delta = self.out_proj(processed[:, : x.shape[1]]) * gate
        updated = x[:, :, -1, -self.output_dim :] + primary_delta
        linear_delta = None
        if linear_features is not None:
            assert linear_adapter is not None
            linear_delta = linear_adapter.output_proj(
                processed[:, x.shape[1] :]
            )
            if self.config.gate_linear_state_delta:
                linear_delta = linear_delta * gate
        return updated, linear_delta, selected_heads

    def _iterate_layers(
        self,
        h: torch.Tensor,
        selected_heads: torch.Tensor,
        topk_heads: int,
    ) -> torch.Tensor:
        if self.flow is not None:
            return self.flow(
                h,
                lambda state, _time: self.out_norm(
                    self._run_flow_field(state, selected_heads)
                ),
            )
        iterations = (
            self.config.iterations
            if self.config.iteration_mode == "internal"
            else 1
        )
        learned_reselection = (
            self.config.internal_iteration_score == "learned_contextual"
        )
        for iteration in range(iterations):
            has_next_iteration = iteration + 1 < iterations
            return_attn_scores = has_next_iteration and not learned_reselection
            layer_scores = []
            for module in self.layers:
                layer = cast(ProcessorLayer, module)
                result = self._run_layer(
                    layer,
                    h,
                    selected_heads,
                    return_attn_scores=return_attn_scores,
                )
                if return_attn_scores:
                    if not isinstance(result, tuple):
                        raise TypeError(
                            "processor layer did not return iteration scores"
                        )
                    h, attn_scores = result
                    layer_scores.append(attn_scores)
                else:
                    if not isinstance(result, torch.Tensor):
                        raise TypeError(
                            "processor layer returned unexpected iteration scores"
                        )
                    h = result
            if has_next_iteration:
                if learned_reselection:
                    next_selected_heads = self.select_heads(h, topk_heads)
                else:
                    head_scores = torch.stack(layer_scores, dim=0).mean(dim=0)
                    next_selected_heads = _select_heads_from_head_scores(
                        head_scores, topk_heads
                    )
                self.record_selected_head_overlap(
                    iteration + 1, selected_heads, next_selected_heads
                )
                selected_heads = next_selected_heads
        return self.out_norm(h)

    def linear_state_query(
        self, layer_idx: int, following_cache_idx: int, context: torch.Tensor
    ) -> torch.Tensor:
        if self.linear_state_adapter is None:
            raise ValueError("linear state adapter is not configured")
        return self.linear_state_adapter.query(
            layer_idx, following_cache_idx, context
        )

    def linear_state_query_cache_key(
        self, layer_idx: int, following_cache_idx: int
    ) -> str:
        if self.linear_state_adapter is None:
            raise ValueError("linear state adapter is not configured")
        return self.linear_state_adapter.query_cache_key(
            layer_idx, following_cache_idx
        )

    def linear_state_update_dims(self) -> tuple[int, int]:
        if self.linear_state_adapter is None:
            raise ValueError("linear state adapter is not configured")
        return (
            self.linear_state_adapter.linear_key_dim,
            self.linear_state_adapter.linear_value_dim,
        )

    def _run_flow_field(
        self, state: torch.Tensor, selected_heads: torch.Tensor
    ) -> torch.Tensor:
        for module in self.layers:
            layer = cast(ProcessorLayer, module)
            result = self._run_layer(
                layer, state, selected_heads, return_attn_scores=False
            )
            if not isinstance(result, torch.Tensor):
                raise TypeError(
                    "flow field unexpectedly returned iteration scores"
                )
            state = result
        return state

    def _run_layer(
        self,
        layer: ProcessorLayer,
        h: torch.Tensor,
        selected_heads: torch.Tensor,
        *,
        return_attn_scores: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            _gradient_checkpointing_enabled(self.config.gradient_checkpointing)
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(
                layer,
                h,
                selected_heads,
                return_attn_scores,
                use_reentrant=False,
            )
        return layer(h, selected_heads, return_attn_scores=return_attn_scores)

    def pop_iteration_metrics(self) -> dict[str, float]:
        if cache_processor_recomputation_id() is not None:
            return {}
        metrics = {
            f"cache_processor/selected_head_overlap_after_iter_{iteration}": float(
                (total / self._selected_head_overlap_counts[iteration])
                .detach()
                .cpu()
            )
            for iteration in sorted(self._selected_head_overlap_totals)
            for total in (self._selected_head_overlap_totals[iteration],)
            if self._selected_head_overlap_counts[iteration] > 0
        }
        self._selected_head_overlap_totals.clear()
        self._selected_head_overlap_counts.clear()
        return metrics

    def record_selected_head_overlap(
        self,
        iteration: int,
        selected_heads: torch.Tensor,
        next_selected_heads: torch.Tensor,
    ) -> None:
        if cache_processor_recomputation_id() is not None:
            return
        assert selected_heads.shape[-1] > 0, (
            "selected_heads must include at least one cache head"
        )
        assert next_selected_heads.shape[-1] > 0, (
            "next_selected_heads must include at least one cache head"
        )
        maintained = (
            selected_heads[:, :, None] == next_selected_heads[:, None, :]
        ).any(dim=-1)
        overlap = maintained.float().mean().detach()
        if iteration in self._selected_head_overlap_totals:
            self._selected_head_overlap_totals[iteration] = (
                self._selected_head_overlap_totals[iteration] + overlap
            )
            self._selected_head_overlap_counts[iteration] += 1
        else:
            self._selected_head_overlap_totals[iteration] = overlap
            self._selected_head_overlap_counts[iteration] = 1

    def _gate_scale(self, dtype: torch.dtype) -> torch.Tensor:
        if self.config.gate_mode == "sigmoid":
            return torch.sigmoid(self.gate).to(dtype)
        if self.config.gate_mode == "direct":
            return self.gate.to(dtype)
        raise ValueError(
            f"unsupported cache processor gate_mode: {self.config.gate_mode}"
        )
