from __future__ import annotations

from dataclasses import dataclass
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import (
    WindowedCache,
    GroupedWindowedCache,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models.backbones.transformer import RMSNorm
from gcw.models.cache_processors.update_metrics import UPDATE_OBSERVER
from gcw.models.cache_processors.longshot_layers import (
    _activate_update_gate,
    _initialize_contextual_output_projection,
    _zero_projection,
)


CacheLike = CacheState | StaticCacheState | WindowedCache | GroupedWindowedCache


@dataclass(frozen=True)
class LinearStateSpec:
    layer_idx: int
    following_cache_idx: int
    head_count: int


@dataclass
class LinearStateInputs:
    features: torch.Tensor
    specs: list[LinearStateSpec]


def _topk_linear_heads(config: CacheProcessorConfig) -> int:
    return (
        config.topk_linear_heads
        if config.topk_linear_heads is not None
        else config.topk_heads
    )


class LinearStateAdapter(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        processor_config: CacheProcessorConfig,
        specs: list[LinearStateSpec],
    ):
        super().__init__()
        self.query_mode = processor_config.linear_state_query_mode
        self.num_kv_heads = model_config.num_kv_heads
        if model_config.head_dim is None:
            raise ValueError(
                "linear state adapter requires a resolved model head_dim"
            )
        self.full_head_dim = model_config.head_dim
        self.linear_key_dim = int(
            model_config.linear_key_head_dim or self.full_head_dim
        )
        self.linear_value_dim = int(
            model_config.linear_value_head_dim or self.linear_key_dim
        )
        self.linear_value_heads = int(
            model_config.linear_num_value_heads or model_config.num_heads
        )
        self.context_dim = processor_config.hidden_size
        self.feature_dim = self.linear_key_dim + self.linear_value_dim
        self.output_dim = self.feature_dim + (
            2
            if processor_config.linear_state_update_mode == "gated_delta"
            else 0
        )
        self.gate_dim = (
            processor_config.update_gate_dim(self.output_dim)
            if processor_config.gate_linear_state_delta
            else 0
        )
        self.gate_mode = processor_config.gate_mode
        self._gate_value: torch.Tensor | None = None
        self.query_projs = nn.ModuleDict()
        for key in self._query_projection_keys(specs):
            out_dim = (
                self.linear_key_dim
                if self.query_mode != "per_linear_state"
                else self.linear_value_heads * self.linear_key_dim
            )
            self.query_projs[key] = nn.Linear(self.context_dim, out_dim)
        self.query_context_norm = RMSNorm(
            self.context_dim, processor_config.norm_eps
        )
        self.input_norm = RMSNorm(self.feature_dim, processor_config.norm_eps)
        self.input_proj = nn.Linear(
            self.feature_dim, processor_config.hidden_size
        )
        self.output_proj = nn.Linear(
            processor_config.hidden_size, self.output_dim + self.gate_dim
        )
        self.init_weights(processor_config)

    def init_weights(
        self, config: CacheProcessorConfig, std: float = 0.02
    ) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        if self.gate_dim:
            _initialize_contextual_output_projection(
                self.output_proj, self.output_dim, config, std
            )
        elif config.out_proj_init == "zero":
            _zero_projection(self.output_proj)
        if config.live_rank1_linear_writer:
            # Live keys give zero-valued factors a first gradient without
            # changing the initial update or subsequent random draws.
            weight = self.output_proj.weight
            devices = [weight.device] if weight.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                nn.init.normal_(weight[: self.linear_key_dim], std=std)

    def query(
        self, layer_idx: int, following_cache_idx: int, context: torch.Tensor
    ) -> torch.Tensor:
        projected = self.query_projs[
            self.query_cache_key(layer_idx, following_cache_idx)
        ](self.query_context_norm(context))
        if self.query_mode != "per_linear_state":
            query = projected[:, None, :].expand(
                -1, self.linear_value_heads, -1
            )
        else:
            query = projected.view(
                projected.shape[0], self.linear_value_heads, self.linear_key_dim
            )
        return F.normalize(query.float(), p=2, dim=-1).to(projected.dtype)

    def query_cache_key(self, layer_idx: int, following_cache_idx: int) -> str:
        if self.query_mode == "per_linear_state":
            return f"layer:{layer_idx}"
        if self.query_mode == "shared":
            return f"cache:{following_cache_idx}"
        if self.query_mode == "global":
            return "global"
        raise ValueError(
            f"unsupported linear_state_query_mode: {self.query_mode}"
        )

    def _query_projection_keys(self, specs: list[LinearStateSpec]) -> list[str]:
        return sorted(
            {
                self.query_cache_key(spec.layer_idx, spec.following_cache_idx)
                for spec in specs
            }
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.input_proj(
            self.input_norm(features.to(self.input_proj.weight.dtype))
        )

    def project_update(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self.output_proj(hidden)
        observer = UPDATE_OBSERVER.get() if not self.training else None
        if observer is not None:
            observer.linear(projected[..., : self.output_dim])
        if self.gate_dim == 0:
            return projected
        delta, gate_logits = projected.split(
            [self.output_dim, self.gate_dim], dim=-1
        )
        gate = _activate_update_gate(gate_logits, self.gate_mode, delta.dtype)
        self._gate_value = gate.detach()
        return delta * gate


class LinearStateWriterMixin:
    def _scatter_linear_state_updates(
        self,
        cache: CacheLike,
        linear_delta: torch.Tensor,
        specs: list[LinearStateSpec],
    ) -> CacheLike:
        if self.block.linear_state_adapter is None or not specs:
            return cache
        if linear_delta.ndim == 3:
            linear_delta = linear_delta.unsqueeze(2)
        states = (
            {} if cache.linear_states is None else dict(cache.linear_states)
        )
        offset = 0
        key_dim, value_dim = (
            self.block.linear_state_adapter.linear_key_dim,
            self.block.linear_state_adapter.linear_value_dim,
        )
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
                "bhrk,bhrv->bhkv", delta_key.float(), delta_value.float()
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
        key_dim, value_dim = (
            self.block.linear_state_adapter.linear_key_dim,
            self.block.linear_state_adapter.linear_value_dim,
        )
        if update.ndim == 3:
            update = update.unsqueeze(2)
        synthetic_key, synthetic_value, beta_logit, decay_input = torch.split(
            update, [key_dim, value_dim, 1, 1], dim=-1
        )
        heads = synthetic_key.shape[1]
        if heads != dynamics.num_heads:
            raise ValueError(
                f"linear state head count {heads} does not match layer {layer_idx} "
                f"value heads {dynamics.num_heads}"
            )
        query = synthetic_key.transpose(1, 2).contiguous()
        key = query
        value = synthetic_value.transpose(1, 2).contiguous()
        beta = torch.sigmoid(beta_logit.squeeze(-1).transpose(1, 2))
        g = dynamics(decay_input.squeeze(-1).transpose(1, 2))
        _, final_state = qwen35_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
        )
        return final_state.to(recurrent_state.dtype)
