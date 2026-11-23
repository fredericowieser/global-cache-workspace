from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from gcw.config import CacheProcessorConfig, FlowProcessorConfig, ModelConfig
from gcw.config.model import linear_attention_following_cache_indices
from gcw.models.cache_processors.longshot_linear_state import (
    LinearStateAdapter,
    LinearStateSpec,
    LinearStateWriterMixin,
    _topk_linear_heads,
)
from gcw.models.cache_processors.base import (
    CacheProcessorBase,
    processor_gradient_checkpointing_enabled,
)
from gcw.models.cache_processors.ops import (
    scatter_cache_layers,
    scatter_cache_values,
    split_merged_cache as _split_merged,
    split_merged_cache_values as _split_merged_values,
    zero_projection,
    expand_cache_head_indices,
    gather_per_cache_head_scores,
    gather_per_cache_head_kv,
    rotary_for_indices,
    rotate_cache_keys,
    select_heads_from_per_cache_head_scores,
    select_per_cache_head_kv_norm_indices_from_scores,
)
from gcw.performance.backends.icarus import (
    IcarusProcessorBackend,
    icarus_processor_backend_metadata,
    resolve_icarus_processor_backend,
    run_icarus_post_selection_region,
)
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.flow import LatentFlow
from gcw.models.backbones.transformer import RMSNorm, apply_rotary, flash_sdpa
from gcw.models.cache_processors.windowed import (
    WindowedCache,
    GroupedWindowedCache,
)
from gcw.models.cache_processors.layout import (
    CacheShapeGroup,
    cache_shape_groups,
    cache_group_views,
)
from gcw.performance.longshot_scores import (
    native_cache_squared_l2,
    chunked_fp32_l2_norm,
    chunked_fp32_squared_l2,
)
from gcw.performance.triton.execution import triton_cache_movement_enabled

CacheLike = CacheState | StaticCacheState | WindowedCache | GroupedWindowedCache


class ProcessorAttention(nn.Module):
    def __init__(self, config: CacheProcessorConfig):
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                "processor hidden_size must be divisible by num_heads"
            )
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError(
                "processor num_heads must be divisible by num_kv_heads"
            )
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.include_self_head_kv = config.include_self_head_kv

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        theta: float,
        selected_tokens: torch.Tensor,
        head_groups: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        batch, tokens, _ = x.shape
        q = self.q_proj(x).view(batch, tokens, self.num_heads, self.head_dim)
        if self.include_self_head_kv:
            source, source_positions = x, positions
        else:
            source = x.gather(
                1, selected_tokens[..., None].expand(-1, -1, x.shape[-1])
            )
            source_positions = positions.gather(1, selected_tokens)
        k = self.k_proj(source).view(
            batch, -1, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(source).view(
            batch, -1, self.num_kv_heads, self.head_dim
        )
        cos, sin = rotary_for_indices(positions, self.head_dim, theta, q.dtype)
        q = apply_rotary(q, cos, sin)
        cos, sin = rotary_for_indices(
            source_positions, self.head_dim, theta, k.dtype
        )
        k = apply_rotary(k, cos, sin)
        if self.include_self_head_kv:
            y = self._attend_selected_and_self(
                q, k, v, selected_tokens, head_groups
            )
        else:
            y = self._attend_selected(q, k, v)
        return self.o_proj(y)

    def _attend_selected(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        batch, tokens = q.shape[:2]
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        y = flash_sdpa(
            q.to(v.dtype),
            k.to(v.dtype),
            v,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        return y.transpose(1, 2).reshape(batch, tokens, -1)

    def _attend_selected_and_self(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selected_tokens: torch.Tensor,
        head_groups: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        batch, tokens = q.shape[:2]
        groups = self.num_heads // self.num_kv_heads
        selected_count = selected_tokens.shape[1]
        index = selected_tokens[..., None, None].expand(
            -1, -1, self.num_kv_heads, self.head_dim
        )
        k_sel, v_sel = k.gather(1, index), v.gather(1, index)
        selected = torch.zeros(batch, tokens, dtype=torch.bool, device=q.device)
        selected.scatter_(1, selected_tokens, True)
        outputs = []
        offset = 0
        for heads, length in head_groups:
            end = offset + heads * length
            query = (
                q[:, offset:end]
                .to(v.dtype)
                .reshape(
                    batch,
                    heads,
                    length,
                    self.num_kv_heads,
                    groups,
                    self.head_dim,
                )
            )
            own_keys = (
                k[:, offset:end]
                .to(v.dtype)
                .reshape(batch, heads, length, self.num_kv_heads, self.head_dim)
            )
            own_values = v[:, offset:end].reshape(
                batch, heads, length, self.num_kv_heads, self.head_dim
            )
            selected_logits = (
                torch.einsum("bhtkgd,bskd->bhtkgs", query, k_sel.to(v.dtype))
                * self.head_dim**-0.5
            )
            self_logits = (
                torch.einsum("bhtkgd,bhskd->bhtkgs", query, own_keys)
                * self.head_dim**-0.5
            )
            duplicate = selected[:, offset:end].reshape(batch, heads, length)
            self_logits = self_logits.masked_fill(
                duplicate[:, :, None, None, None, :], float("-inf")
            )
            weights = torch.softmax(
                torch.cat([selected_logits, self_logits], -1), -1
            )
            if self.training and self.dropout > 0.0:
                weights = F.dropout(weights, p=self.dropout, training=True)
            y = torch.einsum(
                "bhtkgs,bskd->bhtkgd", weights[..., :selected_count], v_sel
            )
            y = y + torch.einsum(
                "bhtkgs,bhskd->bhtkgd",
                weights[..., selected_count:],
                own_values,
            )
            outputs.append(y.reshape(batch, heads * length, -1))
            offset = end
        return torch.cat(outputs, 1)


class ProcessorLayer(nn.Module):
    def __init__(self, config: CacheProcessorConfig):
        super().__init__()
        self.attn = ProcessorAttention(config)
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.input_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.theta = config.rope_theta

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        selected_tokens: torch.Tensor,
        head_groups: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        x = x + self.attn(
            self.input_norm(x),
            positions,
            self.theta,
            selected_tokens,
            head_groups,
        )
        h = self.post_norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class PerCacheHeadLinear(nn.Module):
    def __init__(
        self,
        num_cache_heads: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_cache_heads, in_features, out_features)
        )
        self.bias = (
            nn.Parameter(torch.empty(num_cache_heads, out_features))
            if bias
            else None
        )

    def forward(
        self, x: torch.Tensor, *, head_index: int | None = None
    ) -> torch.Tensor:
        if head_index is not None:
            return F.linear(
                x,
                self.weight[head_index].T,
                None if self.bias is None else self.bias[head_index],
            )
        y = torch.einsum("bcsi,cio->bcso", x, self.weight)
        if self.bias is not None:
            y = y + self.bias[None, :, None, :]
        return y


class ScalarGateMixin:
    def _gate_scale(self, dtype: torch.dtype) -> torch.Tensor:
        if self.config.gate_mode == "sigmoid":
            scale = torch.sigmoid(self.gate).to(dtype)
        elif self.config.gate_mode == "direct":
            scale = self.gate.to(dtype)
        else:
            raise ValueError(
                f"unsupported cache processor gate_mode: {self.config.gate_mode}"
            )
        self._gate_value = scale.detach()
        return scale


class CacheHeadProjection(ScalarGateMixin, nn.Module):
    def __init__(self, group: CacheShapeGroup, config: CacheProcessorConfig):
        super().__init__()
        self.config = config
        width = 2 * group.head_dim
        self.output_dim = (
            group.head_dim if config.update_stream == "v" else width
        )
        self.in_norm = RMSNorm(width, config.norm_eps)
        if config.per_cache_head_projections:
            self.in_proj = PerCacheHeadLinear(
                len(group.head_indices), width, config.hidden_size
            )
            self.out_proj = PerCacheHeadLinear(
                len(group.head_indices), config.hidden_size, self.output_dim
            )
        else:
            self.in_proj = nn.Linear(width, config.hidden_size)
            self.out_proj = nn.Linear(config.hidden_size, self.output_dim)
        self.gate = nn.Parameter(torch.full((1,), config.gate_init))
        self._gate_value = None

    def apply_update(self, original, hidden):
        delta = self.out_proj(hidden)
        return original[..., -self.output_dim :] + delta * self._gate_scale(
            delta.dtype
        )


class CacheProcessor(ScalarGateMixin, nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_cache_heads: int,
        config: CacheProcessorConfig,
        *,
        output_dim: int | None = None,
        flow_config: FlowProcessorConfig | None = None,
        shape_groups: tuple[CacheShapeGroup, ...] = (),
        linear_state_model_config: ModelConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.shape_groups = shape_groups
        self.linear_state_specs: list[LinearStateSpec] = []
        if shape_groups:
            self.projections = nn.ModuleDict(
                {g.name: CacheHeadProjection(g, config) for g in shape_groups}
            )
            order = [h for g in shape_groups for h in g.head_indices]
            self.register_buffer(
                "canonical_order",
                torch.tensor(
                    [order.index(h) for h in range(num_cache_heads)],
                    device="cpu",
                ),
                persistent=False,
            )
            for g in shape_groups:
                self.register_buffer(
                    g.name + "_heads",
                    torch.tensor(g.head_indices, device="cpu"),
                    persistent=False,
                )
            self.layers = nn.ModuleList(
                ProcessorLayer(config) for _ in range(config.num_layers)
            )
            self.out_norm = RMSNorm(config.hidden_size, config.norm_eps)
            self.flow = (
                LatentFlow(config.hidden_size, flow_config)
                if flow_config is not None
                else None
            )
            self.init_weights()
            return
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.in_norm = RMSNorm(input_dim, config.norm_eps)
        self.in_proj: nn.Module
        if config.per_cache_head_projections:
            self.in_proj = PerCacheHeadLinear(
                num_cache_heads, input_dim, config.hidden_size
            )
        else:
            self.in_proj = nn.Linear(input_dim, config.hidden_size)
        self.layers = nn.ModuleList(
            ProcessorLayer(config) for _ in range(config.num_layers)
        )
        self.out_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.out_proj: nn.Module
        if config.per_cache_head_projections:
            self.out_proj = PerCacheHeadLinear(
                num_cache_heads, config.hidden_size, self.output_dim
            )
        else:
            self.out_proj = nn.Linear(config.hidden_size, self.output_dim)
        self.gate = nn.Parameter(torch.full((1,), config.gate_init))
        self._gate_value: torch.Tensor | None = None
        self.flow = (
            LatentFlow(config.hidden_size, flow_config)
            if flow_config is not None
            else None
        )
        self.init_weights()

        if linear_state_model_config is not None:
            self.linear_state_specs = [
                LinearStateSpec(
                    layer,
                    following,
                    int(
                        linear_state_model_config.linear_num_value_heads
                        or linear_state_model_config.num_heads
                    ),
                )
                for layer, following in linear_attention_following_cache_indices(
                    linear_state_model_config, require_layer_types=True
                )
            ]
            self.linear_state_adapter = LinearStateAdapter(
                linear_state_model_config, self.config, self.linear_state_specs
            )

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
        projections = (
            self.projections.values() if self.shape_groups else (self,)
        )
        for projection in projections:
            nn.init.constant_(projection.gate, self.config.gate_init)
            if self.config.out_proj_init == "zero":
                _zero_projection(projection.out_proj)

    def compute_update(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        selected_heads: torch.Tensor,
        *,
        gradient_checkpointing: bool | None = None,
    ) -> torch.Tensor:
        initial = self.in_proj(self.in_norm(x))
        hidden = self._process_hidden(
            initial,
            positions,
            selected_heads,
            gradient_checkpointing=gradient_checkpointing,
        )
        return self.out_proj(hidden)

    def _process_hidden(
        self,
        initial: torch.Tensor,
        positions: torch.Tensor,
        selected_heads: torch.Tensor,
        *,
        gradient_checkpointing: bool | None = None,
        head_groups: tuple[tuple[int, int], ...] | None = None,
    ) -> torch.Tensor:
        def field(state, _time=None):
            return self.out_norm(
                self._run_layers(
                    state,
                    positions,
                    selected_heads,
                    gradient_checkpointing=gradient_checkpointing,
                    head_groups=head_groups,
                )
            )

        return (
            self.flow(initial, field)
            if self.flow is not None
            else field(initial)
        )

    def forward_grouped(self, histories, positions, selected_heads):
        initial = torch.cat(
            [
                self.projections[g.name].in_proj(
                    self.projections[g.name].in_norm(x)
                )
                for g, x in zip(self.shape_groups, histories, strict=True)
            ],
            dim=1,
        ).index_select(1, self.canonical_order)
        positions = torch.cat(positions, dim=1).index_select(
            1, self.canonical_order
        )

        hidden = self._process_hidden(initial, positions, selected_heads)
        return tuple(
            self.projections[g.name].apply_update(
                x, hidden.index_select(1, getattr(self, g.name + "_heads"))
            )
            for g, x in zip(self.shape_groups, histories, strict=True)
        )

    def encode_linear_kv(self, merged, indices, read_indices, query_kv=None):
        kv_hidden = self.in_proj(self.in_norm(merged))
        if read_indices.shape[-1] == 1:
            return kv_hidden, kv_hidden[:, :, -1:]
        if query_kv is None:
            return kv_hidden, kv_hidden
        offsets = torch.searchsorted(
            indices.contiguous(), read_indices.contiguous()
        ).clamp_max(indices.shape[-1] - 1)
        missing = indices.gather(2, offsets) != read_indices
        query_hidden = kv_hidden.gather(
            2, offsets[..., None].expand(-1, -1, -1, kv_hidden.shape[-1])
        )
        head, batch, read = missing.transpose(0, 1).nonzero(as_tuple=True)
        packed = self.in_norm(query_kv[batch, head, read])
        if isinstance(self.in_proj, PerCacheHeadLinear):
            counts = torch.bincount(head, minlength=merged.shape[1]).tolist()
            encoded = torch.cat(
                [
                    self.in_proj(rows, head_index=index)
                    for index, rows in enumerate(packed.split(counts))
                ],
                dim=0,
            )
        else:
            encoded = self.in_proj(packed)
        query_hidden = query_hidden.index_put((batch, head, read), encoded)
        return kv_hidden, query_hidden

    def read_linear_states(
        self,
        encoded: torch.Tensor,
        recurrent_states: list[torch.Tensor],
        read_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adapter = self.linear_state_adapter
        batch, layers, reads = read_positions.shape
        contexts = encoded.reshape(
            batch, layers, encoded.shape[1] // layers, reads, -1
        ).mean(2)
        features = []
        queries = {}
        for spec, state in zip(
            self.linear_state_specs, recurrent_states, strict=True
        ):
            key = (
                adapter.query_cache_key(
                    spec.layer_idx, spec.following_cache_idx
                ),
                spec.following_cache_idx,
            )
            if key not in queries:
                context = contexts[:, spec.following_cache_idx].reshape(
                    batch * reads, -1
                )
                queries[key] = (
                    adapter.query(
                        spec.layer_idx, spec.following_cache_idx, context
                    )
                    .reshape(batch, reads, spec.head_count, -1)
                    .transpose(1, 2)
                )
            query = queries[key].to(state.dtype)
            readout = torch.einsum("bhrk,bhkv->bhrv", query, state)
            features.append(torch.cat([query, readout], -1))
        features = torch.cat(features, 1)
        scores = chunked_fp32_squared_l2(
            features[..., adapter.linear_key_dim :].contiguous()
        ).sum(-1)
        return features, scores

    def forward_linear(
        self,
        merged: torch.Tensor,
        kv_hidden: torch.Tensor,
        indices: torch.Tensor,
        selected_heads: torch.Tensor,
        features: torch.Tensor,
        linear_positions: torch.Tensor,
        linear_heads: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adapter = self.linear_state_adapter
        batch, cache_heads = merged.shape[:2]
        reads = features.shape[2]
        linear_hidden = adapter.encode(features)
        kv_length = merged.shape[2]
        kv_tokens = cache_heads * kv_length
        initial = torch.cat(
            [kv_hidden.flatten(1, 2), linear_hidden.flatten(1, 2)], 1
        )
        positions = torch.cat(
            [indices.flatten(1), linear_positions.flatten(1)], 1
        )
        head_groups = (
            (cache_heads, kv_length),
            (features.shape[1], reads),
        )
        selected_tokens = torch.cat(
            [
                (
                    selected_heads[..., None] * kv_length
                    + torch.arange(kv_length, device=merged.device)
                ).flatten(1),
                (
                    kv_tokens
                    + (linear_heads[..., None] - cache_heads) * reads
                    + torch.arange(reads, device=merged.device)
                ).flatten(1),
            ],
            1,
        )

        hidden = self._process_hidden(
            initial, positions, selected_tokens, head_groups=head_groups
        )
        delta = adapter.project_update(
            hidden[:, kv_tokens:].reshape(batch, features.shape[1], reads, -1)
        )
        if adapter.gate_dim == 0 and self.config.gate_linear_state_delta:
            delta = delta * self._gate_scale(delta.dtype)
        kv_delta = self.out_proj(
            hidden[:, :kv_tokens].reshape(batch, cache_heads, kv_length, -1)
        )
        return merged + kv_delta * self._gate_scale(kv_delta.dtype), delta

    def _run_layers(
        self,
        state: torch.Tensor,
        positions: torch.Tensor,
        selected_heads: torch.Tensor,
        *,
        gradient_checkpointing: bool | None = None,
        head_groups: tuple[tuple[int, int], ...] | None = None,
    ) -> torch.Tensor:
        if gradient_checkpointing is None:
            gradient_checkpointing = processor_gradient_checkpointing_enabled(
                self.config.gradient_checkpointing, variant="icarus"
            )
        original_shape = state.shape
        if head_groups is None:
            _, heads, length, _ = state.shape
            state = state.flatten(1, 2)
            positions = positions.flatten(1)
            head_groups = ((heads, length),)
            selected_tokens = (
                selected_heads[..., None] * length
                + torch.arange(length, device=state.device)
            ).flatten(1)
        else:
            selected_tokens = selected_heads
        for layer in self.layers:
            if (
                gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                state = checkpoint(
                    layer,
                    state,
                    positions,
                    selected_tokens,
                    head_groups,
                    use_reentrant=False,
                )
            else:
                state = layer(state, positions, selected_tokens, head_groups)
        return state.reshape(original_shape)

    def scaled_update(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        selected_heads: torch.Tensor,
        *,
        gradient_checkpointing: bool | None = None,
    ) -> torch.Tensor:
        delta = self.compute_update(
            x,
            positions,
            selected_heads,
            gradient_checkpointing=gradient_checkpointing,
        )
        return delta * self._gate_scale(delta.dtype)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        selected_heads: torch.Tensor,
        *,
        gradient_checkpointing: bool | None = None,
    ) -> torch.Tensor:
        residual = x[..., -self.output_dim :]
        return residual + self.scaled_update(
            x,
            positions,
            selected_heads,
            gradient_checkpointing=gradient_checkpointing,
        )


class GlobalCacheProcessor(LinearStateWriterMixin, CacheProcessorBase):
    def __init__(
        self, model_config: ModelConfig, *, linear_state_dynamics=None
    ):
        super().__init__()
        self.shape_groups = cache_shape_groups(model_config.stored_cache_shapes)
        self.mixed_shapes = len(self.shape_groups) > 1
        if model_config.cache_processor is None:
            raise ValueError("Icarus requires cache_processor config")
        self.config = model_config.cache_processor
        self._model_config = model_config
        if self.config.processor_mode != "standard":
            raise ValueError(
                "Icarus supports only cache_processor.processor_mode='standard'"
            )
        if self.config.update_stream not in {"kv", "v"}:
            raise ValueError(
                "Icarus supports only cache_processor.update_stream='kv' or 'v'"
            )
        if self.config.score not in {"attn_l2", "kv_norm"}:
            raise ValueError(
                "Icarus supports only cache_processor.score='attn_l2' or 'kv_norm'"
            )
        self.num_layers = model_config.cache_num_layers
        self.num_kv_heads, self.head_dim = model_config.stored_cache_shapes[0]
        self.rope_theta = model_config.rope_theta
        self.rotary_style = model_config.rotary_style
        self.rotary_dim = int(
            self.head_dim * model_config.partial_rotary_factor
        )
        self.num_cache_heads = model_config.cache_num_heads
        output_dim = (
            self.head_dim
            if self.config.update_stream == "v"
            else 2 * self.head_dim
        )
        self.block = CacheProcessor(
            2 * self.head_dim,
            self.num_cache_heads,
            self.config,
            output_dim=output_dim,
            flow_config=model_config.flow_processor,
            shape_groups=self.shape_groups if self.mixed_shapes else (),
            linear_state_model_config=model_config
            if self.config.linear_state_cache
            else None,
        )
        self.linear_state_dynamics = linear_state_dynamics or {}
        self.linear_state_specs = self.block.linear_state_specs
        if self.config.linear_state_update_mode == "gated_delta" and any(
            spec.layer_idx not in self.linear_state_dynamics
            for spec in self.linear_state_specs
        ):
            raise ValueError(
                "gated_delta requires matching backbone linear-state dynamics"
            )
        self.execution_backend: IcarusProcessorBackend = (
            resolve_icarus_processor_backend(model_config)
        )

    def execution_backend_metadata(self) -> dict[str, object]:
        return icarus_processor_backend_metadata(
            self._model_config, self.execution_backend
        )

    def forward(
        self, cache: CacheLike, *, select_all: bool = False
    ) -> CacheLike:
        if self.mixed_shapes or self._model_config.family == "gemma4":
            return self._forward_native(cache, select_all=select_all)
        position_scores = None
        if self.config.score == "kv_norm" and not select_all:
            all_indices = expand_cache_head_indices(
                cache.all_indices(), self.num_cache_heads
            )
            merged = gather_per_cache_head_kv(
                cache,
                all_indices,
                unrope_keys=self.config.unrope_keys,
                rope_theta=self.rope_theta,
                rotary_style=self.rotary_style,
                rotary_dim=self.rotary_dim,
                backend=self._cache_movement_backend(),
            )
            position_scores = chunked_fp32_squared_l2(merged).detach()
            indices, selected_heads = self._select_indices_and_heads(
                cache, merged, select_all, scores=position_scores
            )
            if indices.shape[-1] == 0:
                return cache
            index = indices[..., None].expand(-1, -1, -1, merged.shape[-1])
            merged = torch.gather(merged, dim=2, index=index)
            if (
                not self.config.linear_state_cache
                or self.config.icarus_linear_read_mode == "single"
                or indices.shape[-1] == cache.seq_len
            ):
                position_scores = None
        else:
            indices, selected_heads = self._select_indices_and_heads(
                cache, None, select_all
            )
            if indices.shape[-1] == 0:
                return cache
            merged = gather_per_cache_head_kv(
                cache,
                indices,
                unrope_keys=self.config.unrope_keys,
                rope_theta=self.rope_theta,
                rotary_style=self.rotary_style,
                rotary_dim=self.rotary_dim,
                backend=self._cache_movement_backend(),
            )
        if indices.shape[-1] == 0:
            return cache
        if selected_heads is None:
            selected_heads = _select_heads_from_merged(
                merged, self.config.topk_heads
            )
        if self.config.linear_state_cache:
            updated = self._process_linear_states(
                cache,
                merged,
                indices,
                selected_heads,
                select_all,
                position_scores=position_scores,
            )
        else:
            updated = run_icarus_post_selection_region(
                self.execution_backend,
                self.block,
                merged,
                indices,
                selected_heads,
            )
        if self.config.update_stream == "v":
            return scatter_cache_values(
                cache,
                indices,
                _split_merged_values(
                    updated, self.num_layers, self.num_kv_heads, self.head_dim
                ),
            )
        keys, values = updated.split(self.head_dim, dim=-1)
        if self.config.unrope_keys:
            keys = rotate_cache_keys(
                keys,
                indices,
                theta=self.rope_theta,
                rotary_style=self.rotary_style,
                rotary_dim=self.rotary_dim,
            )
        return scatter_cache_layers(
            cache,
            indices,
            _split_merged_values(
                keys, self.num_layers, self.num_kv_heads, self.head_dim
            ),
            _split_merged_values(
                values, self.num_layers, self.num_kv_heads, self.head_dim
            ),
        )

    def _linear_read_positions(self, cache, retained_count, *, scores=None):
        if self.config.icarus_linear_read_mode == "single":
            return cache.all_indices()[:, None, -1:].expand(
                -1, self.num_layers, -1
            )
        if retained_count == cache.seq_len:
            return cache.all_indices()[:, None, :].expand(
                -1, self.num_layers, -1
            )
        if scores is None:
            scores = native_cache_squared_l2(cache)
        scores = scores.reshape(
            cache.batch_size,
            self.num_layers,
            self.num_kv_heads,
            cache.seq_len,
        ).mean(2)
        count = retained_count
        earlier = torch.argsort(
            scores[..., :-1], dim=-1, descending=True, stable=True
        )[..., : count - 1]
        latest = torch.full_like(
            scores[..., :1], cache.seq_len - 1, dtype=torch.long
        )
        return torch.cat([earlier, latest], -1).sort(-1).values

    def _process_linear_states(
        self,
        cache,
        merged,
        indices,
        selected_heads,
        select_all,
        *,
        position_scores=None,
    ):
        read_positions = self._linear_read_positions(
            cache, merged.shape[2], scores=position_scores
        )
        read_indices = read_positions.repeat_interleave(
            self.num_kv_heads, dim=1
        )
        query_kv = None
        if read_indices.shape[-1] > 1 and merged.shape[2] < cache.seq_len:
            query_kv = gather_per_cache_head_kv(
                cache,
                read_indices,
                unrope_keys=self.config.unrope_keys,
                rope_theta=self.rope_theta,
                rotary_style=self.rotary_style,
                rotary_dim=self.rotary_dim,
                backend=self._cache_movement_backend(),
            )
        kv_hidden, query_hidden = self.block.encode_linear_kv(
            merged, indices, read_indices, query_kv
        )
        recurrent_states, positions = [], []
        for spec in self.linear_state_specs:
            state = (cache.linear_states or {}).get(spec.layer_idx)
            if state is None or state[0] is None:
                raise RuntimeError(
                    f"linear state layer {spec.layer_idx} has no recurrent state"
                )
            recurrent_states.append(state[0])
            positions.append(
                read_positions[:, spec.following_cache_idx, None].expand(
                    -1, spec.head_count, -1
                )
            )
        if not recurrent_states:
            raise RuntimeError(
                "Icarus linear states require at least one recurrent layer"
            )
        features, scores = self.block.read_linear_states(
            query_hidden, recurrent_states, read_positions
        )
        linear_positions = torch.cat(positions, 1)
        budget = (
            features.shape[1]
            if select_all
            else min(_topk_linear_heads(self.config), features.shape[1])
        )
        linear_heads = scores.topk(budget, dim=1).indices + self.num_cache_heads
        updated, delta = self.block.forward_linear(
            merged,
            kv_hidden,
            indices,
            selected_heads,
            features,
            linear_positions,
            linear_heads,
        )
        self._scatter_linear_state_updates(
            cache, delta, self.linear_state_specs
        )
        return updated

    @property
    def recurrent_cache_format(self) -> str:
        return "list" if self._model_config.family == "gemma4" else "windowed"

    def _forward_native(
        self, cache: CacheLike, *, select_all: bool
    ) -> CacheLike:
        if cache.seq_len == 0:
            return cache
        groups = cache_group_views(cache, self.shape_groups)
        histories, positions, magnitudes = [], [], []
        for spec, group in zip(self.shape_groups, groups, strict=True):
            if self.config.score == "kv_norm" and not select_all:
                scores = native_cache_squared_l2(group)
                indices = select_per_cache_head_kv_norm_indices_from_scores(
                    scores, self.config.process_every, self.config.topk
                )
                magnitude = (
                    gather_per_cache_head_scores(scores, indices)
                    .float()
                    .sum(-1)
                )
                divisor = 2 * spec.head_dim
            else:
                indices = expand_cache_head_indices(
                    cache.all_indices()
                    if select_all
                    else cache.select_indices(
                        self.config.process_every, self.config.topk
                    ),
                    len(spec.head_indices),
                )
                magnitude = None
                divisor = (2 * spec.head_dim) ** 0.5
            merged = gather_per_cache_head_kv(group, indices, unrope_keys=False)
            if magnitude is None:
                magnitude = chunked_fp32_l2_norm(merged)
            if self.config.head_score_normalization == "rms":
                magnitude = magnitude / divisor
            histories.append(merged)
            positions.append(indices)
            magnitudes.append(magnitude)
        scores = torch.cat(magnitudes, dim=1)
        if self.mixed_shapes:
            scores = scores.index_select(1, self.block.canonical_order)
        selected = (
            torch.arange(self.num_cache_heads, device=scores.device).expand(
                cache.batch_size, -1
            )
            if select_all
            else scores.topk(
                min(self.config.topk_heads, self.num_cache_heads), dim=-1
            )
            .indices.sort(dim=-1)
            .values
        )
        updates = (
            self.block.forward_grouped(histories, positions, selected)
            if self.mixed_shapes
            else (
                run_icarus_post_selection_region(
                    self.execution_backend,
                    self.block,
                    histories[0],
                    positions[0],
                    selected,
                ),
            )
        )
        for spec, group, indices, updated in zip(
            self.shape_groups, groups, positions, updates, strict=True
        ):
            if self.config.update_stream == "v":
                scatter_cache_values(
                    group,
                    indices,
                    _split_merged_values(
                        updated, len(spec.layers), spec.num_heads, spec.head_dim
                    ),
                )
            else:
                keys, values = _split_merged(
                    updated, len(spec.layers), spec.num_heads, spec.head_dim
                )
                scatter_cache_layers(group, indices, keys, values)
            if isinstance(cache, CacheState):
                for local, layer in enumerate(spec.layers):
                    cache.keys[layer], cache.values[layer] = group.layer(local)
        return cache

    def _cache_movement_backend(self) -> str:
        return "triton" if triton_cache_movement_enabled(self) else "eager"

    def _select_indices_and_heads(
        self,
        cache: CacheLike,
        merged: torch.Tensor | None,
        select_all: bool,
        *,
        scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if select_all:
            return (
                expand_cache_head_indices(
                    cache.all_indices(), self.num_cache_heads
                ),
                cache.all_heads(),
            )
        if self.config.score == "kv_norm":
            if merged is None:
                raise ValueError(
                    "merged cache is required for kv_norm selection"
                )
            if scores is None:
                scores = chunked_fp32_squared_l2(merged)
            indices = select_per_cache_head_kv_norm_indices_from_scores(
                scores, self.config.process_every, self.config.topk
            )
            selected_scores = gather_per_cache_head_scores(scores, indices)
            if self.config.head_score_normalization == "rms":
                selected_scores = selected_scores / (2 * self.head_dim)
            selected_heads = select_heads_from_per_cache_head_scores(
                selected_scores, self.config.topk_heads
            )
            return indices, selected_heads
        indices = expand_cache_head_indices(
            cache.select_indices(self.config.process_every, self.config.topk),
            self.num_cache_heads,
        )
        return indices, None


def _zero_projection(module: nn.Module) -> None:
    zero_projection(module, per_head_type=PerCacheHeadLinear)


def _select_heads_from_merged(x: torch.Tensor, topk: int) -> torch.Tensor:
    assert topk > 0, "topk must select at least one cache head"
    scores = chunked_fp32_l2_norm(x)
    k = min(topk, scores.shape[-1])
    _, indices = torch.topk(scores, k=k, dim=-1)
    return indices.sort(dim=-1).values
