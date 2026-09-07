from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from gcw.config import (
    CacheProcessorConfig,
    DiffusionProcessorConfig,
    FlowProcessorConfig,
    ModelConfig,
)
from gcw.performance.backends.cache_processors import (
    BTProcessorBackend,
    CacheProcessorBackend,
    bt_processor_backend_metadata,
    cache_processor_backend_metadata,
    resolve_bt_processor_backend,
    resolve_cache_processor_backend,
    run_bgt_processor_region,
    run_prepared_bt_processor_region,
)
from gcw.performance.prepared_bt import (
    PreparedBatchedCacheProcessorParameters,
    prepare_batched_cache_processor_parameters,
    run_batched_cache_processors,
)
from gcw.models.cache_processors.base import CacheProcessorBase
from gcw.performance.triton import (
    gather_shared_kv as triton_gather_shared_kv,
    scatter_shared_kv as triton_scatter_shared_kv,
)
from gcw.performance.triton.execution import triton_cache_movement_enabled
from gcw.models.cache_processors.state import CacheState
from gcw.models.cache_processors.ops import (
    gather_cache_layers,
    gather_cache_values,
    merge_global_kv,
    merge_global_values,
    merge_layer_kv,
    merge_layer_values,
    rotary_for_indices,
    scatter_cache_layers,
    scatter_cache_values,
    select_kv_norm_indices,
    split_global_kv,
    split_global_values,
    split_layer_kv,
    split_layer_values,
    workspace_writer_kv,
    workspace_writer_values,
)
from gcw.models.cache_processors.diffusion import TaskDirectedDiffusion
from gcw.models.cache_processors.flow import LatentFlow
from gcw.models.backbones.transformer import RMSNorm, apply_rotary, flash_sdpa
from gcw.models.cache_processors.windowed import WindowedCache

CacheLike = CacheState | WindowedCache


class ProcessorAttention(nn.Module):
    def __init__(self, config: CacheProcessorConfig):
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                "processor hidden_size must be divisible by num_heads"
            )
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.num_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, theta: float
    ) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim)
        v = (
            self.v_proj(x)
            .view(batch, seq, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        cos_sin = rotary_for_indices(positions, self.head_dim, theta, q.dtype)
        q = apply_rotary(q, *cos_sin).transpose(1, 2)
        k = apply_rotary(k, *cos_sin).transpose(1, 2)
        q, k = q.to(v.dtype), k.to(v.dtype)
        y = flash_sdpa(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        return self.o_proj(y.transpose(1, 2).contiguous().view(batch, seq, -1))


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

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), positions, self.theta)
        h = self.post_norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class CacheProcessor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        config: CacheProcessorConfig,
        flow_config: FlowProcessorConfig | None = None,
        diffusion_config: DiffusionProcessorConfig | None = None,
    ):
        super().__init__()
        self.config = config
        # This is a training execution policy, not model state. Keeping it as a
        # plain attribute preserves parameter names and serialized state_dict keys.
        self.activation_checkpointing = False
        self.in_norm = RMSNorm(input_dim, config.norm_eps)
        self.in_proj = nn.Linear(input_dim, config.hidden_size)
        self.layers = nn.ModuleList(
            ProcessorLayer(config) for _ in range(config.num_layers)
        )
        self.out_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.out_proj = nn.Linear(config.hidden_size, input_dim)
        self.gate = nn.Parameter(torch.full((1,), config.gate_init))
        self._gate_value: torch.Tensor | None = None
        if flow_config is not None and diffusion_config is not None:
            raise ValueError(
                "cache processor cannot combine flow and diffusion"
            )
        self.flow = (
            LatentFlow(config.hidden_size, flow_config)
            if flow_config is not None
            else None
        )
        self.diffusion = (
            TaskDirectedDiffusion(config.hidden_size, diffusion_config)
            if diffusion_config is not None
            else None
        )
        self.init_weights()

    def init_weights(self, std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        nn.init.constant_(self.gate, self.config.gate_init)
        if self.config.out_proj_init == "zero":
            nn.init.zeros_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

    def compute_update(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        initial = self.in_proj(self.in_norm(x))
        if self.flow is not None:
            displacement = self.flow(
                initial,
                lambda state, _time: self.out_norm(
                    self._run_layers(state, positions)
                ),
            )
            return self.out_proj(displacement)
        if self.diffusion is not None:
            displacement = self.diffusion(
                initial,
                lambda state, _time: self.out_norm(
                    self._run_layers(state, positions)
                ),
            )
            return self.out_proj(displacement)
        return self.out_proj(
            self.out_norm(self._run_layers(initial, positions))
        )

    def _run_layers(
        self, state: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            if (
                self.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                # Save only the layer input and recompute attention/MLP activations
                # during backward. One extra layer forward lowers peak memory.
                state = checkpoint(
                    layer,
                    state,
                    positions,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                state = layer(state, positions)
        return state

    def scaled_update(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        delta = self.compute_update(x, positions)
        return delta * self._gate_scale(delta.dtype)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return x + self.scaled_update(x, positions)

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


class PreparedPerLayerCacheProcessorExecution:
    """One explicitly owned BT execution with lazily prepared parameter stacks.

    The owner may cover a complete ordinary forward or one checkpoint segment
    invocation. Closing it is final: backward replay must create a new owner.
    This keeps graph-derived stacks local without weakening reuse protection.
    """

    def __init__(self, processor: PerLayerCacheProcessor):
        self._processor = processor
        self._backend = processor.execution_backend
        self._prepared: PreparedBatchedCacheProcessorParameters | None = None
        self._closed = False

    def __enter__(self) -> PreparedPerLayerCacheProcessorExecution:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self._prepared = None
        self._closed = True

    def __call__(
        self, cache: CacheLike, *, select_all: bool = False
    ) -> CacheLike:
        if self._closed:
            raise RuntimeError(
                "prepared cache-processor execution cannot be reused across forwards"
            )
        if self._prepared is None:
            self._prepared = prepare_batched_cache_processor_parameters(
                self._processor.blocks
            )
        prepared = self._prepared
        return self._processor(
            cache,
            select_all=select_all,
            _prepared=prepared,
            _prepared_backend=self._backend,
        )


class PerLayerCacheProcessor(CacheProcessorBase):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        model_config.ensure_uniform_stored_cache_shape("BT")
        if model_config.cache_processor is None:
            raise ValueError("BT requires cache_processor config")
        self.config = model_config.cache_processor
        if self.config.processor_mode != "standard":
            raise ValueError(
                f"cache_processor.processor_mode='{self.config.processor_mode}' is currently BGT-only"
            )
        if self.config.score != "attn_l2":
            raise ValueError(
                "cache_processor.score='kv_norm' is currently BGT-only"
            )
        num_kv_heads, head_dim = model_config.stored_cache_shape
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        layer_dim = self.config.stream_factor * num_kv_heads * head_dim
        self.blocks = nn.ModuleList(
            CacheProcessor(layer_dim, self.config)
            for _ in range(model_config.cache_num_layers)
        )
        self.execution_backend: BTProcessorBackend = (
            resolve_bt_processor_backend()
        )

    def prepare_forward_execution(
        self,
    ) -> PreparedPerLayerCacheProcessorExecution:
        return PreparedPerLayerCacheProcessorExecution(self)

    def execution_backend_metadata(self) -> dict[str, object]:
        return bt_processor_backend_metadata(self.execution_backend)

    def forward(
        self,
        cache: CacheLike,
        *,
        select_all: bool = False,
        _prepared: PreparedBatchedCacheProcessorParameters | None = None,
        _prepared_backend: BTProcessorBackend = "eager",
    ) -> CacheLike:
        indices = (
            cache.all_indices()
            if select_all
            else cache.select_indices(
                self.config.process_every, self.config.topk
            )
        )
        if self.config.update_stream == "v":
            values = gather_cache_values(cache, indices)
            merged = merge_layer_values(values)
            updated = (
                run_batched_cache_processors(self.blocks, merged, indices)
                if _prepared is None
                else run_prepared_bt_processor_region(
                    _prepared_backend, _prepared, merged, indices
                )
            )
            new_values = split_layer_values(
                updated, values.shape[2], values.shape[4]
            )
            return scatter_cache_values(cache, indices, new_values)
        if triton_cache_movement_enabled(self):
            if not isinstance(cache, WindowedCache):
                raise RuntimeError(
                    "enabled Triton BT movement requires WindowedCache"
                )
            merged = triton_gather_shared_kv(
                cache.keys,
                cache.values,
                indices,
                layout="layer",
                validate_indices=False,
            )
        else:
            keys, values = gather_cache_layers(cache, indices)
            merged = merge_layer_kv(keys, values)
        updated = (
            run_batched_cache_processors(self.blocks, merged, indices)
            if _prepared is None
            else run_prepared_bt_processor_region(
                _prepared_backend, _prepared, merged, indices
            )
        )
        if triton_cache_movement_enabled(self):
            assert isinstance(cache, WindowedCache)
            cache.keys, cache.values = triton_scatter_shared_kv(
                cache.keys,
                cache.values,
                indices,
                updated,
                layout="layer",
                validate_indices=False,
            )
            return cache
        new_keys, new_values = split_layer_kv(
            updated, self.num_kv_heads, self.head_dim
        )
        return scatter_cache_layers(cache, indices, new_keys, new_values)


class GlobalCacheProcessor(CacheProcessorBase):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self._model_config = model_config
        model_config.ensure_uniform_stored_cache_shape("BGT")
        if model_config.cache_processor is None:
            raise ValueError("BGT requires cache_processor config")
        self.config = model_config.cache_processor
        if (
            self.config.processor_mode == "workspace_broadcast"
            and self.config.update_stream != "v"
        ):
            raise ValueError(
                "workspace_broadcast requires cache_processor.update_stream='v'"
            )
        if (
            self.config.processor_mode == "workspace_broadcast"
            and self.config.score != "kv_norm"
        ):
            raise ValueError(
                "workspace_broadcast requires cache_processor.score='kv_norm'"
            )
        if (
            self.config.processor_mode == "head_mask_broadcast"
            and self.config.update_stream != "kv"
        ):
            raise ValueError(
                "head_mask_broadcast requires cache_processor.update_stream='kv'"
            )
        if (
            self.config.processor_mode == "head_mask_broadcast"
            and self.config.score != "kv_norm"
        ):
            raise ValueError(
                "head_mask_broadcast requires cache_processor.score='kv_norm'"
            )
        num_kv_heads, head_dim = model_config.stored_cache_shape
        per_layer = self.config.stream_factor * num_kv_heads * head_dim
        self.num_layers = model_config.cache_num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_cache_heads = self.num_layers * self.num_kv_heads
        self.block = CacheProcessor(
            self.num_layers * per_layer,
            self.config,
            model_config.flow_processor,
            model_config.diffusion_processor,
        )
        if self.config.score == "kv_norm":
            init = torch.tensor(
                [self.config.kv_score_init_k, self.config.kv_score_init_v],
                dtype=torch.float32,
            )
            self.salience_logits = nn.Parameter(init.clamp_min(1e-8).log())
        self.execution_backend: CacheProcessorBackend = (
            resolve_cache_processor_backend(model_config)
        )

    def execution_backend_metadata(self) -> dict[str, object]:
        return cache_processor_backend_metadata(
            self._model_config, self.execution_backend
        )

    def forward(
        self, cache: CacheLike, *, select_all: bool = False
    ) -> CacheLike:
        indices = self._select_indices(cache, select_all)
        if self.config.processor_mode == "workspace_broadcast":
            return self._forward_workspace_broadcast(cache, indices)
        if self.config.processor_mode == "head_mask_broadcast":
            return self._forward_head_mask_broadcast(cache, indices)
        if self.config.update_stream == "v":
            values = gather_cache_values(cache, indices)
            updated = run_bgt_processor_region(
                self.execution_backend,
                self.block,
                merge_global_values(values),
                indices,
            )
            new_values = split_global_values(
                updated, self.num_layers, self.num_kv_heads, self.head_dim
            )
            return scatter_cache_values(cache, indices, new_values)
        if triton_cache_movement_enabled(self):
            if not isinstance(cache, WindowedCache):
                raise RuntimeError(
                    "enabled Triton BGT movement requires WindowedCache"
                )
            merged = triton_gather_shared_kv(
                cache.keys,
                cache.values,
                indices,
                layout="global",
                validate_indices=False,
            )
        else:
            keys, values = gather_cache_layers(cache, indices)
            merged = merge_global_kv(keys, values)
        updated = run_bgt_processor_region(
            self.execution_backend,
            self.block,
            merged,
            indices,
        )
        if triton_cache_movement_enabled(self):
            assert isinstance(cache, WindowedCache)
            cache.keys, cache.values = triton_scatter_shared_kv(
                cache.keys,
                cache.values,
                indices,
                updated,
                layout="global",
                validate_indices=False,
            )
            return cache
        new_keys, new_values = split_global_kv(
            updated, self.num_layers, self.num_kv_heads, self.head_dim
        )
        return scatter_cache_layers(cache, indices, new_keys, new_values)

    def _select_indices(
        self, cache: CacheLike, select_all: bool
    ) -> torch.Tensor:
        if select_all:
            return cache.all_indices()
        if self.config.score == "kv_norm":
            return select_kv_norm_indices(
                cache,
                self.salience_logits,
                self.config.process_every,
                self.config.topk,
            )
        return cache.select_indices(self.config.process_every, self.config.topk)

    def _forward_workspace_broadcast(
        self, cache: CacheLike, indices: torch.Tensor
    ) -> CacheLike:
        writer_values, original_values = workspace_writer_values(
            cache,
            indices,
            self.salience_logits,
            self.config.writer_head_fraction,
            self.config.salience_temperature,
        )
        delta = run_bgt_processor_region(
            self.execution_backend,
            self.block,
            merge_global_values(writer_values),
            indices,
            scaled_update=True,
        )
        broadcast = split_global_values(
            delta, self.num_layers, self.num_kv_heads, self.head_dim
        )
        return scatter_cache_values(cache, indices, original_values + broadcast)

    def _forward_head_mask_broadcast(
        self, cache: CacheLike, indices: torch.Tensor
    ) -> CacheLike:
        writer_keys, writer_values, original_keys, original_values = (
            workspace_writer_kv(
                cache,
                indices,
                self.salience_logits,
                self.config.writer_head_fraction,
                self.config.salience_temperature,
            )
        )
        delta = run_bgt_processor_region(
            self.execution_backend,
            self.block,
            merge_global_kv(writer_keys, writer_values),
            indices,
            scaled_update=True,
        )
        key_delta, value_delta = split_global_kv(
            delta, self.num_layers, self.num_kv_heads, self.head_dim
        )
        return scatter_cache_layers(
            cache,
            indices,
            original_keys + key_delta,
            original_values + value_delta,
        )


def _merge_kv(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    batch, heads, seq, dim = keys.shape
    k = keys.permute(0, 2, 1, 3).reshape(batch, seq, heads * dim)
    v = values.permute(0, 2, 1, 3).reshape(batch, seq, heads * dim)
    return torch.cat([k, v], dim=-1)


def _split_kv(
    x: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq, dim = x.shape
    half = dim // 2
    keys, values = x[..., :half], x[..., half:]
    keys = (
        keys.view(batch, seq, num_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    values = (
        values.view(batch, seq, num_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return keys, values
