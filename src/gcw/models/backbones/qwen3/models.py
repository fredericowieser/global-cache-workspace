from __future__ import annotations

from dataclasses import replace
from typing import cast
import torch
import torch.nn as nn

from gcw.config import LossBackend, ModelConfig
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.models.backbones.transformer import (
    CacheLike,
    Linear,
    ProcessorExecutionFactory,
    ProcessorFn,
    RotaryCache,
    TF,
    _autocast_dtype,
)
from gcw.models.backbones.qwen3.blocks import Qwen35Block
from gcw.models.backbones.qwen3.linear_attention import (
    Qwen35GatedDeltaNet,
)
import gcw.models.backbones.qwen3 as qwen3_pkg
from gcw.models.backbones.qwen3.moe import (
    Qwen35MoeBlock,
    _Qwen35MoeExperts,
)
from gcw.models.backbones.qwen3.norms import (
    Qwen35RMSNorm,
    Qwen35RMSNormGated,
)


class Qwen35TF(TF):
    block_cls = Qwen35Block

    def __init__(self, config: ModelConfig, pad_vocab_size_to: int = 128):
        nn.Module.__init__(self)
        self.config = replace(config, variant="tf", cache_processor=None)
        vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.padded_vocab_size = vocab
        self.embed_tokens = nn.Embedding(vocab, config.hidden_size)
        cache_index_by_layer = {layer: idx for idx, layer in enumerate(config.cache_layer_indices)}
        self.layers = nn.ModuleList(
            self.block_cls(config, i, cache_index_by_layer.get(i)) for i in range(config.num_layers)
        )
        self.norm = Qwen35RMSNorm(config.hidden_size, config.norm_eps)  # type: ignore[assignment]
        rotary_dim = int(config.head_dim_for_layer(0) * config.partial_rotary_factor)
        self.rotary = RotaryCache(config.sequence_len, rotary_dim, config.rope_theta)
        self.lm_head = None if config.tie_embeddings else Linear(config.hidden_size, vocab, bias=False)

    @torch.no_grad()
    def init_weights(self, std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, 0.0, std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0.0, std)
            elif isinstance(module, _Qwen35MoeExperts):
                nn.init.normal_(module.gate_up_proj, 0.0, std)
                nn.init.normal_(module.down_proj, 0.0, std)
            elif isinstance(module, Qwen35RMSNorm):
                nn.init.zeros_(module.weight)
            elif isinstance(module, Qwen35RMSNormGated):
                nn.init.ones_(module.weight)
            elif isinstance(module, Qwen35GatedDeltaNet):
                module.init_state_weights()

    def forward_full(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_reduction: str = "mean",
        attention_mask: torch.Tensor | None = None,
        loss_backend: LossBackend = "reference",
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        cos_sin = self.rotary.get_range(0, input_ids.shape[1], input_ids.device, _autocast_dtype(x.device) or x.dtype)
        for module in self.layers:
            layer = cast(Qwen35Block, module)
            x, _, _, _, _ = layer(x, cos_sin, None, None)
        x = self.norm(x)
        if targets is not None:
            return self.loss_from_hidden(x, targets, loss_reduction, loss_backend)
        return self.logits_from_hidden(x)

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        cache: CacheState | WindowedCache | None = None,
        *,
        logits: bool = True,
        cache_format: str = "list",
    ) -> tuple[torch.Tensor, CacheState | WindowedCache]:
        x = self.embed_tokens(input_ids)
        cos_sin = self.rotary.get_range(
            start_pos,
            input_ids.shape[1],
            input_ids.device,
            _autocast_dtype(x.device) or x.dtype,
        )
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        layer_scores: list[torch.Tensor] = []
        old_scores = input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32) if cache is None else cache.scores
        has_cache = cache is not None and cache.scores.numel() > 0
        linear_states = {} if cache is None or cache.linear_states is None else dict(cache.linear_states)
        cache_idx = 0
        for module in self.layers:
            layer = cast(Qwen35Block, module)
            if layer.layer_type == "full_attention":
                if has_cache:
                    if cache is None:
                        raise RuntimeError("cache state disappeared during cached attention")
                    past = cache.layer(cache_idx)
                else:
                    past = None
                x, k, v, scores, _ = layer(x, cos_sin, past, None)
                keys.append(k)
                values.append(v)
                cache_idx += 1
            else:
                x, _, _, scores, new_state = layer(x, cos_sin, None, linear_states.get(layer.layer_idx))
                linear_states[layer.layer_idx] = new_state
            layer_scores.append(scores)
        x = self.norm(x)
        out = self.logits_from_hidden(x) if logits else x
        chunk_scores = torch.stack(layer_scores).mean(dim=0)
        new_scores = chunk_scores if old_scores.numel() == 0 else torch.cat([old_scores, chunk_scores], dim=1)
        if cache_format == "windowed" or isinstance(cache, WindowedCache):
            return out, WindowedCache.from_lists(keys, values, new_scores, linear_states)
        return out, CacheState(keys, values, new_scores, linear_states)

    def forward_chunk_static(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        cache: StaticCacheState,
        *,
        logits: bool = True,
    ) -> tuple[torch.Tensor, StaticCacheState]:
        x = self.embed_tokens(input_ids)
        end_pos = start_pos + input_ids.shape[1]
        pos = torch.arange(start_pos, end_pos, device=input_ids.device)
        cos_sin = self.rotary.get(pos, x.dtype)
        layer_scores: list[torch.Tensor] = []
        for module in self.layers:
            layer = cast(Qwen35Block, module)
            x, scores = layer.forward_static(x, cos_sin, cache, start_pos, end_pos)
            layer_scores.append(scores)
        x = self.norm(x)
        out = self.logits_from_hidden(x) if logits else x
        cache.write_scores(start_pos, torch.stack(layer_scores).mean(dim=0))
        return out, cache


class Qwen35MoeTF(Qwen35TF):
    block_cls = Qwen35MoeBlock

    def forward_full(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_reduction: str = "mean",
        attention_mask: torch.Tensor | None = None,
        loss_backend: LossBackend = "reference",
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        cos_sin = self.rotary.get_range(0, input_ids.shape[1], input_ids.device, _autocast_dtype(x.device) or x.dtype)
        router_logits = []
        for layer in self.layers:
            x, _, _, _, _, layer_router_logits = layer(x, cos_sin, None, None)
            router_logits.append(layer_router_logits)
        x = self.norm(x)
        if targets is not None:
            loss = self.loss_from_hidden(x, targets, loss_reduction, loss_backend)
            if loss_reduction == "mean" and self.config.router_aux_loss_coef:
                aux_loss = qwen3_pkg.qwen35_moe_load_balancing_loss(
                    tuple(router_logits),
                    num_experts=self.config.num_experts,
                    top_k=self.config.num_experts_per_tok,
                    attention_mask=attention_mask,
                )
                loss = loss + self.config.router_aux_loss_coef * aux_loss.to(loss.device)
            return loss
        return self.logits_from_hidden(x)

    def forward_windowed_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        *,
        processor: ProcessorFn,
        process_every: int,
        loss_reduction: str = "mean",
        attention_mask: torch.Tensor | None = None,
        activation_checkpoint_windows: int = 0,
        processor_execution_factory: ProcessorExecutionFactory | None = None,
        loss_backend: LossBackend = "reference",
    ) -> torch.Tensor:
        del processor_execution_factory
        if activation_checkpoint_windows:
            raise ValueError("segmented window activation checkpointing does not yet support qwen3_5_moe router logits")
        cache: CacheLike = CacheState([], [], input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32))
        hidden_chunks: list[torch.Tensor] = []
        router_chunks: list[list[torch.Tensor]] = [[] for _ in self.layers]
        for start in range(0, input_ids.shape[1], process_every):
            if start > 0:
                cache = cast(CacheLike, processor(cache))
            end = min(start + process_every, input_ids.shape[1])
            hidden, cache, chunk_router_logits = self._forward_chunk_with_router_logits(
                input_ids[:, start:end],
                start,
                cache,
                logits=False,
                cache_format="windowed",
            )
            hidden_chunks.append(hidden)
            for layer_chunks, layer_logits in zip(router_chunks, chunk_router_logits, strict=True):
                layer_chunks.append(layer_logits.view(input_ids.shape[0], end - start, -1))
        loss = self.loss_from_hidden(torch.cat(hidden_chunks, dim=1), targets, loss_reduction, loss_backend)
        if loss_reduction == "mean" and self.config.router_aux_loss_coef:
            router_logits = tuple(
                torch.cat(layer_chunks, dim=1).reshape(-1, self.config.num_experts) for layer_chunks in router_chunks
            )
            aux_loss = qwen3_pkg.qwen35_moe_load_balancing_loss(
                router_logits,
                num_experts=self.config.num_experts,
                top_k=self.config.num_experts_per_tok,
                attention_mask=attention_mask,
            )
            loss = loss + self.config.router_aux_loss_coef * aux_loss.to(loss.device)
        return loss

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        cache: CacheState | WindowedCache | None = None,
        *,
        logits: bool = True,
        cache_format: str = "list",
    ) -> tuple[torch.Tensor, CacheState | WindowedCache]:
        out, updated_cache, _ = self._forward_chunk_with_router_logits(
            input_ids,
            start_pos,
            cache,
            logits=logits,
            cache_format=cache_format,
        )
        return out, updated_cache

    def _forward_chunk_with_router_logits(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        cache: CacheState | WindowedCache | None = None,
        *,
        logits: bool = True,
        cache_format: str = "list",
    ) -> tuple[torch.Tensor, CacheState | WindowedCache, tuple[torch.Tensor, ...]]:
        x = self.embed_tokens(input_ids)
        cos_sin = self.rotary.get_range(
            start_pos,
            input_ids.shape[1],
            input_ids.device,
            _autocast_dtype(x.device) or x.dtype,
        )
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        layer_scores: list[torch.Tensor] = []
        router_logits: list[torch.Tensor] = []
        old_scores = input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32) if cache is None else cache.scores
        has_cache = cache is not None and cache.scores.numel() > 0
        linear_states = {} if cache is None or cache.linear_states is None else dict(cache.linear_states)
        cache_idx = 0
        for module in self.layers:
            layer = cast(Qwen35MoeBlock, module)
            if layer.layer_type == "full_attention":
                if has_cache:
                    if cache is None:
                        raise RuntimeError("cache state disappeared during cached attention")
                    past = cache.layer(cache_idx)
                else:
                    past = None
                x, key, value, scores, _, layer_router_logits = layer(x, cos_sin, past, None)
                if key is None or value is None:
                    raise RuntimeError("full-attention layer did not return KV tensors")
                keys.append(key)
                values.append(value)
                cache_idx += 1
            else:
                x, _, _, scores, new_state, layer_router_logits = layer(
                    x,
                    cos_sin,
                    None,
                    linear_states.get(layer.layer_idx),
                )
                if new_state is None:
                    raise RuntimeError("linear-attention layer did not return recurrent state")
                linear_states[layer.layer_idx] = new_state
            layer_scores.append(scores)
            router_logits.append(layer_router_logits)
        x = self.norm(x)
        out = self.logits_from_hidden(x) if logits else x
        chunk_scores = torch.stack(layer_scores).mean(dim=0)
        new_scores = chunk_scores if old_scores.numel() == 0 else torch.cat([old_scores, chunk_scores], dim=1)
        if cache_format == "windowed" or isinstance(cache, WindowedCache):
            updated_cache: CacheState | WindowedCache = WindowedCache.from_lists(
                keys,
                values,
                new_scores,
                linear_states,
            )
        else:
            updated_cache = CacheState(keys, values, new_scores, linear_states)
        return out, updated_cache, tuple(router_logits)


Qwen3TF = TF
