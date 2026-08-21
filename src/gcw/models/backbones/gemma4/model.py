from __future__ import annotations

from dataclasses import replace
from typing import cast
import torch
import torch.nn as nn

from gcw.config import LossBackend, ModelConfig
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.models.backbones.transformer import (
    Linear,
    TF,
    _autocast_dtype,
    _matches_stop_sequence,
)
from gcw.models.backbones.gemma4.block import Gemma4Block
from gcw.models.backbones.gemma4.norms import (
    Gemma4RMSNorm,
    Gemma4ScaledWordEmbedding,
)
from gcw.models.backbones.gemma4.rotary import Gemma4RotaryCache


class Gemma4TF(TF):
    block_cls = Gemma4Block

    def __init__(self, config: ModelConfig, pad_vocab_size_to: int = 128):
        nn.Module.__init__(self)
        self.config = replace(config, variant="tf")
        vocab = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        ple_vocab = int(config.vocab_size_per_layer_input or config.vocab_size)
        ple_vocab = (
            (ple_vocab + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        self.padded_vocab_size = vocab
        self.padded_ple_vocab_size = ple_vocab
        self.embed_tokens = Gemma4ScaledWordEmbedding(
            vocab,
            config.hidden_size,
            padding_idx=0,
            embed_scale=config.hidden_size**0.5,
        )
        cache_index_by_layer = {
            layer: idx for idx, layer in enumerate(config.cache_layer_indices)
        }
        self.layers = nn.ModuleList(
            self.block_cls(config, i, cache_index_by_layer.get(i))
            for i in range(config.num_layers)
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, config.norm_eps)  # type: ignore[assignment]
        self.rotary = Gemma4RotaryCache(config)  # type: ignore[assignment]
        self.hidden_size_per_layer_input = int(
            config.hidden_size_per_layer_input
        )
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = Gemma4ScaledWordEmbedding(
                ple_vocab,
                config.num_layers * self.hidden_size_per_layer_input,
                padding_idx=0,
                embed_scale=self.hidden_size_per_layer_input**0.5,
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = Linear(
                config.hidden_size,
                config.num_layers * self.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_projection_norm = Gemma4RMSNorm(
                self.hidden_size_per_layer_input, config.norm_eps
            )
        self.lm_head = (
            None
            if config.tie_embeddings
            else Linear(config.hidden_size, vocab, bias=False)
        )

    @torch.no_grad()
    def init_weights(self, std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0.0, std)
                if isinstance(module, Gemma4ScaledWordEmbedding):
                    module.embed_scale.fill_(module.scalar_embed_scale)
            elif isinstance(module, Gemma4RMSNorm) and module.with_scale:
                nn.init.ones_(module.weight)
            elif isinstance(module, Gemma4Block):
                module.layer_scalar.fill_(1.0)

    def _per_layer_inputs(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | None:
        if not self.hidden_size_per_layer_input:
            return None
        per_layer_inputs = self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_layers,
            self.hidden_size_per_layer_input,
        )
        projection = (
            self.per_layer_model_projection(inputs_embeds)
            * self.per_layer_model_projection_scale
        )
        projection = projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_layers,
            self.hidden_size_per_layer_input,
        )
        projection = self.per_layer_projection_norm(projection)
        return (projection + per_layer_inputs) * self.per_layer_input_scale

    def logits_from_hidden(self, x: torch.Tensor) -> torch.Tensor:
        logits = super().logits_from_hidden(x)
        softcap = self.config.final_logit_softcapping
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def _position_embeddings(
        self,
        start_pos: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        positions = torch.arange(start_pos, start_pos + seq_len, device=device)
        rotary = cast(Gemma4RotaryCache, self.rotary)
        return {
            layer_type: rotary.get(positions, layer_type, dtype)
            for layer_type in set(self.config.layer_types or ["full_attention"])
        }

    def forward_full(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_reduction: str = "mean",
        attention_mask: torch.Tensor | None = None,
        loss_backend: LossBackend = "reference",
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        per_layer_inputs = self._per_layer_inputs(input_ids, x)
        position_embeddings = self._position_embeddings(
            0,
            input_ids.shape[1],
            input_ids.device,
            _autocast_dtype(x.device) or x.dtype,
        )
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, module in enumerate(self.layers):
            layer = cast(Gemma4Block, module)
            per_layer_input = (
                per_layer_inputs[:, :, i, :]
                if per_layer_inputs is not None
                else None
            )
            x, _, _, _ = layer(
                x,
                per_layer_input,
                position_embeddings[layer.layer_type],
                None,
                shared_kv_states,
                0,
            )
        x = self.norm(x)
        if targets is not None:
            return self.loss_from_hidden(
                x, targets, loss_reduction, loss_backend
            )
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
        if isinstance(cache, WindowedCache) and not self._uniform_cache_shape():
            raise ValueError(
                "Gemma4 WindowedCache requires uniform stored KV shapes"
            )
        x = self.embed_tokens(input_ids)
        per_layer_inputs = self._per_layer_inputs(input_ids, x)
        position_embeddings = self._position_embeddings(
            start_pos,
            input_ids.shape[1],
            input_ids.device,
            _autocast_dtype(x.device) or x.dtype,
        )
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        layer_scores: list[torch.Tensor] = []
        old_scores = (
            input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32)
            if cache is None
            else cache.scores
        )
        has_cache = cache is not None and cache.scores.numel() > 0
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        cache_idx = 0
        for i, module in enumerate(self.layers):
            layer = cast(Gemma4Block, module)
            per_layer_input = (
                per_layer_inputs[:, :, i, :]
                if per_layer_inputs is not None
                else None
            )
            if layer.self_attn.is_kv_shared_layer:
                x, _, _, scores = layer(
                    x,
                    per_layer_input,
                    position_embeddings[layer.layer_type],
                    None,
                    shared_kv_states,
                    start_pos,
                )
            else:
                if has_cache:
                    if cache is None:
                        raise RuntimeError(
                            "cache state disappeared during cached attention"
                        )
                    past = cache.layer(cache_idx)
                else:
                    past = None
                x, k, v, scores = layer(
                    x,
                    per_layer_input,
                    position_embeddings[layer.layer_type],
                    past,
                    shared_kv_states,
                    start_pos,
                )
                keys.append(k)
                values.append(v)
                cache_idx += 1
            layer_scores.append(scores)
        x = self.norm(x)
        out = self.logits_from_hidden(x) if logits else x
        chunk_scores = torch.stack(layer_scores).mean(dim=0)
        new_scores = (
            chunk_scores
            if old_scores.numel() == 0
            else torch.cat([old_scores, chunk_scores], dim=1)
        )
        if cache_format == "windowed" or isinstance(cache, WindowedCache):
            if not self._uniform_cache_shape():
                raise ValueError(
                    "Gemma4 WindowedCache requires uniform stored KV shapes"
                )
            return out, WindowedCache.from_lists(keys, values, new_scores)
        return out, CacheState(keys, values, new_scores)

    def forward_chunk_static(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        cache: StaticCacheState,
        *,
        logits: bool = True,
    ) -> tuple[torch.Tensor, StaticCacheState]:
        if not self._uniform_cache_shape():
            raise ValueError(
                "Gemma4 static cache requires uniform stored KV shapes"
            )
        x = self.embed_tokens(input_ids)
        per_layer_inputs = self._per_layer_inputs(input_ids, x)
        end_pos = start_pos + input_ids.shape[1]
        position_embeddings = self._position_embeddings(
            start_pos, input_ids.shape[1], input_ids.device, x.dtype
        )
        layer_scores: list[torch.Tensor] = []
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, module in enumerate(self.layers):
            layer = cast(Gemma4Block, module)
            per_layer_input = (
                per_layer_inputs[:, :, i, :]
                if per_layer_inputs is not None
                else None
            )
            x, scores = layer.forward_static(
                x,
                per_layer_input,
                position_embeddings[layer.layer_type],
                cache,
                shared_kv_states,
                start_pos,
                end_pos,
            )
            layer_scores.append(scores)
        x = self.norm(x)
        out = self.logits_from_hidden(x) if logits else x
        cache.write_scores(start_pos, torch.stack(layer_scores).mean(dim=0))
        return out, cache

    @torch.no_grad()
    def generate_fast(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        processor=None,
        process_every: int = 64,
        process_prefill_windows: bool = True,
        process_after_prefill: bool = False,
        relative_generation_cadence: bool = False,
        stop_sequences: list[list[int]] | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        if self._uniform_cache_shape():
            return super().generate_fast(
                input_ids,
                max_new_tokens,
                eos_token_id=eos_token_id,
                processor=processor,
                process_every=process_every,
                process_prefill_windows=process_prefill_windows,
                process_after_prefill=process_after_prefill,
                relative_generation_cadence=relative_generation_cadence,
                stop_sequences=stop_sequences,
                pad_token_id=pad_token_id,
            )
        out = input_ids[:, -self.config.sequence_len :]
        if out.shape[1] == 0:
            raise ValueError("generate requires at least one input token")
        cache: CacheState | WindowedCache = CacheState(
            [],
            [],
            input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32),
        )
        logits = None
        if process_prefill_windows:
            for start in range(0, out.shape[1], process_every):
                if start > 0 and processor is not None:
                    cache = processor(cache)
                logits, cache = self.forward_chunk(
                    out[:, start : start + process_every], start, cache
                )
        else:
            logits, cache = self.forward_chunk(out, 0, cache)
        if logits is None:
            raise ValueError("generate requires at least one input token")

        finished = torch.zeros(
            out.shape[0], dtype=torch.bool, device=out.device
        )
        fill_id = eos_token_id if pad_token_id is None else pad_token_id
        if fill_id is None:
            fill_id = 0
        fill = torch.full(
            (out.shape[0], 1),
            int(fill_id),
            dtype=input_ids.dtype,
            device=out.device,
        )
        for generated_idx in range(max_new_tokens):
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            if bool(finished.any()):
                next_id = torch.where(finished[:, None], fill, next_id)
            out = torch.cat([out, next_id], dim=1)
            if eos_token_id is not None:
                finished = finished | (next_id.squeeze(1) == eos_token_id)
            finished = finished | _matches_stop_sequence(out, stop_sequences)
            if bool(finished.all()):
                break
            pos = out.shape[1] - 1
            if processor is not None:
                if relative_generation_cadence:
                    should_process = (
                        generated_idx == 0 and process_after_prefill
                    )
                    should_process = should_process or (
                        generated_idx > 0 and generated_idx % process_every == 0
                    )
                else:
                    should_process = pos > 0 and pos % process_every == 0
                if should_process:
                    cache = processor(
                        cache,
                        select_all=relative_generation_cadence
                        and generated_idx == 0,
                    )
            logits, cache = self.forward_chunk(next_id, pos, cache)
        return out

    def _uniform_cache_shape(self) -> bool:
        shapes = [
            (
                self.config.kv_heads_for_layer(i),
                self.config.head_dim_for_layer(i),
            )
            for i in self.config.cache_layer_indices
        ]
        return len(set(shapes)) <= 1
