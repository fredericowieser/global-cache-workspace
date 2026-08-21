from __future__ import annotations

from dataclasses import replace
from typing import Any, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from gcw.config import LossBackend, ModelConfig
from gcw.models.backbones.base import Backbone
from gcw.models.cache_processors.base import cache_processor_recomputation_id
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.models.backbones.transformer.block import Block
from gcw.models.backbones.transformer.cache import (
    CacheLike,
    ProcessorExecutionFactory,
    ProcessorFn,
    WindowedCacheLayout,
    _flatten_windowed_cache,
    _unflatten_windowed_cache,
    _window_checkpoint_contexts,
)
from gcw.models.backbones.transformer.generation import (
    generate_fast_tf,
    generate_tf,
)
from gcw.models.backbones.transformer.losses import (
    answer_only_loss_from_hidden as _answer_only_loss_from_hidden,
    loss_from_hidden_reference as _loss_from_hidden_reference,
)
from gcw.models.backbones.transformer.layers import (
    Linear,
    RMSNorm,
    RotaryCache,
    _autocast_dtype,
    _causal_chunk_bias,
)
from gcw.performance.backends.qwen3 import (
    Qwen3BackboneChunkBackend,
    run_qwen3_backbone_chunk_region,
    run_qwen3_initial_backbone_chunk_region,
)
import gcw.models.backbones.transformer as tf_pkg


class TF(Backbone):
    def __init__(self, config: ModelConfig, pad_vocab_size_to: int = 128):
        super().__init__()
        self.config = replace(config, variant="tf")
        vocab = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        self.padded_vocab_size = vocab
        self.embed_tokens = nn.Embedding(vocab, config.hidden_size)
        self.layers = nn.ModuleList(
            Block(config, i) for i in range(config.num_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        if config.head_dim is None:
            raise ValueError("transformer requires a resolved model head_dim")
        self.rotary = RotaryCache(
            config.sequence_len, config.head_dim, config.rope_theta
        )
        self.lm_head = (
            None
            if config.tie_embeddings
            else Linear(config.hidden_size, vocab, bias=False)
        )
        self.backbone_chunk_backend: Qwen3BackboneChunkBackend = "eager"

    @torch.no_grad()
    def init_weights(self, std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0.0, std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def get_device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_reduction: str = "mean",
        processor: ProcessorFn | None = None,
        process_every: int = 64,
        attention_mask: torch.Tensor | None = None,
        activation_checkpoint_windows: int = 0,
        processor_execution_factory: ProcessorExecutionFactory | None = None,
        loss_backend: LossBackend = "reference",
    ) -> torch.Tensor:
        if processor is None:
            return self.forward_full(
                input_ids,
                targets,
                loss_reduction,
                attention_mask=attention_mask,
                loss_backend=loss_backend,
            )
        if targets is not None:
            kwargs: dict[str, Any] = {
                "processor": processor,
                "process_every": process_every,
                "loss_reduction": loss_reduction,
                "attention_mask": attention_mask,
                "loss_backend": loss_backend,
            }
            if activation_checkpoint_windows:
                kwargs["activation_checkpoint_windows"] = (
                    activation_checkpoint_windows
                )
                kwargs["processor_execution_factory"] = (
                    processor_execution_factory
                )
            return self.forward_windowed_loss(input_ids, targets, **kwargs)
        logits, _ = self.forward_windowed(
            input_ids, processor=processor, process_every=process_every
        )
        return logits

    def forward_full(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_reduction: str = "mean",
        attention_mask: torch.Tensor | None = None,
        loss_backend: LossBackend = "reference",
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        cos_sin = self.rotary.get_range(
            0,
            input_ids.shape[1],
            input_ids.device,
            _autocast_dtype(x.device) or x.dtype,
        )
        for layer in self.layers:
            x, _, _, _ = cast(Block, layer)(x, cos_sin, None)
        x = self.norm(x)
        if targets is not None:
            return self.loss_from_hidden(
                x, targets, loss_reduction, loss_backend
            )
        return self.logits_from_hidden(x)

    def logits_from_hidden(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.tie_embeddings:
            logits = F.linear(x, self.embed_tokens.weight.to(x.dtype))
        else:
            if self.lm_head is None:
                raise RuntimeError(
                    "untied transformer has no language-model head"
                )
            logits = self.lm_head(x)
        return logits[..., : self.config.vocab_size].float()

    def loss_from_hidden(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        loss_reduction: str,
        loss_backend: LossBackend = "reference",
    ) -> torch.Tensor:
        if loss_reduction not in {"mean", "sum", "none", "row_sum"}:
            raise ValueError(
                "loss_reduction must be 'mean', 'sum', 'none', or 'row_sum'"
            )
        if loss_backend == "reference":
            execution = tf_pkg.reference_loss_execution(self)
            if execution.mode == "legacy":
                return self.loss_from_hidden_reference(
                    x, targets, loss_reduction
                )
            return tf_pkg.bounded_reference_loss(
                self.logits_from_hidden, x, targets, loss_reduction, execution
            )
        if loss_backend == "target_only":
            return self.answer_only_loss_from_hidden(x, targets, loss_reduction)
        if loss_reduction in {"none", "row_sum"}:
            tf_pkg.require_loss_backend(loss_backend, x.device)
            return self.answer_only_loss_from_hidden(x, targets, loss_reduction)
        return tf_pkg.liger_fused_linear_cross_entropy(
            self.output_weight(),
            x,
            targets,
            reduction=loss_reduction,
            softcap=self.config.final_logit_softcapping,
        )

    def output_weight(self) -> torch.Tensor:
        """Return the real-vocabulary projection weight without registering new model state."""
        if self.config.tie_embeddings:
            weight = self.embed_tokens.weight
        else:
            if self.lm_head is None:
                raise RuntimeError(
                    "untied transformer has no language-model head"
                )
            weight = self.lm_head.weight
        return weight[: self.config.vocab_size]

    def answer_only_loss_from_hidden(
        self, x: torch.Tensor, targets: torch.Tensor, loss_reduction: str
    ) -> torch.Tensor:
        """Compute loss without projecting prompt or padding positions through the vocabulary head."""
        return _answer_only_loss_from_hidden(
            self.logits_from_hidden,
            x,
            targets,
            loss_reduction,
            tie_embeddings=self.config.tie_embeddings,
            embed_weight=self.embed_tokens.weight,
            lm_head=self.lm_head,
        )

    def loss_from_hidden_reference(
        self, x: torch.Tensor, targets: torch.Tensor, loss_reduction: str
    ) -> torch.Tensor:
        """Eager loss oracle which projects every position, including ignored positions."""
        return _loss_from_hidden_reference(
            self.logits_from_hidden, x, targets, loss_reduction
        )

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
        del attention_mask
        if activation_checkpoint_windows < 0:
            raise ValueError(
                "activation_checkpoint_windows must be non-negative"
            )
        cache: CacheLike = CacheState(
            [],
            [],
            input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32),
        )
        hidden_chunks: list[torch.Tensor] = []
        checkpoint_active = (
            activation_checkpoint_windows > 0
            and self.training
            and torch.is_grad_enabled()
        )
        windows_per_segment = (
            activation_checkpoint_windows if checkpoint_active else 1
        )
        segment_span = windows_per_segment * process_every
        for segment_start in range(0, input_ids.shape[1], segment_span):
            segment_end = min(segment_start + segment_span, input_ids.shape[1])
            if checkpoint_active:
                if processor_execution_factory is None:
                    raise RuntimeError(
                        "checkpointed window segments require a processor execution factory"
                    )
                hidden, cache = self._checkpointed_window_segment(
                    input_ids[:, segment_start:segment_end],
                    segment_start,
                    cache,
                    processor_execution_factory=processor_execution_factory,
                    process_every=process_every,
                )
                hidden_chunks.append(hidden)
                continue
            for start in range(segment_start, segment_end, process_every):
                if start > 0:
                    cache = cast(CacheLike, processor(cache))
                end = min(start + process_every, input_ids.shape[1])
                hidden, cache = self.forward_chunk(
                    input_ids[:, start:end],
                    start,
                    cache,
                    logits=False,
                    cache_format="windowed",
                )
                hidden_chunks.append(hidden)
        return self.loss_from_hidden(
            torch.cat(hidden_chunks, dim=1),
            targets,
            loss_reduction,
            loss_backend,
        )

    def _checkpointed_window_segment(
        self,
        segment_input_ids: torch.Tensor,
        segment_start: int,
        cache: CacheLike,
        *,
        processor_execution_factory: ProcessorExecutionFactory,
        process_every: int,
    ) -> tuple[torch.Tensor, WindowedCache]:
        boundary_cache = cache if isinstance(cache, WindowedCache) else None
        input_tensors, input_layout = _flatten_windowed_cache(boundary_cache)
        output_layout: list[WindowedCacheLayout] = []

        def run_segment(
            ids: torch.Tensor, *flat_cache: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            if boundary_cache is None:
                current: CacheLike = CacheState(
                    [], [], ids.new_empty(ids.shape[0], 0, dtype=torch.float32)
                )
            else:
                current = _unflatten_windowed_cache(flat_cache, input_layout)
            segment_hidden: list[torch.Tensor] = []
            with processor_execution_factory() as segment_processor:
                for relative_start in range(0, ids.shape[1], process_every):
                    start = segment_start + relative_start
                    if start > 0:
                        current = cast(CacheLike, segment_processor(current))
                    hidden, current = self.forward_chunk(
                        ids[:, relative_start : relative_start + process_every],
                        start,
                        current,
                        logits=False,
                        cache_format="windowed",
                    )
                    segment_hidden.append(hidden)
            if not isinstance(current, WindowedCache):
                raise TypeError(
                    "checkpointed window segment requires WindowedCache output"
                )
            flat_output, layout = _flatten_windowed_cache(current)
            if cache_processor_recomputation_id() is None:
                output_layout[:] = [layout]
            return (torch.cat(segment_hidden, dim=1), *flat_output)

        outputs = checkpoint(
            run_segment,
            segment_input_ids,
            *input_tensors,
            use_reentrant=False,
            preserve_rng_state=True,
            context_fn=_window_checkpoint_contexts,
        )
        if not isinstance(outputs, tuple) or len(outputs) < 4:
            raise RuntimeError(
                "checkpointed window segment returned an invalid tensor boundary"
            )
        if len(output_layout) != 1:
            raise RuntimeError(
                "checkpointed window segment did not record its cache layout"
            )
        return outputs[0], _unflatten_windowed_cache(
            outputs[1:], output_layout[0]
        )

    def forward_windowed(
        self,
        input_ids: torch.Tensor,
        processor: ProcessorFn | None = None,
        process_every: int = 64,
    ) -> tuple[torch.Tensor, CacheLike]:
        cache: CacheLike = CacheState(
            [],
            [],
            input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32),
        )
        logits = []
        for start in range(0, input_ids.shape[1], process_every):
            if start > 0 and processor is not None:
                cache = cast(CacheLike, processor(cache))
            chunk_logits, cache = self.forward_chunk(
                input_ids[:, start : start + process_every], start, cache
            )
            logits.append(chunk_logits)
        return torch.cat(logits, dim=1), cache

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        cache: CacheLike | None = None,
        *,
        logits: bool = True,
        cache_format: str = "list",
    ) -> tuple[torch.Tensor, CacheLike]:
        return self.forward_embedding_chunk(
            self.embed_tokens(input_ids),
            start_pos,
            cache,
            logits=logits,
            cache_format=cache_format,
        )

    def forward_embedding_chunk(
        self,
        inputs_embeds: torch.Tensor,
        start_pos: int,
        cache: CacheLike | None = None,
        *,
        logits: bool = True,
        cache_format: str = "list",
    ) -> tuple[torch.Tensor, CacheLike]:
        if inputs_embeds.ndim != 3:
            raise ValueError(
                "inputs_embeds must have shape (batch, steps, hidden_size)"
            )
        if inputs_embeds.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"inputs_embeds width must be {self.config.hidden_size}, got {inputs_embeds.shape[-1]}"
            )
        if inputs_embeds.shape[1] == 0:
            raise ValueError(
                "inputs_embeds must contain at least one causal position"
            )
        if not inputs_embeds.is_floating_point():
            raise TypeError("inputs_embeds must be floating point")
        if inputs_embeds.device != self.embed_tokens.weight.device:
            raise ValueError("inputs_embeds must be on the transformer device")
        if (
            cache is not None
            and cache.scores.shape[0] != inputs_embeds.shape[0]
        ):
            raise ValueError("inputs_embeds and cache batch sizes must match")

        has_past = cache is not None and cache.scores.numel() > 0
        full_qwen3_window = (
            self.config.family == "qwen3" and inputs_embeds.shape[1] == 64
        )
        if (
            self.backbone_chunk_backend
            in {"compile_full_windows", "cuda_graph"}
            and full_qwen3_window
            and not has_past
            and start_pos == 0
        ):
            return self._forward_compiled_initial_embedding_chunk(
                inputs_embeds,
                start_pos,
                cache,
                logits=logits,
                cache_format=cache_format,
            )
        if (
            self.backbone_chunk_backend
            in {"compile", "compile_full_windows", "cuda_graph"}
            and full_qwen3_window
            and has_past
        ):
            return self._forward_compiled_embedding_chunk(
                inputs_embeds,
                start_pos,
                cache,
                logits=logits,
                cache_format=cache_format,
            )

        x = inputs_embeds
        cos_sin = self.rotary.get_range(
            start_pos,
            inputs_embeds.shape[1],
            inputs_embeds.device,
            _autocast_dtype(x.device) or x.dtype,
        )
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        layer_scores: list[torch.Tensor] = []
        has_cache = cache is not None and cache.scores.numel() > 0
        for i, layer in enumerate(self.layers):
            past = None if cache is None or not has_cache else cache.layer(i)
            x, k, v, scores = cast(Block, layer)(x, cos_sin, past)
            keys.append(k)
            values.append(v)
            layer_scores.append(scores)
        x = self.norm(x)
        out = self.logits_from_hidden(x) if logits else x
        chunk_scores = torch.stack(layer_scores).mean(dim=0)
        old_scores = (
            inputs_embeds.new_empty(
                inputs_embeds.shape[0], 0, dtype=torch.float32
            )
            if cache is None
            else cache.scores
        )
        new_scores = (
            chunk_scores
            if old_scores.numel() == 0
            else torch.cat([old_scores, chunk_scores], dim=1)
        )
        if cache_format == "windowed" or isinstance(cache, WindowedCache):
            return out, WindowedCache.from_lists(
                keys,
                values,
                new_scores,
                kv_score_sidecar=(
                    None if cache is None else cache.kv_score_sidecar
                ),
            )
        return out, CacheState(
            keys,
            values,
            new_scores,
            kv_score_sidecar=None if cache is None else cache.kv_score_sidecar,
        )

    def _forward_compiled_initial_embedding_chunk(
        self,
        inputs_embeds: torch.Tensor,
        start_pos: int,
        cache: CacheLike | None,
        *,
        logits: bool,
        cache_format: str,
    ) -> tuple[torch.Tensor, CacheLike]:
        if start_pos != 0:
            raise RuntimeError(
                "compiled initial Qwen3 chunks require start_pos=0"
            )
        cos, sin = self.rotary.get_range(
            start_pos,
            inputs_embeds.shape[1],
            inputs_embeds.device,
            _autocast_dtype(inputs_embeds.device) or inputs_embeds.dtype,
        )
        hidden, keys, values, chunk_scores = (
            run_qwen3_initial_backbone_chunk_region(
                self, self.backbone_chunk_backend, inputs_embeds, cos, sin
            )
        )
        out = self.logits_from_hidden(hidden) if logits else hidden
        if cache_format == "windowed" or isinstance(cache, WindowedCache):
            return out, WindowedCache(keys, values, chunk_scores)
        return out, CacheState(
            list(keys.unbind(0)), list(values.unbind(0)), chunk_scores
        )

    def _forward_compiled_embedding_chunk(
        self,
        inputs_embeds: torch.Tensor,
        start_pos: int,
        cache: CacheLike | None,
        *,
        logits: bool,
        cache_format: str,
    ) -> tuple[torch.Tensor, CacheLike]:
        cos, sin = self.rotary.get_range(
            start_pos,
            inputs_embeds.shape[1],
            inputs_embeds.device,
            _autocast_dtype(inputs_embeds.device) or inputs_embeds.dtype,
        )
        if cache is None or cache.scores.numel() == 0:
            raise RuntimeError(
                "compiled Qwen3 backbone chunks require non-empty past cache tensors"
            )
        if isinstance(cache, WindowedCache):
            past_keys, past_values = cache.keys, cache.values
        else:
            past_keys = torch.stack(cache.keys)
            past_values = torch.stack(cache.values)
        attention_bias, is_causal = _causal_chunk_bias(
            inputs_embeds.shape[1], past_keys.shape[3], inputs_embeds.device
        )
        if attention_bias is None or is_causal:
            raise RuntimeError(
                "compiled Qwen3 backbone chunks require a lower-right causal attention bias"
            )
        hidden, keys, values, chunk_scores = run_qwen3_backbone_chunk_region(
            self,
            self.backbone_chunk_backend,
            inputs_embeds,
            cos,
            sin,
            past_keys,
            past_values,
            attention_bias,
        )
        new_scores = torch.cat([cache.scores, chunk_scores], dim=1)
        out = self.logits_from_hidden(hidden) if logits else hidden
        if cache_format == "windowed" or isinstance(cache, WindowedCache):
            return out, WindowedCache(
                keys,
                values,
                new_scores,
                kv_score_sidecar=(
                    None if cache is None else cache.kv_score_sidecar
                ),
            )
        return out, CacheState(
            list(keys.unbind(0)),
            list(values.unbind(0)),
            new_scores,
            kv_score_sidecar=None if cache is None else cache.kv_score_sidecar,
        )

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
        for layer in self.layers:
            x, scores = cast(Block, layer).forward_static(
                x, cos_sin, cache, start_pos, end_pos
            )
            layer_scores.append(scores)
        x = self.norm(x)
        out = self.logits_from_hidden(x) if logits else x
        cache.write_scores(start_pos, torch.stack(layer_scores).mean(dim=0))
        return out, cache

    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        processor: ProcessorFn | None = None,
        process_every: int = 64,
        process_prefill_windows: bool = True,
        process_after_prefill: bool = False,
        relative_generation_cadence: bool = False,
        generation_process_every: int | None = None,
    ) -> torch.Tensor:
        return generate_tf(
            self,
            input_ids,
            max_new_tokens,
            eos_token_id=eos_token_id,
            processor=processor,
            process_every=process_every,
            generation_process_every=generation_process_every,
            process_prefill_windows=process_prefill_windows,
            process_after_prefill=process_after_prefill,
            relative_generation_cadence=relative_generation_cadence,
        )

    @torch.no_grad()
    def generate_fast(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        processor: ProcessorFn | None = None,
        process_every: int = 64,
        process_prefill_windows: bool = True,
        process_after_prefill: bool = False,
        relative_generation_cadence: bool = False,
        stop_sequences: list[list[int]] | None = None,
        pad_token_id: int | None = None,
        generation_process_every: int | None = None,
    ) -> torch.Tensor:
        return generate_fast_tf(
            self,
            input_ids,
            max_new_tokens,
            eos_token_id=eos_token_id,
            processor=processor,
            process_every=process_every,
            generation_process_every=generation_process_every,
            process_prefill_windows=process_prefill_windows,
            process_after_prefill=process_after_prefill,
            relative_generation_cadence=relative_generation_cadence,
            stop_sequences=stop_sequences,
            pad_token_id=pad_token_id,
        )

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not non_embedding:
            return total
        total -= self.embed_tokens.weight.numel()
        if self.lm_head is not None:
            total -= self.lm_head.weight.numel()
        return total
