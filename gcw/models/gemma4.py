from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcw.config import ModelConfig
from gcw.models.cache_state import CacheState, StaticCacheState
from gcw.models.transformer import Linear, TF, _autocast_dtype, flash_sdpa
from gcw.models.transformer import _causal_chunk_bias, _matches_stop_sequence
from gcw.models.windowed_cache import WindowedCache

try:
    from transformers.activations import ACT2FN as _HF_ACT2FN
except Exception:
    _HF_ACT2FN = None


def _act_fn(name: str):
    if _HF_ACT2FN is not None and name in _HF_ACT2FN:
        return _HF_ACT2FN[name]
    if name == "gelu_pytorch_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "silu":
        return F.silu
    raise ImportError(
        f"Gemma4 activation {name!r} requires transformers.activations.ACT2FN"
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return (x * cos[None, :, None, :]) + (
        _rotate_half(x) * sin[None, :, None, :]
    )


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.pow(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps, -0.5
        )
        if self.with_scale:
            y = y * self.weight.float()
        return y.to(x.dtype)


class Gemma4ScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None,
        embed_scale: float,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer(
            "embed_scale", torch.tensor(embed_scale), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(
            self.weight.dtype
        )


class Gemma4RotaryCache(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.rope_parameters = config.rope_parameters or {}
        for layer_type in set(config.layer_types or ["full_attention"]):
            head_dim = self._head_dim(layer_type)
            inv_freq = self._inv_freq(layer_type, head_dim, torch.device("cpu"))
            self.register_buffer(
                f"{layer_type}_inv_freq", inv_freq, persistent=False
            )

    def get(
        self,
        positions: torch.Tensor,
        layer_type: str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq").to(positions.device)
        freqs = positions.float().unsqueeze(-1) * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _head_dim(self, layer_type: str) -> int:
        if layer_type == "full_attention":
            return int(self.config.global_head_dim or self.config.head_dim)
        return int(self.config.head_dim)

    def _inv_freq(
        self, layer_type: str, head_dim: int, device: torch.device
    ) -> torch.Tensor:
        params = self.rope_parameters.get(layer_type, {})
        base = float(params.get("rope_theta", 10000.0))
        rope_type = params.get("rope_type", "default")
        if rope_type == "proportional":
            proportion = float(params.get("partial_rotary_factor", 1.0))
            rope_angles = int(proportion * head_dim // 2)
            if rope_angles > 0:
                inv_rotated = 1.0 / (
                    base
                    ** (
                        torch.arange(
                            0,
                            2 * rope_angles,
                            2,
                            dtype=torch.float32,
                            device=device,
                        )
                        / float(head_dim)
                    )
                )
            else:
                inv_rotated = torch.empty(0, dtype=torch.float32, device=device)
            nope_angles = head_dim // 2 - rope_angles
            if nope_angles > 0:
                inv_freq = torch.cat(
                    [
                        inv_rotated,
                        torch.zeros(
                            nope_angles, dtype=torch.float32, device=device
                        ),
                    ]
                )
            else:
                inv_freq = inv_rotated
        elif rope_type == "default":
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(
                        0, head_dim, 2, dtype=torch.float32, device=device
                    )
                    / float(head_dim)
                )
            )
        else:
            raise NotImplementedError(
                f"Gemma4 rope_type={rope_type!r} is not implemented"
            )
        return inv_freq / float(params.get("factor", 1.0))


def _sliding_attention_mask(
    *,
    steps: int,
    total: int,
    start_pos: int,
    sliding_window: int | None,
    device: torch.device,
) -> torch.Tensor | None:
    if sliding_window is None:
        raise ValueError("sliding_attention requires sliding_window")
    q_pos = torch.arange(start_pos, start_pos + steps, device=device)[:, None]
    kv_pos = torch.arange(total, device=device)[None, :]
    keep = (kv_pos <= q_pos) & ((q_pos - kv_pos) < sliding_window)
    if bool(keep.all()):
        return None
    mask = torch.zeros(steps, total, dtype=torch.float32, device=device)
    return mask.masked_fill(~keep, torch.finfo(torch.float32).min)[
        None, None, :, :
    ]


def _gemma4_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    is_causal: bool,
    groups: int,
    scale: float,
    try_flash: bool,
) -> torch.Tensor:
    y = flash_sdpa(
        q,
        k,
        v,
        attn_mask=mask,
        is_causal=is_causal,
        enable_gqa=groups != 1,
        scale=scale,
        try_flash=try_flash,
    )
    return y.transpose(1, 2).contiguous()


class Gemma4Attention(nn.Module):
    def __init__(
        self, config: ModelConfig, layer_idx: int, cache_idx: int | None
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.cache_idx = cache_idx
        self.layer_type = (
            config.layer_types[layer_idx]
            if config.layer_types is not None
            else "full_attention"
        )
        self.is_sliding = self.layer_type == "sliding_attention"
        first_shared = config.num_layers - config.num_kv_shared_layers
        self.is_kv_shared_layer = layer_idx >= first_shared
        prev_layers = (config.layer_types or [])[:first_shared]
        self.store_full_length_kv = False
        if not self.is_kv_shared_layer and self.layer_type in prev_layers:
            self.store_full_length_kv = layer_idx == len(
                prev_layers
            ) - 1 - prev_layers[::-1].index(self.layer_type)
        self.num_heads = config.num_heads
        self.num_kv_heads = config.kv_heads_for_layer(layer_idx)
        self.head_dim = config.head_dim_for_layer(layer_idx)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = 1.0
        self.use_alternative_attention = (
            config.attention_k_eq_v and not self.is_sliding
        )
        self.q_proj = Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.q_norm = Gemma4RMSNorm(self.head_dim, config.norm_eps)
        if not self.is_kv_shared_layer:
            self.k_proj = Linear(
                config.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=False,
            )
            self.v_proj = (
                None
                if self.use_alternative_attention
                else Linear(
                    config.hidden_size,
                    self.num_kv_heads * self.head_dim,
                    bias=False,
                )
            )
            self.k_norm = Gemma4RMSNorm(self.head_dim, config.norm_eps)
            self.v_norm = Gemma4RMSNorm(
                self.head_dim, config.norm_eps, with_scale=False
            )
        self.o_proj = Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        past: tuple[torch.Tensor, torch.Tensor] | None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        start_pos: int,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor
    ]:
        batch, steps, _ = x.shape
        q = self.q_proj(x).view(batch, steps, self.num_heads, self.head_dim)
        q = _apply_rotary_pos_emb(self.q_norm(q), *cos_sin).transpose(1, 2)
        if self.is_kv_shared_layer:
            k, v = shared_kv_states[self.layer_type]
        else:
            k = self.k_proj(x).view(
                batch, steps, self.num_kv_heads, self.head_dim
            )
            v = (
                k
                if self.v_proj is None
                else self.v_proj(x).view(
                    batch, steps, self.num_kv_heads, self.head_dim
                )
            )
            k = _apply_rotary_pos_emb(self.k_norm(k), *cos_sin).transpose(1, 2)
            v = self.v_norm(v).transpose(1, 2)
            if past is not None:
                pk, pv = past
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            if self.store_full_length_kv:
                shared_kv_states[self.layer_type] = (k, v)
        mask, is_causal, try_flash = self._attention_bias(
            steps, k.shape[2], start_pos, x.device
        )
        y = _gemma4_attention(
            q,
            k,
            v,
            mask,
            is_causal,
            self.num_key_value_groups,
            self.scaling,
            try_flash,
        )
        y = y.reshape(batch, steps, self.num_heads * self.head_dim)
        out = self.o_proj(y)
        return (
            out,
            None if self.is_kv_shared_layer else k,
            None if self.is_kv_shared_layer else v,
            out.float().norm(dim=-1),
        )

    def forward_static(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cache: StaticCacheState,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        start_pos: int,
        end_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = x.shape
        q = self.q_proj(x).view(batch, steps, self.num_heads, self.head_dim)
        q = _apply_rotary_pos_emb(self.q_norm(q), *cos_sin).transpose(1, 2)
        if self.is_kv_shared_layer:
            k, v = shared_kv_states[self.layer_type]
        else:
            if self.cache_idx is None:
                raise ValueError(
                    "non-shared Gemma4 attention layers require a cache index"
                )
            current_k = self.k_proj(x).view(
                batch, steps, self.num_kv_heads, self.head_dim
            )
            current_v = (
                current_k
                if self.v_proj is None
                else self.v_proj(x).view(
                    batch, steps, self.num_kv_heads, self.head_dim
                )
            )
            current_k = _apply_rotary_pos_emb(
                self.k_norm(current_k), *cos_sin
            ).transpose(1, 2)
            current_v = self.v_norm(current_v).transpose(1, 2)
            cache.write_layer(self.cache_idx, start_pos, current_k, current_v)
            k, v = cache.layer_view(self.cache_idx, end_pos)
            if self.store_full_length_kv:
                shared_kv_states[self.layer_type] = (k, v)
        mask, is_causal, try_flash = self._attention_bias(
            steps, k.shape[2], start_pos, x.device
        )
        y = _gemma4_attention(
            q,
            k,
            v,
            mask,
            is_causal,
            self.num_key_value_groups,
            self.scaling,
            try_flash,
        )
        y = y.reshape(batch, steps, self.num_heads * self.head_dim)
        out = self.o_proj(y)
        return out, out.float().norm(dim=-1)

    def _attention_bias(
        self,
        steps: int,
        total: int,
        start_pos: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, bool, bool]:
        if not self.is_sliding:
            mask, is_causal = _causal_chunk_bias(steps, total - steps, device)
            return mask, is_causal, True
        mask = _sliding_attention_mask(
            steps=steps,
            total=total,
            start_pos=start_pos,
            sliding_window=self.config.sliding_window,
            device=device,
        )
        return mask, False, mask is None


class Gemma4MLP(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        first_shared = config.num_layers - config.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_shared > 0
        intermediate = config.intermediate_size * (
            2 if config.use_double_wide_mlp and is_kv_shared_layer else 1
        )
        self.gate_proj = Linear(config.hidden_size, intermediate, bias=False)
        self.up_proj = Linear(config.hidden_size, intermediate, bias=False)
        self.down_proj = Linear(intermediate, config.hidden_size, bias=False)
        self.act = _act_fn(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class Gemma4Block(nn.Module):
    def __init__(
        self, config: ModelConfig, layer_idx: int, cache_idx: int | None
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = (
            config.layer_types[layer_idx]
            if config.layer_types is not None
            else "full_attention"
        )
        self.self_attn = Gemma4Attention(config, layer_idx, cache_idx)
        self.mlp = Gemma4MLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.norm_eps
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.norm_eps
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.norm_eps
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.norm_eps
        )
        self.register_buffer("layer_scalar", torch.ones(1))
        self.hidden_size_per_layer_input = int(
            config.hidden_size_per_layer_input
        )
        if self.hidden_size_per_layer_input:
            self.act = _act_fn(config.hidden_act)
            self.per_layer_input_gate = Linear(
                config.hidden_size, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = Linear(
                self.hidden_size_per_layer_input, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                config.hidden_size, config.norm_eps
            )

    def forward(
        self,
        x: torch.Tensor,
        per_layer_input: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        past: tuple[torch.Tensor, torch.Tensor] | None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        start_pos: int,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor
    ]:
        residual = x
        attn_out, k, v, scores = self.self_attn(
            self.input_layernorm(x),
            cos_sin,
            past,
            shared_kv_states,
            start_pos,
        )
        x = residual + self.post_attention_layernorm(attn_out)
        residual = x
        x = residual + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x))
        )
        if self.hidden_size_per_layer_input:
            if per_layer_input is None:
                raise ValueError("Gemma4 PLE layer requires per_layer_input")
            residual = x
            x = self.act(self.per_layer_input_gate(x)) * per_layer_input
            x = residual + self.post_per_layer_input_norm(
                self.per_layer_projection(x)
            )
        return x * self.layer_scalar.to(x.dtype), k, v, scores

    def forward_static(
        self,
        x: torch.Tensor,
        per_layer_input: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cache: StaticCacheState,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        start_pos: int,
        end_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        attn_out, scores = self.self_attn.forward_static(
            self.input_layernorm(x),
            cos_sin,
            cache,
            shared_kv_states,
            start_pos,
            end_pos,
        )
        x = residual + self.post_attention_layernorm(attn_out)
        residual = x
        x = residual + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x))
        )
        if self.hidden_size_per_layer_input:
            if per_layer_input is None:
                raise ValueError("Gemma4 PLE layer requires per_layer_input")
            residual = x
            x = self.act(self.per_layer_input_gate(x)) * per_layer_input
            x = residual + self.post_per_layer_input_norm(
                self.per_layer_projection(x)
            )
        return x * self.layer_scalar.to(x.dtype), scores


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
        self.norm = Gemma4RMSNorm(config.hidden_size, config.norm_eps)
        self.rotary = Gemma4RotaryCache(config)
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
        return {
            layer_type: self.rotary.get(positions, layer_type, dtype)
            for layer_type in set(self.config.layer_types or ["full_attention"])
        }

    def forward_full(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_reduction: str = "mean",
        attention_mask: torch.Tensor | None = None,
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
        for i, layer in enumerate(self.layers):
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
            return self.loss_from_hidden(x, targets, loss_reduction)
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
        for i, layer in enumerate(self.layers):
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
                past = None if not has_cache else cache.layer(cache_idx)
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
        for i, layer in enumerate(self.layers):
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
    def generate_fast(
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
        cache = CacheState(
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
