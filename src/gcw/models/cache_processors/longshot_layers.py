from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import torch
import torch.nn as nn
import torch.nn.functional as F

from gcw.config import (
    CacheEncoderMethod,
    CacheEncoderQuerySource,
    CacheProcessorConfig,
)
from gcw.models.backbones.transformer import RMSNorm, flash_sdpa
from gcw.models.cache_processors.ops import zero_projection

EncoderMethod = CacheEncoderMethod
EncoderQuerySource = CacheEncoderQuerySource

_GRADIENT_CHECKPOINTING_OVERRIDE: ContextVar[bool | None] = ContextVar(
    "gcw_longshot_gradient_checkpointing_override",
    default=None,
)


@contextmanager
def longshot_gradient_checkpointing_override(enabled: bool) -> Iterator[None]:
    """Override Longshot processor-layer checkpointing for one forward context."""
    token = _GRADIENT_CHECKPOINTING_OVERRIDE.set(enabled)
    try:
        yield
    finally:
        _GRADIENT_CHECKPOINTING_OVERRIDE.reset(token)


def _gradient_checkpointing_enabled(configured: bool) -> bool:
    override = _GRADIENT_CHECKPOINTING_OVERRIDE.get()
    return configured if override is None else override


def _gather_cache_heads(
    x: torch.Tensor, selected_heads: torch.Tensor
) -> torch.Tensor:
    if selected_heads.ndim != 2:
        raise ValueError(
            "selected_heads must have shape (batch, selected_heads)"
        )
    if x.ndim != 3:
        raise ValueError("x must have shape (batch, cache_heads, hidden)")
    batch, _, dim = x.shape
    index = selected_heads[:, :, None].expand(-1, -1, dim)
    return torch.gather(x, dim=1, index=index)


def _zero_projection(projection: nn.Module) -> None:
    zero_projection(projection, per_head_type=PerCacheHeadLinear)


def _initialize_contextual_output_projection(
    projection: nn.Module,
    delta_dim: int,
    config: CacheProcessorConfig,
    std: float,
) -> None:
    if isinstance(projection, nn.Linear):
        delta_weight = projection.weight[:delta_dim]
        gate_weight = projection.weight[delta_dim:]
        delta_bias = (
            None if projection.bias is None else projection.bias[:delta_dim]
        )
        gate_bias = (
            None if projection.bias is None else projection.bias[delta_dim:]
        )
    elif isinstance(projection, PerCacheHeadLinear):
        delta_weight = projection.weight[..., :delta_dim]
        gate_weight = projection.weight[..., delta_dim:]
        delta_bias = (
            None
            if projection.bias is None
            else projection.bias[..., :delta_dim]
        )
        gate_bias = (
            None
            if projection.bias is None
            else projection.bias[..., delta_dim:]
        )
    else:
        raise TypeError(
            "contextual output projection must be nn.Linear or PerCacheHeadLinear"
        )
    if config.out_proj_init == "zero":
        nn.init.zeros_(delta_weight)
    else:
        nn.init.normal_(delta_weight, mean=0.0, std=std)
    if delta_bias is not None:
        nn.init.zeros_(delta_bias)
    nn.init.zeros_(gate_weight)
    if gate_bias is not None:
        nn.init.constant_(gate_bias, config.gate_init)


def _activate_update_gate(
    gate_logits: torch.Tensor, gate_mode: str, output_dtype: torch.dtype
) -> torch.Tensor:
    if gate_mode == "sigmoid":
        return torch.sigmoid(gate_logits.float()).to(output_dtype)
    if gate_mode == "direct":
        return gate_logits.to(output_dtype)
    raise ValueError(f"unsupported cache processor gate_mode: {gate_mode}")


def _fit_time_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    seq_len = x.shape[2]
    if seq_len == target_len:
        return x
    warnings.warn(
        f"Longshot linear encoder expected {target_len} selected positions but got {seq_len}; "
        "padding/truncating the selected cache history.",
        stacklevel=2,
    )
    if seq_len > target_len:
        return x[:, :, -target_len:, :]
    pad = target_len - seq_len
    return F.pad(x, (0, 0, pad, 0))


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
        self.internal_iteration_score = config.internal_iteration_score
        self.include_self_head_kv = config.include_self_head_kv

    def forward(
        self,
        x: torch.Tensor,
        selected_heads: torch.Tensor,
        return_attn_scores: bool = False,
        routing_gates: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if routing_gates is not None and routing_gates.shape != x.shape[:2]:
            raise ValueError(
                "routing_gates must have shape (batch, cache_heads)"
            )
        if self.include_self_head_kv:
            return self.forward_with_self_head_kv(
                x, selected_heads, return_attn_scores, routing_gates
            )
        return self.forward_selected_head_kv(
            x, selected_heads, return_attn_scores, routing_gates
        )

    def forward_with_iteration_scores(
        self,
        x: torch.Tensor,
        selected_heads: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.internal_iteration_score == "cache_head_attention":
            return self.forward_with_cache_head_attention_scores(
                x,
                selected_heads,
                include_self_head_kv=self.include_self_head_kv,
            )
        result = self.forward(x, selected_heads, return_attn_scores=True)
        if not isinstance(result, tuple):
            raise TypeError("iteration scoring requires attention scores")
        return result

    def forward_selected_head_kv(
        self,
        x: torch.Tensor,
        selected_heads: torch.Tensor,
        return_attn_scores: bool = False,
        routing_gates: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, cache_heads, _ = x.shape
        if routing_gates is not None and return_attn_scores:
            raise ValueError(
                "routing gates are not supported with iterative attention scores"
            )
        score_kv_norm = (
            return_attn_scores and self.internal_iteration_score == "kv_norm"
        )
        assert (
            selected_heads.shape[-1] > 0
        ), "selected_heads must include at least one cache head"
        q = self.q_proj(x).view(
            batch, cache_heads, self.num_heads, self.head_dim
        )
        if score_kv_norm:
            k_all = self.k_proj(x).view(
                batch, cache_heads, self.num_kv_heads, self.head_dim
            )
            v_all = self.v_proj(x).view(
                batch, cache_heads, self.num_kv_heads, self.head_dim
            )
            iteration_scores = k_all.float().pow(2).sum(
                dim=(2, 3)
            ) + v_all.float().pow(2).sum(dim=(2, 3))
            gather_index = selected_heads[:, :, None, None].expand(
                -1, -1, self.num_kv_heads, self.head_dim
            )
            k = torch.gather(k_all, dim=1, index=gather_index)
            v = torch.gather(v_all, dim=1, index=gather_index)
        else:
            selected_x = (
                x
                if routing_gates is not None
                else _gather_cache_heads(x, selected_heads)
            )
            k = self.k_proj(selected_x).view(
                batch,
                selected_x.shape[1],
                self.num_kv_heads,
                self.head_dim,
            )
            v = self.v_proj(selected_x).view(
                batch,
                selected_x.shape[1],
                self.num_kv_heads,
                self.head_dim,
            )
        q = q.permute(0, 2, 1, 3)
        k, v = (t.permute(0, 2, 1, 3) for t in (k, v))
        q, k = q.to(v.dtype), k.to(v.dtype)
        y = flash_sdpa(
            q,
            k,
            v,
            attn_mask=(
                None
                if routing_gates is None
                else routing_gates[:, None, None, :].to(v.dtype)
            ),
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.num_kv_heads != self.num_heads,
            try_flash=routing_gates is None,
        )
        y = y.permute(0, 2, 1, 3).contiguous().view(batch, cache_heads, -1)
        y = self.o_proj(y)
        if return_attn_scores:
            if score_kv_norm:
                return y, iteration_scores
            return y, y.float().pow(2).sum(dim=-1)
        return y

    def forward_with_self_head_kv(
        self,
        x: torch.Tensor,
        selected_heads: torch.Tensor,
        return_attn_scores: bool = False,
        routing_gates: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, cache_heads, _ = x.shape
        selected_count = selected_heads.shape[-1]
        assert (
            selected_count > 0
        ), "selected_heads must include at least one cache head"
        groups = self.num_heads // self.num_kv_heads

        q = self.q_proj(x).view(
            batch, cache_heads, self.num_kv_heads, groups, self.head_dim
        )
        k_all = self.k_proj(x).view(
            batch, cache_heads, self.num_kv_heads, self.head_dim
        )
        v_all = self.v_proj(x).view(
            batch, cache_heads, self.num_kv_heads, self.head_dim
        )
        if routing_gates is not None:
            if return_attn_scores:
                raise ValueError(
                    "routing gates are not supported with iterative attention scores"
                )
            q, k_all = q.to(v_all.dtype), k_all.to(v_all.dtype)
            logits = (
                torch.einsum("bckgd,bskd->bckgs", q, k_all)
                * self.head_dim**-0.5
            ).float()
            diagonal = torch.eye(
                cache_heads, dtype=torch.bool, device=x.device
            )[None, :, None, None, :]
            attn = torch.softmax(
                torch.where(
                    diagonal,
                    logits,
                    logits + routing_gates[:, None, None, None, :],
                ),
                dim=-1,
            ).to(v_all.dtype)
            if self.training and self.dropout > 0.0:
                attn = F.dropout(attn, p=self.dropout, training=True)
            y = torch.einsum("bckgs,bskd->bckgd", attn, v_all)
            y = y.contiguous().view(
                batch, cache_heads, self.num_heads * self.head_dim
            )
            return self.o_proj(y)
        if return_attn_scores and self.internal_iteration_score == "kv_norm":
            iteration_scores = k_all.float().pow(2).sum(
                dim=(2, 3)
            ) + v_all.float().pow(2).sum(dim=(2, 3))
        gather_index = selected_heads[:, :, None, None].expand(
            -1, -1, self.num_kv_heads, self.head_dim
        )
        k_sel = torch.gather(k_all, dim=1, index=gather_index)
        v_sel = torch.gather(v_all, dim=1, index=gather_index)

        q, k_sel, k_all = (
            q.to(v_all.dtype),
            k_sel.to(v_all.dtype),
            k_all.to(v_all.dtype),
        )
        scale = self.head_dim**-0.5
        selected_logits = torch.einsum("bckgd,bskd->bckgs", q, k_sel) * scale
        self_logits = (
            torch.einsum("bckgd,bckd->bckg", q, k_all)[:, :, :, :, None] * scale
        )
        query_heads = torch.arange(cache_heads, device=x.device)
        self_is_selected = (
            selected_heads[:, None, :] == query_heads[None, :, None]
        ).any(dim=-1)
        self_logits = self_logits.masked_fill(
            self_is_selected[:, :, None, None, None], float("-inf")
        )

        logits = torch.cat([selected_logits, self_logits], dim=-1)
        attn = torch.softmax(logits, dim=-1)
        if self.training and self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout, training=True)

        selected_attn, self_attn = (
            attn[..., :selected_count],
            attn[..., selected_count],
        )
        y = torch.einsum("bckgs,bskd->bckgd", selected_attn, v_sel)
        y = y + self_attn[:, :, :, :, None] * v_all[:, :, :, None, :]
        y = y.contiguous().view(
            batch, cache_heads, self.num_heads * self.head_dim
        )
        y = self.o_proj(y)
        if return_attn_scores:
            if self.internal_iteration_score == "kv_norm":
                return y, iteration_scores
            return y, y.float().pow(2).sum(dim=-1)
        return y

    def forward_with_cache_head_attention_scores(
        self,
        x: torch.Tensor,
        selected_heads: torch.Tensor,
        *,
        include_self_head_kv: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, cache_heads, _ = x.shape
        selected_count = selected_heads.shape[-1]
        assert (
            selected_count > 0
        ), "selected_heads must include at least one cache head"
        groups = self.num_heads // self.num_kv_heads

        q = self.q_proj(x).view(
            batch, cache_heads, self.num_kv_heads, groups, self.head_dim
        )
        k_all = self.k_proj(x).view(
            batch, cache_heads, self.num_kv_heads, self.head_dim
        )
        v_all = self.v_proj(x).view(
            batch, cache_heads, self.num_kv_heads, self.head_dim
        )

        q, k_all = q.to(v_all.dtype), k_all.to(v_all.dtype)
        scale = self.head_dim**-0.5
        logits = torch.einsum("bckgd,bskd->bckgs", q, k_all) * scale
        all_attn = torch.softmax(logits.float(), dim=-1)
        cache_head_attention_scores = all_attn.float().sum(dim=(1, 2, 3))

        logit_index = selected_heads[:, None, None, None, :].expand(
            -1,
            cache_heads,
            self.num_kv_heads,
            groups,
            -1,
        )
        selected_logits = torch.gather(logits, dim=-1, index=logit_index)
        value_index = selected_heads[:, :, None, None].expand(
            -1, -1, self.num_kv_heads, self.head_dim
        )
        v_sel = torch.gather(v_all, dim=1, index=value_index)

        if include_self_head_kv:
            query_heads = torch.arange(cache_heads, device=x.device)
            self_index = query_heads[None, :, None, None, None].expand(
                batch,
                -1,
                self.num_kv_heads,
                groups,
                1,
            )
            self_logits = torch.gather(logits, dim=-1, index=self_index)
            self_is_selected = (
                selected_heads[:, None, :] == query_heads[None, :, None]
            ).any(dim=-1)
            self_logits = self_logits.masked_fill(
                self_is_selected[:, :, None, None, None], float("-inf")
            )
            output_logits = torch.cat([selected_logits, self_logits], dim=-1)
        else:
            output_logits = selected_logits

        output_attn = torch.softmax(output_logits, dim=-1)
        if self.training and self.dropout > 0.0:
            output_attn = F.dropout(output_attn, p=self.dropout, training=True)

        selected_attn = output_attn[..., :selected_count]
        y = torch.einsum("bckgs,bskd->bckgd", selected_attn, v_sel)
        if include_self_head_kv:
            self_attn = output_attn[..., selected_count]
            y = y + self_attn[:, :, :, :, None] * v_all[:, :, :, None, :]
        y = y.contiguous().view(
            batch, cache_heads, self.num_heads * self.head_dim
        )
        return self.o_proj(y), cache_head_attention_scores


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

    def forward(
        self,
        x: torch.Tensor,
        selected_heads: torch.Tensor,
        return_attn_scores: bool = False,
        routing_gates: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_attn_scores and routing_gates is not None:
            raise ValueError(
                "routing gates are not supported with iterative attention scores"
            )
        h = self.input_norm(x)
        if return_attn_scores:
            attn_out, attn_scores = self.attn.forward_with_iteration_scores(
                h, selected_heads
            )
        else:
            result = self.attn(h, selected_heads, routing_gates=routing_gates)
            if not isinstance(result, torch.Tensor):
                raise TypeError(
                    "attention returned scores when none were requested"
                )
            attn_out = result
        x = x + attn_out
        h = self.post_norm(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        if return_attn_scores:
            return x, attn_scores
        return x


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.einsum("bci,cio->bco", x, self.weight)
        if self.bias is not None:
            y = y + self.bias[None, :, :]
        return y


class QueryAttentionTimeEncoder(nn.Module):
    def __init__(
        self, input_dim: int, num_cache_heads: int, config: CacheProcessorConfig
    ):
        super().__init__()
        self.input_dim = input_dim
        self.encoder_query_source = config.encoder_query_source
        if self.encoder_query_source == "learned":
            self.query = nn.Parameter(torch.empty(num_cache_heads, input_dim))
        elif self.encoder_query_source == "final_token":
            self.query_proj: nn.Module
            if config.per_cache_head_projections:
                self.query_proj = PerCacheHeadLinear(
                    num_cache_heads, input_dim, input_dim
                )
            else:
                self.query_proj = nn.Linear(input_dim, input_dim)
        else:
            raise ValueError(
                f"unknown encoder_query_source: {self.encoder_query_source}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self._query(x)
        scale = self.input_dim**-0.5
        scores = torch.einsum("bcsi,bci->bcs", x, query.to(x.dtype)) * scale
        weights = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        return torch.einsum("bcs,bcsi->bci", weights, x)

    def _query(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_query_source == "learned":
            return self.query[None, :, :].expand(x.shape[0], -1, -1)
        if self.encoder_query_source == "final_token":
            return self.query_proj(x[:, :, -1, :])
        raise ValueError(
            f"unsupported encoder_query_source: {self.encoder_query_source}"
        )
