from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, cast

import torch
import torch.nn as nn

from gcw.models.backbones.transformer import RMSNorm
from gcw.config.training import LongshotSelectorTrainingConfig
from gcw.models.cache_processors.base import cache_processor_recomputation_id
from gcw.models.cache_processors.ops import (
    expand_cache_head_indices,
    gather_per_cache_head_kv,
    select_per_cache_head_kv_norm_indices_from_scores,
)
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.performance.kv_score_sidecar import (
    kv_score_sidecar_execution,
    kv_score_sidecar_spec,
    refresh_kv_score_sidecar,
    synchronize_kv_score_sidecar,
)
from gcw.performance.longshot_cache import (
    gather_windowed_per_cache_head_kv,
    longshot_full_history_execution,
    record_longshot_full_history_event,
)
from gcw.performance.longshot_scores import (
    chunked_fp32_concatenated_kv_squared_l2,
    chunked_fp32_squared_l2,
    chunked_fp32_unroped_concatenated_kv_squared_l2,
)
from gcw.performance.triton.execution import triton_cache_movement_enabled

# The cache object any selection path accepts: static, mutable, or windowed.
CacheLike = CacheState | StaticCacheState | WindowedCache


SelectorRoutingMode = Literal["warmup", "soft", "transition", "hard"]


@dataclass(frozen=True)
class SelectorRouting:
    mode: SelectorRoutingMode
    temperature: float | torch.Tensor
    hard_blend: float | torch.Tensor


def selector_routing(
    config: LongshotSelectorTrainingConfig | None,
    progress: float,
    *,
    training: bool,
) -> SelectorRouting:
    if not training or config is None or progress >= config.hard_start:
        return SelectorRouting(
            "hard", config.final_temperature if config else 0.0, 1.0
        )
    if progress < config.warmup_end:
        return SelectorRouting("warmup", config.initial_temperature, 0.0)
    if progress <= config.soft_end:
        return SelectorRouting("soft", config.initial_temperature, 0.0)
    if progress < config.second_hardening_start:
        fraction = (progress - config.soft_end) / (
            config.second_hardening_start - config.soft_end
        )
        cosine = 0.5 - 0.5 * math.cos(math.pi * fraction)
        temperature = config.initial_temperature + cosine * (
            config.intermediate_temperature - config.initial_temperature
        )
        suppression = config.initial_suppression + fraction * (
            config.intermediate_suppression - config.initial_suppression
        )
    else:
        fraction = (progress - config.second_hardening_start) / (
            config.hard_start - config.second_hardening_start
        )
        cosine = 0.5 - 0.5 * math.cos(math.pi * fraction)
        temperature = config.intermediate_temperature + cosine * (
            config.final_temperature - config.intermediate_temperature
        )
        suppression = config.intermediate_suppression + fraction * (
            config.final_suppression - config.intermediate_suppression
        )
    return SelectorRouting(
        "transition", temperature, 1.0 - math.exp(-suppression)
    )


def _soft_topk_mask(
    scores: torch.Tensor,
    topk_heads: int,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("soft top-k scores must have shape (batch, heads)")
    if topk_heads <= 0:
        raise ValueError("topk_heads must select at least one cache head")
    if not isinstance(temperature, torch.Tensor):
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                "soft top-k temperature must be finite and positive"
            )
    scores = scores.float()
    heads = scores.shape[-1]
    if topk_heads >= heads:
        return torch.ones_like(scores)

    with torch.no_grad():
        span = temperature * (32.0 + math.log(max(heads, 2)))
        lower = scores.amin(dim=-1) - span
        upper = scores.amax(dim=-1) + span
        for _ in range(4):
            lower_mass = torch.sigmoid(
                (scores - lower[:, None]) / temperature
            ).sum(dim=-1)
            upper_mass = torch.sigmoid(
                (scores - upper[:, None]) / temperature
            ).sum(dim=-1)
            width = upper - lower
            lower = torch.where(lower_mass > topk_heads, lower, lower - width)
            upper = torch.where(upper_mass < topk_heads, upper, upper + width)
        for _ in range(26):
            midpoint = (lower + upper) * 0.5
            mass = torch.sigmoid(
                (scores - midpoint[:, None]) / temperature
            ).sum(dim=-1)
            lower = torch.where(mass > topk_heads, midpoint, lower)
            upper = torch.where(mass > topk_heads, upper, midpoint)
        threshold = (lower + upper) * 0.5
        detached_mask = torch.sigmoid(
            (scores - threshold[:, None]) / temperature
        )
        derivatives = detached_mask * (1.0 - detached_mask)
        implicit_weights = derivatives / derivatives.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

    threshold = threshold + ((scores - scores.detach()) * implicit_weights).sum(
        dim=-1
    )
    return torch.sigmoid((scores - threshold[:, None]) / temperature)


class ContextualHeadSelector(nn.Module):
    """Score encoded cache heads against their shared encoded context."""

    def __init__(
        self,
        hidden_size: int,
        selector_hidden_size: int,
        num_cache_heads: int,
        *,
        norm_eps: float,
        use_identity_bias: bool,
        kv_cache_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.num_cache_heads = num_cache_heads
        self.kv_cache_heads = kv_cache_heads or num_cache_heads
        self.scale = selector_hidden_size**-0.5
        self.norm = RMSNorm(hidden_size, norm_eps)
        self.context_proj = nn.Linear(
            hidden_size, selector_hidden_size, bias=False
        )
        self.candidate_proj = nn.Linear(
            hidden_size, selector_hidden_size, bias=False
        )
        if use_identity_bias:
            self.identity_bias = nn.Parameter(torch.zeros(num_cache_heads))
        else:
            self.register_parameter("identity_bias", None)
        self._training_enabled = False
        self._routing_mode: SelectorRoutingMode = "hard"
        self.register_buffer(
            "_temperature", torch.zeros((), device="cpu"), persistent=False
        )
        self.register_buffer(
            "_hard_blend", torch.ones((), device="cpu"), persistent=False
        )
        self._metric_totals: dict[str, torch.Tensor] = {}
        self._metric_counts: dict[str, int] = {}

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(
                "contextual head selector hidden must have shape "
                "(batch, active_heads, hidden_size)"
            )
        normalized = self.norm(hidden)
        context_hidden = self.context_proj(normalized.mean(dim=1)).float()
        candidate_hidden = self.candidate_proj(normalized).float()
        with torch.autocast(hidden.device.type, enabled=False):
            scores = (
                torch.einsum("bd,bhd->bh", context_hidden, candidate_hidden)
                * self.scale
            )
        if self.identity_bias is not None:
            scores = scores + self.identity_bias.float()
        return scores

    def set_training_progress(
        self, config: LongshotSelectorTrainingConfig, progress: float
    ) -> None:
        if not 0.0 <= progress <= 1.0:
            raise ValueError("selector training progress must be within [0, 1]")
        self._training_enabled = True
        routing = selector_routing(config, progress, training=True)
        self._routing_mode = routing.mode
        cast(torch.Tensor, self._temperature).fill_(routing.temperature)
        cast(torch.Tensor, self._hard_blend).fill_(routing.hard_blend)

    def route(
        self, hidden: torch.Tensor, topk_heads: int, metric_scope: str
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        routing = SelectorRouting("hard", 0.0, 1.0)
        if self.training and self._training_enabled:
            routing = SelectorRouting(
                self._routing_mode,
                cast(torch.Tensor, self._temperature),
                cast(torch.Tensor, self._hard_blend),
            )
        if routing.mode == "warmup":
            selected = torch.arange(
                hidden.shape[1], dtype=torch.long, device=hidden.device
            ).expand(hidden.shape[0], -1)
            self._record_metrics(metric_scope, routing, None, None, selected)
            return selected, None
        scores = self(hidden)
        selected = _select_heads_from_head_scores(scores, topk_heads)
        gates = None
        if routing.mode != "hard":
            soft_gates = _soft_topk_mask(
                scores, topk_heads, routing.temperature
            )
            hard_mask = torch.zeros_like(soft_gates).scatter(1, selected, 1.0)
            gates = (
                1.0 - routing.hard_blend
            ) * soft_gates + routing.hard_blend * hard_mask
        self._record_metrics(metric_scope, routing, scores, gates, selected)
        if gates is None:
            return selected, None
        return selected, torch.log(gates.float() + 1e-6)

    def pop_metrics(self) -> dict[str, float]:
        metrics = {
            name: float((total / self._metric_counts[name]).cpu())
            for name, total in self._metric_totals.items()
        }
        self._metric_totals.clear()
        self._metric_counts.clear()
        return metrics

    def _record_metrics(
        self,
        scope: str,
        routing: SelectorRouting,
        scores: torch.Tensor | None,
        gates: torch.Tensor | None,
        selected_heads: torch.Tensor,
    ) -> None:
        if cache_processor_recomputation_id() is not None or not self.training:
            return
        head_count = (
            scores.shape[-1] if scores is not None else selected_heads.shape[-1]
        )
        hard_mask = torch.zeros(
            selected_heads.shape[0],
            head_count,
            dtype=torch.float32,
            device=selected_heads.device,
        ).scatter(1, selected_heads, 1.0)
        measured = hard_mask if gates is None else gates.float()
        selection_rate = hard_mask.mean(dim=0)
        starved = (selection_rate == 0.0).float().mean()
        prefix = f"cache_processor/selector/{scope}"
        values = {
            f"{prefix}/temperature": torch.as_tensor(
                routing.temperature, device=measured.device
            ),
            f"{prefix}/hard_blend": torch.as_tensor(
                routing.hard_blend, device=measured.device
            ),
            f"{prefix}/mask_mass": measured.sum(dim=-1).mean(),
            f"{prefix}/mask_entropy": (
                -(
                    measured.clamp_min(1e-6).log() * measured
                    + (1.0 - measured) * (1.0 - measured).clamp_min(1e-6).log()
                ).mean()
            ),
            f"{prefix}/distinct_selected_heads": hard_mask.amax(dim=0).sum(),
            f"{prefix}/max_selection_rate": selection_rate.amax(),
            f"{prefix}/starved_head_fraction": starved,
            f"{prefix}/kv_routing_fraction": (
                measured[:, : min(self.kv_cache_heads, head_count)].sum()
                / measured.sum().clamp_min(1e-12)
            ),
        }
        if scores is not None:
            scores = scores.float()
            values[f"{prefix}/score_mean"] = scores.mean()
            values[f"{prefix}/score_std"] = scores.std(unbiased=False)
            if selected_heads.shape[-1] < head_count:
                ordered = torch.topk(
                    scores, selected_heads.shape[-1] + 1, dim=-1
                ).values
                values[f"{prefix}/topk_margin"] = (
                    ordered[:, -2] - ordered[:, -1]
                ).mean()
        for name, value in values.items():
            value = value.detach()
            if name in self._metric_totals:
                self._metric_totals[name] = self._metric_totals[name] + value
                self._metric_counts[name] += 1
            else:
                self._metric_totals[name] = value
                self._metric_counts[name] = 1


def set_longshot_selector_training_progress(
    model: nn.Module,
    config: LongshotSelectorTrainingConfig,
    progress: float,
) -> None:
    if not 0.0 <= progress <= 1.0:
        raise ValueError("selector training progress must be within [0, 1]")
    for module in model.modules():
        if isinstance(module, ContextualHeadSelector):
            module.set_training_progress(config, progress)


def _select_heads_from_head_scores(
    head_scores: torch.Tensor, topk_heads: int
) -> torch.Tensor:
    if topk_heads <= 0:
        raise ValueError(
            f"topk_heads must select at least one cache head, got {topk_heads}"
        )
    if topk_heads >= head_scores.shape[-1]:
        return torch.arange(
            head_scores.shape[-1], device=head_scores.device
        ).expand(head_scores.shape[0], -1)
    return torch.topk(
        head_scores, k=topk_heads, dim=-1, largest=True, sorted=True
    ).indices


def _select_random_heads(
    batch_size: int,
    kv_heads: int,
    linear_heads: int,
    topk_kv_heads: int,
    topk_linear_heads: int,
    device: torch.device,
) -> torch.Tensor:
    scores = torch.rand(
        batch_size,
        kv_heads + linear_heads,
        dtype=torch.float32,
        device=device,
    )
    selected = [
        _select_heads_from_head_scores(
            scores[:, :kv_heads], min(topk_kv_heads, kv_heads)
        )
    ]
    linear_budget = min(topk_linear_heads, linear_heads)
    if linear_budget > 0:
        selected.append(
            _select_heads_from_head_scores(scores[:, kv_heads:], linear_budget)
            + kv_heads
        )
    return torch.cat(selected, dim=-1)


def select_kv_norm_history(
    processor: Any,
    cache: CacheLike,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sidecar_policy = kv_score_sidecar_execution(processor)
    if sidecar_policy.enabled:
        policy = longshot_full_history_execution(processor)
        if policy.enabled:
            raise RuntimeError(
                "KV-score sidecar cannot run with Longshot full-history execution"
            )
        if not isinstance(cache, WindowedCache):
            raise TypeError(
                "enabled KV-score sidecar requires contiguous WindowedCache storage"
            )
        synchronize_sidecar(processor, cache)
        state = cache.kv_score_sidecar
        if state is None:
            raise RuntimeError(
                "KV-score sidecar bootstrap did not produce state"
            )
        all_indices = expand_cache_head_indices(
            cache.all_indices(), processor.num_kv_cache_heads
        )
        merged_all = gather_per_cache_head_kv(
            cache,
            all_indices,
            unrope_keys=processor.config.unrope_keys,
            rope_theta=processor.rope_theta,
            rotary_style=processor.rotary_style,
            rotary_dim=processor.rotary_dim,
            backend=cache_movement_backend(processor),
        )
        scores = state.scores
        indices = select_per_cache_head_kv_norm_indices_from_scores(
            scores, processor.config.process_every, processor.config.topk
        )
        index = indices[..., None].expand(-1, -1, -1, merged_all.shape[-1])
        return torch.gather(merged_all, dim=2, index=index), scores, indices

    policy = longshot_full_history_execution(processor)
    if not policy.enabled:
        all_indices = expand_cache_head_indices(
            cache.all_indices(), processor.num_kv_cache_heads
        )
        merged_all = gather_per_cache_head_kv(
            cache,
            all_indices,
            unrope_keys=processor.config.unrope_keys,
            rope_theta=processor.rope_theta,
            rotary_style=processor.rotary_style,
            rotary_dim=processor.rotary_dim,
            backend=cache_movement_backend(processor),
        )
        scores = chunked_fp32_squared_l2(merged_all)
        indices = select_per_cache_head_kv_norm_indices_from_scores(
            scores, processor.config.process_every, processor.config.topk
        )
        index = indices[..., None].expand(-1, -1, -1, merged_all.shape[-1])
        return torch.gather(merged_all, dim=2, index=index), scores, indices

    if not isinstance(cache, WindowedCache):
        raise TypeError(
            "enabled Longshot selected-history execution requires contiguous WindowedCache storage"
        )
    with torch.no_grad():
        if processor.config.unrope_keys:
            scores = chunked_fp32_unroped_concatenated_kv_squared_l2(
                cache.keys,
                cache.values,
                theta=processor.rope_theta,
                rotary_style=processor.rotary_style,
                rotary_dim=processor.rotary_dim,
                max_temp_bytes=policy.max_temp_bytes,
            )
        else:
            scores = chunked_fp32_concatenated_kv_squared_l2(
                cache.keys,
                cache.values,
                max_temp_bytes=policy.max_temp_bytes,
            )
        indices = select_per_cache_head_kv_norm_indices_from_scores(
            scores, processor.config.process_every, processor.config.topk
        )

    keys, values = gather_windowed_per_cache_head_kv(
        cache.keys,
        cache.values,
        indices,
        unrope_keys=processor.config.unrope_keys,
        rope_theta=processor.rope_theta,
        rotary_style=processor.rotary_style,
        rotary_dim=processor.rotary_dim,
    )
    record_longshot_full_history_event(
        processor,
        keys=cache.keys,
        selected_positions=indices.shape[-1],
        max_temp_bytes=policy.max_temp_bytes,
        unrope_keys=processor.config.unrope_keys,
        rotary_dim=processor.rotary_dim,
    )
    return torch.cat([keys, values], dim=-1), scores, indices


def synchronize_sidecar(processor: Any, cache: WindowedCache) -> None:
    spec = kv_score_sidecar_spec(processor)
    cache.kv_score_sidecar = synchronize_kv_score_sidecar(
        cache.kv_score_sidecar, cache.keys, cache.values, spec
    )


def refresh_sidecar(
    processor: Any, cache: CacheLike, positions: torch.Tensor
) -> CacheLike:
    if not kv_score_sidecar_execution(processor).enabled:
        return cache
    if not isinstance(cache, WindowedCache):
        raise TypeError(
            "enabled KV-score sidecar requires contiguous WindowedCache storage"
        )
    spec = kv_score_sidecar_spec(processor)
    cache.kv_score_sidecar = refresh_kv_score_sidecar(
        cache.kv_score_sidecar, cache.keys, cache.values, positions, spec
    )
    return cache


def cache_movement_backend(processor: Any) -> str:
    return "triton" if triton_cache_movement_enabled(processor) else "eager"
