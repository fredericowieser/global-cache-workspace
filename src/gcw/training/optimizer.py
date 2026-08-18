from __future__ import annotations

import math
from typing import Any
import torch

from gcw.config import TrainConfig


def _build_optimizer(params: list[torch.nn.Parameter], config: TrainConfig) -> torch.optim.AdamW:
    decay = [p for p in params if p.ndim >= 2]
    nodecay = [p for p in params if p.ndim < 2]
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "lr": config.learning_rate, "weight_decay": config.weight_decay, "decay": True})
    if nodecay:
        groups.append({"params": nodecay, "lr": config.learning_rate, "weight_decay": 0.0, "decay": False})
    opt = torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
    return opt


def _wsd_lr(step: int, max_steps: int, warmup_steps: int, warmdown_ratio: float, final_lr_frac: float) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    warmdown = round(warmdown_ratio * max_steps)
    if warmdown <= 0 or step < max_steps - warmdown:
        return 1.0
    if warmdown == 1:
        return final_lr_frac
    progress = (max_steps - 1 - step) / (warmdown - 1)
    return final_lr_frac + progress * (1.0 - final_lr_frac)


def _weight_decay(step: int, max_steps: int, base_weight_decay: float) -> float:
    return base_weight_decay * 0.5 * (1.0 + math.cos(math.pi * step / max(1, max_steps)))


def _clip_grad_norm(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    max_norm: float,
    *,
    use_fsdp: bool = False,
) -> torch.Tensor:
    if use_fsdp:
        clip_grad_norm = getattr(model, "clip_grad_norm_", None)
        if not callable(clip_grad_norm):
            raise TypeError("FSDP model does not expose clip_grad_norm_")
        return clip_grad_norm(max_norm)
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if v is not None}


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)
