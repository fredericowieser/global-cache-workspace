from __future__ import annotations

import math
import random
from collections import Counter
from contextlib import contextmanager
from typing import Any, Iterable

import torch


class HeadMeanAblator:
    def __init__(self, model: torch.nn.Module):
        backbone: Any = getattr(model, "backbone", model)
        self.backbone = backbone
        self.num_heads = int(self.backbone.config.num_heads)
        self.head_dim = int(self.backbone.config.head_dim)
        self.layer_indices = list(range(len(self.backbone.layers)))
        self.head_means: dict[int, torch.Tensor] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

    @torch.no_grad()
    def capture(self, batches: Iterable[torch.Tensor]) -> None:
        sums = {index: torch.zeros(self.num_heads, self.head_dim, dtype=torch.float32) for index in self.layer_indices}
        counts = {index: 0 for index in self.layer_indices}

        def make_hook(layer_index: int):
            def hook(_module, inputs):
                values = inputs[0].detach().float().cpu().reshape(-1, self.num_heads, self.head_dim)
                sums[layer_index] += values.sum(dim=0)
                counts[layer_index] += values.shape[0]

            return hook

        handles = [
            self.backbone.layers[index].self_attn.o_proj.register_forward_pre_hook(make_hook(index))
            for index in self.layer_indices
        ]
        try:
            for input_ids in batches:
                self.backbone.forward_full(input_ids)
        finally:
            for handle in handles:
                handle.remove()

        device = self.backbone.get_device()
        for index in self.layer_indices:
            if counts[index] == 0:
                raise ValueError(f"no calibration tokens reached layer {index}")
            self.head_means[index] = (sums[index] / counts[index]).to(device)

    def apply(self, masks: dict[int, torch.Tensor]) -> None:
        self.remove()

        def make_hook(layer_index: int):
            def hook(_module, inputs):
                values = inputs[0]
                batch, sequence_len, width = values.shape
                heads = values.reshape(batch, sequence_len, self.num_heads, self.head_dim)
                mask = masks[layer_index].to(device=values.device, dtype=values.dtype).view(1, 1, self.num_heads, 1)
                means = self.head_means[layer_index].to(device=values.device, dtype=values.dtype).view(
                    1,
                    1,
                    self.num_heads,
                    self.head_dim,
                )
                mixed = heads * mask + means * (1.0 - mask)
                return (mixed.reshape(batch, sequence_len, width),)

            return hook

        self.hooks = [
            self.backbone.layers[index].self_attn.o_proj.register_forward_pre_hook(make_hook(index))
            for index in self.layer_indices
        ]

    def remove(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    @contextmanager
    def ablated(self, masks: dict[int, torch.Tensor] | None):
        if masks is None:
            self.remove()
            yield
            return
        self.apply(masks)
        try:
            yield
        finally:
            self.remove()


class UnigramBaseline:
    def __init__(self, vocab_size: int, epsilon: float = 1e-8):
        self.vocab_size = vocab_size
        self.epsilon = epsilon
        self.counts: Counter[int] = Counter()
        self.total = 0

    def update(self, token_ids: Iterable[int]) -> None:
        ids = [int(token_id) for token_id in token_ids if 0 <= int(token_id) < self.vocab_size]
        self.counts.update(ids)
        self.total += len(ids)

    def nll(self, token_ids: list[int]) -> torch.Tensor:
        denominator = max(self.total + self.vocab_size * self.epsilon, self.vocab_size * self.epsilon)
        return torch.tensor(
            [-math.log((self.counts.get(int(token_id), 0) + self.epsilon) / denominator) for token_id in token_ids],
            dtype=torch.float32,
        )


def deterministic_masks(
    ablator: HeadMeanAblator,
    *,
    split_id: int,
    seed: int,
    group_fraction: float,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    rng = random.Random(seed + split_id * 1009)
    group_size = max(1, min(ablator.num_heads - 1, round(ablator.num_heads * group_fraction)))
    first_masks = {}
    second_masks = {}
    for layer_index in ablator.layer_indices:
        heads = list(range(ablator.num_heads))
        rng.shuffle(heads)
        first = torch.zeros(ablator.num_heads)
        second = torch.zeros(ablator.num_heads)
        first[heads[:group_size]] = 1.0
        second[heads[group_size : group_size * 2]] = 1.0
        if second.sum() == 0:
            second[heads[group_size:]] = 1.0
        first_masks[layer_index] = first
        second_masks[layer_index] = second
    return first_masks, second_masks


def synergy_from_losses(
    full: torch.Tensor,
    group_a: torch.Tensor,
    group_b: torch.Tensor,
    baseline: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[float, float]:
    raw_mean = float(((group_a + group_b) - (full + baseline)).mean().item())
    useful = float((baseline - full).mean().item())
    normalized = raw_mean / useful if useful > eps else float("nan")
    return raw_mean, normalized
