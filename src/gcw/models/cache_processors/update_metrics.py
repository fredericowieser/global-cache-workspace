"""Evaluation-only observations of Longshot diffusion updates."""

from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from collections.abc import Iterator, Sequence
from typing import Any

import torch
import torch.nn.functional as F

UPDATE_BATCH: ContextVar[UpdateBatch | None] = ContextVar(
    "update_batch", default=None
)
UPDATE_OBSERVER: ContextVar[DiffusionUpdateObservation | None] = ContextVar(
    "update_observer", default=None
)


def outer_difference_norm(
    key: torch.Tensor,
    value: torch.Tensor,
    previous: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    if previous is None:
        return key.norm(dim=-1) * value.norm(dim=-1)
    old_key, old_value = previous
    dk, dv = key - old_key, value - old_value
    squared = (
        key.square().sum(-1) * dv.square().sum(-1)
        + dk.square().sum(-1) * old_value.square().sum(-1)
        + 2 * (key * dk).sum(-1) * (dv * old_value).sum(-1)
    )
    return squared.clamp_min(0).sqrt()


class UpdateBatch:
    def __init__(
        self,
        size: int,
        device: torch.device | str,
        lengths: Sequence[int] | None = None,
    ) -> None:
        self.active = torch.ones(size, dtype=torch.bool, device=device)
        self.lengths = (
            torch.as_tensor(lengths, device=device)
            if lengths is not None
            else None
        )
        self.totals: dict[
            tuple[int, int, str], tuple[torch.Tensor, torch.Tensor, int]
        ] = {}
        self.external_iteration = 1

    @contextmanager
    def application(
        self, processor: Any, cache: Any, specs: list
    ) -> Iterator[None]:
        active = self.active
        if self.lengths is not None:
            active = active & (cache.seq_len < self.lengths)
        observer = DiffusionUpdateObservation(
            self, processor, cache, specs, active
        )
        token = UPDATE_OBSERVER.set(observer)
        try:
            yield
        finally:
            UPDATE_OBSERVER.reset(token)

    def add(
        self,
        iteration: int,
        step: int,
        name: str,
        magnitudes: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        if magnitudes.shape[1] == 0:
            return
        minimum, maximum = torch.aminmax(magnitudes, dim=1)
        std, mean = torch.std_mean(magnitudes, dim=1, correction=0)
        values = torch.stack([minimum, maximum, mean, std], dim=1)
        values = torch.where(active[:, None], values, 0)
        key = (iteration, step, name)
        if key not in self.totals:
            self.totals[key] = (
                torch.zeros_like(values),
                torch.zeros_like(active, dtype=torch.long),
                magnitudes.shape[1],
            )
        totals, counts, heads = self.totals[key]
        if heads != magnitudes.shape[1]:
            raise RuntimeError(
                "cache-update diagnostic head population changed"
            )
        totals.add_(values)
        counts.add_(active)

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = [
            dict() for _ in range(self.active.shape[0])
        ]
        if not self.totals:
            return rows
        totals = (
            torch.stack([value[0] for value in self.totals.values()], dim=1)
            .cpu()
            .tolist()
        )
        counts = (
            torch.stack([value[1] for value in self.totals.values()], dim=1)
            .cpu()
            .tolist()
        )
        for row, row_totals, row_counts in zip(
            rows, totals, counts, strict=True
        ):
            for ((iteration, step, name), (_, _, heads)), total, count in zip(
                self.totals.items(), row_totals, row_counts, strict=True
            ):
                if count:
                    row[f"{iteration}:{step}:{name}"] = {
                        "sum": total,
                        "applications": count,
                        "heads": heads,
                    }
        return rows


class DiffusionUpdateObservation:
    def __init__(
        self,
        batch: UpdateBatch,
        processor: Any,
        cache: Any,
        specs: list,
        active: torch.Tensor,
    ) -> None:
        self.batch = batch
        self.processor = processor
        self.block = processor.block
        self.cache = cache
        self.specs = specs or []
        self.active = active
        self.iteration = batch.external_iteration
        self.step_index = 0
        self.previous_kv: torch.Tensor | None = None
        self.previous_linear_factors: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None
        self.previous_linear_raw: torch.Tensor | None = None

    def record(self, name: str, magnitudes: torch.Tensor) -> None:
        self.batch.add(
            self.iteration, self.step_index, name, magnitudes, self.active
        )

    def step(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        initial: torch.Tensor,
        *,
        final: bool,
    ) -> None:
        self.step_index += 1
        kv_heads = self.block.kv_cache_heads
        change = after.float() - before.float()
        self.record("latent_kv", change[:, :kv_heads].norm(dim=-1))
        self.record("latent_linear", change[:, kv_heads:].norm(dim=-1))
        if not final:
            displacement = after - initial
            projected = self.block.out_proj(displacement[:, :kv_heads])
            self.kv(projected[..., : self.block.output_dim])
            adapter = self.block.linear_state_adapter
            if adapter is not None and self.specs:
                projected = adapter.output_proj(displacement[:, kv_heads:])
                self.linear(projected[..., : adapter.output_dim])

    def kv(self, raw: torch.Tensor) -> None:
        raw = raw.detach().float()
        change = raw if self.previous_kv is None else raw - self.previous_kv
        self.record("kv", change.norm(dim=-1))
        self.previous_kv = raw

    def linear(self, raw: torch.Tensor) -> None:
        raw = raw.detach()
        adapter = self.block.linear_state_adapter
        kv_dim = adapter.linear_key_dim + adapter.linear_value_dim
        change = raw[..., :kv_dim].float()
        if self.previous_linear_raw is not None:
            change = change - self.previous_linear_raw[..., :kv_dim].float()
        self.record("linear_kv", change.norm(dim=-1))
        if self.processor.config.linear_state_update_mode == "outer_product":
            key, value = raw.split(
                [adapter.linear_key_dim, adapter.linear_value_dim], dim=-1
            )
            # Match the writer's rounding before measuring in FP32.
            key = F.normalize(key.float(), p=2, dim=-1).to(value.dtype).float()
            value = value.float()
            self.record(
                "linear_state",
                outer_difference_norm(key, value, self.previous_linear_factors),
            )
            self.previous_linear_factors = (key, value)
            self.previous_linear_raw = raw
            return
        magnitudes = []
        offset = 0
        for spec in self.specs:
            update = raw[:, offset : offset + spec.head_count]
            previous_update = (
                self.previous_linear_raw[:, offset : offset + spec.head_count]
                if self.previous_linear_raw is not None
                else None
            )
            offset += spec.head_count
            recurrent = self.cache.linear_states[spec.layer_idx][0]
            proposed = self.processor._gated_delta_linear_state_update(
                spec.layer_idx, update, recurrent
            ).float()
            old = (
                self.processor._gated_delta_linear_state_update(
                    spec.layer_idx, previous_update, recurrent
                ).float()
                if previous_update is not None
                else recurrent.float()
            )
            magnitudes.append((proposed - old).flatten(2).norm(dim=-1))
        self.record("linear_state", torch.cat(magnitudes, dim=1))
        self.previous_linear_raw = raw
