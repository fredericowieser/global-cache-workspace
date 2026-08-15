from __future__ import annotations

from contextlib import nullcontext
import json

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from benchmarks.adaptive_cuda_parity import (
    ParityVariant,
    reserve_parity_steps,
    run_adaptive_parity,
)
from benchmarks.adaptive_trace import CacheEventRecorder, TraceScope, active_trace_scope, trace_cache_events
from gcw.data.training import SFTCursor, TrainingBatch
from gcw.models.cache_processors import standard as standard_module
from gcw.models.cache_processors.base import cache_processor_recomputation
from gcw.models.cache_processors.state import CacheState
from gcw.performance.config import AdaptiveBatchingConfig, PhysicalBatchBandConfig


class _PairStream:
    def __init__(self, steps: int) -> None:
        self.rows = 0
        self.steps = steps

    @property
    def cursor(self) -> SFTCursor:
        return SFTCursor(row=self.rows)

    def __next__(self) -> TrainingBatch:
        if self.rows >= self.steps * 64:
            raise StopIteration
        pair = self.rows // 2
        x = torch.tensor(
            [[pair % 11 + 1, 2, 3, 4], [pair % 13 + 1, 5, 6, 7]],
            dtype=torch.long,
        )
        y = torch.tensor(
            [[-100, -100, 3, 4], [-100, -100, 6, 7]],
            dtype=torch.long,
        )
        self.rows += 2
        return TrainingBatch(x=x, y=y, attention_mask=torch.ones_like(x, dtype=torch.bool))


class _TracedPairModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.125))
        self.bias = torch.nn.Parameter(torch.tensor(-0.25))

    def forward(self, batch: TrainingBatch) -> torch.Tensor:
        signal = batch.x[:, :2].float() * self.scale + self.bias

        def cache_region(values: torch.Tensor) -> torch.Tensor:
            cache_values = values[:, None, :, None]
            cache = CacheState(
                keys=[cache_values * 0.5],
                values=[cache_values],
                scores=torch.tensor([0.1, 0.9], dtype=torch.float32).expand(values.shape[0], -1),
            )
            indices = cache.select_indices(process_every=1, topk=1)
            selected = torch.gather(cache.values[0], 2, indices[:, None, :, None])
            cache = standard_module.scatter_cache_values(cache, indices, (selected + self.scale).unsqueeze(0))
            return cache.values[0].squeeze(1).squeeze(-1)

        processed = checkpoint(
            cache_region,
            signal,
            use_reentrant=False,
            preserve_rng_state=True,
            context_fn=lambda: (nullcontext(), cache_processor_recomputation()),
        )
        row_losses = (processed.sum(dim=1) - batch.y.clamp_min(0).float().sum(dim=1) * 0.01).square()
        return row_losses.reshape(-1, 2).mean(dim=1)


def _policy() -> AdaptiveBatchingConfig:
    return AdaptiveBatchingConfig(
        enabled=True,
        bands=(
            PhysicalBatchBandConfig(max_padded_tokens=4, physical_rows=16),
            PhysicalBatchBandConfig(max_padded_tokens=8, physical_rows=8),
            PhysicalBatchBandConfig(max_padded_tokens=12, physical_rows=4),
            PhysicalBatchBandConfig(max_padded_tokens=16, physical_rows=2),
        ),
    )


def test_trace_context_restores_symbols_and_aligns_checkpoint_recomputation() -> None:
    original_select = CacheState.select_indices
    original_scatter = standard_module.scatter_cache_values
    model = _TracedPairModel()
    batch = next(_PairStream(1))
    recorder = CacheEventRecorder()

    with trace_cache_events(recorder), active_trace_scope(
        TraceScope(step=0, group=0, row_keys=("step-0000/pair-00/row-0", "step-0000/pair-00/row-1"))
    ):
        model(batch).sum().backward()

    assert CacheState.select_indices is original_select
    assert standard_module.scatter_cache_values is original_scatter
    primary = [event for event in recorder.events if event.phase == "primary"]
    recomputed = [event for event in recorder.events if event.phase == "recomputation"]
    assert {event.operation for event in primary} == {"select", "write_v"}
    assert recomputed
    assert all(recorder.recomputation_alignment().values())
    assert primary[0].event_key.startswith("step-0000/pair-00/row-0/")


def test_formal_harness_compares_fixed_adaptive_and_compiled_short_trajectory() -> None:
    torch.manual_seed(92)
    initial = _TracedPairModel()
    reservations = reserve_parity_steps(_PairStream(steps=2), 2)

    report = run_adaptive_parity(
        initial,
        variants=(
            ParityVariant("fixed_batch2_eager", "fixed", _TracedPairModel),
            ParityVariant("adaptive_eager", "adaptive", _TracedPairModel),
            # Tiny CPU tests exercise the compiled-policy alignment path without
            # invoking Inductor. Real compilation belongs to the CUDA CLI gate.
            ParityVariant("adaptive_compiled", "adaptive", _TracedPairModel),
        ),
        reserved_steps=reservations,
        pair_loss=lambda model, batch: model(batch),
        optimizer_factory=lambda model: torch.optim.AdamW(model.parameters(), lr=3e-4),
        adaptive_policy=_policy(),
    )

    assert report["passed"] is True
    assert report["schema"] == "gcw.adaptive_cuda_parity.v1"
    assert report["timing_policy"] == "forbidden_in_instrumented_runs"
    assert report["reserved_steps"][-1]["cursor_after"] == {"epoch": 0, "row": 128, "kind": "sft"}
    assert len(report["comparisons"]) == 2
    assert all(comparison["cursor_equal"] for comparison in report["comparisons"])
    assert all(comparison["initial_rng_equal"] for comparison in report["comparisons"])
    assert all(step["cache_events"]["primary_equal"] for comparison in report["comparisons"] for step in comparison["steps"])
    assert all(step["optimizer_steps_equal"] for comparison in report["comparisons"] for step in comparison["steps"])
    assert "seconds" not in json.dumps(report)
    json.dumps(report, allow_nan=False)


def test_formal_harness_fails_closed_on_ambiguous_variants() -> None:
    reservations = reserve_parity_steps(_PairStream(steps=1), 1)
    variant = ParityVariant("duplicate", "fixed", _TracedPairModel)
    with pytest.raises(ValueError, match="names must be unique"):
        run_adaptive_parity(
            _TracedPairModel(),
            (variant, variant),
            reservations,
            lambda model, batch: model(batch),
            lambda model: torch.optim.AdamW(model.parameters()),
            adaptive_policy=_policy(),
        )
