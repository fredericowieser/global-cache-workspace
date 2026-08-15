from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import benchmarks.adaptive_cuda_parity as parity_module
from benchmarks.adaptive_cuda_parity import (
    ParityVariant,
    VectorTolerance,
    _compare_vector_family,
    _real_model_cli,
    reserve_parity_steps,
    run_adaptive_parity,
)
from benchmarks.adaptive_trace import (
    CacheEventRecorder,
    TraceScope,
    active_trace_scope,
    trace_cache_events,
)
from gcw.config import CacheProcessorConfig, DataConfig, ModelConfig, ResumeConfig, TrainConfig
from gcw.data.training import SFTCursor, TrainingBatch
from gcw.models.cache_processors import standard as standard_module
from gcw.models.cache_processors.base import cache_processor_recomputation
from gcw.models.cache_processors.state import CacheState
from gcw.performance.config import (
    AdaptiveBatchingConfig,
    PerformanceConfig,
    PhysicalBatchBandConfig,
)


class _PairStream:
    def __init__(self, widths: list[int]) -> None:
        self.widths = widths
        self.calls = 0

    @property
    def cursor(self) -> SFTCursor:
        return SFTCursor(row=self.calls * 2)

    def __next__(self) -> TrainingBatch:
        if self.calls >= len(self.widths):
            raise StopIteration
        width = self.widths[self.calls]
        pair = self.calls
        x = torch.stack(
            (
                torch.arange(width, dtype=torch.long).add(pair + 1).remainder(29).add(1),
                torch.arange(width, dtype=torch.long).add(pair + 3).remainder(29).add(1),
            )
        )
        y = x.clone()
        y[:, : max(1, width // 2)] = -100
        self.calls += 1
        return TrainingBatch(x=x, y=y, attention_mask=torch.ones_like(x, dtype=torch.bool))


class _TracedPairModel(torch.nn.Module):
    """Tiny checkpointed window model whose replays occur in reverse order."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.125))
        self.bias = torch.nn.Parameter(torch.tensor(-0.25))

    def forward(self, batch: TrainingBatch) -> torch.Tensor:
        signal = batch.x.float() * self.scale + self.bias
        row_values = (signal * batch.attention_mask).sum(dim=1)

        for boundary in range(2, batch.x.shape[1], 2):
            def cache_region(values: torch.Tensor, end: int = boundary) -> torch.Tensor:
                prefix = values[:, :end]
                cache_values = prefix[:, None, :, None]
                scores = torch.arange(end, dtype=torch.float32, device=values.device).expand(values.shape[0], -1)
                cache = CacheState(keys=[cache_values * 0.5], values=[cache_values], scores=scores)
                indices = cache.select_indices(process_every=2, topk=1)
                selected = torch.gather(cache.values[0], 2, indices[:, None, :, None])
                cache = standard_module.scatter_cache_values(
                    cache, indices, (selected + self.scale).unsqueeze(0)
                )
                return cache.values[0].sum(dim=(1, 2, 3))

            update = checkpoint(
                cache_region,
                signal,
                use_reentrant=False,
                preserve_rng_state=True,
                context_fn=lambda: (nullcontext(), cache_processor_recomputation()),
            )
            has_future_source_tokens = batch.attention_mask[:, boundary:].any(dim=1)
            row_values = row_values + update * has_future_source_tokens * 0.001

        target = batch.y.clamp_min(0).float().sum(dim=1) * 0.01
        return (row_values - target).square().reshape(-1, 2).mean(dim=1)


def _policy() -> AdaptiveBatchingConfig:
    return AdaptiveBatchingConfig(
        enabled=True,
        bands=(
            PhysicalBatchBandConfig(max_padded_tokens=8, physical_rows=16),
            PhysicalBatchBandConfig(max_padded_tokens=9, physical_rows=8),
            PhysicalBatchBandConfig(max_padded_tokens=10, physical_rows=4),
            PhysicalBatchBandConfig(max_padded_tokens=11, physical_rows=2),
        ),
    )


def _run_tiny(widths: list[int], *, steps: int = 1) -> dict[str, object]:
    initial = _TracedPairModel()
    reservations = reserve_parity_steps(_PairStream(widths), steps)
    return run_adaptive_parity(
        initial,
        variants=(
            ParityVariant("fixed_batch2_eager", "fixed", _TracedPairModel),
            ParityVariant("adaptive_eager", "adaptive", _TracedPairModel),
            ParityVariant("adaptive_compiled", "adaptive", _TracedPairModel),
        ),
        reserved_steps=reservations,
        pair_loss=lambda model, batch: model(batch),
        optimizer_factory=lambda model: torch.optim.AdamW(model.parameters(), lr=3e-4),
        adaptive_policy=_policy(),
        process_every=2,
        required_event_operations=("select", "write_v"),
        checkpoint_expected=lambda _batch: True,
    )


def test_trace_context_restores_symbols_and_aligns_checkpoint_recomputation() -> None:
    original_select = CacheState.select_indices
    original_scatter = standard_module.scatter_cache_values
    model = _TracedPairModel()
    batch = next(_PairStream([4]))
    recorder = CacheEventRecorder(("select", "write_v"))
    scope = TraceScope(
        step=0,
        group=0,
        row_keys=("step-0000/pair-00/row-0", "step-0000/pair-00/row-1"),
        source_padded_widths=(4, 4),
        process_every=2,
        checkpoint_expected=True,
    )
    recorder.expect_scope(scope)

    with trace_cache_events(recorder), active_trace_scope(scope):
        model(batch).sum().backward()

    assert CacheState.select_indices is original_select
    assert standard_module.scatter_cache_values is original_scatter
    assert recorder.coverage()["complete"] is True
    replay = recorder.recomputation_report()
    assert replay["passed"] is True
    assert replay["expected_replay_count"] == replay["actual_replay_count"] == 4


def test_group_shape_selection_mismatch_is_an_unconditional_parity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Numeric tolerances can never admit changed semantic cache decisions."""
    original_select = CacheState.select_indices

    def shape_sensitive_select(
        state: CacheState, process_every: int = 64, topk: int = 32
    ) -> torch.Tensor:
        indices = original_select(state, process_every=process_every, topk=topk)
        if state.scores.shape[0] > 2:
            indices = indices.clone()
            indices[:, 0] = (indices[:, 0] + 1).remainder(state.scores.shape[1])
        return indices

    monkeypatch.setattr(CacheState, "select_indices", shape_sensitive_select)
    report = _run_tiny([8] * 32)

    assert report["passed"] is False
    for comparison in report["comparisons"]:
        cache_events = comparison["steps"][0]["cache_events"]
        assert cache_events["semantic_primary_equal"] is False
        assert cache_events["passed"] is False
        assert comparison["passed"] is False


def test_variable_width_group_padding_is_classified_trimmed_and_semantically_aligned() -> None:
    torch.manual_seed(92)
    widths = [4, 8] * 16
    report = _run_tiny(widths)
    assert report["passed"] is True
    comparison = report["comparisons"][0]["steps"][0]
    events = comparison["cache_events"]
    assert events["semantic_primary_equal"] is True
    assert events["reference_coverage"]["padding_only_count"] == 0
    assert events["candidate_coverage"]["padding_only_count"] > 0
    assert events["candidate_coverage"]["padding_trimmed"] is True
    candidate_events = report["variants"][1]["steps"][0]["events"]
    padding = [event for event in candidate_events if event["padding_only"]]
    assert padding
    assert all(event["destinations"] == () for event in padding)
    assert all(event["cache_boundary"] >= event["source_padded_width"] for event in padding)
    json.dumps(report, allow_nan=False)


def test_checkpoint_replays_align_by_boundary_when_backward_order_is_reversed() -> None:
    report = _run_tiny([8] * 32)
    recomputation = report["variants"][0]["steps"][0]["recomputation"]
    assert recomputation["passed"] is True
    assert recomputation["expected_replay_count"] == recomputation["actual_replay_count"]
    replay_events = [
        event
        for event in report["variants"][0]["steps"][0]["events"]
        if event["phase"] == "recomputation" and event["row_key"].endswith("pair-00/row-0")
    ]
    boundaries = [event["cache_boundary"] for event in replay_events if event["operation"] == "select"]
    assert boundaries == sorted(boundaries, reverse=True)
    assert all(event["recomputation_of"] == event["stable_event_id"] for event in replay_events)


def test_no_recomputation_only_passes_when_not_expected() -> None:
    recorder = CacheEventRecorder(("select", "write_v"))
    scope = TraceScope(
        step=0,
        group=0,
        row_keys=("step-0000/pair-00/row-0", "step-0000/pair-00/row-1"),
        source_padded_widths=(4, 4),
        process_every=2,
        checkpoint_expected=True,
    )
    recorder.expect_scope(scope)
    cache_values = torch.ones(2, 1, 2, 1)
    cache = CacheState([cache_values], [cache_values], torch.ones(2, 2))
    with active_trace_scope(scope):
        recorder.record("select", cache, torch.ones(2, 1, dtype=torch.long))
        recorder.record("write_v", cache, torch.ones(2, 1, dtype=torch.long))
    replay = recorder.recomputation_report()
    assert replay["expected_replay_count"] == 4
    assert replay["actual_replay_count"] == 0
    assert replay["passed"] is False


def test_near_zero_wrong_gradient_fails_strict_absolute_gate() -> None:
    tolerance = VectorTolerance(relative_l2=1.0, minimum_cosine=-1.0, near_zero_norm=1e-10, near_zero_atol=1e-9)
    comparison = _compare_vector_family(
        {"zero_grad": torch.zeros(8)},
        {"zero_grad": torch.full((8,), 1e-6)},
        tolerance,
    )
    assert comparison["global"]["near_zero"] is True
    assert comparison["tensors"]["zero_grad"]["max_abs"] == pytest.approx(1e-6)
    assert comparison["passed"] is False


def test_nonfinite_vector_failure_is_structured_and_json_finite() -> None:
    comparison = _compare_vector_family(
        {"grad": torch.ones(2)},
        {"grad": torch.tensor([float("nan"), float("inf")])},
        VectorTolerance(1e-3, 0.99, 1e-12, 1e-12),
    )
    assert comparison["tensors"]["grad"]["actual_nonfinite_count"] == 2
    assert comparison["tensors"]["grad"]["relative_l2"] is None
    assert comparison["passed"] is False
    json.dumps(comparison, allow_nan=False)


def _real_cli_config(variant: str) -> TrainConfig:
    model = ModelConfig(
        variant=variant,  # type: ignore[arg-type]
        family="qwen3",
        sequence_len=8,
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        head_dim=8,
        cache_processor=CacheProcessorConfig(
            process_every=2,
            topk=1,
            score="kv_norm" if variant == "bgt" else "attn_l2",
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
            update_stream="v",
        ),
    )
    return TrainConfig(
        model=model,
        data=DataConfig(kind="sft", dataset="unused.jsonl"),
        mode="sft",
        trainable="cache_processor",
        batch_size=2,
        sequence_len=8,
        global_batch_rows=64,
        logical_pair_grouping=True,
        max_steps=1,
        performance=PerformanceConfig(adaptive_batching=_policy()),
    )


@pytest.mark.parametrize("variant", ["bt", "bgt"])
@pytest.mark.parametrize(
    "variant_set", ["adaptive", "fixed_regional", "checkpoint_threshold"]
)
def test_real_tiny_bt_bgt_cli_integration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variant: str, variant_set: str
) -> None:
    config = _real_cli_config(variant)
    if variant_set in {"fixed_regional", "checkpoint_threshold"}:
        config.performance = PerformanceConfig()
    if variant_set == "checkpoint_threshold":
        config.activation_checkpoint_windows = 2
        config.activation_checkpoint_min_padded_sequence_length = 4
    config_path = tmp_path / f"{variant}-{variant_set}.yaml"
    config_path.write_text("test: integration\n", encoding="utf-8")
    monkeypatch.setattr(parity_module, "load_train_config", lambda _path: config)
    monkeypatch.setattr(
        parity_module,
        "repository_source_identity",
        lambda *_args, **_kwargs: {
            "git_commit": "0" * 40,
            "git_dirty": False,
            "source_tree_sha256": "1" * 64,
        },
    )
    monkeypatch.setattr(
        parity_module,
        "build_training_stream",
        lambda *_args, **_kwargs: _PairStream([8] * 32),
    )
    # CPU integration exercises the real model and CLI wiring while substituting
    # eager regions for CUDA compilation, which remains the opt-in GPU gate.
    monkeypatch.setattr(parity_module, "_temporary_environment", lambda _updates: nullcontext())
    monkeypatch.setattr(parity_module, "configure_qwen3_backbone_chunk_backend", lambda *_args: None)
    payload = _real_model_cli(
        argparse.Namespace(
            train_config=config_path,
            output=tmp_path / "out.json",
            steps=1,
            device="cpu",
            require_resume_evidence=False,
            variant_set=variant_set,
            candidate_checkpoint_min_padded_sequence_length=(
                9 if variant_set == "checkpoint_threshold" else None
            ),
        )
    )
    assert payload["passed"] is True
    assert payload["variant"] == variant
    assert payload["variant_set"] == variant_set
    if variant_set == "adaptive":
        expected_candidates = ["adaptive_eager", "adaptive_compiled"]
    elif variant_set == "fixed_regional":
        expected_candidates = ["fixed_batch2_compiled", "fixed_batch2_compile_full_windows"]
    else:
        expected_candidates = ["fixed_batch2_checkpoint_candidate"]
    assert [comparison["candidate"] for comparison in payload["comparisons"]] == expected_candidates
    assert payload["production_resume_gate"]["available"] is False
    assert payload["comparisons"][0]["steps"][0]["cache_events"]["reference_coverage"]["complete"]


def test_cli_fail_closed_resume_request_and_step_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _real_cli_config("bt")
    config_path = tmp_path / "bt.yaml"
    config_path.write_text("test: validation\n", encoding="utf-8")
    monkeypatch.setattr(parity_module, "load_train_config", lambda _path: config)
    monkeypatch.setattr(
        parity_module,
        "repository_source_identity",
        lambda *_args, **_kwargs: {
            "git_commit": "0" * 40,
            "git_dirty": False,
            "source_tree_sha256": "1" * 64,
        },
    )
    base = dict(train_config=config_path, output=tmp_path / "out.json", device="cpu")
    with pytest.raises(ValueError, match="resume evidence is unavailable"):
        _real_model_cli(argparse.Namespace(**base, steps=1, require_resume_evidence=True))
    with pytest.raises(ValueError, match="no greater than train max_steps"):
        _real_model_cli(argparse.Namespace(**base, steps=2, require_resume_evidence=False))

    config.resume = ResumeConfig(checkpoint="unused")
    with pytest.raises(ValueError, match="rejects training resume configs"):
        _real_model_cli(argparse.Namespace(**base, steps=1, require_resume_evidence=False))
    config.resume = None
    monkeypatch.setattr(
        parity_module,
        "repository_source_identity",
        lambda *_args, **_kwargs: {
            "git_commit": "0" * 40,
            "git_dirty": True,
            "source_tree_sha256": "1" * 64,
        },
    )
    with pytest.raises(ValueError, match="clean source tree"):
        _real_model_cli(argparse.Namespace(**base, steps=1, require_resume_evidence=False))


def test_formal_harness_fails_closed_on_ambiguous_variants() -> None:
    reservations = reserve_parity_steps(_PairStream([4] * 32), 1)
    variant = ParityVariant("duplicate", "fixed", _TracedPairModel)
    with pytest.raises(ValueError, match="names must be unique"):
        run_adaptive_parity(
            _TracedPairModel(),
            (variant, variant),
            reservations,
            lambda model, batch: model(batch),
            lambda model: torch.optim.AdamW(model.parameters()),
            adaptive_policy=_policy(),
            process_every=2,
            required_event_operations=("select", "write_v"),
        )
