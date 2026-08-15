from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from typing import Any

import pytest
import torch

import gcw.data.training as training_data_module
import gcw.training.core as train_module
from gcw.architecture import resolve_experiment, resolve_model
from gcw.checkpoint import load_checkpoint
from gcw.config import (
    CacheProcessorConfig,
    DataConfig,
    DiffusionProcessorConfig,
    LoRAConfig,
    ModelConfig,
    ResumeConfig,
    TrainConfig,
    to_dict,
)
from gcw.data.training import SFTBatchStream, SFTCursor, TrainingBatch
from gcw.models import build_model
from gcw.performance.adjacent_pairs import (
    PAIRS_PER_STEP,
    ROWS_PER_STEP,
    adaptive_batching_metadata,
    build_adaptive_pair_step,
    collate_pair_group,
    plan_contiguous_pair_groups,
    reserve_adjacent_pairs,
)
from gcw.performance.config import AdaptiveBatchingConfig, PerformanceConfig, PhysicalBatchBandConfig
from gcw.training.checkpoint import save_training_checkpoint as real_save_training_checkpoint
from gcw.training.core import _build_optimizer, _training_fingerprint, train
from gcw.training.objectives import adjacent_pair_loss_sum, adjacent_pair_loss_sum_from_rows, model_loss


def _pair(pair_index: int, width: int) -> TrainingBatch:
    x = torch.full((2, width), pair_index + 1, dtype=torch.long)
    y = torch.full((2, width), -100, dtype=torch.long)
    y[:, -1] = pair_index + 2
    attention_mask = torch.ones((2, width), dtype=torch.bool)
    return TrainingBatch(x=x, y=y, attention_mask=attention_mask)


class _PairStream:
    def __init__(self, widths: list[int]) -> None:
        self.widths = widths
        self.calls = 0

    @property
    def cursor(self) -> SFTCursor:
        return SFTCursor(row=self.calls * 2)

    def __next__(self) -> TrainingBatch:
        width = self.widths[self.calls]
        batch = _pair(self.calls, width)
        self.calls += 1
        return batch


def _enabled_policy(**kwargs: Any) -> AdaptiveBatchingConfig:
    return AdaptiveBatchingConfig(enabled=True, **kwargs)


def test_adaptive_v2_has_no_admissible_production_config_until_exact_cuda_parity() -> None:
    """A grouped V2 must not become configurable before the H200 parity blocker is resolved."""
    assert "adaptive_batching_v2" not in to_dict(PerformanceConfig())
    with pytest.raises(TypeError, match="adaptive_batching_v2"):
        PerformanceConfig(adaptive_batching_v2={})  # type: ignore[call-arg]


def test_reservation_consumes_exactly_32_source_pairs() -> None:
    stream = _PairStream([17] * PAIRS_PER_STEP)

    reservation = reserve_adjacent_pairs(stream)

    assert stream.calls == PAIRS_PER_STEP
    assert stream.cursor == SFTCursor(row=ROWS_PER_STEP)
    assert reservation.padded_lengths == (17,) * PAIRS_PER_STEP
    for pair_index, batch in enumerate(reservation.pairs):
        assert torch.equal(batch.x[:, 0], torch.full((2,), pair_index + 1))


def test_planner_uses_exact_boundaries_and_keeps_contiguous_pairs() -> None:
    widths = [1024] * 8 + [2048] * 4 + [3072] * 2 + [4096] * 18
    reservation = reserve_adjacent_pairs(_PairStream(widths))

    step = plan_contiguous_pair_groups(reservation, _enabled_policy())

    assert [(group.pair_start, group.pair_stop, group.physical_rows, group.padded_tokens) for group in step.groups] == [
        (0, 8, 16, 1024),
        (8, 12, 8, 2048),
        (12, 14, 4, 3072),
        *[(index, index + 1, 2, 4096) for index in range(14, 32)],
    ]


def test_width_one_over_each_boundary_falls_to_the_next_band() -> None:
    widths = [1025] * PAIRS_PER_STEP
    step = build_adaptive_pair_step(_PairStream(widths), _enabled_policy())
    assert all(group.physical_rows == 8 for group in step.groups)

    widths = [2049] * PAIRS_PER_STEP
    step = build_adaptive_pair_step(_PairStream(widths), _enabled_policy())
    assert all(group.physical_rows == 4 for group in step.groups)

    widths = [3073] * PAIRS_PER_STEP
    step = build_adaptive_pair_step(_PairStream(widths), _enabled_policy())
    assert all(group.physical_rows == 2 for group in step.groups)


def test_planner_fails_closed_above_sequence_contract() -> None:
    reservation = reserve_adjacent_pairs(_PairStream([4097] + [1] * (PAIRS_PER_STEP - 1)))
    with pytest.raises(ValueError, match="no physical band for pair 0"):
        plan_contiguous_pair_groups(reservation, _enabled_policy())


def test_collation_preserves_pair_and_row_order_with_right_padding() -> None:
    widths = [3, 5, 4, 2, 6, 1, 4, 3] + [4096] * 24
    step = build_adaptive_pair_step(_PairStream(widths), _enabled_policy())
    first = step.groups[0]
    physical = collate_pair_group(step, first)

    assert first.physical_rows == 16
    assert physical.batch.x.shape == (16, 6)
    for pair_index in range(8):
        rows = physical.batch.x[pair_index * 2 : pair_index * 2 + 2]
        assert torch.equal(rows[:, 0], torch.full((2,), pair_index + 1))
        assert torch.all(rows[:, widths[pair_index] :] == 0)
        assert torch.all(physical.batch.y[pair_index * 2 : pair_index * 2 + 2, widths[pair_index] :] == -100)
        assert not physical.batch.attention_mask[
            pair_index * 2 : pair_index * 2 + 2, widths[pair_index] :
        ].any()


def test_pair_loss_sum_weights_each_original_pair_by_its_own_token_count() -> None:
    row_sums = torch.tensor([2.0, 4.0, 3.0, 9.0, 10.0, 2.0])
    row_counts = torch.tensor([1.0, 2.0, 1.0, 3.0, 4.0, 2.0])

    result = adjacent_pair_loss_sum_from_rows(row_sums, row_counts)

    assert result == pytest.approx((6.0 / 3.0) + (12.0 / 4.0) + (12.0 / 6.0))


def test_adaptive_metadata_authenticates_every_policy_field() -> None:
    policy = _enabled_policy()
    metadata = adaptive_batching_metadata(policy, global_batch_rows=64)
    changed = AdaptiveBatchingConfig(
        enabled=True,
        bands=(
            PhysicalBatchBandConfig(1000, 16),
            *policy.bands[1:],
        ),
    )
    changed_metadata = adaptive_batching_metadata(changed, global_batch_rows=64)

    assert metadata["schema"] == "gcw.adaptive_pair_batching.v1"
    assert metadata["reserved_pairs"] == 32
    assert metadata["group_weighting"] == "adjacent_pair_loss_sum_divided_by_32"
    assert metadata["fingerprint_sha256"] != changed_metadata["fingerprint_sha256"]


def _adaptive_config(**kwargs: Any) -> TrainConfig:
    model = kwargs.pop("model", ModelConfig())
    data = kwargs.pop("data", DataConfig(kind="sft", dataset="unused.jsonl"))
    return TrainConfig(
        model=model,
        data=data,
        mode=kwargs.pop("mode", "sft"),
        batch_size=kwargs.pop("batch_size", 2),
        sequence_len=kwargs.pop("sequence_len", 128),
        global_batch_rows=kwargs.pop("global_batch_rows", 64),
        logical_pair_grouping=kwargs.pop("logical_pair_grouping", True),
        loss_backend=kwargs.pop("loss_backend", "reference"),
        performance=kwargs.pop(
            "performance",
            PerformanceConfig(adaptive_batching=_enabled_policy()),
        ),
        **kwargs,
    )


def test_adaptive_configuration_is_default_off_and_fail_closed() -> None:
    assert not TrainConfig().performance.adaptive_batching.enabled
    assert _adaptive_config().performance.adaptive_batching.enabled

    with pytest.raises(ValueError, match="logical_pair_grouping=true"):
        _adaptive_config(logical_pair_grouping=False)
    with pytest.raises(ValueError, match="batch_size=2"):
        _adaptive_config(batch_size=4)
    with pytest.raises(ValueError, match="global_batch_rows=64"):
        _adaptive_config(global_batch_rows=32)
    with pytest.raises(ValueError, match="single-process"):
        _adaptive_config(distributed="ddp")
    with pytest.raises(ValueError, match="loss_backend='reference'"):
        _adaptive_config(loss_backend="target_only")
    with pytest.raises(ValueError, match="sequence_len up to 4096"):
        _adaptive_config(sequence_len=4097)

    processor_dropout = replace(
        ModelConfig(),
        variant="bgt",
        cache_processor=CacheProcessorConfig(attention_dropout=0.1),
    )
    with pytest.raises(ValueError, match="attention_dropout=0"):
        _adaptive_config(model=processor_dropout)
    lora_dropout = replace(ModelConfig(), lora=LoRAConfig(dropout=0.1))
    with pytest.raises(ValueError, match="lora.dropout=0"):
        _adaptive_config(model=lora_dropout)

    noisy = replace(
        ModelConfig(),
        variant="bgt",
        diffusion_processor=DiffusionProcessorConfig(noise_scale=0.15),
    )
    with pytest.raises(ValueError, match="rejects noisy Diffusion"):
        _adaptive_config(model=noisy)


def test_performance_yaml_mappings_are_strictly_typed() -> None:
    config = _adaptive_config(
        performance={
            "adaptive_batching": {
                "enabled": True,
                "bands": [
                    {"max_padded_tokens": 1024, "physical_rows": 16},
                    {"max_padded_tokens": 2048, "physical_rows": 8},
                    {"max_padded_tokens": 3072, "physical_rows": 4},
                    {"max_padded_tokens": 4096, "physical_rows": 2},
                ],
            }
        }
    )
    assert config.performance.adaptive_batching.bands[0] == PhysicalBatchBandConfig(1024, 16)

    with pytest.raises(ValueError, match="strictly decreasing"):
        AdaptiveBatchingConfig(
            bands=(PhysicalBatchBandConfig(1024, 8), PhysicalBatchBandConfig(2048, 16))
        )


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        variant="tf",
        family="smollm",
        vocab_size=41,
        hidden_size=16,
        intermediate_size=24,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=8,
        tie_embeddings=True,
    )


def _mixed_policy() -> PerformanceConfig:
    return PerformanceConfig(
        adaptive_batching=AdaptiveBatchingConfig(
            enabled=True,
            bands=(
                PhysicalBatchBandConfig(2, 16),
                PhysicalBatchBandConfig(4, 8),
                PhysicalBatchBandConfig(6, 4),
                PhysicalBatchBandConfig(8, 2),
            ),
        )
    )


def test_adaptive_gradients_and_multistep_adamw_match_32_serial_pair_means() -> None:
    widths = [2] * 8 + [4] * 4 + [6] * 2 + [8] * 18
    pairs = [_pair(index, width) for index, width in enumerate(widths)]
    config = _adaptive_config(
        model=_tiny_model_config(),
        sequence_len=8,
        performance=_mixed_policy(),
        max_steps=3,
        learning_rate=3e-4,
        weight_decay=0.07,
    )
    torch.manual_seed(812)
    serial = build_model(config.model)
    serial.init_weights()
    adaptive = deepcopy(serial)
    serial_optimizer = _build_optimizer(list(serial.parameters()), config)
    adaptive_optimizer = _build_optimizer(list(adaptive.parameters()), config)

    for _update in range(3):
        serial_optimizer.zero_grad(set_to_none=True)
        adaptive_optimizer.zero_grad(set_to_none=True)
        for pair in pairs:
            loss = model_loss(
                serial,
                pair.x,
                pair.y,
                attention_mask=pair.attention_mask,
            )
            (loss / PAIRS_PER_STEP).backward()

        step = build_adaptive_pair_step(_PairStream(widths), config.performance.adaptive_batching)
        for group in step.groups:
            batch = collate_pair_group(step, group).batch
            loss = adjacent_pair_loss_sum(
                adaptive,
                batch.x,
                batch.y,
                attention_mask=batch.attention_mask,
            )
            (loss / PAIRS_PER_STEP).backward()

        for expected, actual in zip(serial.parameters(), adaptive.parameters(), strict=True):
            assert expected.grad is not None
            assert actual.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-7, rtol=2e-6)
        serial_optimizer.step()
        adaptive_optimizer.step()

    for expected, actual in zip(serial.parameters(), adaptive.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5)
        expected_state = serial_optimizer.state[expected]
        actual_state = adaptive_optimizer.state[actual]
        torch.testing.assert_close(actual_state["exp_avg"], expected_state["exp_avg"], atol=2e-8, rtol=2e-6)
        torch.testing.assert_close(
            actual_state["exp_avg_sq"],
            expected_state["exp_avg_sq"],
            atol=2e-10,
            rtol=2e-6,
        )
        assert actual_state["step"] == expected_state["step"]


class _Tracker:
    def __init__(self) -> None:
        self.logs: list[tuple[int, dict[str, Any]]] = []
        self.summary: dict[str, Any] = {}

    def log_message(self, _message: str) -> None:
        pass

    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        self.logs.append((step, dict(metrics)))

    def finish(self, *, summary: dict[str, Any]) -> None:
        self.summary = summary


def test_training_integrates_adaptive_plan_and_reports_actual_groups(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{}\n" * ROWS_PER_STEP)
    widths = [2] * 8 + [4] * 4 + [6] * 2 + [8] * 18
    config = _adaptive_config(
        model=_tiny_model_config(),
        sequence_len=8,
        max_steps=1,
        save_final=False,
        output_dir=str(tmp_path / "run"),
        data=DataConfig(kind="sft", dataset=str(dataset)),
        performance=_mixed_policy(),
        grad_clip=100.0,
        seed=812,
    )

    torch.manual_seed(config.seed)
    oracle = build_model(config.model)
    oracle.init_weights()
    expected_loss = 0.0
    with torch.no_grad():
        for pair_index, width in enumerate(widths):
            pair = _pair(pair_index, width)
            expected_loss += float(
                model_loss(oracle, pair.x, pair.y, attention_mask=pair.attention_mask)
            ) / PAIRS_PER_STEP

    monkeypatch.setattr(train_module, "init_distributed", lambda _kind: torch.device("cpu"))
    stream = _PairStream(widths)
    tracker = _Tracker()
    monkeypatch.setattr(train_module, "build_training_stream", lambda *_args, **_kwargs: stream)
    monkeypatch.setattr(
        train_module.RunTracker,
        "for_training",
        lambda _config, _run_meta, model=None: tracker,
    )

    train(config)

    assert stream.calls == PAIRS_PER_STEP
    assert len(tracker.logs) == 1
    _, metrics = tracker.logs[0]
    assert metrics["train/loss"] == pytest.approx(expected_loss, abs=2e-7)
    assert metrics["train/grad_accum_steps"] == 21
    assert metrics["train/nominal_grad_accum_steps"] == 32
    assert metrics["train/adaptive_physical_groups"] == 21
    assert metrics["train/adaptive_groups_rows_16"] == 1
    assert metrics["train/adaptive_groups_rows_8"] == 1
    assert metrics["train/adaptive_groups_rows_4"] == 1
    assert metrics["train/adaptive_groups_rows_2"] == 18
    assert tracker.summary["adaptive_batching"]["enabled"] is True


def test_adaptive_policy_is_fingerprinted_without_changing_the_objective_protocol(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{}\n" * ROWS_PER_STEP)
    config = _adaptive_config(
        model=_tiny_model_config(),
        sequence_len=8,
        performance=_mixed_policy(),
        data=DataConfig(kind="sft", dataset=str(dataset)),
    )
    fixed = replace(config, performance=PerformanceConfig())
    changed_policy = PerformanceConfig(
        adaptive_batching=AdaptiveBatchingConfig(
            enabled=True,
            bands=(
                PhysicalBatchBandConfig(1, 16),
                PhysicalBatchBandConfig(4, 8),
                PhysicalBatchBandConfig(6, 4),
                PhysicalBatchBandConfig(8, 2),
            ),
        )
    )
    changed = replace(config, performance=changed_policy)

    adaptive_fingerprint = _training_fingerprint(config, 1, 16, 512, 32)
    fixed_fingerprint = _training_fingerprint(fixed, 1, 16, 512, 32)
    changed_fingerprint = _training_fingerprint(changed, 1, 16, 512, 32)

    assert "adaptive_batching" in adaptive_fingerprint
    assert "adaptive_batching" not in fixed_fingerprint
    assert adaptive_fingerprint != changed_fingerprint
    _, model_receipt = resolve_model(config.model).require()
    experiment, _ = resolve_experiment(config, model_receipt).require()
    assert experiment.objective_protocol == "gcw.sft.answer_only_ce.adjacent_pair_token_mean.v1"
    assert "adaptive_adjacent_pair_batching" in experiment.capabilities


class _ResumeTokenizer:
    eot_id = 1

    def encode(self, text: str) -> list[int]:
        return [2 + ord(character) % 29 for character in text]


def test_one_adaptive_step_leaves_the_same_sixteen_row_epoch_tail_unconsumed() -> None:
    rows = [{"prompt": f"Q{index}", "answer": "A"} for index in range(80)]
    stream = SFTBatchStream(
        rows,
        _ResumeTokenizer(),
        batch_size=2,
        sequence_len=8,
        device="cpu",
        rank=0,
        world_size=1,
        drop_last=True,
    )

    reservation = reserve_adjacent_pairs(stream)

    assert len(reservation.pairs) == PAIRS_PER_STEP
    assert stream.cursor == SFTCursor(epoch=0, row=64)


def test_interrupted_adaptive_training_resumes_exactly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    dataset = tmp_path / "adaptive-train.jsonl"
    rows = [
        {"prompt": "Q" * (1 + index % 6), "answer": "A"}
        for index in range(ROWS_PER_STEP * 2)
    ]
    dataset.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(training_data_module, "get_tokenizer", lambda _name: _ResumeTokenizer())
    monkeypatch.setattr(train_module, "init_distributed", lambda _kind: torch.device("cpu"))

    def make_config(output_dir: str) -> TrainConfig:
        return _adaptive_config(
            model=_tiny_model_config(),
            data=DataConfig(kind="sft", dataset=str(dataset), tokenizer="test"),
            sequence_len=8,
            performance=_mixed_policy(),
            max_steps=2,
            output_dir=output_dir,
            save_final=True,
            eval_every=0,
            eval_on_start=False,
            seed=291,
        )

    baseline = make_config(str(tmp_path / "baseline"))
    train(baseline)

    interrupted = make_config(str(tmp_path / "resumed"))
    interrupted.save_every = 1

    def save_then_interrupt(
        path,
        model,
        model_config,
        optimizer,
        resume_state,
        training_fingerprint,
    ):
        real_save_training_checkpoint(
            path,
            model,
            model_config,
            optimizer,
            resume_state,
            training_fingerprint,
        )
        if resume_state.step == 1 and str(path).endswith("checkpoint"):
            raise RuntimeError("simulated adaptive interruption")

    monkeypatch.setattr(train_module, "save_training_checkpoint", save_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated adaptive interruption"):
        train(interrupted)

    monkeypatch.setattr(train_module, "save_training_checkpoint", real_save_training_checkpoint)
    resumed = make_config(str(tmp_path / "resumed"))
    resumed.save_every = 1
    resumed.resume = ResumeConfig(checkpoint=str(tmp_path / "resumed" / "checkpoint"))
    train(resumed)

    baseline_model, _, baseline_state = load_checkpoint(tmp_path / "baseline" / "checkpoint")
    resumed_model, _, resumed_state = load_checkpoint(tmp_path / "resumed" / "checkpoint")
    assert baseline_state["step"] == resumed_state["step"] == 2
    for expected, actual in zip(baseline_model.parameters(), resumed_model.parameters(), strict=True):
        assert torch.equal(expected, actual)
