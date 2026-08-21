from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
import torch

import gcw.training.core as train_module
import gcw.training.assembly as assembly_module
from gcw.architecture import resolve_experiment, resolve_model
from gcw.config import (
    DataConfig,
    DiffusionProcessorConfig,
    ModelConfig,
    TrainConfig,
)
from gcw.data.training import SFTBatchStream, SFTCursor, TrainingBatch
from gcw.models import build_model
from gcw.training.audit import _training_fingerprint
from gcw.training.objectives import (
    logical_pair_grouped_loss as _logical_pair_grouped_loss,
    model_loss as _model_loss,
)
from gcw.training.optimizer import _build_optimizer
from gcw.training.core import (
    _eval_loss,
    train,
)


class _Tokenizer:
    eot_id = 1

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def encode(self, text: str) -> list[int]:
        if text.startswith("Q"):
            self.prompts.append(text)
        return [2 + ord(char) % 29 for char in text]


def _model_config() -> ModelConfig:
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


def _grouped_config(
    *, batch_size: int = 4, global_batch_rows: int = 8, **kwargs: Any
) -> TrainConfig:
    return TrainConfig(
        model=kwargs.pop("model", _model_config()),
        data=kwargs.pop("data", DataConfig(kind="sft", dataset="unused.jsonl")),
        mode=kwargs.pop("mode", "sft"),
        batch_size=batch_size,
        sequence_len=8,
        global_batch_rows=global_batch_rows,
        logical_pair_grouping=kwargs.pop("logical_pair_grouping", True),
        save_final=False,
        eval_every=0,
        **kwargs,
    )


def _unequal_batch(
    step: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(900 + step)
    x = torch.randint(2, 41, (8, 8), generator=generator)
    counts = (1, 3, 2, 5, 4, 1, 6, 2)
    lengths = (3, 5, 4, 8, 7, 2, 8, 6)
    y = torch.full_like(x, -100)
    attention_mask = torch.zeros_like(x, dtype=torch.bool)
    for row, (count, length) in enumerate(zip(counts, lengths, strict=True)):
        y[row, length - count : length] = (
            x[row, length - count : length] + 1
        ) % 41
        attention_mask[row, :length] = True
        x[row, length:] = 0
    return x, y, attention_mask


def _serial_pair_backward(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    for start in range(0, x.shape[0], 2):
        loss = _model_loss(
            model,
            x[start : start + 2],
            y[start : start + 2],
            attention_mask=attention_mask[start : start + 2],
        )
        (loss / (x.shape[0] // 2)).backward()


def _physical_group_backward(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    physical_batch = 4
    for start in range(0, x.shape[0], physical_batch):
        loss = _logical_pair_grouped_loss(
            model,
            x[start : start + physical_batch],
            y[start : start + physical_batch],
            attention_mask=attention_mask[start : start + physical_batch],
        )
        (loss / (x.shape[0] // physical_batch)).backward()


def test_grouped_gradients_and_multistep_adamw_match_serial_batch_two() -> None:
    torch.manual_seed(77)
    serial = build_model(_model_config())
    serial.init_weights()
    grouped = deepcopy(serial)
    config = _grouped_config(learning_rate=3e-4, weight_decay=0.07, max_steps=3)
    serial_opt = _build_optimizer(list(serial.parameters()), config)
    grouped_opt = _build_optimizer(list(grouped.parameters()), config)

    for step in range(3):
        x, y, attention_mask = _unequal_batch(step)
        serial_opt.zero_grad(set_to_none=True)
        grouped_opt.zero_grad(set_to_none=True)
        _serial_pair_backward(serial, x, y, attention_mask)
        _physical_group_backward(grouped, x, y, attention_mask)

        for expected, actual in zip(
            serial.parameters(), grouped.parameters(), strict=True
        ):
            assert expected.grad is not None
            assert actual.grad is not None
            torch.testing.assert_close(
                actual.grad, expected.grad, atol=2e-7, rtol=2e-6
            )
        serial_opt.step()
        grouped_opt.step()

    for expected, actual in zip(
        serial.parameters(), grouped.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        expected_state = serial_opt.state[expected]
        actual_state = grouped_opt.state[actual]
        torch.testing.assert_close(
            actual_state["exp_avg"],
            expected_state["exp_avg"],
            atol=2e-8,
            rtol=2e-6,
        )
        torch.testing.assert_close(
            actual_state["exp_avg_sq"],
            expected_state["exp_avg_sq"],
            atol=2e-10,
            rtol=2e-6,
        )
        assert actual_state["step"] == expected_state["step"]


def test_grouped_stream_drops_tail_without_cross_epoch_pairs_and_resumes() -> (
    None
):
    rows = [{"prompt": f"Q{index}", "answer": "A"} for index in range(10)]
    tokenizer = _Tokenizer()
    stream = SFTBatchStream(
        rows,
        tokenizer,
        batch_size=4,
        sequence_len=8,
        device="cpu",
        rank=0,
        world_size=1,
        drop_last=True,
    )

    next(stream)
    next(stream)
    assert tokenizer.prompts == [f"Q{index}" for index in range(8)]
    assert stream.cursor == SFTCursor(epoch=0, row=8)

    resumed_tokenizer = _Tokenizer()
    resumed = SFTBatchStream(
        rows,
        resumed_tokenizer,
        batch_size=4,
        sequence_len=8,
        device="cpu",
        rank=0,
        world_size=1,
        cursor=stream.cursor,
        drop_last=True,
    )
    next(resumed)

    assert resumed_tokenizer.prompts == ["Q0", "Q1", "Q2", "Q3"]
    assert resumed.cursor == SFTCursor(epoch=1, row=4)


class _FixedStream:
    def __init__(self, batches: list[TrainingBatch]) -> None:
        self.batches = batches
        self.index = 0

    @property
    def cursor(self) -> SFTCursor:
        rows = self.index * self.batches[0].x.shape[0]
        return SFTCursor(epoch=rows // 64, row=rows % 64)

    def __next__(self) -> TrainingBatch:
        batch = self.batches[self.index % len(self.batches)]
        self.index += 1
        return batch


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


def test_training_metrics_preserve_32_pair_global_objective(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{}\n" * 64)
    config = _grouped_config(
        batch_size=8,
        global_batch_rows=64,
        max_steps=1,
        output_dir=str(tmp_path / "run"),
        grad_clip=100.0,
        seed=31,
        data=DataConfig(kind="sft", dataset=str(dataset)),
    )
    batches = []
    for batch_index in range(8):
        x, y, attention_mask = _unequal_batch(batch_index)
        batches.append(TrainingBatch(x=x, y=y, attention_mask=attention_mask))

    torch.manual_seed(config.seed)
    oracle = build_model(config.model)
    oracle.init_weights()
    with torch.no_grad():
        expected_loss = sum(
            float(
                _logical_pair_grouped_loss(
                    oracle,
                    batch.x,
                    batch.y,
                    attention_mask=batch.attention_mask,
                )
            )
            for batch in batches
        ) / len(batches)

    tracker = _Tracker()
    # This test builds a CPU oracle and CPU batches; pin the training device to CPU
    # so the single step reproduces the oracle loss on GPU hosts too.
    monkeypatch.setattr(
        assembly_module, "init_distributed", lambda _kind: torch.device("cpu")
    )
    monkeypatch.setattr(
        assembly_module,
        "build_training_stream",
        lambda *_args, **_kwargs: _FixedStream(batches),
    )
    monkeypatch.setattr(
        assembly_module.RunTracker,
        "for_training",
        lambda _config, _run_meta, model=None: tracker,
    )
    train(config)

    assert len(tracker.logs) == 1
    _, metrics = tracker.logs[0]
    assert metrics["train/loss"] == pytest.approx(expected_loss, abs=1e-7)
    assert metrics["train/grad_accum_steps"] == 8
    assert metrics["train/rows_per_step"] == 64
    assert metrics["train/logical_pairs_per_step"] == 32
    assert tracker.summary["logical_pair_grouping"] is True
    assert tracker.summary["logical_pairs_per_step"] == 32


def test_grouped_validation_loss_averages_pair_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _grouped_config(
        batch_size=4,
        global_batch_rows=8,
        eval_batches=2,
        data=DataConfig(
            kind="sft", dataset="unused", validation_dataset="unused-validation"
        ),
    )
    x, y, attention_mask = _unequal_batch()
    batches = [
        TrainingBatch(x=x[:4], y=y[:4], attention_mask=attention_mask[:4]),
        TrainingBatch(x=x[4:], y=y[4:], attention_mask=attention_mask[4:]),
    ]
    torch.manual_seed(12)
    model = build_model(config.model)
    model.init_weights()
    with torch.no_grad():
        expected = sum(
            float(
                _logical_pair_grouped_loss(
                    model, batch.x, batch.y, attention_mask=batch.attention_mask
                )
            )
            for batch in batches
        ) / len(batches)
    monkeypatch.setattr(
        train_module,
        "build_training_stream",
        lambda *_args, **_kwargs: _FixedStream(batches),
    )

    metrics = _eval_loss(model, config, torch.device("cpu"))

    assert metrics["val/loss"] == pytest.approx(expected, abs=2e-7)
    assert metrics["val/tokens"] == 24


def test_grouped_validation_without_held_out_sft_data_uses_random_fallback() -> (
    None
):
    config = _grouped_config(batch_size=4, global_batch_rows=8, eval_batches=2)
    model = build_model(config.model)
    model.init_weights()

    metrics = _eval_loss(model, config, torch.device("cpu"))

    assert torch.isfinite(torch.tensor(metrics["val/loss"]))
    assert metrics["val/tokens"] == 64


def test_logical_pair_grouping_is_explicit_in_fingerprint_and_admission_protocol(
    tmp_path,
) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{}\n")
    grouped = _grouped_config(data=DataConfig(kind="sft", dataset=str(dataset)))
    ungrouped = replace(grouped, logical_pair_grouping=False)
    grouped_fingerprint = _training_fingerprint(grouped, 1, 32, 64, 2)
    ungrouped_fingerprint = _training_fingerprint(ungrouped, 1, 32, 64, 2)

    assert (
        grouped_fingerprint["objective"]
        == "adjacent_pair_token_mean_answer_only_ce"
    )
    assert grouped_fingerprint["batch"]["logical_pair_grouping"] is True
    assert grouped_fingerprint["batch"]["logical_pairs_per_step"] == 4
    assert grouped_fingerprint != ungrouped_fingerprint

    _, model_receipt = resolve_model(grouped.model).require()
    experiment, _ = resolve_experiment(grouped, model_receipt).require()
    assert (
        experiment.objective_protocol
        == "gcw.sft.answer_only_ce.adjacent_pair_token_mean.v1"
    )


def test_logical_pair_grouping_validation_and_default() -> None:
    assert TrainConfig().logical_pair_grouping is False
    with pytest.raises(ValueError, match="must be a boolean"):
        _grouped_config(logical_pair_grouping="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires SFT data"):
        TrainConfig(
            batch_size=4, global_batch_rows=8, logical_pair_grouping=True
        )
    with pytest.raises(ValueError, match="answer-only"):
        _grouped_config(
            data=DataConfig(
                kind="sft", dataset="unused", sft_loss_scope="full_sequence"
            )
        )
    with pytest.raises(ValueError, match="even physical batch_size"):
        _grouped_config(batch_size=3)
    with pytest.raises(ValueError, match="positive even global_batch_rows"):
        _grouped_config(global_batch_rows=0)
    with pytest.raises(ValueError, match="divisible by physical batch_size"):
        _grouped_config(batch_size=4, global_batch_rows=10)
    with pytest.raises(ValueError, match="router auxiliary"):
        _grouped_config(
            model=replace(_model_config(), router_aux_loss_coef=0.01)
        )

    noisy = replace(
        _model_config(),
        variant="bgt",
        diffusion_processor=DiffusionProcessorConfig(noise_scale=0.15),
    )
    with pytest.raises(ValueError, match="rejects noisy Diffusion"):
        _grouped_config(model=noisy)
    deterministic = replace(
        _model_config(),
        variant="bgt",
        diffusion_processor=DiffusionProcessorConfig(noise_scale=0.0),
    )
    assert _grouped_config(model=deterministic).logical_pair_grouping is True
