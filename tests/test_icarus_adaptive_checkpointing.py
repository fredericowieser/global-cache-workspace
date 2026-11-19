from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from threading import Barrier

import pytest
import torch

from gcw.architecture.contracts import _config_payload
from gcw.config import (
    CacheProcessorConfig,
    DataConfig,
    FlowProcessorConfig,
    ModelConfig,
    ResumeConfig,
    TrainConfig,
    load_train_config_bytes,
    to_dict,
)
from gcw.models import build_model
from gcw.models.cache_processors import icarus
from gcw.models.cache_processors.base import (
    processor_gradient_checkpointing_enabled,
    processor_gradient_checkpointing_override,
)
from gcw.performance.checkpointing import (
    processor_gradient_checkpointing_decision,
    processor_gradient_checkpointing_metadata,
    processor_gradient_checkpointing_step_metrics,
)
from gcw.training.audit import _training_fingerprint
from gcw.training.checkpoint import CheckpointCompatibilityError
from gcw.training.core import _eval_loss, train
from gcw.training.objectives import (
    adjacent_pair_loss_sum,
    logical_pair_grouped_loss,
    model_loss,
)

FIELD = "processor_gradient_checkpointing_seq_len_threshold"


def _config(tmp_path, threshold=8, *, flow=False, dropout=0.0):
    return TrainConfig(
        model=ModelConfig(
            variant="icarus",
            family="qwen3",
            sequence_len=16,
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            cache_processor=CacheProcessorConfig(
                process_every=4,
                topk=2,
                topk_heads=1,
                hidden_size=8,
                intermediate_size=16,
                num_layers=2,
                num_heads=1,
                num_kv_heads=1,
                gradient_checkpointing=True,
                attention_dropout=dropout,
            ),
            flow_processor=FlowProcessorConfig(num_steps=2) if flow else None,
        ),
        data=DataConfig(kind="random"),
        trainable="cache_processor",
        processor_gradient_checkpointing_seq_len_threshold=threshold,
        output_dir=str(tmp_path / "run"),
        batch_size=2,
        sequence_len=8,
        max_steps=1,
        learning_rate=1e-3,
        warmup_steps=1,
        save_final=True,
        eval_every=0,
    )


@pytest.mark.parametrize(
    "objective", [model_loss, logical_pair_grouped_loss, adjacent_pair_loss_sum]
)
def test_threshold_uses_physical_padded_width_and_preserves_state(
    monkeypatch, tmp_path, objective
):
    calls = []

    def checkpoint(function, *args, **kwargs):
        calls.append(kwargs)
        return function(*args)

    monkeypatch.setattr(icarus, "checkpoint", checkpoint)
    config = _config(tmp_path)
    model = build_model(config.model).train()
    model.init_weights()
    before = copy.deepcopy(model.state_dict())
    model_config = to_dict(config.model)
    for length, expected in ((7, False), (8, True), (9, True)):
        ids = torch.randint(0, 32, (2, length))
        mask = torch.ones_like(ids)
        mask[:, 4:] = 0
        decision = processor_gradient_checkpointing_decision(config, ids)
        assert decision is expected
        calls.clear()
        objective(
            model,
            ids,
            ids,
            attention_mask=mask,
            icarus_gradient_checkpointing=decision,
        )
        assert bool(calls) is expected
        assert all(call == {"use_reentrant": False} for call in calls)
    assert to_dict(config.model) == model_config
    assert before.keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name
    calls.clear()
    model_loss(model, ids, ids)
    assert calls
    calls.clear()
    with torch.no_grad():
        model_loss(model, ids, ids, icarus_gradient_checkpointing=True)
    model.eval()
    model_loss(model, ids, ids, icarus_gradient_checkpointing=True)
    assert not calls


@pytest.mark.parametrize("flow,segments", [(False, 0), (True, 0), (False, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adaptive_loss_gradients_and_optimizer_steps_match_eager(
    tmp_path, flow, segments, dtype
):
    torch.manual_seed(11)
    config = _config(tmp_path, flow=flow, dropout=0.15)
    adaptive = build_model(config.model).train()
    adaptive.init_weights()
    eager = copy.deepcopy(adaptive)
    eager.cache_processor.block.config.gradient_checkpointing = False
    models = (eager, adaptive)
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=1e-3) for model in models
    ]
    for length in (7, 8, 9):
        ids = torch.randint(0, 32, (2, length))
        rng = torch.get_rng_state()
        losses = []
        for model, optimizer in zip(models, optimizers, strict=True):
            torch.set_rng_state(rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cpu", dtype=dtype, enabled=dtype == torch.bfloat16
            ):
                loss = model_loss(
                    model,
                    ids,
                    ids,
                    activation_checkpoint_windows=segments,
                    icarus_gradient_checkpointing=(
                        (length >= 8) if model is adaptive else None
                    ),
                )
            loss.backward()
            losses.append(loss.detach())
        assert torch.equal(*losses)
        for (name, actual), (expected_name, expected) in zip(
            adaptive.named_parameters(), eager.named_parameters(), strict=True
        ):
            assert name == expected_name
            assert actual.grad is not None and expected.grad is not None
            assert torch.equal(actual.grad, expected.grad), name
        for optimizer in optimizers:
            optimizer.step()
        for actual, expected in zip(
            adaptive.parameters(), eager.parameters(), strict=True
        ):
            assert torch.equal(actual, expected)


def test_overrides_are_nested_variant_isolated_and_thread_local():
    with processor_gradient_checkpointing_override(False, variant="icarus"):
        assert not processor_gradient_checkpointing_enabled(
            True, variant="icarus"
        )
        assert processor_gradient_checkpointing_enabled(
            True, variant="longshot"
        )
        with pytest.raises(RuntimeError, match="forward failed"):
            with processor_gradient_checkpointing_override(
                True, variant="icarus"
            ):
                assert processor_gradient_checkpointing_enabled(
                    False, variant="icarus"
                )
                raise RuntimeError("forward failed")
        assert not processor_gradient_checkpointing_enabled(
            True, variant="icarus"
        )
    assert processor_gradient_checkpointing_enabled(True, variant="icarus")
    barrier = Barrier(2)

    def scoped(enabled):
        with processor_gradient_checkpointing_override(
            enabled, variant="icarus"
        ):
            barrier.wait()
            return processor_gradient_checkpointing_enabled(
                not enabled, variant="icarus"
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(scoped, (False, True))) == [False, True]


@pytest.mark.parametrize("threshold", [0, -1, True, 8.5, "8"])
def test_invalid_thresholds_are_rejected(tmp_path, threshold):
    with pytest.raises(ValueError, match="positive integer"):
        _config(tmp_path, threshold=threshold)


def test_config_defaults_admission_serialization_and_fingerprint(tmp_path):
    config = _config(tmp_path, threshold=None)
    assert (
        processor_gradient_checkpointing_decision(config, torch.zeros(2, 8))
        is None
    )
    assert FIELD not in to_dict(config)
    assert FIELD not in _config_payload(config)
    assert FIELD not in _training_fingerprint(config, 1, 8, 16, 1)
    enabled = replace(config, **{FIELD: 8})
    assert to_dict(enabled)[FIELD] == 8
    assert _config_payload(enabled)[FIELD] == 8
    assert _training_fingerprint(enabled, 1, 8, 16, 1)[FIELD] == 8
    loaded = load_train_config_bytes(json.dumps(to_dict(enabled)).encode())
    assert loaded.processor_gradient_checkpointing_seq_len_threshold == 8
    for variant in ("tf", "bt", "bgt"):
        with pytest.raises(ValueError, match="requires a Longshot or Icarus"):
            replace(enabled, model=replace(enabled.model, variant=variant))
    with pytest.raises(ValueError, match="gradient_checkpointing=true"):
        replace(
            enabled,
            model=replace(
                enabled.model,
                cache_processor=replace(
                    enabled.model.cache_processor, gradient_checkpointing=False
                ),
            ),
        )
    runtime = processor_gradient_checkpointing_metadata(
        enabled, variant="icarus"
    )
    assert runtime["boundary"] == "icarus_processor_layer"
    assert runtime["padded_sequence_length_threshold"] == 8
    assert processor_gradient_checkpointing_metadata(enabled) is None
    metrics = processor_gradient_checkpointing_step_metrics(
        [False, True], [7, 8], variant="icarus"
    )
    assert metrics == {
        "cache_processor/icarus_checkpointed_microbatches": 1.0,
        "cache_processor/icarus_eager_microbatches": 1.0,
        "cache_processor/icarus_checkpointed_microbatch_fraction": 0.5,
        "cache_processor/icarus_padded_sequence_length_min": 7.0,
        "cache_processor/icarus_padded_sequence_length_max": 8.0,
    }


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("threshold", [None, 8, 9])
def test_training_metadata_metrics_resume_and_evaluation(
    monkeypatch, tmp_path, threshold, legacy
):
    monkeypatch.setenv("GCW_DATA_ROOT", str(tmp_path / "records"))
    monkeypatch.setenv("GCW_EXECUTION_ID", "icarus-adaptive-initial")
    config = replace(_config(tmp_path, threshold=threshold), max_steps=2)
    with monkeypatch.context() as patch:
        patch.setattr(
            "gcw.training.core._checkpoint_stop_requested",
            lambda *args, **kwargs: True,
        )
        train(config)
    checkpoint = tmp_path / "run" / "checkpoint"
    state = json.loads((checkpoint / "train_state.json").read_text())
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    metrics_path = next(
        (tmp_path / "records").glob("training-runs/*/*/metrics.jsonl")
    )
    metrics = json.loads(metrics_path.read_text().strip())
    if threshold is None:
        assert "icarus_gradient_checkpointing" not in state
        assert FIELD not in manifest["training_fingerprint"]
        assert not any("icarus_checkpoint" in name for name in metrics)
    else:
        assert (
            state["icarus_gradient_checkpointing"][
                "padded_sequence_length_threshold"
            ]
            == threshold
        )
        assert manifest["training_fingerprint"][FIELD] == threshold
        assert metrics[
            "cache_processor/icarus_checkpointed_microbatches"
        ] == float(threshold <= 8)
        assert metrics["cache_processor/icarus_eager_microbatches"] == float(
            threshold > 8
        )
    if legacy and threshold is not None:
        fingerprint = manifest["training_fingerprint"]
        fingerprint["icarus_gradient_checkpointing_seq_len_threshold"] = (
            fingerprint.pop(FIELD)
        )
        (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("GCW_EXECUTION_ID", "icarus-adaptive-resume")
    train(replace(config, resume=ResumeConfig(checkpoint=str(checkpoint))))
    with pytest.raises(CheckpointCompatibilityError, match=FIELD):
        train(
            replace(
                config,
                **{FIELD: 10},
                resume=ResumeConfig(checkpoint=str(checkpoint)),
            )
        )

    def forbidden_checkpoint(*args, **kwargs):
        raise AssertionError("evaluation must remain eager")

    monkeypatch.setattr(icarus, "checkpoint", forbidden_checkpoint)
    config.eval_batches = 1
    model = build_model(config.model).train()
    model.init_weights()
    evaluation = _eval_loss(model, config, torch.device("cpu"))
    assert model.training
    assert evaluation["val/tokens"] == 16.0
    assert processor_gradient_checkpointing_enabled(True, variant="icarus")


def test_compiled_region_receives_resolved_checkpoint_policy(
    monkeypatch, tmp_path
):
    from gcw.performance.backends import icarus as backend

    config = _config(tmp_path)
    block = icarus.CacheProcessor(16, 2, config.model.cache_processor).train()
    block.__dict__["_gcw_icarus_active_training_batch"] = 2
    compiled = torch.compile(
        backend._run_icarus_post_selection, backend="eager", fullgraph=True
    )
    monkeypatch.setattr(
        backend, "_get_compiled_icarus_runner", lambda block: compiled
    )
    x = torch.randn(2, 2, 4, 16, requires_grad=True)
    positions = torch.arange(4).expand(2, 2, 4)
    heads = torch.tensor([[0], [1]])
    for enabled in (False, True, False):
        with processor_gradient_checkpointing_override(
            enabled, variant="icarus"
        ):
            expected = backend.run_icarus_post_selection_region(
                "eager", block, x, positions, heads
            )
            actual = backend.run_icarus_post_selection_region(
                "compile", block, x, positions, heads
            )
        assert torch.equal(actual, expected)
        expected_grad = torch.autograd.grad(expected.sum(), x)[0]
        actual_grad = torch.autograd.grad(actual.sum(), x)[0]
        assert torch.equal(actual_grad, expected_grad)
