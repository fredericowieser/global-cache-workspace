from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from threading import Barrier

import pytest
import torch

from gcw.config import CacheProcessorConfig, DataConfig, ModelConfig, ResumeConfig, TrainConfig, to_dict
from gcw.models import build_model
from gcw.models.cache_processors import longshot as longshot_module
from gcw.models.cache_processors.longshot import (
    _gradient_checkpointing_enabled,
    longshot_gradient_checkpointing_override,
)
from gcw.performance.checkpointing import (
    longshot_gradient_checkpointing_decision as _longshot_gradient_checkpointing_decision,
    longshot_gradient_checkpointing_metadata as _longshot_gradient_checkpointing_runtime,
)
from gcw.training.objectives import model_loss as _model_loss
from gcw.training.checkpoint import CheckpointCompatibilityError
from gcw.training.audit import _training_fingerprint
from gcw.training.core import (
    _eval_loss,
    train,
)


def _model_config(*, iteration_mode: str = "internal", iterations: int = 1) -> ModelConfig:
    return ModelConfig(
        variant="longshot",
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
            encoder_method="query_attention",
            encoder_query_source="final_token",
            iterations=iterations,
            iteration_mode=iteration_mode,  # type: ignore[arg-type]
        ),
    )


def _train_config(tmp_path, *, threshold: int | None = 8, iterations: int = 1) -> TrainConfig:
    return TrainConfig(
        model=_model_config(iterations=iterations),
        data=DataConfig(kind="random"),
        trainable="cache_processor",
        longshot_gradient_checkpointing_seq_len_threshold=threshold,
        output_dir=str(tmp_path / "run"),
        batch_size=1,
        sequence_len=8,
        max_steps=1,
        learning_rate=1e-3,
        warmup_steps=1,
        save_final=True,
        eval_every=0,
    )


def _processor_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.cache_processor.named_parameters()
        if parameter.grad is not None
    }


@pytest.mark.parametrize("iteration_mode", ["internal", "external"])
def test_threshold_boundary_controls_all_longshot_recurrence(monkeypatch, tmp_path, iteration_mode: str) -> None:
    calls: list[dict[str, object]] = []

    def fake_checkpoint(function, *args, **kwargs):
        calls.append(kwargs)
        return function(*args)

    monkeypatch.setattr(longshot_module, "checkpoint", fake_checkpoint)
    config = _train_config(tmp_path, threshold=8)
    config.model.cache_processor.iteration_mode = iteration_mode  # type: ignore[union-attr,assignment]
    config.model.cache_processor.iterations = 2  # type: ignore[union-attr]
    model = build_model(config.model).train()
    model.init_weights()

    short = torch.randint(0, 32, (1, 7))
    assert _longshot_gradient_checkpointing_decision(config, short) is False
    _model_loss(
        model,
        short,
        short,
        longshot_gradient_checkpointing=_longshot_gradient_checkpointing_decision(config, short),
    )
    assert calls == []

    boundary = torch.randint(0, 32, (1, 8))
    assert _longshot_gradient_checkpointing_decision(config, boundary) is True
    _model_loss(
        model,
        boundary,
        boundary,
        longshot_gradient_checkpointing=_longshot_gradient_checkpointing_decision(config, boundary),
    )
    assert calls
    assert all(call == {"use_reentrant": False} for call in calls)


def test_adaptive_longshot_forward_backward_matches_eager(tmp_path) -> None:
    torch.manual_seed(7)
    config = _train_config(tmp_path)
    eager = build_model(config.model).train()
    eager.init_weights()
    checkpointed = copy.deepcopy(eager)
    input_ids = torch.randint(0, 32, (2, 8))

    eager_loss = _model_loss(eager, input_ids, input_ids, longshot_gradient_checkpointing=False)
    checkpointed_loss = _model_loss(checkpointed, input_ids, input_ids, longshot_gradient_checkpointing=True)
    eager_loss.backward()
    checkpointed_loss.backward()

    assert torch.equal(eager_loss, checkpointed_loss)
    assert _processor_gradients(eager).keys() == _processor_gradients(checkpointed).keys()
    for name, expected in _processor_gradients(eager).items():
        assert torch.equal(expected, _processor_gradients(checkpointed)[name]), name


def test_adaptive_longshot_repeated_optimizer_steps_match_eager(tmp_path) -> None:
    torch.manual_seed(11)
    config = _train_config(tmp_path)
    eager = build_model(config.model).train()
    eager.init_weights()
    checkpointed = copy.deepcopy(eager)
    eager_optimizer = torch.optim.AdamW(eager.cache_processor.parameters(), lr=1e-3)
    checkpointed_optimizer = torch.optim.AdamW(checkpointed.cache_processor.parameters(), lr=1e-3)

    for input_ids in (torch.randint(0, 32, (1, 7)), torch.randint(0, 32, (1, 8))):
        for model, optimizer, enabled in (
            (eager, eager_optimizer, False),
            (checkpointed, checkpointed_optimizer, input_ids.shape[1] >= 8),
        ):
            optimizer.zero_grad(set_to_none=True)
            _model_loss(model, input_ids, input_ids, longshot_gradient_checkpointing=enabled).backward()
            optimizer.step()

    for (eager_name, eager_parameter), (checkpointed_name, checkpointed_parameter) in zip(
        eager.cache_processor.named_parameters(),
        checkpointed.cache_processor.named_parameters(),
        strict=True,
    ):
        assert eager_name == checkpointed_name
        assert torch.equal(eager_parameter, checkpointed_parameter), eager_name


def test_override_is_exception_safe_and_thread_local() -> None:
    assert _gradient_checkpointing_enabled(True) is True
    with pytest.raises(RuntimeError, match="forward failed"):
        with longshot_gradient_checkpointing_override(False):
            assert _gradient_checkpointing_enabled(True) is False
            raise RuntimeError("forward failed")
    assert _gradient_checkpointing_enabled(True) is True

    barrier = Barrier(2)

    def scoped(enabled: bool) -> bool:
        with longshot_gradient_checkpointing_override(enabled):
            barrier.wait()
            return _gradient_checkpointing_enabled(not enabled)

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(scoped, (False, True))) == [False, True]
    assert _gradient_checkpointing_enabled(True) is True


def test_eval_restores_training_mode_and_runtime_override(tmp_path) -> None:
    config = _train_config(tmp_path)
    config.eval_batches = 1
    model = build_model(config.model).train()
    model.init_weights()

    metrics = _eval_loss(model, config, torch.device("cpu"))

    assert model.training
    assert metrics["val/tokens"] == 8.0
    assert _gradient_checkpointing_enabled(True) is True


def test_adaptive_policy_is_state_dict_and_model_config_neutral(tmp_path) -> None:
    config = _train_config(tmp_path)
    model = build_model(config.model).train()
    model.init_weights()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    model_config_before = to_dict(config.model)
    input_ids = torch.randint(0, 32, (1, 8))

    _model_loss(model, input_ids, input_ids, longshot_gradient_checkpointing=True)

    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(tensor, after[name]) for name, tensor in before.items())
    assert to_dict(config.model) == model_config_before
    assert config.model.cache_processor is not None
    assert config.model.cache_processor.gradient_checkpointing is True
    assert not any("checkpoint" in name for name in after)


def test_longshot_adaptive_config_defaults_to_fixed_model_policy_and_validates() -> None:
    fixed = TrainConfig(model=_model_config(), data=DataConfig(kind="random"))
    assert fixed.longshot_gradient_checkpointing_seq_len_threshold is None
    assert _longshot_gradient_checkpointing_decision(fixed, torch.zeros(1, 8)) is None
    assert "longshot_gradient_checkpointing_seq_len_threshold" not in _training_fingerprint(
        fixed, 1, 8, 8, 1
    )
    assert _longshot_gradient_checkpointing_runtime(fixed) == {
        "policy": "model_config",
        "model_config_gradient_checkpointing": True,
        "padded_sequence_length_threshold": None,
        "checkpoint_when": "model.cache_processor.gradient_checkpointing",
        "short_microbatches": None,
        "long_microbatches": None,
        "implementation": "torch.utils.checkpoint.checkpoint",
        "boundary": "longshot_processor_layer",
        "use_reentrant": False,
        "preserve_rng_state": True,
        "scope": "physical_train_and_eval_forward",
        "requires_training_and_grad": True,
        "sequence_length_source": "input_ids.shape[1] after batch padding",
    }

    with pytest.raises(ValueError, match="positive integer"):
        TrainConfig(
            model=_model_config(),
            data=DataConfig(kind="random"),
            longshot_gradient_checkpointing_seq_len_threshold=0,
        )
    with pytest.raises(ValueError, match="requires a Longshot"):
        TrainConfig(longshot_gradient_checkpointing_seq_len_threshold=8)
    model_config = _model_config()
    assert model_config.cache_processor is not None
    model_config.cache_processor.gradient_checkpointing = False
    with pytest.raises(ValueError, match="gradient_checkpointing=true"):
        TrainConfig(
            model=model_config,
            data=DataConfig(kind="random"),
            longshot_gradient_checkpointing_seq_len_threshold=8,
        )


def test_adaptive_policy_is_provenance_visible_and_exact_resume_checked(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GCW_DATA_ROOT", str(tmp_path / "records"))
    monkeypatch.setenv("GCW_EXECUTION_ID", "adaptive-initial")
    config = _train_config(tmp_path, threshold=8)
    train(config)
    checkpoint = tmp_path / "run" / "checkpoint"
    train_state = json.loads((checkpoint / "train_state.json").read_text(encoding="utf-8"))
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    metrics_path = next((tmp_path / "records").glob("training-runs/*/*/metrics.jsonl"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8").strip())

    runtime = train_state["longshot_gradient_checkpointing"]
    assert runtime["policy"] == "padded_sequence_length_threshold"
    assert runtime["padded_sequence_length_threshold"] == 8
    assert runtime["checkpoint_when"] == "padded_input_sequence_length >= threshold"
    assert metrics["cache_processor/longshot_checkpointed_microbatches"] == 1.0
    assert metrics["cache_processor/longshot_eager_microbatches"] == 0.0
    assert manifest["training_fingerprint"]["longshot_gradient_checkpointing_seq_len_threshold"] == 8

    fingerprint = _training_fingerprint(config, 1, 8, 8, 1)
    changed = replace(config, longshot_gradient_checkpointing_seq_len_threshold=9)
    changed_fingerprint = _training_fingerprint(changed, 1, 8, 8, 1)
    assert fingerprint | {"longshot_gradient_checkpointing_seq_len_threshold": 9} == changed_fingerprint

    monkeypatch.setenv("GCW_EXECUTION_ID", "adaptive-resume")
    resumed = replace(changed, resume=ResumeConfig(checkpoint=str(checkpoint)))
    with pytest.raises(
        CheckpointCompatibilityError,
        match="longshot_gradient_checkpointing_seq_len_threshold",
    ):
        train(resumed)
