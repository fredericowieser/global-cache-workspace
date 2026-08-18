from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
import torch

from gcw.checkpoint import load_checkpoint
from gcw.config import (
    CacheProcessorConfig,
    DataConfig,
    DiffusionProcessorConfig,
    ModelConfig,
    ResumeConfig,
    TrainConfig,
)
from gcw.models import build_model
from gcw.models.cache_processors.standard import CacheProcessor
from gcw.performance.checkpointing import (
    configure_native_activation_checkpointing as _configure_native_activation_checkpointing,
)
from gcw.training.objectives import logical_pair_grouped_loss as _logical_pair_grouped_loss
from gcw.training.checkpoint import CheckpointCompatibilityError, save_training_checkpoint
from gcw.training.audit import _training_fingerprint
from gcw.training.model_setup import _apply_trainable_policy
from gcw.training.core import train


def _model_config(
    variant: str,
    *,
    attention_dropout: float = 0.0,
    diffusion: bool = False,
) -> ModelConfig:
    return ModelConfig(
        variant=variant,  # type: ignore[arg-type]
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
            topk_heads=2,
            encoder_method="query_attention",
            encoder_query_source="final_token",
            hidden_size=8,
            intermediate_size=16,
            num_layers=2,
            num_heads=1,
            num_kv_heads=1,
            attention_dropout=attention_dropout,
        ),
        diffusion_processor=DiffusionProcessorConfig(num_steps=2, noise_scale=0.2) if diffusion else None,
    )


def _processor_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.cache_processor.named_parameters()
        if parameter.grad is not None
    }


@pytest.mark.parametrize("variant", ["bt", "bgt"])
def test_standard_processor_checkpointing_forward_backward_parity(variant: str) -> None:
    torch.manual_seed(7)
    eager = build_model(_model_config(variant))
    eager.init_weights()
    checkpointed = copy.deepcopy(eager)
    _apply_trainable_policy(eager, "cache_processor")
    _apply_trainable_policy(checkpointed, "cache_processor")
    runtime = _configure_native_activation_checkpointing(
        checkpointed,
        "standard_processor_layers",
    )
    input_ids = torch.randint(0, 32, (2, 8))

    eager_loss = eager(input_ids, input_ids)
    checkpointed_loss = checkpointed(input_ids, input_ids)
    eager_loss.backward()
    checkpointed_loss.backward()

    assert torch.equal(eager_loss, checkpointed_loss)
    assert _processor_gradients(eager).keys() == _processor_gradients(checkpointed).keys()
    for name, expected in _processor_gradients(eager).items():
        assert torch.equal(expected, _processor_gradients(checkpointed)[name]), name
    assert runtime == {
        "policy": "standard_processor_layers",
        "implementation": "torch.utils.checkpoint.checkpoint",
        "boundary": "standard_processor_layer",
        "use_reentrant": False,
        "preserve_rng_state": True,
        "processor_blocks": 2 if variant == "bt" else 1,
        "layer_boundaries": 4 if variant == "bt" else 2,
    }


@pytest.mark.parametrize("variant", ["bt", "bgt", "longshot"])
def test_segmented_window_checkpointing_forward_backward_parity(variant: str) -> None:
    torch.manual_seed(17)
    eager = build_model(_model_config(variant))
    eager.init_weights()
    checkpointed = copy.deepcopy(eager)
    input_ids = torch.randint(0, 32, (2, 12))

    eager_loss = eager(input_ids, input_ids)
    checkpointed_loss = checkpointed(input_ids, input_ids, activation_checkpoint_windows=2)
    eager_loss.backward()
    checkpointed_loss.backward()

    assert torch.equal(checkpointed_loss, eager_loss)
    assert checkpointed.cache_processor.pop_call_count() == 2
    for (name, expected), (actual_name, actual) in zip(
        eager.named_parameters(),
        checkpointed.named_parameters(),
        strict=True,
    ):
        assert name == actual_name
        assert expected.grad is not None, name
        assert actual.grad is not None, name
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize(
    ("minimum_length", "expected_owners"),
    [(0, 4), (12, 4), (13, 1)],
    ids=["zero-preserves-always-on", "at-threshold", "below-threshold"],
)
def test_adaptive_segmented_bt_prepared_owner_scope(
    monkeypatch: pytest.MonkeyPatch,
    minimum_length: int,
    expected_owners: int,
) -> None:
    model = build_model(_model_config("bt"))
    model.init_weights()
    owners: list[object] = []
    prepare = model.cache_processor.prepare_forward_execution

    def track_owner():
        owner = prepare()
        owners.append(owner)
        return owner

    monkeypatch.setattr(model.cache_processor, "prepare_forward_execution", track_owner)
    input_ids = torch.randint(0, 32, (1, 12))
    model(
        input_ids,
        input_ids,
        activation_checkpoint_windows=2,
        activation_checkpoint_min_padded_sequence_length=minimum_length,
    ).backward()

    assert len(owners) == expected_owners
    assert all(getattr(owner, "_closed") for owner in owners)


def test_adaptive_segment_policy_composes_with_grouped_logical_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_model(_model_config("bgt"))
    model.init_weights()
    effective_windows: list[int] = []
    forward = model.backbone.forward

    def track_forward(*args, **kwargs):
        effective_windows.append(kwargs["activation_checkpoint_windows"])
        return forward(*args, **kwargs)

    monkeypatch.setattr(model.backbone, "forward", track_forward)
    input_ids = torch.randint(0, 32, (4, 12))

    below = _logical_pair_grouped_loss(
        model,
        input_ids,
        input_ids,
        activation_checkpoint_windows=2,
        activation_checkpoint_min_padded_sequence_length=13,
    )
    at = _logical_pair_grouped_loss(
        model,
        input_ids,
        input_ids,
        activation_checkpoint_windows=2,
        activation_checkpoint_min_padded_sequence_length=12,
    )
    model.eval()
    with torch.no_grad():
        evaluated = model(
            input_ids,
            input_ids,
            activation_checkpoint_windows=2,
            activation_checkpoint_min_padded_sequence_length=12,
        )

    assert below.shape == at.shape == evaluated.shape == ()
    assert effective_windows == [0, 2, 0]


def test_segment_recomputation_does_not_duplicate_processor_metrics() -> None:
    config = _model_config("longshot")
    assert config.cache_processor is not None
    config.cache_processor.iterations = 2
    model = build_model(config)
    model.init_weights()
    input_ids = torch.randint(0, 32, (1, 12))

    loss = model(input_ids, input_ids, activation_checkpoint_windows=2)
    block = model.cache_processor.block
    forward_counts = dict(block._selected_head_overlap_counts)
    forward_calls = model.cache_processor.pop_call_count()
    loss.backward()

    assert forward_counts == {1: 2}
    assert block._selected_head_overlap_counts == forward_counts
    assert forward_calls == 2
    assert model.cache_processor.pop_call_count() == 0


def test_segmented_checkpointing_preserves_rng_and_composes_with_processor_checkpointing() -> None:
    torch.manual_seed(29)
    eager = build_model(_model_config("bgt", attention_dropout=0.25, diffusion=True))
    eager.init_weights()
    checkpointed = copy.deepcopy(eager)
    _configure_native_activation_checkpointing(checkpointed, "standard_processor_layers")
    input_ids = torch.randint(0, 32, (2, 12))
    torch.manual_seed(101)
    initial_rng = torch.random.get_rng_state()

    eager_loss = eager(input_ids, input_ids)
    eager_loss.backward()
    eager_rng = torch.random.get_rng_state()

    torch.random.set_rng_state(initial_rng)
    checkpointed_loss = checkpointed(
        input_ids,
        input_ids,
        activation_checkpoint_windows=2,
        activation_checkpoint_min_padded_sequence_length=12,
    )
    checkpointed_loss.backward()

    assert torch.equal(checkpointed_loss, eager_loss)
    assert torch.equal(torch.random.get_rng_state(), eager_rng)
    for (name, expected), (actual_name, actual) in zip(
        eager.named_parameters(),
        checkpointed.named_parameters(),
        strict=True,
    ):
        assert name == actual_name
        assert expected.grad is not None and actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)


def test_checkpointing_preserves_diffusion_and_attention_rng() -> None:
    torch.manual_seed(11)
    eager = build_model(_model_config("bgt", attention_dropout=0.25, diffusion=True))
    eager.init_weights()
    checkpointed = copy.deepcopy(eager)
    _apply_trainable_policy(eager, "cache_processor")
    _apply_trainable_policy(checkpointed, "cache_processor")
    _configure_native_activation_checkpointing(checkpointed, "standard_processor_layers")
    input_ids = torch.randint(0, 32, (2, 8))
    torch.manual_seed(91)
    initial_rng = torch.random.get_rng_state()

    eager_loss = eager(input_ids, input_ids)
    eager_loss.backward()
    eager_final_rng = torch.random.get_rng_state()
    eager_gradients = _processor_gradients(eager)

    torch.random.set_rng_state(initial_rng)
    checkpointed_loss = checkpointed(input_ids, input_ids)
    checkpointed_loss.backward()

    assert torch.equal(eager_loss, checkpointed_loss)
    assert torch.equal(torch.random.get_rng_state(), eager_final_rng)
    for name, expected in eager_gradients.items():
        assert torch.equal(expected, _processor_gradients(checkpointed)[name]), name


def test_checkpoint_policy_does_not_change_state_dict_schema() -> None:
    model = build_model(_model_config("bt"))
    model.init_weights()
    before = {name: value.clone() for name, value in model.state_dict().items()}

    _configure_native_activation_checkpointing(model, "standard_processor_layers")

    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(value, after[name]) for name, value in before.items())
    assert not any("checkpoint" in name for name in after)


def test_processor_layer_boundary_reduces_saved_activation_tensors() -> None:
    config = CacheProcessorConfig(
        hidden_size=16,
        intermediate_size=64,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
    )
    eager = CacheProcessor(32, config).train()
    checkpointed = copy.deepcopy(eager)
    checkpointed.activation_checkpointing = True
    positions = torch.arange(8).expand(2, -1)

    def saved_numel(processor: CacheProcessor) -> int:
        saved: list[int] = []
        state = torch.randn(2, 8, 32, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: (saved.append(tensor.numel()) or tensor),
            lambda tensor: tensor,
        ):
            output = processor(state, positions)
        output.sum().backward()
        return sum(saved)

    eager_saved = saved_numel(eager)
    checkpointed_saved = saved_numel(checkpointed)

    assert checkpointed_saved < eager_saved / 2


def test_segmented_window_boundary_reduces_saved_tensors_and_preserves_state_dict() -> None:
    torch.manual_seed(31)
    eager = build_model(_model_config("bt"))
    eager.init_weights()
    checkpointed = copy.deepcopy(eager)
    input_ids = torch.randint(0, 32, (2, 12))
    before = {name: tensor.clone() for name, tensor in checkpointed.state_dict().items()}

    def saved_numel(model: torch.nn.Module, minimum_length: int) -> int:
        saved: list[int] = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: (saved.append(tensor.numel()) or tensor),
            lambda tensor: tensor,
        ):
            loss = model(
                input_ids,
                input_ids,
                activation_checkpoint_windows=2,
                activation_checkpoint_min_padded_sequence_length=minimum_length,
            )
        loss.backward()
        return sum(saved)

    eager_saved = saved_numel(eager, 13)
    checkpointed_saved = saved_numel(checkpointed, 12)

    assert checkpointed_saved < eager_saved / 2
    assert tuple(checkpointed.state_dict()) == tuple(before)
    assert all(torch.equal(tensor, before[name]) for name, tensor in checkpointed.state_dict().items())
    restored = build_model(_model_config("bt"))
    incompatible = restored.load_state_dict(checkpointed.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_checkpointed_bt_repeated_adamw_steps_match_eager() -> None:
    torch.manual_seed(23)
    eager = build_model(_model_config("bt"))
    eager.init_weights()
    checkpointed = copy.deepcopy(eager)
    _apply_trainable_policy(eager, "cache_processor")
    _apply_trainable_policy(checkpointed, "cache_processor")
    _configure_native_activation_checkpointing(checkpointed, "standard_processor_layers")
    eager_optimizer = torch.optim.AdamW(eager.cache_processor.parameters(), lr=1e-3)
    checkpointed_optimizer = torch.optim.AdamW(checkpointed.cache_processor.parameters(), lr=1e-3)
    batches = [torch.randint(0, 32, (2, 8)) for _ in range(2)]

    for input_ids in batches:
        for model, optimizer in (
            (eager, eager_optimizer),
            (checkpointed, checkpointed_optimizer),
        ):
            optimizer.zero_grad(set_to_none=True)
            model(input_ids, input_ids).backward()
            optimizer.step()

    for (eager_name, eager_parameter), (checkpointed_name, checkpointed_parameter) in zip(
        eager.cache_processor.named_parameters(),
        checkpointed.cache_processor.named_parameters(),
        strict=True,
    ):
        assert eager_name == checkpointed_name
        assert torch.equal(eager_parameter, checkpointed_parameter), eager_name
    assert eager_optimizer.state_dict()["param_groups"] == checkpointed_optimizer.state_dict()["param_groups"]
    for eager_state, checkpointed_state in zip(
        eager_optimizer.state_dict()["state"].values(),
        checkpointed_optimizer.state_dict()["state"].values(),
        strict=True,
    ):
        for name, expected in eager_state.items():
            actual = checkpointed_state[name]
            assert torch.equal(expected, actual) if isinstance(expected, torch.Tensor) else expected == actual


def _train_config(tmp_path) -> TrainConfig:
    return TrainConfig(
        model=_model_config("bgt"),
        data=DataConfig(kind="random"),
        trainable="cache_processor",
        output_dir=str(tmp_path / "run"),
        batch_size=1,
        sequence_len=8,
        max_steps=1,
        learning_rate=1e-3,
        warmup_steps=1,
        save_final=True,
        eval_every=0,
    )


def test_checkpoint_policy_is_runtime_provenance_and_resume_fingerprint(tmp_path) -> None:
    eager = _train_config(tmp_path)
    train(eager)
    checkpoint = tmp_path / "run" / "checkpoint"
    train_state = json.loads((checkpoint / "train_state.json").read_text(encoding="utf-8"))
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))

    assert train_state["activation_checkpointing"]["policy"] == "none"
    assert train_state["activation_checkpointing"]["implementation"] == "eager"
    assert manifest["training_fingerprint"]["activation_checkpointing"] == "none"
    assert train_state["activation_checkpoint_windows"] == 0
    assert manifest["training_fingerprint"]["activation_checkpoint_windows"] == 0
    assert train_state["activation_checkpoint_min_padded_sequence_length"] == 0
    assert manifest["training_fingerprint"]["activation_checkpoint_min_padded_sequence_length"] == 0

    resumed = replace(
        eager,
        activation_checkpointing="standard_processor_layers",
        resume=ResumeConfig(checkpoint=str(checkpoint)),
    )
    with pytest.raises(CheckpointCompatibilityError, match="activation_checkpointing"):
        train(resumed)


    resumed_windows = replace(
        eager,
        activation_checkpoint_windows=2,
        resume=ResumeConfig(checkpoint=str(checkpoint)),
    )
    with pytest.raises(CheckpointCompatibilityError, match="activation_checkpoint_windows"):
        train(resumed_windows)

    resumed_threshold = replace(
        eager,
        activation_checkpoint_min_padded_sequence_length=9,
        resume=ResumeConfig(checkpoint=str(checkpoint)),
    )
    with pytest.raises(
        CheckpointCompatibilityError,
        match="activation_checkpoint_min_padded_sequence_length",
    ):
        train(resumed_threshold)


def test_training_and_eval_apply_segment_policy_and_record_provenance(tmp_path) -> None:
    config = replace(
        _train_config(tmp_path),
        output_dir=str(tmp_path / "segmented"),
        activation_checkpoint_windows=2,
        activation_checkpoint_min_padded_sequence_length=8,
        eval_every=1,
        eval_batches=1,
    )

    train(config)

    checkpoint = tmp_path / "segmented" / "checkpoint"
    train_state = json.loads((checkpoint / "train_state.json").read_text(encoding="utf-8"))
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert train_state["activation_checkpoint_windows"] == 2
    assert manifest["training_fingerprint"]["activation_checkpoint_windows"] == 2
    assert train_state["activation_checkpoint_min_padded_sequence_length"] == 8
    assert manifest["training_fingerprint"]["activation_checkpoint_min_padded_sequence_length"] == 8


@pytest.mark.parametrize(("minimum_length", "active_count"), [(8, 1), (9, 0)])
def test_adaptive_segment_policy_is_visible_in_per_step_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    minimum_length: int,
    active_count: int,
) -> None:
    logged: list[dict[str, object]] = []

    class Tracker:
        def log_message(self, _message: str) -> None:
            pass

        def log(self, metrics: dict[str, object], *, step: int) -> None:
            del step
            logged.append(metrics)

        def finish(self, *, summary: dict[str, object]) -> None:
            del summary

    config = replace(
        _train_config(tmp_path),
        output_dir=str(tmp_path / f"metrics-{minimum_length}"),
        activation_checkpoint_windows=2,
        activation_checkpoint_min_padded_sequence_length=minimum_length,
        save_final=False,
    )
    monkeypatch.setattr(
        "gcw.training.assembly.RunTracker.for_training",
        lambda *_args, **_kwargs: Tracker(),
    )

    train(config)

    assert len(logged) == 1
    metrics = logged[0]
    assert metrics["train/segmented_window_checkpointing_active"] is bool(active_count)
    assert metrics["train/segmented_window_checkpointed_microbatches"] == active_count
    assert metrics["train/segmented_window_eager_microbatches"] == 1 - active_count


def test_segmented_training_exact_resume_matches_uninterrupted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    baseline = replace(
        _train_config(tmp_path),
        output_dir=str(tmp_path / "baseline"),
        max_steps=3,
        activation_checkpoint_windows=2,
        activation_checkpoint_min_padded_sequence_length=8,
    )
    train(baseline)

    interrupted = replace(
        baseline,
        output_dir=str(tmp_path / "resumed"),
        save_every=1,
    )

    def save_then_interrupt(*args, **kwargs):
        save_training_checkpoint(*args, **kwargs)
        resume_state = args[4]
        if resume_state.step == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr("gcw.training.state.save_training_checkpoint", save_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(interrupted)

    monkeypatch.setattr("gcw.training.state.save_training_checkpoint", save_training_checkpoint)
    resumed = replace(
        interrupted,
        resume=ResumeConfig(checkpoint=str(tmp_path / "resumed" / "checkpoint")),
    )
    train(resumed)

    baseline_model, _, baseline_state = load_checkpoint(tmp_path / "baseline" / "checkpoint")
    resumed_model, _, resumed_state = load_checkpoint(tmp_path / "resumed" / "checkpoint")
    assert baseline_state["step"] == resumed_state["step"] == 3
    for expected, actual in zip(baseline_model.parameters(), resumed_model.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_training_fingerprint_changes_only_checkpoint_policy(tmp_path) -> None:
    eager = _train_config(tmp_path)
    checkpointed = replace(eager, activation_checkpointing="standard_processor_layers")
    eager_fingerprint = _training_fingerprint(eager, 1, 8, 8, 1)
    checkpointed_fingerprint = _training_fingerprint(checkpointed, 1, 8, 8, 1)

    assert eager_fingerprint | {"activation_checkpointing": "standard_processor_layers"} == checkpointed_fingerprint


    segmented = replace(eager, activation_checkpoint_windows=2)
    segmented_fingerprint = _training_fingerprint(segmented, 1, 8, 8, 1)
    assert eager_fingerprint | {"activation_checkpoint_windows": 2} == segmented_fingerprint

    adaptive = replace(eager, activation_checkpoint_min_padded_sequence_length=8)
    adaptive_fingerprint = _training_fingerprint(adaptive, 1, 8, 8, 1)
    assert (
        eager_fingerprint | {"activation_checkpoint_min_padded_sequence_length": 8}
        == adaptive_fingerprint
    )


def test_activation_checkpointing_is_explicit_opt_in() -> None:
    assert TrainConfig().activation_checkpointing == "none"
    assert TrainConfig().activation_checkpoint_min_padded_sequence_length == 0
    with pytest.raises(ValueError, match="must be an integer"):
        TrainConfig(activation_checkpoint_min_padded_sequence_length=True)
    with pytest.raises(ValueError, match="must be non-negative"):
        TrainConfig(activation_checkpoint_min_padded_sequence_length=-1)
    with pytest.raises(ValueError, match="must be 'none'"):
        TrainConfig(activation_checkpointing="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires a BT or BGT"):
        TrainConfig(activation_checkpointing="standard_processor_layers")
