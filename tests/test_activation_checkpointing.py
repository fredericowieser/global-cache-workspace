from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
import torch

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
from gcw.training.checkpoint import CheckpointCompatibilityError
from gcw.training.core import (
    _apply_trainable_policy,
    _configure_native_activation_checkpointing,
    _training_fingerprint,
    train,
)


def _model_config(
    variant: str,
    *,
    attention_dropout: float = 0.0,
    diffusion: bool = False,
) -> ModelConfig:
    return ModelConfig(
        variant=variant,  # type: ignore[arg-type]
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

    resumed = replace(
        eager,
        activation_checkpointing="standard_processor_layers",
        resume=ResumeConfig(checkpoint=str(checkpoint)),
    )
    with pytest.raises(CheckpointCompatibilityError, match="activation_checkpointing"):
        train(resumed)


def test_training_fingerprint_changes_only_checkpoint_policy(tmp_path) -> None:
    eager = _train_config(tmp_path)
    checkpointed = replace(eager, activation_checkpointing="standard_processor_layers")
    eager_fingerprint = _training_fingerprint(eager, 1, 8, 8, 1)
    checkpointed_fingerprint = _training_fingerprint(checkpointed, 1, 8, 8, 1)

    assert eager_fingerprint | {"activation_checkpointing": "standard_processor_layers"} == checkpointed_fingerprint


def test_activation_checkpointing_is_explicit_opt_in() -> None:
    assert TrainConfig().activation_checkpointing == "none"
    with pytest.raises(ValueError, match="must be 'none'"):
        TrainConfig(activation_checkpointing="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires a BT or BGT"):
        TrainConfig(activation_checkpointing="standard_processor_layers")
