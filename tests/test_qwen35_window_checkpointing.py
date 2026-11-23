from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from gcw.config import FlowProcessorConfig, TrainConfig
from gcw.models import build_model
from test_models import tiny_qwen35_config, tiny_qwen35_moe_config


@pytest.mark.parametrize(
    "factory", [tiny_qwen35_config, tiny_qwen35_moe_config]
)
@pytest.mark.parametrize("variant", ["icarus", "longshot"])
@pytest.mark.parametrize("linear", [False, True])
def test_qwen35_window_checkpoint_admission(factory, variant, linear):
    model = factory(
        variant,
        linear_state_cache=linear,
        encoder_method="query_attention",
        encoder_query_source="final_token",
    )
    config = TrainConfig(model=model, activation_checkpoint_windows=2)
    assert config.activation_checkpoint_windows == 2
    with pytest.raises(ValueError, match="FSDP"):
        replace(config, distributed="fsdp")


@pytest.mark.parametrize(
    "factory", [tiny_qwen35_config, tiny_qwen35_moe_config]
)
@pytest.mark.parametrize("variant", ["tf", "bt", "bgt"])
def test_qwen35_window_checkpoint_keeps_other_variants_unadmitted(
    factory, variant
):
    with pytest.raises(ValueError, match="activation_checkpoint_windows"):
        TrainConfig(model=factory(variant), activation_checkpoint_windows=2)


@pytest.mark.parametrize("variant", ["icarus", "longshot"])
def test_moe_window_checkpoint_rejects_router_auxiliary_loss(variant):
    cfg = replace(tiny_qwen35_moe_config(variant), router_aux_loss_coef=0.1)
    with pytest.raises(ValueError, match="router_aux_loss_coef=0"):
        TrainConfig(model=cfg, activation_checkpoint_windows=2)
    model = build_model(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 9))
    with pytest.raises(ValueError, match="router_aux_loss_coef=0"):
        model(tokens, tokens, activation_checkpoint_windows=2)


@pytest.mark.parametrize(
    "factory", [tiny_qwen35_config, tiny_qwen35_moe_config]
)
@pytest.mark.parametrize("variant", ["icarus", "longshot"])
@pytest.mark.parametrize(
    "writer,reads,flow,truncation",
    [
        ("outer_product", "single", False, 0),
        ("gated_delta", "multiple", True, 2),
    ],
)
def test_qwen35_admitted_window_checkpoint_replays_with_gradient_parity(
    monkeypatch, factory, variant, writer, reads, flow, truncation
):
    import gcw.models.backbones.transformer.tf as tf

    calls = []
    checkpoint = tf.checkpoint

    def observed(function, *args, **kwargs):
        def segment(*inputs):
            calls.append(True)
            return function(*inputs)

        return checkpoint(segment, *args, **kwargs)

    monkeypatch.setattr(tf, "checkpoint", observed)
    model_config = factory(
        variant,
        linear_state_cache=True,
        linear_state_update_mode=writer,
        icarus_linear_read_mode=reads,
        encoder_method="query_attention",
        encoder_query_source="final_token",
        gradient_checkpointing=True,
        attention_dropout=0.1,
    )
    if flow:
        model_config = replace(
            model_config, flow_processor=FlowProcessorConfig(num_steps=2)
        )
    config = TrainConfig(
        model=model_config,
        activation_checkpoint_windows=2,
        truncated_bptt_windows=truncation,
    )
    torch.manual_seed(79)
    eager = build_model(config.model)
    eager.init_weights()
    replay = deepcopy(eager)
    replay.refresh_linear_state_dynamics()
    tokens = torch.randint(0, model_config.vocab_size, (2, 14))
    losses = []
    for model, windows in (
        (eager, 0),
        (replay, config.activation_checkpoint_windows),
    ):
        torch.manual_seed(83)
        loss = model(
            tokens[:, :-1],
            tokens[:, 1:],
            activation_checkpoint_windows=windows,
            truncated_bptt_windows=config.truncated_bptt_windows,
        )
        forward_calls = len(calls)
        loss.backward()
        assert (len(calls) > forward_calls) == bool(windows)
        losses.append(loss)
    torch.testing.assert_close(*losses, rtol=0, atol=0)
    for (name, actual), (_, expected) in zip(
        eager.named_parameters(), replay.named_parameters(), strict=True
    ):
        assert (actual.grad is None) == (expected.grad is None), name
        if actual.grad is not None:
            torch.testing.assert_close(
                actual.grad, expected.grad, rtol=1e-4, atol=1e-6, msg=name
            )
    for model in (eager, replay):
        torch.optim.SGD(model.parameters(), lr=0.01).step()
    for actual, expected in zip(
        eager.parameters(), replay.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
