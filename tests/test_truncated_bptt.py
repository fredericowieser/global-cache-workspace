from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from gcw.config import (
    CacheProcessorConfig,
    ModelConfig,
    TrainConfig,
    to_dict,
)
from gcw.models import build_model
from gcw.models.backbones.qwen3 import Qwen35MoeTF
from gcw.models.cache_processors.windowed import WindowedCache


def _model_config(variant: str) -> ModelConfig:
    return ModelConfig(
        variant=variant,  # type: ignore[arg-type]
        family="smollm",
        sequence_len=32,
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=2,
            topk_heads=2,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
        ),
    )


def _backward_with_window_gradients(model, input_ids, targets, **kwargs):
    windows = []

    def retain_window(_module, _args, output):
        output.retain_grad()
        windows.append(output)

    handle = model.backbone.embed_tokens.register_forward_hook(retain_window)
    try:
        loss = model(input_ids, targets, **kwargs)
    finally:
        # Retain original activations, not checkpoint recomputation copies.
        handle.remove()
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert gradients
    assert all(
        torch.isfinite(gradient).all() for gradient in gradients.values()
    )
    return loss.detach(), windows, gradients


def _assert_gradient_parity(expected, actual):
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(
            actual[name], expected[name], atol=2e-6, rtol=2e-5, msg=name
        )


@pytest.mark.parametrize("variant", ["bt", "bgt", "icarus", "longshot"])
def test_truncated_bptt_preserves_forward_loss_and_is_opt_in(
    variant: str,
) -> None:
    torch.manual_seed(41)
    full = build_model(_model_config(variant))
    full.init_weights()
    truncated = copy.deepcopy(full)
    input_ids = torch.randint(0, 128, (2, 12))
    targets = input_ids.roll(-1, dims=1)

    full_loss = full(input_ids, targets)
    truncated_loss = truncated(
        input_ids,
        targets,
        truncated_bptt_windows=1,
    )

    assert torch.equal(truncated_loss, full_loss)


@pytest.mark.parametrize(
    "variant,horizon,checkpoint_windows",
    [
        ("bt", 2, 0),
        ("bgt", 2, 0),
        ("icarus", 2, 0),
        ("longshot", 2, 0),
        ("bgt", 1, 3),
        ("bgt", 2, 2),
        ("bgt", 2, 3),
        ("bt", 3, 2),
    ],
)
def test_final_window_gradients_stop_at_the_horizon_boundary(
    variant, horizon, checkpoint_windows
):
    torch.manual_seed(43)
    config = replace(_model_config(variant), family="qwen3")
    full = build_model(config)
    full.init_weights()
    truncated = copy.deepcopy(full)
    input_ids = torch.randint(0, 128, (1, 14))
    targets = input_ids.roll(-1, dims=1)
    # Only the partial final window contributes direct loss. Any earlier
    # activation gradient must therefore have travelled through cache history.
    targets[:, :12] = -100
    full_loss, full_windows, full_grads = _backward_with_window_gradients(
        full, input_ids, targets
    )
    loss, windows, grads = _backward_with_window_gradients(
        truncated,
        input_ids,
        targets,
        truncated_bptt_windows=horizon,
        activation_checkpoint_windows=checkpoint_windows,
    )

    assert torch.equal(loss, full_loss)
    assert [window.shape[1] for window in windows] == [4, 4, 4, 2]
    boundary = (3 // horizon) * horizon
    for index, (full_window, window) in enumerate(
        zip(full_windows, windows, strict=True)
    ):
        assert full_window.grad is not None
        assert torch.count_nonzero(full_window.grad) > 0, index
        assert window.grad is not None
        assert torch.isfinite(window.grad).all()
        if index < boundary:
            assert torch.count_nonzero(window.grad) == 0, index
        else:
            assert torch.count_nonzero(window.grad) > 0, index
            torch.testing.assert_close(
                window.grad, full_window.grad, atol=2e-6, rtol=2e-5
            )
    processor_names = [
        name for name in full_grads if name.startswith("cache_processor.")
    ]
    assert processor_names
    assert any(torch.count_nonzero(grads[name]) for name in processor_names)
    assert any(
        not torch.allclose(grads[name], full_grads[name], atol=1e-9, rtol=1e-5)
        for name in processor_names
    )


@pytest.mark.parametrize("variant", ["bt", "bgt", "icarus", "longshot"])
def test_horizon_beyond_history_matches_full_bptt_gradients(variant):
    torch.manual_seed(47)
    full = build_model(_model_config(variant))
    full.init_weights()
    bounded = copy.deepcopy(full)
    input_ids = torch.randint(0, 128, (1, 14))
    full_loss, _, full_grads = _backward_with_window_gradients(
        full, input_ids, input_ids
    )
    loss, _, grads = _backward_with_window_gradients(
        bounded, input_ids, input_ids, truncated_bptt_windows=5
    )
    assert torch.equal(loss, full_loss)
    assert full_grads.keys() == grads.keys()
    for name in full_grads:
        assert torch.equal(grads[name], full_grads[name]), name


@pytest.mark.parametrize(
    "variant,horizon,checkpoint_windows",
    [("bt", 2, 2), ("bgt", 2, 3), ("icarus", 3, 2), ("longshot", 1, 3)],
)
def test_checkpointed_truncation_matches_eager_gradients_and_dropout_rng(
    variant, horizon, checkpoint_windows
):
    torch.manual_seed(53)
    config = replace(_model_config(variant), family="qwen3")
    config.cache_processor.attention_dropout = 0.2
    eager = build_model(config)
    eager.init_weights()
    checkpointed = copy.deepcopy(eager)
    input_ids = torch.randint(0, 128, (1, 14))
    rng = torch.get_rng_state().clone()
    eager_loss, _, eager_grads = _backward_with_window_gradients(
        eager, input_ids, input_ids, truncated_bptt_windows=horizon
    )
    eager_rng = torch.get_rng_state().clone()
    assert not torch.equal(rng, eager_rng), "dropout must actually consume RNG"
    torch.set_rng_state(rng)
    loss, _, grads = _backward_with_window_gradients(
        checkpointed,
        input_ids,
        input_ids,
        truncated_bptt_windows=horizon,
        activation_checkpoint_windows=checkpoint_windows,
    )
    assert torch.equal(loss, eager_loss)
    assert torch.equal(torch.get_rng_state(), eager_rng)
    _assert_gradient_parity(eager_grads, grads)


@pytest.mark.parametrize("horizon", [1, 2, 5])
def test_qwen35_moe_windowed_loss_truncates_real_hybrid_cache_history(horizon):
    torch.manual_seed(59)
    config = replace(
        _model_config("bgt"),
        family="qwen3_5_moe",
        hidden_size=16,
        intermediate_size=8,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        tie_embeddings=False,
        qk_norm=True,
        rotary_style="llama",
        layer_types=["linear_attention", "full_attention"] * 2,
        processor_layer_indices=[1, 3],
        partial_rotary_factor=0.5,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=8,
        router_aux_loss_coef=0.01,
    )
    full = build_model(config)
    full.init_weights()
    truncated = copy.deepcopy(full)
    assert isinstance(full.backbone, Qwen35MoeTF)
    input_ids = torch.arange(1, 15).unsqueeze(0)
    targets = input_ids.roll(-1, dims=1)
    targets[:, :12] = -100
    results = []
    for model, windows in ((full, 0), (truncated, horizon)):
        with model.cache_processor.prepare_forward_execution() as processor:
            loss = model.backbone.forward_windowed_loss(
                input_ids,
                targets,
                processor=processor,
                process_every=4,
                truncated_bptt_windows=windows,
                # Exclude auxiliary loss only from the cutoff probe: router
                # auxiliary loss directly supervises even earlier windows.
                loss_reduction="sum",
            )
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        for layer in model.backbone.layers:
            for parameter in (
                layer.mlp.gate.weight,
                layer.mlp.experts.gate_up_proj,
                layer.mlp.experts.down_proj,
            ):
                assert torch.count_nonzero(parameter) > 0
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
                assert torch.count_nonzero(parameter.grad) > 0
        results.append((loss.detach(), gradients))

    (full_loss, full_grads), (loss, grads) = results
    assert torch.equal(loss, full_loss)
    assert all(torch.isfinite(gradient).all() for gradient in grads.values())
    # Unique input IDs and an untied head make embedding rows a per-window
    # probe of both full-attention KV and gated-delta/conv cache recurrence.
    embedding = "backbone.embed_tokens.weight"
    boundary = 0 if horizon > 3 else (3 // horizon) * horizon
    for window in range(4):
        rows = input_ids[0, window * 4 : (window + 1) * 4]
        assert torch.count_nonzero(full_grads[embedding][rows]) > 0
        if window < boundary:
            assert torch.count_nonzero(grads[embedding][rows]) == 0
        else:
            assert torch.count_nonzero(grads[embedding][rows]) > 0
    if horizon > 3:
        _assert_gradient_parity(full_grads, grads)
    else:
        assert any(
            not torch.allclose(grads[name], gradient, atol=1e-9, rtol=1e-5)
            for name, gradient in full_grads.items()
            if name.startswith("cache_processor.")
        )
    with torch.no_grad():
        losses = []
        for model, windows in ((full, 0), (truncated, horizon)):
            with model.cache_processor.prepare_forward_execution() as processor:
                losses.append(
                    model.backbone.forward_windowed_loss(
                        input_ids,
                        targets,
                        processor=processor,
                        process_every=4,
                        truncated_bptt_windows=windows,
                    )
                )
        assert torch.equal(losses[0], losses[1])
        assert (
            losses[0] > full_loss / 2
        )  # Nonzero native router auxiliary loss.


@pytest.mark.parametrize("horizon, expected_detaches", [(1, 2), (2, 1), (3, 0)])
def test_truncated_bptt_detaches_only_at_configured_window_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    horizon: int,
    expected_detaches: int,
) -> None:
    model = build_model(_model_config("bgt"))
    model.init_weights()
    input_ids = torch.randint(0, 128, (1, 12))
    calls = 0
    original_detach = WindowedCache.detach

    def track_detach(cache: WindowedCache) -> WindowedCache:
        nonlocal calls
        calls += 1
        return original_detach(cache)

    monkeypatch.setattr(WindowedCache, "detach", track_detach)
    model(
        input_ids,
        input_ids,
        truncated_bptt_windows=horizon,
    ).backward()

    assert calls == expected_detaches


def test_truncated_bptt_config_is_serializable_and_rejects_processor_free_models() -> (
    None
):
    config = TrainConfig(
        model=_model_config("bgt"),
        truncated_bptt_windows=2,
    )
    assert config.truncated_bptt_windows == 2
    assert to_dict(config)["truncated_bptt_windows"] == 2
    assert "truncated_bptt_windows" not in to_dict(TrainConfig())

    with pytest.raises(
        ValueError,
        match="truncated_bptt_windows requires a cache-processor variant",
    ):
        TrainConfig(
            model=ModelConfig(
                variant="tf",
                sequence_len=32,
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                head_dim=8,
            ),
            truncated_bptt_windows=1,
        )
