from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from test_gemma4_longshot import mixed_config
from gcw.config import FlowProcessorConfig, TrainConfig
from gcw.models import build_model
from gcw.models.cache_processors.icarus import GlobalCacheProcessor
from gcw.models.cache_processors.base import cache_processor_recomputation_id
from gcw.performance.checkpointing import (
    processor_gradient_checkpointing_context,
    processor_gradient_checkpointing_decision,
)


@pytest.mark.parametrize("subset", [None, [0, 1, 3], [0, 2], [1, 3]])
def test_admission_and_defaults(subset):
    cfg = replace(
        mixed_config(), variant="icarus", processor_layer_indices=subset
    )
    assert TrainConfig(model=cfg).activation_checkpoint_windows == 0
    assert (
        TrainConfig(
            model=cfg, activation_checkpoint_windows=2
        ).activation_checkpoint_windows
        == 2
    )
    with pytest.raises(ValueError, match="FSDP"):
        TrainConfig(
            model=cfg, activation_checkpoint_windows=2, distributed="fsdp"
        )
    with pytest.raises(ValueError, match="supports qwen3"):
        TrainConfig(
            model=replace(
                cfg,
                variant="bgt",
                processor_layer_indices=[0, 2],
                cache_processor=replace(
                    cfg.cache_processor, head_score_normalization="none"
                ),
            ),
            activation_checkpoint_windows=2,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "windows,truncation,subset,stream,per_head,flow,frozen,packed",
    [
        (1, 0, None, "kv", False, False, True, False),
        (2, 0, [0, 1, 3], "v", True, False, False, False),
        (2, 2, [1, 3], "kv", False, True, True, False),
        (3, 2, [0, 1, 3], "kv", True, False, False, False),
        (2, 3, [0, 1, 3], "v", False, True, True, True),
        (2, 0, [0, 2], "kv", True, False, False, True),
    ],
)
def test_segment_loss_gradients_and_optimizer_parity(
    monkeypatch,
    dtype,
    windows,
    truncation,
    subset,
    stream,
    per_head,
    flow,
    frozen,
    packed,
):
    if packed:
        monkeypatch.setattr(
            GlobalCacheProcessor,
            "recurrent_cache_format",
            property(lambda self: "windowed"),
        )
    cfg = replace(
        mixed_config(
            out_proj_init="normal",
            update_stream=stream,
            per_cache_head_projections=per_head,
        ),
        variant="icarus",
        processor_layer_indices=subset,
        flow_processor=FlowProcessorConfig(num_steps=2) if flow else None,
    )
    torch.manual_seed(81)
    eager = build_model(cfg).to(dtype=dtype)
    eager.init_weights()
    eager.backbone.requires_grad_(not frozen)
    replay = deepcopy(eager)
    tokens = torch.randint(1, 128, (2, 12))
    losses = []
    for model, segment_size in ((eager, 0), (replay, windows)):
        loss = model(
            tokens[:, :-1],
            tokens[:, 1:],
            activation_checkpoint_windows=segment_size,
            truncated_bptt_windows=truncation,
        )
        loss.backward()
        losses.append(loss)
        assert model.cache_processor.pop_call_count() == 3
    torch.testing.assert_close(*losses, atol=0, rtol=0)
    tolerance = (
        dict(atol=1e-6, rtol=1e-4)
        if dtype == torch.float32
        else dict(atol=2e-3, rtol=2e-2)
    )
    for (name, a), (_, b) in zip(
        eager.named_parameters(), replay.named_parameters(), strict=True
    ):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, msg=name, **tolerance)
    for model in (eager, replay):
        torch.optim.SGD(model.parameters(), lr=0.01).step()
    for a, b in zip(eager.parameters(), replay.parameters(), strict=True):
        torch.testing.assert_close(a, b, **tolerance)


@pytest.mark.parametrize("threshold", [10, 11, 12])
@pytest.mark.parametrize("configured", [False, True])
def test_nested_adaptive_policy_survives_forward_context(
    monkeypatch, threshold, configured
):
    import gcw.models.cache_processors.icarus as icarus

    cfg = replace(
        mixed_config(
            out_proj_init="normal",
            gradient_checkpointing=configured,
            attention_dropout=0.2,
        ),
        variant="icarus",
        processor_layer_indices=[0, 1, 3],
    )
    train_cfg = TrainConfig(
        model=cfg,
        activation_checkpoint_windows=2,
        processor_gradient_checkpointing_seq_len_threshold=threshold
        if configured
        else None,
    )
    torch.manual_seed(82)
    eager = build_model(cfg)
    eager.init_weights()
    replay = deepcopy(eager)
    tokens = torch.randint(1, 128, (2, 12))
    calls = []
    checkpoint = icarus.checkpoint

    def observed(*args, **kwargs):
        calls.append(cache_processor_recomputation_id() is not None)
        return checkpoint(*args, **kwargs)

    monkeypatch.setattr(icarus, "checkpoint", observed)
    losses = []
    for model, windows in ((eager, 0), (replay, 2)):
        torch.manual_seed(83)
        with processor_gradient_checkpointing_context(
            processor_gradient_checkpointing_decision(
                train_cfg, tokens[:, :-1]
            ),
            variant="icarus",
        ):
            loss = model(
                tokens[:, :-1],
                tokens[:, 1:],
                activation_checkpoint_windows=windows,
            )
        loss.backward()
        losses.append(loss)
    torch.testing.assert_close(*losses, atol=0, rtol=0)
    for a, b in zip(eager.parameters(), replay.parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)
    assert any(calls) == (configured and threshold <= 11)
    assert bool(calls) == (configured and threshold <= 11)


@pytest.mark.parametrize(
    "threshold,training,grad_enabled,expected",
    [
        (10, True, True, True),
        (11, True, True, True),
        (12, True, True, False),
        (0, False, True, False),
        (0, True, False, False),
    ],
)
def test_segment_threshold_and_evaluation(
    monkeypatch, threshold, training, grad_enabled, expected
):
    cfg = replace(mixed_config(), variant="icarus")
    model = build_model(cfg).train(training)
    calls = []
    segment = model.backbone._checkpointed_window_segment

    def observed(*args, **kwargs):
        calls.append(True)
        return segment(*args, **kwargs)

    monkeypatch.setattr(
        model.backbone, "_checkpointed_window_segment", observed
    )
    tokens = torch.randint(1, 128, (2, 12))
    with torch.set_grad_enabled(grad_enabled):
        model(
            tokens[:, :-1],
            tokens[:, 1:],
            activation_checkpoint_windows=2,
            activation_checkpoint_min_padded_sequence_length=threshold,
        )
    assert bool(calls) == expected


@pytest.mark.parametrize("packed", [False, True])
def test_boundary_preserves_attention_state_without_stacking(
    monkeypatch, packed
):
    import gcw.models.backbones.transformer.tf as tf
    from gcw.models.backbones.transformer.cache import _unflatten_windowed_cache
    from gcw.models.cache_processors.state import CacheState
    from gcw.models.cache_processors.windowed import GroupedWindowedCache

    if packed:
        monkeypatch.setattr(
            GlobalCacheProcessor,
            "recurrent_cache_format",
            property(lambda self: "windowed"),
        )
    cfg = replace(
        mixed_config(out_proj_init="normal"),
        variant="icarus",
        processor_layer_indices=[0, 1, 3],
    )
    model = build_model(cfg)
    flatten = tf._flatten_windowed_cache
    boundaries = []

    def observed(cache):
        flat, layout = flatten(cache)
        if cache is not None:
            assert isinstance(
                cache, GroupedWindowedCache if packed else CacheState
            )
            restored = _unflatten_windowed_cache(flat, layout)
            assert set(restored.attention_states) == {2}
            for i in range(cache.num_layers):
                for a, b in zip(cache.layer(i), restored.layer(i)):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                    assert a.requires_grad == b.requires_grad
                    assert a.data_ptr() == b.data_ptr()
            for a, b in zip(
                cache.attention_states[2], restored.attention_states[2]
            ):
                assert a.data_ptr() == b.data_ptr()
            boundaries.append(cache.seq_len)
        return flat, layout

    monkeypatch.setattr(tf, "_flatten_windowed_cache", observed)
    tokens = torch.randint(1, 128, (2, 12))
    model(
        tokens[:, :-1], tokens[:, 1:], activation_checkpoint_windows=2
    ).backward()
    assert 6 in boundaries and 11 in boundaries


def test_segments_reduce_saved_activations_without_changing_weights():
    cfg = replace(
        mixed_config(out_proj_init="normal"),
        variant="icarus",
        processor_layer_indices=[0, 1, 3],
    )
    torch.manual_seed(84)
    eager = build_model(cfg)
    eager.init_weights()
    replay = deepcopy(eager)
    tokens = torch.randint(1, 128, (2, 12))
    totals = []
    for model, windows in ((eager, 0), (replay, 2)):
        sizes = []

        def pack(tensor):
            sizes.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(
            pack, lambda tensor: tensor
        ):
            loss = model(
                tokens[:, :-1],
                tokens[:, 1:],
                activation_checkpoint_windows=windows,
            )
        loss.backward()
        totals.append(sum(sizes))
    assert totals[1] < totals[0]
    for name, value in eager.state_dict().items():
        torch.testing.assert_close(
            value, replay.state_dict()[name], atol=0, rtol=0
        )
    replay.load_state_dict(eager.state_dict(), strict=True)
