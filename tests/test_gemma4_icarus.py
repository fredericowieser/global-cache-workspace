from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from test_gemma4_longshot import mixed_config
from gcw.models import build_model
from gcw.params import processor_params
from gcw.models.cache_processors.icarus import GlobalCacheProcessor
from gcw.models.cache_processors.windowed import GroupedWindowedCache


@pytest.mark.parametrize("per_head", [False, True])
@pytest.mark.parametrize("stream", ["kv", "v"])
@pytest.mark.parametrize("score", ["kv_norm", "attn_l2"])
@pytest.mark.parametrize("select_all", [False, True])
def test_native_grouped_parity(per_head, stream, score, select_all):
    cfg = replace(
        mixed_config(
            per_cache_head_projections=per_head,
            update_stream=stream,
            score=score,
            out_proj_init="normal",
        ),
        variant="icarus",
        processor_layer_indices=[0, 1, 3],
    )
    model = build_model(cfg)
    model.init_weights()
    _, cache = model.backbone.forward_chunk(
        torch.randint(1, 128, (2, 7)), 0, cache_format="list"
    )
    cache = cache.detach()
    for value in cache.keys + cache.values:
        value.requires_grad_(True)
    packed = GroupedWindowedCache.from_cache_state(deepcopy(cache))
    left = GlobalCacheProcessor(cfg)
    right = deepcopy(left)
    assert processor_params(cfg) == sum(p.numel() for p in left.parameters())
    original = [
        (k.detach().clone(), v.detach().clone())
        for k, v in zip(cache.keys, cache.values)
    ]
    a = left(cache, select_all=select_all)
    b = right(packed, select_all=select_all)
    loss_a, loss_b = 0, 0
    for i, (old_k, old_v) in enumerate(original):
        ak, av = a.layer(i)
        bk, bv = b.layer(i)
        torch.testing.assert_close(ak, bk)
        torch.testing.assert_close(av, bv)
        if stream == "v":
            torch.testing.assert_close(ak, old_k)
        loss_a = loss_a + ak.square().sum() + av.square().sum()
        loss_b = loss_b + bk.square().sum() + bv.square().sum()
    loss_a.backward()
    loss_b.backward()
    for x, y in zip(left.parameters(), right.parameters()):
        assert x.grad is not None
        torch.testing.assert_close(x.grad, y.grad)
    restored = GlobalCacheProcessor(cfg)
    restored.load_state_dict(left.state_dict(), strict=True)


def test_mixed_windowed_training():
    cfg = replace(mixed_config(), variant="icarus")
    model = build_model(cfg)
    model.init_weights()
    tokens = torch.randint(1, 128, (2, 10))
    result = model(tokens)
    result.float().square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.cache_processor.parameters()
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("flow", [False, True])
def test_checkpointed_training_parity(dtype, flow):
    from gcw.config import FlowProcessorConfig
    from gcw.models.cache_processors.base import (
        processor_gradient_checkpointing_override,
    )

    cfg = replace(
        mixed_config(out_proj_init="normal", gradient_checkpointing=True),
        variant="icarus",
        flow_processor=FlowProcessorConfig(num_steps=2) if flow else None,
    )
    torch.manual_seed(71)
    eager = build_model(cfg).to(dtype=dtype)
    eager.init_weights()
    replay = deepcopy(eager)
    tokens = torch.randint(1, 128, (2, 11))
    with processor_gradient_checkpointing_override(False, variant="icarus"):
        a = eager(tokens[:, :-1], tokens[:, 1:])
        a.backward()
    with processor_gradient_checkpointing_override(True, variant="icarus"):
        b = replay(tokens[:, :-1], tokens[:, 1:])
        b.backward()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for x, y in zip(eager.parameters(), replay.parameters()):
        if x.grad is not None:
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    assert processor_params(cfg) == sum(
        p.numel() for p in eager.cache_processor.parameters()
    )


@pytest.mark.parametrize("subset", [None, [1, 3]])
def test_native_gather_is_selected_only(monkeypatch, subset):
    import gcw.models.cache_processors.icarus as module

    cfg = replace(
        mixed_config(out_proj_init="zero"),
        variant="icarus",
        processor_layer_indices=subset,
    )
    model = build_model(cfg)
    _, cache = model.backbone.forward_chunk(torch.randint(1, 128, (2, 17)), 0)
    original = deepcopy(cache.detach())
    gather = module.gather_per_cache_head_kv

    def checked(cache, indices, **kwargs):
        assert (
            indices.shape[-1]
            <= cfg.cache_processor.process_every + cfg.cache_processor.topk
        )
        return gather(cache, indices, **kwargs)

    monkeypatch.setattr(module, "gather_per_cache_head_kv", checked)
    model.cache_processor(cache)
    for i in range(cache.num_layers):
        for x, y in zip(cache.layer(i), original.layer(i)):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert processor_params(cfg) == sum(
        p.numel() for p in model.cache_processor.parameters()
    )


def test_width_normalization_changes_global_head_ranking(monkeypatch):
    cfg = replace(mixed_config(topk_heads=1), variant="icarus")
    model = build_model(cfg)
    _, cache = model.backbone.forward_chunk(torch.randint(1, 128, (1, 7)), 0)
    for i in range(cache.num_layers):
        k, v = cache.layer(i)
        amplitude = 1.1 if k.shape[-1] == 4 else 1.0
        cache.keys[i] = torch.full_like(k, amplitude)
        cache.values[i] = torch.full_like(v, amplitude)
    chosen = []

    def observe(histories, positions, selected):
        chosen.append(selected.item())
        return histories

    monkeypatch.setattr(model.cache_processor.block, "forward_grouped", observe)
    model.cache_processor(deepcopy(cache.detach()))
    assert chosen[-1] in [0, 1, 3, 4]
    model.cache_processor.config.head_score_normalization = "none"
    model.cache_processor(deepcopy(cache.detach()))
    assert chosen[-1] in [2, 5]


def test_mixed_admission(monkeypatch):
    from gcw.performance.backends.icarus import resolve_icarus_processor_backend

    with pytest.raises(ValueError, match="unrope_keys"):
        build_model(replace(mixed_config(unrope_keys=True), variant="icarus"))
    from gcw.performance.backends.icarus import _ICARUS_BACKEND_ENV

    monkeypatch.setenv(_ICARUS_BACKEND_ENV, "compile")
    with pytest.raises(RuntimeError, match="mixed-shape"):
        resolve_icarus_processor_backend(
            replace(mixed_config(), variant="icarus")
        )


@pytest.mark.parametrize("subset", [None, [0, 2], [1, 3]])
def test_checkpoint_and_dynamic_generation(tmp_path, subset):
    from gcw.checkpoint import save_checkpoint, load_checkpoint

    cfg = replace(
        mixed_config(out_proj_init="normal"),
        variant="icarus",
        processor_layer_indices=subset,
    )
    model = build_model(cfg).eval()
    model.init_weights()
    save_checkpoint(tmp_path / "model", model, cfg)
    restored, _, _ = load_checkpoint(tmp_path / "model")
    restored.eval()
    tokens = torch.randint(1, 128, (1, 8))
    with torch.no_grad():
        a = model.generate(tokens, max_new_tokens=4)
        b = restored.generate(tokens, max_new_tokens=4)
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_canonical_temporal_positions_and_features(monkeypatch):
    cfg = replace(mixed_config(), variant="icarus")
    block = GlobalCacheProcessor(cfg).block
    histories, positions = [], []
    expected_features, expected_positions = {}, {}
    for group in block.shape_groups:
        x = torch.randn(2, len(group.head_indices), 3, 2 * group.head_dim)
        pos = torch.stack(
            [torch.tensor([h, h + 7, 20]) for h in group.head_indices]
        ).expand(2, -1, -1)
        histories.append(x)
        positions.append(pos)
        adapter = block.projections[group.name]
        encoded = adapter.in_proj(adapter.in_norm(x))
        for local, head in enumerate(group.head_indices):
            expected_features[head] = encoded[:, local]
            expected_positions[head] = pos[:, local]

    def check(state, pos, selected):
        for head in expected_features:
            torch.testing.assert_close(state[:, head], expected_features[head])
            torch.testing.assert_close(pos[:, head], expected_positions[head])
        return state

    monkeypatch.setattr(block, "_run_layers", check)
    block.forward_grouped(histories, positions, torch.tensor([[0, 2], [1, 5]]))


@pytest.mark.parametrize("stream", ["kv", "v"])
def test_mixed_flops_sum_native_projection_widths(stream):
    from gcw.flops import (
        processor_forward_flops_per_sequence,
        _icarus_processor_block_flops_per_call,
    )

    cfg = replace(mixed_config(update_stream=stream), variant="icarus")
    cp = cfg.cache_processor
    expected = 0
    for start in (3, 6, 9):
        selected = min(start, cp.process_every + cp.topk)
        heads = cfg.cache_num_heads
        contributors = min(cp.topk_heads, heads) + int(cp.include_self_head_kv)
        shared = _icarus_processor_block_flops_per_call(
            0, cp, selected, heads, contributors
        )
        projections = sum(
            2
            * selected
            * h
            * cp.hidden_size
            * (2 * d + (d if stream == "v" else 2 * d))
            for h, d in cfg.stored_cache_shapes
        )
        expected += shared + projections
    assert processor_forward_flops_per_sequence(cfg, 10) == pytest.approx(
        expected
    )
