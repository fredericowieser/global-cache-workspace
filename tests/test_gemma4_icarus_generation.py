from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from test_gemma4_longshot import mixed_config
from gcw.config import FlowProcessorConfig
from gcw.models import build_model
from gcw.models.cache_processors.state import StaticCacheState


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "score,stream,per_head,flow,subset",
    [
        ("kv_norm", "kv", False, False, None),
        ("kv_norm", "v", True, True, [0, 1, 3]),
        ("attn_l2", "kv", True, False, [1, 3]),
        ("attn_l2", "v", False, True, [0, 2]),
    ],
)
@torch.no_grad()
def test_static_logits_updates_and_native_routing(
    monkeypatch, dtype, score, stream, per_head, flow, subset
):
    cfg = replace(
        mixed_config(
            out_proj_init="normal",
            score=score,
            update_stream=stream,
            per_cache_head_projections=per_head,
        ),
        variant="icarus",
        processor_layer_indices=subset,
        flow_processor=FlowProcessorConfig(num_steps=2) if flow else None,
    )
    torch.manual_seed(92)
    reference = build_model(cfg).to(dtype=dtype).eval()
    reference.init_weights()
    fast = deepcopy(reference)
    tokens = torch.randint(1, 128, (2, 11))
    dynamic = None
    static = fast.backbone.allocate_generation_cache(2, 18, tokens.device)
    pointers = [x.data_ptr() for x in static.keys + static.values]
    start = 0
    tolerance = (
        dict(atol=2e-5, rtol=2e-5)
        if dtype == torch.float32
        else dict(atol=2e-2, rtol=2e-2)
    )
    for steps in (5, 1, 5):
        if start:
            dynamic = reference.cache_processor(dynamic)
            before = [
                (k.clone(), v.clone())
                for k, v in (static.layer(i) for i in range(static.num_layers))
            ]
            old_length = static.length
            returned = fast.cache_processor(static)
            assert returned is static and static.length == old_length
            for i, (old_k, old_v) in enumerate(before):
                k, v = static.layer(i)
                if stream == "v":
                    torch.testing.assert_close(k, old_k, atol=0, rtol=0)
        a, dynamic = reference.backbone.forward_chunk(
            tokens[:, start : start + steps], start, dynamic
        )
        b, static = fast.backbone.forward_chunk_static(
            tokens[:, start : start + steps], start, static
        )
        torch.testing.assert_close(a, b, **tolerance)
        for i in range(static.num_layers):
            for x, y in zip(dynamic.layer(i), static.layer(i), strict=True):
                torch.testing.assert_close(x, y, **tolerance)
        start += steps
    assert pointers == [x.data_ptr() for x in static.keys + static.values]
    allocations, static_calls = [], []
    allocate = fast.backbone.allocate_generation_cache
    forward = fast.backbone.forward_chunk_static

    def observed_allocate(batch, capacity, device):
        allocations.append(capacity)
        return allocate(batch, capacity, device)

    def observed_forward(ids, start, cache, **kwargs):
        assert isinstance(cache, StaticCacheState)
        static_calls.append(start)
        return forward(ids, start, cache, **kwargs)

    monkeypatch.setattr(
        fast.backbone, "allocate_generation_cache", observed_allocate
    )
    monkeypatch.setattr(fast.backbone, "forward_chunk_static", observed_forward)

    def unexpected(*args, **kwargs):
        pytest.fail("generate_fast fell back to dynamic generation")

    monkeypatch.setattr(fast.backbone, "forward_chunk", unexpected)
    a = reference.generate(tokens, max_new_tokens=9)
    b = fast.generate_fast(tokens, max_new_tokens=9)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert allocations == [20] and static_calls


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("budget", [0, 1, 8])
@torch.no_grad()
def test_generation_cadence_truncation_and_budget(
    monkeypatch, relative, budget
):
    base = mixed_config(
        out_proj_init="normal", process_every=3, generation_process_every=2
    )
    cfg = replace(
        base,
        variant="icarus",
        sequence_len=8,
        processor_layer_indices=[0, 3],
        processor_generation_mode="post_prefill_relative"
        if relative
        else base.processor_generation_mode,
    )
    torch.manual_seed(93)
    model = build_model(cfg).eval()
    model.init_weights()
    prompt = torch.randint(1, 128, (2, 12))
    observations = []
    process = model.cache_processor.forward

    def observe(cache, *, select_all=False):
        observations.append((cache.seq_len, select_all))
        return process(cache, select_all=select_all)

    monkeypatch.setattr(model.cache_processor, "forward", observe)
    expected = model.generate(prompt, budget)
    expected_calls = list(observations)
    observations.clear()
    actual = model.generate_fast(prompt, budget)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape[1] == 8 + budget
    assert observations == expected_calls


def test_static_forward_rejects_gradients():
    model = build_model(replace(mixed_config(), variant="icarus"))
    cache = model.backbone.allocate_generation_cache(1, 3, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="without gradients"):
        model.backbone.forward_chunk_static(
            torch.ones(1, 1, dtype=torch.long), 0, cache
        )


@pytest.mark.parametrize("stop", ["eos", "sequence"])
def test_staggered_stopping_with_native_icarus(monkeypatch, stop):
    cfg = replace(
        mixed_config(out_proj_init="normal"),
        variant="icarus",
        processor_layer_indices=[1, 3],
        processor_generation_mode="post_prefill_relative",
    )
    model = build_model(cfg).eval()
    script = [[3, 6], [9, 7], [9, 2]]
    calls = []
    forward = model.backbone.forward_chunk_static

    def scripted(ids, start, cache, **kwargs):
        logits, cache = forward(ids, start, cache, **kwargs)
        prediction = torch.tensor(script[min(len(calls), 2)])
        calls.append(ids.clone())
        logits[:, -1] = -100
        logits[:, -1].scatter_(1, prediction[:, None], 100)
        return logits, cache

    monkeypatch.setattr(model.backbone, "forward_chunk_static", scripted)
    prompt = torch.tensor([[8, 5], [8, 5]])
    args = (
        {"eos_token_id": [2, 3]}
        if stop == "eos"
        else {"stop_sequences": [[3], [7, 2]]}
    )
    result = model.generate_fast(prompt, 8, pad_token_id=0, **args)
    assert result.tolist() == [[8, 5, 3, 0, 0], [8, 5, 6, 7, 2]]
    assert len(calls) == 3


def test_checkpoint_restores_native_static_generation(tmp_path):
    from gcw.checkpoint import load_checkpoint, save_checkpoint

    cfg = replace(
        mixed_config(out_proj_init="normal"),
        variant="icarus",
        processor_layer_indices=[0, 3],
    )
    model = build_model(cfg).eval()
    model.init_weights()
    save_checkpoint(tmp_path / "model", model, cfg)
    restored, _, _ = load_checkpoint(tmp_path / "model")
    restored.eval()
    assert restored.backbone._native_static_generation
    prompt = torch.tensor([[8, 5, 3, 9, 2, 6]])
    with torch.no_grad():
        torch.testing.assert_close(
            restored.generate_fast(prompt, 7),
            model.generate(prompt, 7),
            atol=0,
            rtol=0,
        )
