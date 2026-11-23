from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from gcw.models import build_model
from gcw.models.cache_processors.state import CacheState
from test_models import tiny_qwen35_config, tiny_qwen35_moe_config


def make_cache(length=6, heads=1):
    return CacheState(
        [
            torch.randn(2, heads, length, 8, requires_grad=True)
            for _ in range(2)
        ],
        [
            torch.randn(2, heads, length, 8, requires_grad=True)
            for _ in range(2)
        ],
        torch.ones(2, length),
        {
            layer: (
                torch.randn(2, 2, 4, 4, requires_grad=True),
                torch.randn(2, 2, 3),
            )
            for layer in (0, 2)
        },
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reads", ["single", "multiple"])
@pytest.mark.parametrize("mode", ["global", "shared", "per_linear_state"])
@pytest.mark.parametrize("self_kv", [False, True])
@pytest.mark.parametrize("writer", ["outer_product", "gated_delta"])
def test_linear_forward_backward(reads, mode, self_kv, writer, dtype):
    cfg = tiny_qwen35_config(
        "icarus",
        linear_state_cache=True,
        icarus_linear_read_mode=reads,
        linear_state_query_mode=mode,
        gate_mode="direct",
        gate_init=1.0,
        include_self_head_kv=self_kv,
        linear_state_update_mode=writer,
    )
    model = build_model(cfg)
    model.init_weights()
    cache = make_cache()
    model.to(dtype)
    cache.keys = [x.to(dtype) for x in cache.keys]
    cache.values = [x.to(dtype) for x in cache.values]
    cache.linear_states = {
        k: (v[0].to(dtype), v[1].to(dtype))
        for k, v in cache.linear_states.items()
    }
    original_states = {
        k: (v[0].clone(), v[1]) for k, v in cache.linear_states.items()
    }
    result = model.cache_processor(cache)
    loss = sum(x.square().sum() for x in result.keys) + sum(
        v[0].square().sum() for v in result.linear_states.values()
    )
    loss.backward()
    adapter = model.cache_processor.block.linear_state_adapter
    assert adapter.output_proj.weight.grad.abs().sum() > 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in adapter.parameters()
    )
    for layer, (state, conv) in result.linear_states.items():
        assert not torch.equal(state, original_states[layer][0])
        assert conv is original_states[layer][1]


def test_position_selection_all_heads_and_ties():
    cfg = tiny_qwen35_config(
        "icarus", linear_state_cache=True, icarus_linear_read_mode="multiple"
    )
    cfg = replace(cfg, num_kv_heads=2)
    processor = build_model(cfg).cache_processor
    cache = make_cache(heads=2)
    for k, v in zip(cache.keys, cache.values):
        with torch.no_grad():
            k.zero_()
            v.zero_()
            k[:, 0, 1] = 4
            k[:, 1, 3] = 5
    assert (
        processor._linear_read_positions(cache, 3).tolist()
        == [[[1, 3, 5]] * 2] * 2
    )
    for k in cache.keys:
        with torch.no_grad():
            k.zero_()
    assert (
        processor._linear_read_positions(cache, 3).tolist()
        == [[[0, 1, 5]] * 2] * 2
    )


@pytest.mark.parametrize("reads", ["single", "multiple"])
def test_live_initialization_and_checkpoint_parity(reads):
    cfg = tiny_qwen35_config(
        "icarus",
        linear_state_cache=True,
        icarus_linear_read_mode=reads,
        out_proj_init="zero",
        live_rank1_linear_writer=True,
    )
    eager = build_model(cfg).cache_processor
    replay = deepcopy(eager)
    replay.block.config = replace(
        replay.block.config, gradient_checkpointing=True
    )
    cache = make_cache()
    states = {k: v[0].detach().clone() for k, v in cache.linear_states.items()}
    copy_cache = CacheState(
        [x.detach().clone().requires_grad_() for x in cache.keys],
        [x.detach().clone().requires_grad_() for x in cache.values],
        cache.scores.clone(),
        {
            k: (v[0].detach().clone().requires_grad_(), v[1].clone())
            for k, v in cache.linear_states.items()
        },
    )
    for proc, data in [(eager, cache), (replay, copy_cache)]:
        result = proc(data)
        for k, v in result.linear_states.items():
            torch.testing.assert_close(v[0], states[k], rtol=0, atol=0)
        sum(v[0].sum() for v in result.linear_states.values()).backward()
        assert (
            proc.block.linear_state_adapter.output_proj.weight.grad.abs().sum()
            > 0
        )
    for p, q in zip(eager.parameters(), replay.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


@pytest.mark.parametrize(
    "factory", [tiny_qwen35_config, tiny_qwen35_moe_config]
)
@pytest.mark.parametrize("writer", ["outer_product", "gated_delta"])
def test_model_roundtrip(factory, writer):
    cfg = factory(
        "icarus",
        linear_state_cache=True,
        icarus_linear_read_mode="multiple",
        linear_state_update_mode=writer,
    )
    model = build_model(cfg)
    model.init_weights()
    tokens = torch.randint(0, cfg.vocab_size, (2, 9))
    output = model(tokens)
    logits = output[0] if isinstance(output, tuple) else output
    logits.float().square().mean().backward()
    clone = build_model(cfg)
    clone.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    clone.eval()
    with torch.no_grad():
        a, b = model(tokens), clone(tokens)
    torch.testing.assert_close(
        a[0] if isinstance(a, tuple) else a, b[0] if isinstance(b, tuple) else b
    )


@pytest.mark.parametrize("writer", ["outer_product", "gated_delta"])
@pytest.mark.parametrize("flow", [False, True])
def test_window_checkpoint_and_generation(writer, flow):
    from gcw.config import FlowProcessorConfig

    torch.manual_seed(7)
    cfg = tiny_qwen35_config(
        "icarus",
        linear_state_cache=True,
        icarus_linear_read_mode="multiple",
        linear_state_update_mode=writer,
        gradient_checkpointing=True,
        attention_dropout=0.1,
    )
    if flow:
        cfg = replace(cfg, flow_processor=FlowProcessorConfig())
    model = build_model(cfg)
    model.init_weights()
    copy = deepcopy(model)
    copy.refresh_linear_state_dynamics()
    tokens = torch.randint(0, cfg.vocab_size, (2, 10))
    losses = []
    for current, checkpointed in [(model, False), (copy, True)]:
        torch.manual_seed(9)
        loss = current(
            tokens[:, :-1],
            tokens[:, 1:],
            activation_checkpoint_windows=checkpointed,
        )
        loss.backward()
        losses.append(loss)
    torch.testing.assert_close(*losses, rtol=0, atol=0)
    for p, q in zip(model.parameters(), copy.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    model.eval()
    copy.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model.generate(tokens[:, :3], max_new_tokens=3),
            copy.generate(tokens[:, :3], max_new_tokens=3),
        )


@pytest.mark.parametrize("self_kv", [False, True])
def test_workspace_attention_matches_reference(self_kv):
    from gcw.models.cache_processors.icarus import ProcessorAttention
    from gcw.models.cache_processors.ops import rotary_for_indices
    from gcw.models.backbones.transformer import apply_rotary

    cfg = tiny_qwen35_config(
        "icarus", linear_state_cache=True, include_self_head_kv=self_kv
    ).cache_processor
    attention = ProcessorAttention(cfg).eval()
    x = torch.randn(2, 8, 8)
    positions = torch.tensor([[1, 2, 3, 4, 2, 4, 1, 4]]).expand(2, -1)
    head_ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    selected = torch.tensor([[4, 5], [6, 7]])
    actual = attention(x, positions, cfg.rope_theta, selected, ((1, 4), (2, 2)))
    for b in range(2):
        for head in range(3):
            query_indices = torch.where(head_ids == head)[0]
            source_indices = selected[b]
            if self_kv:
                source_indices = torch.cat(
                    [source_indices, query_indices]
                ).unique()
            query = x[b : b + 1, query_indices]
            source = x[b : b + 1, source_indices]
            q = attention.q_proj(query).unsqueeze(-2)
            k = attention.k_proj(source).unsqueeze(-2)
            cos, sin = rotary_for_indices(
                positions[b : b + 1, query_indices], 8, cfg.rope_theta, q.dtype
            )
            q = apply_rotary(q, cos, sin).transpose(1, 2)
            cos, sin = rotary_for_indices(
                positions[b : b + 1, source_indices], 8, cfg.rope_theta, k.dtype
            )
            k = apply_rotary(k, cos, sin).transpose(1, 2)
            v = attention.v_proj(source)[:, None]
            expected = attention.o_proj(
                torch.nn.functional.scaled_dot_product_attention(q, k, v)
            )
            torch.testing.assert_close(actual[b, query_indices], expected[0, 0])


@pytest.mark.parametrize("writer", ["outer_product", "gated_delta"])
@pytest.mark.parametrize("reads", [1, 3, 65])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_writer_sequence_matches_reference(monkeypatch, writer, reads, dtype):
    from gcw.models.backbones.qwen3 import qwen35_gated_delta_rule

    torch.manual_seed(93)
    cfg = tiny_qwen35_config(
        "icarus", linear_state_cache=True, linear_state_update_mode=writer
    )
    model = build_model(cfg)
    model.init_weights()
    proc = model.cache_processor
    cache = make_cache()
    initial_states = [v[0] for v in cache.linear_states.values()]
    conv_states = [v[1] for v in cache.linear_states.values()]
    expected = {k: v[0].clone() for k, v in cache.linear_states.items()}
    delta = (
        (torch.randn(2, 4, reads, 10 if writer == "gated_delta" else 8) * 0.1)
        .to(dtype)
        .requires_grad_()
    )
    for t in range(reads):
        for i, spec in enumerate(proc.linear_state_specs):
            update = delta[:, i * 2 : (i + 1) * 2, t]
            k, v = update[..., :4], update[..., 4:8]
            if writer == "outer_product":
                expected[spec.layer_idx] = expected[
                    spec.layer_idx
                ] + torch.einsum(
                    "bhk,bhv->bhkv",
                    torch.nn.functional.normalize(k.float(), dim=-1)
                    .to(dtype)
                    .float(),
                    v.float(),
                )
            else:
                _, expected[spec.layer_idx] = qwen35_gated_delta_rule(
                    k[:, None],
                    k[:, None],
                    v[:, None],
                    g=proc.linear_state_dynamics[spec.layer_idx](
                        update[..., 9]
                    )[:, None],
                    beta=update[..., 8].sigmoid()[:, None],
                    initial_state=expected[spec.layer_idx],
                )
    calls = []

    def record_rule(query, *args, **kwargs):
        calls.append(query.shape[1])
        return qwen35_gated_delta_rule(query, *args, **kwargs)

    monkeypatch.setattr(
        "gcw.models.backbones.qwen3.qwen35_gated_delta_rule", record_rule
    )
    proc._scatter_linear_state_updates(cache, delta, proc.linear_state_specs)
    assert calls == ([reads, reads] if writer == "gated_delta" else [])
    tolerance = (
        dict(rtol=3e-4, atol=3e-5)
        if dtype == torch.float32
        else dict(rtol=5e-2, atol=3e-3)
    )
    for k, v in cache.linear_states.items():
        torch.testing.assert_close(v[0], expected[k], **tolerance)
    assert all(
        state[1] is conv
        for state, conv in zip(cache.linear_states.values(), conv_states)
    )
    inputs = [delta, *initial_states]
    if writer == "gated_delta":
        inputs += [
            parameter
            for dynamics in proc.linear_state_dynamics.values()
            for parameter in dynamics.parameters()
        ]
    actual_grad = torch.autograd.grad(
        sum(v[0].square().sum() for v in cache.linear_states.values()),
        inputs,
        retain_graph=True,
    )
    expected_grad = torch.autograd.grad(
        sum(v.square().sum() for v in expected.values()), inputs
    )
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference, **tolerance)


@pytest.mark.parametrize("mode", ["global", "shared", "per_linear_state"])
def test_queries_pool_all_following_heads(monkeypatch, mode):
    from gcw.models.cache_processors.ops import gather_per_cache_head_kv

    cfg = replace(
        tiny_qwen35_config(
            "icarus",
            linear_state_cache=True,
            icarus_linear_read_mode="multiple",
            linear_state_query_mode=mode,
            per_cache_head_projections=True,
        ),
        num_kv_heads=2,
    )
    processor = build_model(cfg).cache_processor
    cache = make_cache(heads=2)
    retained, _ = processor._select_indices_and_heads(cache, None, False)
    count = retained.shape[-1]
    positions = processor._linear_read_positions(cache, count)
    gathered = gather_per_cache_head_kv(
        cache, positions.repeat_interleave(2, 1)
    )
    expected = (
        processor.block.in_proj(processor.block.in_norm(gathered))
        .reshape(2, 2, 2, count, -1)
        .mean(2)
    )
    adapter = processor.block.linear_state_adapter
    original_query = adapter.query
    original_encode = adapter.encode
    reads = []

    def query(layer, following, context):
        torch.testing.assert_close(
            context, expected[:, following].reshape(2 * count, -1)
        )
        q = original_query(layer, following, context)
        q = q.reshape(2, count, 2, 4).transpose(1, 2)
        r = torch.einsum("bhrk,bhkv->bhrv", q, cache.linear_states[layer][0])
        reads.append(torch.cat([q, r], -1))
        return q.transpose(1, 2).reshape(2 * count, 2, 4)

    def encode(features):
        torch.testing.assert_close(features, torch.cat(reads, 1))
        return original_encode(features)

    monkeypatch.setattr(adapter, "query", query)
    monkeypatch.setattr(adapter, "encode", encode)
    processor(cache)
    assert len(reads) == 2


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "other"])
def test_read_mode_validation(value):
    with pytest.raises(ValueError, match="icarus_linear_read_mode"):
        tiny_qwen35_config("icarus", icarus_linear_read_mode=value)


def test_backbone_dynamics_are_shared_and_not_registered_twice():
    from gcw.models.backbones.qwen3 import _qwen35_linear_state_dynamics

    model = build_model(
        tiny_qwen35_config(
            "icarus",
            linear_state_cache=True,
            linear_state_update_mode="gated_delta",
        )
    )
    model.refresh_linear_state_dynamics()
    dynamics = _qwen35_linear_state_dynamics(model.backbone)
    for layer, module in dynamics.items():
        assert model.cache_processor.linear_state_dynamics[layer] is module
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    assert not backbone_ids.intersection(
        id(p) for p in model.cache_processor.parameters()
    )


@pytest.mark.parametrize("mode", ["single", "multiple"])
@pytest.mark.parametrize("length", [1, 3, 11])
@pytest.mark.parametrize("select_all", [False, True])
@pytest.mark.parametrize("score", ["attn_l2", "kv_norm"])
def test_workspace_contains_exactly_retained_tokens(
    monkeypatch, mode, length, select_all, score
):
    cfg = tiny_qwen35_config(
        "icarus",
        linear_state_cache=True,
        icarus_linear_read_mode=mode,
        score=score,
    )
    processor = build_model(cfg).cache_processor
    cache = make_cache(length=length)
    observed = {}
    original = processor._linear_read_positions

    def positions(data, retained_count, **kwargs):
        result = original(data, retained_count, **kwargs)
        observed["retained"] = retained_count
        observed["reads"] = result.shape[-1]
        assert result.shape[-1] == (1 if mode == "single" else retained_count)
        return result

    def layer_input(module, args):
        state, positions, selected_tokens, head_groups = args
        kv_length, reads = observed["retained"], observed["reads"]
        assert state.shape == (2, 2 * kv_length + 4 * reads, 8)
        assert positions.shape == state.shape[:2]
        assert positions.min() >= 0
        assert head_groups == ((2, kv_length), (4, reads))
        assert selected_tokens.min() >= 0
        assert selected_tokens.max() < state.shape[1]
        observed["layer"] = True

    monkeypatch.setattr(processor, "_linear_read_positions", positions)
    handle = processor.block.layers[0].register_forward_pre_hook(layer_input)
    processor(cache, select_all=select_all)
    handle.remove()
    assert observed["layer"]


@pytest.mark.parametrize(
    "query_mode,queries_per_call",
    [("global", 1), ("shared", 1), ("per_linear_state", 3)],
)
@pytest.mark.parametrize("unrope", [False, True])
def test_reuses_history_scores_and_query_projections(
    monkeypatch, query_mode, queries_per_call, unrope
):
    import gcw.models.cache_processors.icarus as icarus

    cfg = replace(
        tiny_qwen35_config(
            "icarus",
            linear_state_cache=True,
            icarus_linear_read_mode="multiple",
            linear_state_query_mode=query_mode,
            score="kv_norm",
            unrope_keys=unrope,
        ),
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        processor_layer_indices=[3],
    )
    model = build_model(cfg)
    model.init_weights()
    with torch.no_grad():
        _, initial = model.backbone.forward_chunk(
            torch.randint(0, 128, (2, 8)), 0
        )
    processor = model.cache_processor
    adapter = processor.block.linear_state_adapter
    gathered, scored, queries = [], [], []
    original_gather = icarus.gather_per_cache_head_kv
    original_score = icarus.chunked_fp32_squared_l2
    original_query = adapter.query

    def gather(cache, indices, **kwargs):
        gathered.append(indices.shape[-1])
        return original_gather(cache, indices, **kwargs)

    def score(features, **kwargs):
        scored.append(tuple(features.shape))
        return original_score(features, **kwargs)

    def query(layer, following, context):
        queries.append((layer, following))
        return original_query(layer, following, context)

    monkeypatch.setattr(icarus, "gather_per_cache_head_kv", gather)
    monkeypatch.setattr(icarus, "chunked_fp32_squared_l2", score)
    monkeypatch.setattr(adapter, "query", query)
    losses = []
    for _ in range(2):
        result = processor(deepcopy(initial))
        losses.append(
            sum(
                state.square().sum()
                for state, _ in result.linear_states.values()
            )
        )
    assert gathered == [8, 6, 8, 6]
    assert scored.count((2, 1, 8, 16)) == 2
    assert len(queries) == 2 * queries_per_call
    sum(losses).backward()
    for projection in adapter.query_projs.values():
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()


@pytest.mark.parametrize("query_mode", ["global", "shared", "per_linear_state"])
def test_shared_query_readouts_and_gradients_match_independent_reads(
    query_mode,
):
    torch.manual_seed(53)
    cfg = replace(
        tiny_qwen35_config(
            "icarus",
            linear_state_cache=True,
            icarus_linear_read_mode="multiple",
            linear_state_query_mode=query_mode,
        ),
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        processor_layer_indices=[3],
    )
    block = build_model(cfg).cache_processor.block
    reference = deepcopy(block)
    query_kv = torch.randn(2, 1, 3, 16, requires_grad=True)
    reference_kv = query_kv.detach().clone().requires_grad_()
    states = [torch.randn(2, 2, 4, 4, requires_grad=True) for _ in range(3)]
    reference_states = [s.detach().clone().requires_grad_() for s in states]
    positions = torch.tensor([[[0, 2, 7]]]).expand(2, -1, -1)
    features, _ = block.read_linear_states(
        block.in_proj(block.in_norm(query_kv)), states, positions
    )
    adapter = reference.linear_state_adapter
    context = (
        reference.in_proj(reference.in_norm(reference_kv))
        .mean(1)
        .reshape(6, -1)
    )
    expected = []
    for spec, state in zip(
        reference.linear_state_specs, reference_states, strict=True
    ):
        query = (
            adapter.query(spec.layer_idx, spec.following_cache_idx, context)
            .reshape(2, 3, 2, 4)
            .transpose(1, 2)
        )
        expected.append(
            torch.cat(
                [query, torch.einsum("bhrk,bhkv->bhrv", query, state)], -1
            )
        )
    expected = torch.cat(expected, 1)
    torch.testing.assert_close(features, expected, rtol=0, atol=0)
    features.square().mean().backward()
    expected.square().mean().backward()
    for actual, oracle in [
        (query_kv, reference_kv),
        *zip(states, reference_states),
        *zip(block.parameters(), reference.parameters()),
    ]:
        if actual.grad is None:
            assert oracle.grad is None
        else:
            torch.testing.assert_close(
                actual.grad, oracle.grad, rtol=2e-5, atol=2e-6
            )


@pytest.mark.parametrize("per_head", [False, True])
@pytest.mark.parametrize("layout", ["single", "full", "partial"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reuses_overlapping_kv_encodings(per_head, layout, dtype):
    from gcw.models.cache_processors.icarus import CacheProcessor

    torch.manual_seed(83)
    cfg = tiny_qwen35_config(
        "icarus", per_cache_head_projections=per_head
    ).cache_processor
    block = CacheProcessor(16, 4, cfg).to(dtype)
    reference = deepcopy(block)
    source = torch.randn(2, 4, 9, 16, dtype=dtype, requires_grad=True)
    ref_source = source.detach().clone().requires_grad_()
    indices = torch.tensor(
        [[0, 3, 5, 8], [1, 3, 6, 8], [0, 2, 4, 8], [2, 3, 7, 8]]
    )[None].expand(2, -1, -1)
    reads = torch.tensor([1, 3, 6, 8])[None, None].expand(2, 4, -1)
    if layout == "single":
        reads = reads[..., -1:]
    elif layout == "full":
        indices = reads = torch.arange(9)[None, None].expand(2, 4, -1)
    merged = source.gather(2, indices[..., None].expand(-1, -1, -1, 16))
    query = source.gather(2, reads[..., None].expand(-1, -1, -1, 16))
    rows = []
    handle = block.in_norm.register_forward_pre_hook(
        lambda module, args: rows.append(args[0].numel() // 16)
    )
    actual = block.encode_linear_kv(
        merged, indices, reads, query if layout == "partial" else None
    )
    handle.remove()
    expected = tuple(
        reference.in_proj(
            reference.in_norm(
                ref_source.gather(
                    2, positions[..., None].expand(-1, -1, -1, 16)
                )
            )
        )
        for positions in (indices, reads)
    )
    unique_rows = sum(
        len(set(kept.tolist()) | set(queried.tolist()))
        for kept, queried in zip(indices.flatten(0, 1), reads.flatten(0, 1))
    )
    assert sum(rows) == unique_rows
    tolerance = (
        dict(rtol=2e-5, atol=2e-6)
        if dtype == torch.float32
        else dict(rtol=2e-2, atol=2e-3)
    )
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, **tolerance)
    sum(x.float().square().sum() for x in actual).backward()
    sum(x.float().square().sum() for x in expected).backward()
    for a, b in [
        (source, ref_source),
        *zip(block.parameters(), reference.parameters()),
    ]:
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, **tolerance)


@pytest.mark.parametrize("unrope", [False, True])
def test_multiple_attn_reads_score_native_cache_without_history_gather(
    monkeypatch, unrope
):
    import gcw.models.cache_processors.icarus as icarus

    cfg = tiny_qwen35_config(
        "icarus",
        linear_state_cache=True,
        icarus_linear_read_mode="multiple",
        unrope_keys=unrope,
        score="attn_l2",
    )
    processor = build_model(cfg).cache_processor
    cache = make_cache(length=11)
    gather = icarus.gather_per_cache_head_kv
    score = icarus.native_cache_squared_l2
    gathers, scores = [], []

    def record_gather(cache, indices, **kwargs):
        assert kwargs["backend"] == processor._cache_movement_backend()
        gathers.append(indices.shape[-1])
        return gather(cache, indices, **kwargs)

    def record_score(cache):
        scores.append(cache.seq_len)
        return score(cache, max_temp_bytes=128)

    monkeypatch.setattr(icarus, "gather_per_cache_head_kv", record_gather)
    monkeypatch.setattr(icarus, "native_cache_squared_l2", record_score)
    processor(cache)
    assert gathers == [6, 6]
    assert scores == [11]
