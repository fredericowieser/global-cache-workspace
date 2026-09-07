import math

import pytest
import torch
import torch.nn as nn

from gcw.architecture import _config_payload
from gcw.config import (
    CacheProcessorConfig,
    LongshotSelectorTrainingConfig,
    ModelConfig,
    TrainConfig,
    to_dict,
)
from gcw.models.cache_processors.longshot import GlobalCacheProcessor
from gcw.models.cache_processors.longshot_linear_state import LinearStateSpec
from gcw.models.cache_processors.longshot_processor import CacheProcessor
from gcw.models.cache_processors.longshot_selection import (
    ContextualHeadSelector,
    _soft_topk_mask,
    selector_routing,
    set_longshot_selector_training_progress,
)
from gcw.models.cache_processors.state import CacheState
from gcw.training.audit import _training_fingerprint


def _processor_config(**overrides) -> CacheProcessorConfig:
    values = {
        "process_every": 2,
        "topk": 1,
        "topk_heads": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_heads": 1,
        "num_kv_heads": 1,
        "encoder_seq_len": 3,
    }
    values.update(overrides)
    return CacheProcessorConfig(**values)


def test_contextual_selector_scores_match_controlled_projections():
    selector = ContextualHeadSelector(
        4, 2, 3, norm_eps=0.0, use_identity_bias=True
    )
    with torch.no_grad():
        selector.norm.weight.fill_(1.0)
        selector.context_proj.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        )
        selector.candidate_proj.weight.copy_(
            torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        )
        selector.identity_bias.copy_(torch.tensor([0.5, -0.25, 1.0]))
    hidden = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, 0.0, 1.0], [0.0, 3.0, 1.0, 2.0]],
            [[4.0, 1.0, 0.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 3.0, 4.0, 5.0]],
        ]
    )
    normalized = torch.nn.functional.rms_norm(hidden, (4,), eps=0.0)
    context = normalized.mean(dim=1)
    context_projected = context[..., :2]
    candidate_projected = normalized[..., 2:]
    expected = (context_projected[:, None] * candidate_projected).sum(
        dim=-1
    ) / math.sqrt(2) + selector.identity_bias[None]
    actual = selector(hidden)

    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


def test_contextual_selector_is_batch_independent_and_permutation_equivariant():
    torch.manual_seed(12)
    selector = ContextualHeadSelector(
        5, 3, 6, norm_eps=1e-6, use_identity_bias=False
    )
    hidden = torch.randn(2, 4, 5)
    scores = selector(hidden)
    changed_other_batch = hidden.clone()
    changed_other_batch[1].mul_(100.0)
    permutation = torch.tensor([2, 0, 3, 1])

    torch.testing.assert_close(selector(changed_other_batch)[0], scores[0])
    torch.testing.assert_close(
        selector(hidden[:, permutation]), scores[:, permutation]
    )


def test_contextual_selector_returns_fp32_from_mixed_precision_hidden():
    selector = ContextualHeadSelector(
        8, 4, 3, norm_eps=1e-6, use_identity_bias=True
    ).to(torch.bfloat16)
    scores = selector(torch.randn(2, 3, 8).to(torch.bfloat16))

    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize("use_identity_bias", [False, True])
def test_contextual_selector_scores_stay_fp32_under_autocast(use_identity_bias):
    selector = ContextualHeadSelector(
        8, 4, 3, norm_eps=1e-6, use_identity_bias=use_identity_bias
    )
    hidden = torch.randn(2, 3, 8)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        scores = selector(hidden)

    assert scores.dtype == torch.float32
    assert not torch.equal(scores, scores.to(torch.bfloat16).float())


def test_soft_topk_mask_has_exact_mass_and_implicit_gradient():
    scores = torch.tensor([[-1.0, 0.25, 0.5, 2.0]], requires_grad=True)
    weights = torch.tensor([[0.2, -0.5, 1.5, 0.7]])
    temperature = 0.6

    mask = _soft_topk_mask(scores, 2, temperature)
    (mask * weights).sum().backward()

    derivative = mask.detach() * (1.0 - mask.detach())
    expected = (
        derivative
        / temperature
        * (
            weights
            - (weights * derivative).sum(dim=-1, keepdim=True)
            / derivative.sum(dim=-1, keepdim=True)
        )
    )
    assert mask.dtype == torch.float32
    torch.testing.assert_close(mask.sum(dim=-1), torch.tensor([2.0]))
    torch.testing.assert_close(scores.grad, expected, atol=1e-6, rtol=1e-6)


def test_soft_topk_mask_bypasses_routing_when_budget_covers_heads():
    scores = torch.randn(2, 3, requires_grad=True)

    mask = _soft_topk_mask(scores, 3, 0.5)

    assert torch.equal(mask, torch.ones_like(mask))


def test_selector_training_schedule_boundaries():
    config = LongshotSelectorTrainingConfig()

    assert selector_routing(config, 0.049, training=True).mode == "warmup"
    assert selector_routing(config, 0.05, training=True).mode == "soft"
    assert selector_routing(config, 0.15, training=True).mode == "soft"
    middle = selector_routing(config, 0.60, training=True)
    assert middle.mode == "transition"
    assert middle.temperature == pytest.approx(0.35)
    assert middle.hard_blend == pytest.approx(1.0 - math.exp(-2.0))
    assert selector_routing(config, 0.90, training=True).mode == "hard"
    assert selector_routing(config, 0.20, training=False).mode == "hard"


def test_selector_training_config_admission():
    with pytest.raises(ValueError, match="learned_contextual"):
        TrainConfig(
            model=ModelConfig(
                variant="longshot",
                cache_processor=_processor_config(),
            ),
            longshot_selector_training={},
        )

    config = TrainConfig(
        model=ModelConfig(
            variant="longshot",
            cache_processor=_processor_config(
                head_selection_method="learned_contextual"
            ),
        ),
        longshot_selector_training={},
    )
    assert isinstance(
        config.longshot_selector_training, LongshotSelectorTrainingConfig
    )


def test_selector_training_schedule_is_resume_fingerprinted_only_when_enabled():
    default = TrainConfig(data={"kind": "random"})
    enabled = TrainConfig(
        model=ModelConfig(
            variant="longshot",
            cache_processor=_processor_config(
                head_selection_method="learned_contextual"
            ),
        ),
        data={"kind": "random"},
        longshot_selector_training={},
    )
    changed = TrainConfig(
        model=enabled.model,
        data={"kind": "random"},
        longshot_selector_training={"hard_start": 0.95},
    )

    default_fingerprint = _training_fingerprint(default, 1, 8, 8, 1)
    enabled_fingerprint = _training_fingerprint(enabled, 1, 8, 8, 1)
    changed_fingerprint = _training_fingerprint(changed, 1, 8, 8, 1)

    assert "longshot_selector_training" not in default_fingerprint
    assert (
        enabled_fingerprint["longshot_selector_training"]
        != changed_fingerprint["longshot_selector_training"]
    )


def test_selector_is_optional_and_shared_by_initial_and_iterative_selection():
    legacy = CacheProcessor(16, 4, _processor_config())
    initial = CacheProcessor(
        16, 4, _processor_config(head_selection_method="learned_contextual")
    )
    iterative = CacheProcessor(
        16,
        4,
        _processor_config(internal_iteration_score="learned_contextual"),
    )
    both = CacheProcessor(
        16,
        4,
        _processor_config(
            head_selection_method="learned_contextual",
            internal_iteration_score="learned_contextual",
        ),
    )

    assert legacy.head_selector is None
    assert not any("head_selector" in key for key in legacy.state_dict())
    assert "head_selection_method" not in _config_payload(legacy.config)
    assert initial.head_selector is not None
    assert iterative.head_selector is not None
    assert both.head_selector is not None
    assert _config_payload(both.config)["head_selector_hidden_size"] == 64
    assert (
        sum(
            isinstance(module, ContextualHeadSelector)
            for module in both.modules()
        )
        == 1
    )
    with pytest.raises(RuntimeError, match="was not configured"):
        legacy.select_heads(torch.randn(1, 4, 8), 2)


def test_random_selection_has_no_selector_parameters_and_is_fingerprinted():
    legacy = CacheProcessor(16, 4, _processor_config())
    random = CacheProcessor(
        16,
        4,
        _processor_config(
            head_selection_method="random",
            internal_iteration_score="random",
            head_selector_hidden_size=17,
            head_selector_use_identity_bias=True,
        ),
    )

    assert random.head_selector is None
    assert set(random.state_dict()) == set(legacy.state_dict())
    assert sum(p.numel() for p in random.parameters()) == sum(
        p.numel() for p in legacy.parameters()
    )
    payload = _config_payload(random.config)
    assert payload["head_selection_method"] == "random"
    assert payload["internal_iteration_score"] == "random"
    assert "head_selector_hidden_size" not in payload
    assert "head_selector_use_identity_bias" not in payload


@pytest.mark.parametrize(
    ("kv_budget", "linear_budget", "expected_kv", "expected_linear"),
    [
        (2, 3, 2, 3),
        (2, 0, 2, 0),
        (20, 20, 4, 3),
    ],
)
def test_random_selection_preserves_separate_kv_and_linear_quotas(
    kv_budget, linear_budget, expected_kv, expected_linear
):
    block = CacheProcessor(
        16,
        7,
        _processor_config(
            head_selection_method="random",
            topk_heads=kv_budget,
            topk_linear_heads=linear_budget,
        ),
        kv_cache_heads=4,
    )

    selected, gates = block._route_heads(
        torch.zeros(5, 7, 8), 99, "initial", "random"
    )

    assert gates is None
    assert selected.shape == (5, expected_kv + expected_linear)
    for row in selected:
        assert len(set(row.tolist())) == row.numel()
        assert ((0 <= row) & (row < 7)).all()
        assert (row < 4).sum().item() == expected_kv
        assert (row >= 4).sum().item() == expected_linear


def test_random_selection_is_per_example_and_reproduces_restored_rng_state():
    block = CacheProcessor(
        16,
        12,
        _processor_config(head_selection_method="random", topk_heads=3),
        kv_cache_heads=12,
    )
    hidden = torch.zeros(4, 12, 8)
    rng_state = torch.random.get_rng_state()

    first, _ = block._route_heads(hidden, 3, "initial", "random")
    torch.random.set_rng_state(rng_state)
    reproduced, _ = block._route_heads(hidden, 3, "initial", "random")

    assert torch.equal(first, reproduced)
    assert torch.unique(first, dim=0).shape[0] > 1


def test_random_selection_supports_kv_only_caches():
    block = CacheProcessor(
        16,
        4,
        _processor_config(
            head_selection_method="random",
            topk_heads=2,
            topk_linear_heads=3,
        ),
    )

    selected, _ = block._route_heads(
        torch.zeros(2, 4, 8), 5, "initial", "random"
    )

    assert selected.shape == (2, 2)
    assert (selected < 4).all()


def test_learned_selection_runs_initially_and_after_each_iteration_boundary():
    class RecordingLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(
            self,
            x,
            selected_heads,
            return_attn_scores=False,
            routing_gates=None,
        ):
            self.calls.append((selected_heads.clone(), return_attn_scores))
            assert return_attn_scores is False
            return x + 1.0

    block = CacheProcessor(
        16,
        4,
        _processor_config(
            head_selection_method="learned_contextual",
            internal_iteration_score="learned_contextual",
            iterations=3,
        ),
    )
    layer = RecordingLayer()
    block.layers = nn.ModuleList([layer])
    selector_calls = []
    assert block.head_selector is not None
    block.head_selector.register_forward_hook(
        lambda _module, args, _output: selector_calls.append(args)
    )

    _, _, selected = block(torch.randn(2, 4, 3, 16), None, topk_heads=2)

    assert selected.shape == (2, 2)
    assert len(selector_calls) == 3
    assert len(layer.calls) == 3
    assert all(return_scores is False for _, return_scores in layer.calls)
    assert {
        "cache_processor/selected_head_overlap_after_iter_1",
        "cache_processor/selected_head_overlap_after_iter_2",
    }.issubset(block.pop_iteration_metrics())


def test_random_selection_runs_at_every_internal_iteration_without_scores():
    class RecordingLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(
            self,
            x,
            selected_heads,
            return_attn_scores=False,
            routing_gates=None,
        ):
            self.calls.append((selected_heads.clone(), return_attn_scores))
            assert return_attn_scores is False
            assert routing_gates is None
            return x + 1.0

    block = CacheProcessor(
        16,
        8,
        _processor_config(
            head_selection_method="random",
            internal_iteration_score="random",
            iterations=3,
            topk_heads=3,
        ),
    )
    layer = RecordingLayer()
    block.layers = nn.ModuleList([layer])
    route_calls = []
    original_route = block._route_heads

    def recording_route(*args, **kwargs):
        result = original_route(*args, **kwargs)
        route_calls.append(result[0].clone())
        return result

    block._route_heads = recording_route
    block(torch.randn(2, 8, 3, 16))

    assert len(route_calls) == 3
    assert len(layer.calls) == 3
    assert all(not return_scores for _, return_scores in layer.calls)
    assert {
        "cache_processor/selected_head_overlap_after_iter_1",
        "cache_processor/selected_head_overlap_after_iter_2",
    }.issubset(block.pop_iteration_metrics())


def test_selector_training_warmup_bypasses_scoring_and_soft_routing_gradients():
    block = CacheProcessor(
        16,
        4,
        _processor_config(head_selection_method="learned_contextual"),
    ).train()
    config = LongshotSelectorTrainingConfig()
    calls = []
    assert block.head_selector is not None
    block.head_selector.register_forward_hook(
        lambda _module, _args, _output: calls.append(True)
    )
    block.head_selector.set_training_progress(config, 0.0)

    _, _, warmup_selected = block(torch.randn(1, 4, 3, 16), None, topk_heads=2)
    assert calls == []
    assert torch.equal(warmup_selected, torch.arange(4)[None])

    block.head_selector.set_training_progress(config, 0.10)
    output, _, soft_selected = block(
        torch.randn(1, 4, 3, 16), None, topk_heads=2
    )
    output.sum().backward()
    assert len(calls) == 1
    assert soft_selected.shape == (1, 2)
    assert block.head_selector.context_proj.weight.grad is not None


def test_selector_training_progress_traverses_wrappers_without_state_keys():
    block = CacheProcessor(
        16,
        4,
        _processor_config(head_selection_method="learned_contextual"),
    )
    wrapper = nn.Sequential(nn.Identity(), block)
    state_keys = set(wrapper.state_dict())
    config = LongshotSelectorTrainingConfig()

    set_longshot_selector_training_progress(wrapper, config, 0.4)

    assert block.head_selector is not None
    assert block.head_selector._training_enabled is True
    assert block.head_selector._routing_mode == "transition"
    assert set(wrapper.state_dict()) == state_keys
    with pytest.raises(ValueError, match="within"):
        set_longshot_selector_training_progress(wrapper, config, 1.1)


def test_select_all_bypasses_contextual_scoring():
    block = CacheProcessor(
        16,
        4,
        _processor_config(head_selection_method="learned_contextual"),
    )
    calls = []
    assert block.head_selector is not None
    block.head_selector.register_forward_hook(
        lambda _module, _args, _output: calls.append(True)
    )

    _, _, selected = block(
        torch.randn(2, 4, 3, 16), None, topk_heads=1, select_all=True
    )

    assert calls == []
    assert torch.equal(selected, torch.arange(4).expand(2, -1))


def test_select_all_bypasses_random_sampling_without_consuming_rng():
    block = CacheProcessor(
        16,
        4,
        _processor_config(
            head_selection_method="random",
            internal_iteration_score="random",
            iterations=3,
            topk_heads=1,
        ),
    )
    before = torch.random.get_rng_state()

    _, _, selected = block(torch.zeros(2, 4, 3, 16), select_all=True)

    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(selected, torch.arange(4).expand(2, -1))


def test_kv_only_contextual_forward_encodes_input_exactly_once():
    block = CacheProcessor(
        16,
        4,
        _processor_config(head_selection_method="learned_contextual"),
    )
    original_encode = block.encode
    calls = []

    def recording_encode(x):
        calls.append(x)
        return original_encode(x)

    block.encode = recording_encode
    block(torch.randn(1, 4, 3, 16), None, topk_heads=2)

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("initial_method", "iterative_method", "expected_selector_calls"),
    [
        ("magnitude", "learned_contextual", 1),
        ("learned_contextual", "attn_l2", 1),
    ],
)
def test_initial_and_iterative_selection_methods_are_independent(
    initial_method, iterative_method, expected_selector_calls
):
    class ScoringLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.return_score_flags = []

        def forward(
            self,
            x,
            selected_heads,
            return_attn_scores=False,
            routing_gates=None,
        ):
            self.return_score_flags.append(return_attn_scores)
            if return_attn_scores:
                return x + 1.0, torch.arange(4, dtype=torch.float)[None]
            return x + 1.0

    block = CacheProcessor(
        16,
        4,
        _processor_config(
            head_selection_method=initial_method,
            internal_iteration_score=iterative_method,
            iterations=2,
        ),
    )
    layer = ScoringLayer()
    block.layers = nn.ModuleList([layer])
    selector_calls = []
    assert block.head_selector is not None
    block.head_selector.register_forward_hook(
        lambda _module, _args, _output: selector_calls.append(True)
    )
    x = torch.randn(1, 4, 3, 16)
    if initial_method == "learned_contextual":
        block(x, None, topk_heads=2)
    else:
        block(x, torch.tensor([[0, 1]]))

    assert len(selector_calls) == expected_selector_calls
    assert layer.return_score_flags == (
        [False, False]
        if iterative_method == "learned_contextual"
        else [True, False]
    )


def test_hybrid_contextual_selection_uses_one_combined_ranking():
    config = _processor_config(
        head_selection_method="learned_contextual",
        head_selector_use_identity_bias=True,
        topk_heads=1,
        topk_linear_heads=2,
    )
    model_config = ModelConfig(
        variant="longshot",
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        cache_processor=config,
    )
    specs = [LinearStateSpec(0, 0, 2), LinearStateSpec(1, 1, 2)]
    block = CacheProcessor(
        16,
        6,
        config,
        kv_cache_heads=2,
        model_config=model_config,
        linear_state_specs=specs,
    )
    assert block.head_selector is not None
    with torch.no_grad():
        block.head_selector.context_proj.weight.zero_()
        block.head_selector.candidate_proj.weight.zero_()
        block.head_selector.identity_bias.copy_(
            torch.tensor([10.0, 1.0, 9.0, 8.0, 7.0, 6.0])
        )
    observed = []
    block.head_selector.register_forward_hook(
        lambda _module, args, _output: observed.append(args[0].shape)
    )

    updated, linear_delta, selected = block(
        torch.randn(1, 2, 3, 16),
        None,
        kv_hidden=torch.randn(1, 2, 8),
        linear_features=torch.randn(1, 4, 16),
        topk_heads=3,
    )

    assert updated.shape == (1, 2, 16)
    assert linear_delta.shape == (1, 4, 16)
    assert len(observed) == 1
    assert observed[0] == torch.Size([1, 6, 8])
    assert set(selected[0].tolist()) == {0, 2, 3}


def test_missing_linear_state_group_is_rejected():
    config = _processor_config(
        encoder_method="query_attention",
        encoder_query_source="final_token",
        linear_state_cache=True,
        head_selection_method="learned_contextual",
    )
    model_config = ModelConfig(
        variant="longshot",
        family="qwen3_5",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=16,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        processor_layer_indices=[1, 3],
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        cache_processor=config,
    )
    processor = GlobalCacheProcessor(model_config)
    later_spec = processor.linear_state_specs[1]
    cache = CacheState(
        keys=[],
        values=[],
        scores=torch.zeros(1, 0),
        linear_states={
            later_spec.layer_idx: (
                torch.randn(1, later_spec.head_count, 4, 4),
                torch.zeros(1, 1, 1),
            )
        },
    )

    with pytest.raises(RuntimeError, match="has no recurrent state"):
        processor._linear_state_inputs(
            cache, torch.randn(1, processor.num_kv_cache_heads, 8)
        )


def test_global_hybrid_contextual_path_reuses_kv_encoding_and_unifies_budget():
    config = _processor_config(
        encoder_method="query_attention",
        encoder_query_source="final_token",
        linear_state_cache=True,
        head_selection_method="learned_contextual",
        head_selector_use_identity_bias=True,
        topk_heads=1,
        topk_linear_heads=2,
    )
    model_config = ModelConfig(
        variant="longshot",
        family="qwen3_5",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=16,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        processor_layer_indices=[1, 3],
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        cache_processor=config,
    )
    processor = GlobalCacheProcessor(model_config)
    cache = CacheState(
        keys=[torch.randn(1, 1, 3, 8) for _ in range(2)],
        values=[torch.randn(1, 1, 3, 8) for _ in range(2)],
        scores=torch.randn(1, 3),
        linear_states={
            spec.layer_idx: (
                torch.randn(1, spec.head_count, 4, 4),
                torch.zeros(1, 1, 1),
            )
            for spec in processor.linear_state_specs
        },
    )
    original_encode = processor.block.encode
    encode_calls = []

    def recording_encode(x):
        encode_calls.append(x)
        return original_encode(x)

    processor.block.encode = recording_encode
    assert processor.block.head_selector is not None
    with torch.no_grad():
        processor.block.head_selector.context_proj.weight.zero_()
        processor.block.head_selector.candidate_proj.weight.zero_()
        processor.block.head_selector.identity_bias.copy_(
            torch.tensor([10.0, 1.0, 9.0, 8.0, 7.0, 6.0])
        )
    selector_inputs = []
    processor.block.head_selector.register_forward_hook(
        lambda _module, args, _output: selector_inputs.append(args[0].clone())
    )

    _, selected = processor._forward_once(cache)

    assert len(encode_calls) == 1
    assert selector_inputs and selector_inputs[0].shape == (1, 6, 8)
    assert selected is not None
    assert selected.shape == (1, 3)
    assert set(selected[0].tolist()) == {0, 2, 3}


def test_global_hybrid_random_path_skips_magnitude_and_keeps_type_quotas(
    monkeypatch,
):
    config = _processor_config(
        encoder_method="query_attention",
        encoder_query_source="final_token",
        linear_state_cache=True,
        head_selection_method="random",
        topk_heads=1,
        topk_linear_heads=2,
    )
    model_config = ModelConfig(
        variant="longshot",
        family="qwen3_5",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=16,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        processor_layer_indices=[1, 3],
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        cache_processor=config,
    )
    processor = GlobalCacheProcessor(model_config)
    cache = CacheState(
        keys=[torch.randn(2, 1, 3, 8) for _ in range(2)],
        values=[torch.randn(2, 1, 3, 8) for _ in range(2)],
        scores=torch.randn(2, 3),
        linear_states={
            spec.layer_idx: (
                torch.randn(2, spec.head_count, 4, 4),
                torch.zeros(2, 1, 1),
            )
            for spec in processor.linear_state_specs
        },
    )

    def unexpected_magnitude(*_args, **_kwargs):
        raise AssertionError("random initial routing computed magnitude scores")

    monkeypatch.setattr(
        "gcw.models.cache_processors.longshot.chunked_fp32_l2_norm",
        unexpected_magnitude,
    )
    _, selected = processor._forward_once(cache)

    assert processor.block.head_selector is None
    assert selected is not None
    assert selected.shape == (2, 3)
    assert (selected[:, :1] < processor.num_kv_cache_heads).all()
    assert (selected[:, 1:] >= processor.num_kv_cache_heads).all()


def test_learned_selector_state_dict_round_trip_reproduces_scores():
    config = _processor_config(
        head_selection_method="learned_contextual",
        internal_iteration_score="learned_contextual",
    )
    source = CacheProcessor(16, 4, config)
    restored = CacheProcessor(16, 4, config)
    restored.load_state_dict(source.state_dict(), strict=True)
    hidden = torch.randn(2, 4, 8)

    assert restored.head_selector is not None
    assert source.head_selector is not None
    torch.testing.assert_close(
        restored.head_selector(hidden), source.head_selector(hidden)
    )
    assert torch.equal(
        restored.select_heads(hidden, 2), source.select_heads(hidden, 2)
    )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_selector_hidden_size_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="head_selector_hidden_size"):
        CacheProcessorConfig(head_selector_hidden_size=value)


def test_contextual_selection_configuration_round_trips_through_model_dict():
    model = ModelConfig(
        variant="longshot",
        cache_processor={
            "head_selection_method": "learned_contextual",
            "internal_iteration_score": "learned_contextual",
            "head_selector_hidden_size": 17,
            "head_selector_use_identity_bias": True,
        },
    )
    assert model.cache_processor is not None
    assert model.cache_processor.head_selection_method == "learned_contextual"
    assert (
        model.cache_processor.internal_iteration_score == "learned_contextual"
    )
    assert model.cache_processor.head_selector_hidden_size == 17
    assert model.cache_processor.head_selector_use_identity_bias is True

    defaults = to_dict(CacheProcessorConfig())
    assert defaults["head_selection_method"] == "magnitude"
    assert defaults["head_selector_hidden_size"] == 64
    assert defaults["head_selector_use_identity_bias"] is False


def test_random_selection_configuration_round_trips_through_model_dict():
    model = ModelConfig(
        variant="longshot",
        cache_processor={
            "head_selection_method": "random",
            "internal_iteration_score": "random",
        },
    )

    assert model.cache_processor is not None
    assert model.cache_processor.head_selection_method == "random"
    assert model.cache_processor.internal_iteration_score == "random"


@pytest.mark.parametrize(
    "field", ["head_selection_method", "internal_iteration_score"]
)
def test_contextual_selection_rejects_unknown_methods(field):
    with pytest.raises(ValueError, match=field):
        CacheProcessorConfig(**{field: "unknown"})
