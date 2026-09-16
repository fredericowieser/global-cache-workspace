from dataclasses import replace
from io import BytesIO

import pytest
import torch

from gcw.architecture import CompatibilityError, _config_payload, resolve_model
from gcw.config import (
    CacheProcessorConfig,
    DiffusionProcessorConfig,
    ModelConfig,
    to_dict,
)
from gcw.flops import processor_forward_flops_per_sequence
from gcw.models.cache_processors.longshot import GlobalCacheProcessor
from gcw.models.cache_processors.state import CacheState
from gcw.params import processor_params
from gcw.models.cache_processors.longshot_layers import PerCacheHeadLinear
from gcw.models.cache_processors.longshot_linear_state import (
    LinearStateAdapter,
    LinearStateSpec,
)
from gcw.models.cache_processors.longshot_processor import CacheProcessor
from gcw.performance.telemetry import cache_processor_metrics


def _processor_config(**overrides) -> CacheProcessorConfig:
    values = {
        "process_every": 2,
        "topk": 1,
        "topk_heads": 2,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_heads": 1,
        "num_kv_heads": 1,
        "encoder_seq_len": 3,
        "gate_init": -2.0,
    }
    values.update(overrides)
    return CacheProcessorConfig(**values)


@pytest.mark.parametrize("gate_mode", ["sigmoid", "direct"])
@pytest.mark.parametrize(
    ("gate_type", "gate_dim"),
    [("contextual_scalar", 1), ("contextual_vector", 3)],
)
def test_contextual_gate_initialization_matches_static_gate(
    gate_mode, gate_type, gate_dim
):
    static = CacheProcessor(
        6, 2, _processor_config(gate_mode=gate_mode), output_dim=3
    )
    contextual = CacheProcessor(
        6,
        2,
        _processor_config(
            gate_mode=gate_mode,
            update_gate_type=gate_type,
        ),
        output_dim=3,
    )
    hidden = torch.randn(2, 2, 4)

    _, static_gate = static._project_primary_update(hidden)
    _, contextual_gate = contextual._project_primary_update(hidden)

    assert contextual_gate.shape == (2, 2, gate_dim)
    torch.testing.assert_close(
        contextual_gate,
        static_gate.expand_as(contextual_gate),
    )
    assert torch.count_nonzero(contextual.out_proj.weight[3:]) == 0
    assert torch.equal(
        contextual.out_proj.bias[3:], torch.full((gate_dim,), -2.0)
    )


@pytest.mark.parametrize("per_head", [False, True])
@pytest.mark.parametrize(
    "gate_type", ["contextual_scalar", "contextual_vector"]
)
def test_contextual_gate_projection_is_data_dependent_and_differentiable(
    per_head, gate_type
):
    processor = CacheProcessor(
        6,
        2,
        _processor_config(
            update_gate_type=gate_type,
            per_cache_head_projections=per_head,
        ),
        output_dim=3,
    )
    hidden = torch.randn(2, 2, 4, requires_grad=True)
    with torch.no_grad():
        if isinstance(processor.out_proj, PerCacheHeadLinear):
            processor.out_proj.weight[..., 3:].normal_()
        else:
            processor.out_proj.weight[3:].normal_()

    delta, gate = processor._project_primary_update(hidden)
    (delta * gate).sum().backward()

    expected_dim = 1 if gate_type == "contextual_scalar" else 3
    assert gate.shape == (2, 2, expected_dim)
    assert not torch.equal(gate[0], gate[1])
    if gate_type == "contextual_vector":
        assert not torch.equal(gate[..., 0], gate[..., 1])
    assert hidden.grad is not None
    assert processor.out_proj.weight.grad is not None
    if isinstance(processor.out_proj, PerCacheHeadLinear):
        gate_weight_grad = processor.out_proj.weight.grad[..., 3:]
        gate_bias_grad = processor.out_proj.bias.grad[..., 3:]
    else:
        gate_weight_grad = processor.out_proj.weight.grad[3:]
        gate_bias_grad = processor.out_proj.bias.grad[3:]
    assert gate_weight_grad.abs().sum() > 0
    assert gate_bias_grad.abs().sum() > 0


@pytest.mark.parametrize("update_stream", ["kv", "v"])
def test_contextual_vector_gate_matches_update_stream_width(update_stream):
    output_dim = 6 if update_stream == "kv" else 3
    processor = CacheProcessor(
        6,
        2,
        _processor_config(
            update_stream=update_stream,
            update_gate_type="contextual_vector",
        ),
        output_dim=output_dim,
    )

    delta, gate = processor._project_primary_update(torch.randn(2, 2, 4))

    assert delta.shape == gate.shape == (2, 2, output_dim)
    assert processor.out_proj.out_features == 2 * output_dim


def test_contextual_gate_preserves_output_dtype():
    processor = CacheProcessor(
        6,
        2,
        _processor_config(update_gate_type="contextual_scalar"),
        output_dim=3,
    ).to(torch.bfloat16)

    delta, gate = processor._project_primary_update(
        torch.randn(2, 2, 4, dtype=torch.bfloat16)
    )

    assert delta.dtype == gate.dtype == torch.bfloat16


@pytest.mark.parametrize("per_head", [False, True])
@pytest.mark.parametrize("update_stream", ["kv", "v"])
@pytest.mark.parametrize(
    "gate_type", ["contextual_scalar", "contextual_vector"]
)
def test_contextual_gate_runs_in_complete_longshot_block(
    per_head, update_stream, gate_type
):
    output_dim = 6 if update_stream == "kv" else 3
    processor = CacheProcessor(
        6,
        2,
        _processor_config(
            per_cache_head_projections=per_head,
            update_stream=update_stream,
            update_gate_type=gate_type,
        ),
        output_dim=output_dim,
    )
    history = torch.randn(2, 2, 3, 6, requires_grad=True)

    updated, linear_delta, selected = processor(
        history, torch.tensor([[0, 1], [0, 1]])
    )
    updated.sum().backward()

    assert updated.shape == (2, 2, output_dim)
    assert linear_delta is None
    assert selected.shape == (2, 2)
    assert processor._gate_value is not None
    assert history.grad is not None


def _linear_state_model_config() -> ModelConfig:
    return ModelConfig(
        family="qwen3_5",
        hidden_size=8,
        intermediate_size=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=2,
        linear_value_head_dim=3,
        linear_num_value_heads=2,
    )


def _linear_state_adapter(
    processor_config: CacheProcessorConfig,
) -> LinearStateAdapter:
    return LinearStateAdapter(
        _linear_state_model_config(),
        processor_config,
        [LinearStateSpec(layer_idx=0, following_cache_idx=1, head_count=2)],
    )


@pytest.mark.parametrize(
    "gate_type", ["contextual_scalar", "contextual_vector"]
)
def test_contextual_gate_runs_with_linear_state_updates(gate_type):
    config = _processor_config(update_gate_type=gate_type)
    specs = [LinearStateSpec(layer_idx=0, following_cache_idx=1, head_count=2)]
    processor = CacheProcessor(
        6,
        4,
        config,
        output_dim=6,
        kv_cache_heads=2,
        model_config=_linear_state_model_config(),
        linear_state_specs=specs,
    )
    history = torch.randn(2, 2, 3, 6, requires_grad=True)
    linear_features = torch.randn(2, 2, 5, requires_grad=True)

    updated, linear_delta, _ = processor(
        history,
        torch.arange(4).expand(2, -1),
        linear_features=linear_features,
    )
    assert linear_delta is not None
    (updated.sum() + linear_delta.sum()).backward()

    assert updated.shape == (2, 2, 6)
    assert linear_delta.shape == (2, 2, 5)
    assert processor._gate_value is not None
    assert processor.linear_state_adapter is not None
    assert processor.linear_state_adapter._gate_value is not None
    assert history.grad is not None
    assert linear_features.grad is not None


@pytest.mark.parametrize(
    ("gate_type", "gate_dim"),
    [("contextual_scalar", 1), ("contextual_vector", 5)],
)
def test_linear_state_adapter_owns_contextual_gate(gate_type, gate_dim):
    adapter = _linear_state_adapter(
        _processor_config(update_gate_type=gate_type)
    )
    hidden = torch.randn(2, 3, 4, requires_grad=True)

    update = adapter.project_update(hidden)
    update.sum().backward()

    assert adapter.gate_dim == gate_dim
    assert adapter.output_proj.out_features == 5 + gate_dim
    assert adapter._gate_value is not None
    assert adapter._gate_value.shape == (2, 3, gate_dim)
    expected_gate = torch.sigmoid(torch.tensor(-2.0))
    torch.testing.assert_close(
        adapter._gate_value,
        expected_gate.expand_as(adapter._gate_value),
    )
    assert hidden.grad is not None
    assert adapter.output_proj.weight.grad is not None
    assert adapter.output_proj.weight.grad[adapter.output_dim :].abs().sum() > 0
    assert adapter.output_proj.bias.grad is not None
    assert adapter.output_proj.bias.grad[adapter.output_dim :].abs().sum() > 0


def test_gated_delta_does_not_add_an_outer_contextual_gate():
    config = _processor_config(
        update_gate_type="contextual_vector",
        linear_state_update_mode="gated_delta",
    )
    adapter = _linear_state_adapter(config)

    assert config.gate_linear_state_delta is False
    assert adapter.gate_dim == 0
    assert adapter.output_proj.out_features == adapter.output_dim == 7


def test_static_gate_keeps_legacy_state_keys_and_projection_shape():
    static = CacheProcessor(6, 2, _processor_config(), output_dim=3)
    contextual = CacheProcessor(
        6,
        2,
        _processor_config(update_gate_type="contextual_scalar"),
        output_dim=3,
    )

    assert "gate" in static.state_dict()
    assert static.out_proj.weight.shape == (3, 4)
    assert "gate" not in contextual.state_dict()
    assert contextual.out_proj.weight.shape == (4, 4)
    assert not any("update_gate" in key for key in contextual.state_dict())


def test_update_gate_configuration_and_admission_contract():
    assert CacheProcessorConfig().update_gate_type == "static_scalar"
    assert "update_gate_type" not in _config_payload(CacheProcessorConfig())
    contextual = CacheProcessorConfig(update_gate_type="contextual_scalar")
    assert (
        _config_payload(contextual)["update_gate_type"] == "contextual_scalar"
    )

    for variant in ("bt", "bgt", "icarus"):
        with pytest.raises(
            CompatibilityError, match="contextual cache update gates"
        ):
            ModelConfig(variant=variant, cache_processor=contextual)
    assert resolve_model(
        ModelConfig(variant="longshot", cache_processor=contextual)
    ).report.compatible


@pytest.mark.parametrize(
    "gate_type", ["contextual_scalar", "contextual_vector"]
)
def test_contextual_update_gates_require_learned_output_scaling(gate_type):
    fixed_scale = CacheProcessorConfig(
        update_gate_type=gate_type,
        output_scale_mode="fixed_one",
        out_proj_init="zero",
    )
    with pytest.raises(CompatibilityError) as fixed_error:
        ModelConfig(variant="longshot", cache_processor=fixed_scale)
    assert {issue.code for issue in fixed_error.value.report.issues} == {
        "update_gate_output_scale"
    }

    diffusion = DiffusionProcessorConfig(
        training_method="scheduled",
        schedule_horizon=50,
        training_solver_steps=4,
    )
    scheduled = replace(fixed_scale, output_scale_mode="learned")
    assert resolve_model(
        ModelConfig(
            variant="longshot",
            cache_processor=scheduled,
            diffusion_processor=diffusion,
        )
    ).report.compatible
    with pytest.raises(CompatibilityError) as scheduled_error:
        ModelConfig(
            variant="longshot",
            cache_processor=fixed_scale,
            diffusion_processor=diffusion,
        )
    assert {issue.code for issue in scheduled_error.value.report.issues} == {
        "update_gate_output_scale"
    }


def test_fixed_output_scaling_admits_the_static_scalar_gate():
    static_fixed = CacheProcessorConfig(out_proj_init="zero")
    assert resolve_model(
        ModelConfig(
            variant="longshot",
            cache_processor=replace(
                static_fixed, output_scale_mode="fixed_one"
            ),
        )
    ).report.compatible
    assert resolve_model(
        ModelConfig(
            variant="longshot",
            cache_processor=static_fixed,
            diffusion_processor=DiffusionProcessorConfig(
                training_method="scheduled"
            ),
        )
    ).report.compatible


def test_update_gate_type_rejects_unknown_value():
    with pytest.raises(ValueError, match="update_gate_type"):
        CacheProcessorConfig(update_gate_type="unknown")


def test_static_and_contextual_projection_keys_round_trip_strictly():
    for gate_type in (
        "static_scalar",
        "contextual_scalar",
        "contextual_vector",
    ):
        config = _processor_config(update_gate_type=gate_type)
        source = CacheProcessor(6, 2, config, output_dim=3)
        restored = CacheProcessor(6, 2, replace(config), output_dim=3)

        restored.load_state_dict(source.state_dict(), strict=True)

        assert set(restored.state_dict()) == set(source.state_dict())


def _scheduled_model_config(**processor_overrides) -> ModelConfig:
    return replace(
        _linear_state_model_config(),
        variant="longshot",
        vocab_size=32,
        sequence_len=8,
        processor_layer_indices=[1],
        linear_num_key_heads=1,
        cache_processor=_processor_config(
            **{
                "topk_heads": 1,
                "encoder_method": "query_attention",
                "encoder_query_source": "final_token",
                "linear_state_cache": True,
                "live_rank1_linear_writer": True,
                "out_proj_init": "zero",
                **processor_overrides,
            }
        ),
        diffusion_processor=DiffusionProcessorConfig(
            training_method="scheduled",
            schedule_horizon=50,
            training_solver_steps=4,
            inference_solver_steps=4,
            noise_scale=0.1,
        ),
    )


def _scheduled_cache() -> CacheState:
    # Distinct examples, without reseeding the process-global RNG.
    keys = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4).sin()
    values = (keys + 0.3).cos()
    recurrent = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3).cos()
    return CacheState(
        keys=[keys],
        values=[values],
        scores=torch.ones(2, 3),
        linear_states={0: (recurrent, torch.ones(2, 8, 2))},
    )


@pytest.mark.parametrize(
    "gate_type", ["contextual_scalar", "contextual_vector"]
)
def test_scheduled_contextual_cache_writeback_noop_and_live_gradients(
    gate_type,
):
    processor = GlobalCacheProcessor(
        _scheduled_model_config(update_gate_type=gate_type)
    ).eval()
    block = processor.block
    adapter = block.linear_state_adapter
    assert adapter is not None and block.diffusion is not None
    assert block.gate is None and block.diffusion.refinement_gate is None
    assert torch.count_nonzero(adapter.output_proj.weight[:2]) > 0
    assert torch.count_nonzero(adapter.output_proj.weight[2:5]) == 0
    projections = {}
    calls = []

    def record(name):
        def hook(_module, _inputs, output):
            calls.append(name)
            projections[name] = output

        return hook

    handles = [
        block.out_proj.register_forward_hook(record("primary")),
        adapter.output_proj.register_forward_hook(record("linear")),
    ]
    try:
        # K changes field evaluations, never the number of cache-write gates.
        for developed, solver_steps in ((False, 4), (True, 7)):
            processor.train(not developed)
            block.diffusion.set_eval_solver_steps(solver_steps)
            calls.clear()
            cache = _scheduled_cache()
            old_keys, old_values = cache.keys[0], cache.values[0]
            old_state, old_conv = cache.linear_states[0]
            result = processor(cache, select_all=True)
            assert calls == ["primary", "linear"]
            assert block.diffusion.pop_field_evaluation_count() == solver_steps
            primary_dim = 1 if gate_type == "contextual_scalar" else 8
            linear_dim = 1 if gate_type == "contextual_scalar" else 5
            assert block._gate_value.shape == (2, 1, primary_dim)
            assert adapter._gate_value.shape == (2, 2, linear_dim)
            primary, logits = projections["primary"].split([8, primary_dim], -1)
            factors, linear_logits = projections["linear"].split(
                [5, linear_dim], -1
            )
            gated = factors * linear_logits.sigmoid()
            key, value = gated.split([2, 3], -1)
            expected_matrix_delta = torch.einsum(
                "bhk,bhv->bhkv",
                torch.nn.functional.normalize(key, dim=-1),
                value,
            )
            matrix_delta = result.linear_states[0][0] - old_state
            torch.testing.assert_close(matrix_delta, expected_matrix_delta)
            torch.testing.assert_close(
                torch.cat(
                    [
                        result.keys[0][:, :, -1] - old_keys[:, :, -1],
                        result.values[0][:, :, -1] - old_values[:, :, -1],
                    ],
                    -1,
                ),
                primary * logits.sigmoid(),
            )
            assert result.linear_states[0][1] is old_conv
            if not developed:
                assert torch.equal(result.keys[0], old_keys)
                assert torch.equal(result.values[0], old_values)
                assert torch.equal(result.linear_states[0][0], old_state)
                # Live rank-one keys are allowed; only the matrix update must vanish.
                assert torch.count_nonzero(key) > 0
                assert torch.count_nonzero(value) == 0
                for gate in (block._gate_value, adapter._gate_value):
                    torch.testing.assert_close(
                        gate, torch.sigmoid(torch.tensor(-2.0)).expand_as(gate)
                    )
            else:
                assert torch.count_nonzero(matrix_delta) > 0
                assert torch.count_nonzero(primary) > 0
                for gate in (block._gate_value, adapter._gate_value):
                    assert not torch.equal(gate[0], gate[1])
                    if gate_type == "contextual_vector":
                        assert not torch.equal(gate[..., 0], gate[..., 1])

            # Exercise the real scatter writer, not a loss on unused key factors.
            loss = result.keys[0].sum() + result.values[0].sum()
            target = torch.linspace(-0.7, 0.9, matrix_delta.numel()).reshape_as(
                matrix_delta
            )
            (loss + (matrix_delta * target).sum()).backward()
            for projection, delta_start, delta_end in (
                (block.out_proj, 0, 8),
                (adapter.output_proj, 2, 5),
            ):
                grad = projection.weight.grad
                assert grad is not None and torch.isfinite(grad).all()
                assert torch.count_nonzero(grad[delta_start:delta_end]) > 0
                assert bool(torch.count_nonzero(grad[delta_end:])) == developed
                bias_grad = projection.bias.grad
                assert bias_grad is not None
                assert (
                    bool(torch.count_nonzero(bias_grad[delta_end:]))
                    == developed
                )
                if not developed:
                    # Simulate a learned nonzero update, without running training.
                    with torch.no_grad():
                        projection.weight[delta_start:delta_end].add_(
                            grad[delta_start:delta_end], alpha=-0.1
                        )
                        gate_weights = projection.weight[delta_end:]
                        gate_weights.copy_(
                            torch.linspace(
                                -0.2, 0.3, gate_weights.numel()
                            ).reshape_as(gate_weights)
                        )
            block.zero_grad(set_to_none=True)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize(
    "gate_type", ["static_scalar", "contextual_scalar", "contextual_vector"]
)
def test_scheduled_gate_checkpoint_roundtrip_and_exact_parameter_count(
    gate_type,
):
    config = _scheduled_model_config(update_gate_type=gate_type)
    source = GlobalCacheProcessor(config).eval()
    assert processor_params(config) == sum(
        p.numel() for p in source.parameters()
    )
    # A nontrivial learned checkpoint must roundtrip, not just the initial no-op.
    with torch.no_grad():
        for projection, width in (
            (source.block.out_proj, 8),
            (source.block.linear_state_adapter.output_proj, 5),
        ):
            projection.weight[:width].fill_(0.03)
            if gate_type != "static_scalar":
                projection.weight[width:].fill_(0.07)
                projection.bias[width:].fill_(-1.3)
    payload = _config_payload(config.cache_processor)
    restored_config = replace(
        config, cache_processor=CacheProcessorConfig(**payload)
    )
    restored = GlobalCacheProcessor(restored_config).eval()
    checkpoint = BytesIO()
    torch.save(source.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored.load_state_dict(
        torch.load(checkpoint, weights_only=True), strict=True
    )
    assert source.state_dict().keys() == restored.state_dict().keys()
    for name, value in source.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    with torch.no_grad():
        expected = source(_scheduled_cache(), select_all=True)
        actual = restored(_scheduled_cache(), select_all=True)
    assert torch.equal(expected.keys[0], actual.keys[0])
    assert torch.equal(expected.values[0], actual.values[0])
    assert torch.equal(expected.linear_states[0][0], actual.linear_states[0][0])
    assert source.block.diffusion.state_dict() == {}
    if gate_type == "static_scalar":
        assert "update_gate_type" not in payload
        assert restored.config.update_gate_type == "static_scalar"
        assert source.block.gate is None
        assert source.block._gate_value.item() == 1.0
        assert source.block.linear_state_adapter._gate_value is None
        assert source.block.out_proj.weight.shape == (8, 4)
        assert source.block.linear_state_adapter.output_proj.weight.shape == (
            5,
            4,
        )
        assert "block.gate" not in source.state_dict()
        assert "block.diffusion.refinement_gate" not in source.state_dict()
    else:
        assert payload["update_gate_type"] == gate_type


def test_scheduled_implicit_static_gate_preserves_rng_payload_and_state_abi():
    implicit = _scheduled_model_config(live_rank1_linear_writer=False)
    explicit = replace(
        implicit,
        cache_processor=replace(
            implicit.cache_processor, update_gate_type="static_scalar"
        ),
    )
    for serialize in (to_dict, _config_payload):
        assert serialize(implicit) == serialize(explicit)
    assert "update_gate_type" not in _config_payload(implicit.cache_processor)
    # Compare the legacy omitted opt-in with its explicit default, preserving RNG.
    with torch.random.fork_rng(devices=[]):
        initial_rng = torch.random.get_rng_state()
        source = GlobalCacheProcessor(implicit)
        next_rng = torch.random.get_rng_state()
        torch.random.set_rng_state(initial_rng)
        restored = GlobalCacheProcessor(explicit)
        assert torch.equal(next_rng, torch.random.get_rng_state())
    assert source.state_dict().keys() == restored.state_dict().keys()
    for name, value in source.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    restored.load_state_dict(source.state_dict(), strict=True)
    assert torch.count_nonzero(source.block.out_proj.weight) == 0
    assert (
        torch.count_nonzero(
            source.block.linear_state_adapter.output_proj.weight
        )
        == 0
    )


@pytest.mark.parametrize(
    "gate_type", ["static_scalar", "contextual_scalar", "contextual_vector"]
)
def test_scheduled_gate_sigmoid_flops_are_outside_solver(gate_type):
    config = _scheduled_model_config(update_gate_type=gate_type)
    calls = len(
        range(
            config.cache_processor.process_every,
            8,
            config.cache_processor.process_every,
        )
    )
    # One KV head and two linear heads, with output widths 8 and 5.
    gate_ops = {
        "static_scalar": 0,
        "contextual_scalar": 3,
        "contextual_vector": 18,
    }
    costs = []
    for solver_steps in (4, 7):
        sigmoid = replace(
            config,
            diffusion_processor=replace(
                config.diffusion_processor, training_solver_steps=solver_steps
            ),
        )
        direct = replace(
            sigmoid,
            cache_processor=replace(
                sigmoid.cache_processor, gate_mode="direct"
            ),
        )
        cost = processor_forward_flops_per_sequence(sigmoid, 8)
        assert (
            cost - processor_forward_flops_per_sequence(direct, 8)
            == calls * gate_ops[gate_type]
        )
        costs.append(cost)
    assert costs[1] > costs[0]


def test_scheduled_contextual_direct_construction_still_requires_zero_projection():
    with pytest.raises(
        ValueError, match="scheduled diffusion requires.*out_proj_init"
    ):
        CacheProcessor(
            8,
            1,
            _processor_config(update_gate_type="contextual_scalar"),
            diffusion_config=_scheduled_model_config().diffusion_processor,
        )


def test_contextual_primary_and_linear_state_gates_are_reported():
    model = torch.nn.Module()
    model.cache_processor = torch.nn.Module()
    model.cache_processor.block = torch.nn.Module()
    model.cache_processor.block._gate_value = torch.tensor([0.25, 0.75])
    model.cache_processor.block.linear_state_adapter = torch.nn.Module()
    model.cache_processor.block.linear_state_adapter._gate_value = torch.tensor(
        [0.1, 0.3]
    )

    metrics = cache_processor_metrics(model)

    assert metrics["cache_processor/gate"] == pytest.approx(0.5)
    assert metrics["cache_processor/linear_state_gate"] == pytest.approx(0.2)
