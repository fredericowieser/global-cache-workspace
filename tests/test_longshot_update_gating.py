from dataclasses import replace

import pytest
import torch

from gcw.architecture import CompatibilityError, _config_payload, resolve_model
from gcw.config import (
    CacheProcessorConfig,
    DiffusionProcessorConfig,
    ModelConfig,
)
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

    scheduled = CacheProcessorConfig(
        update_gate_type=gate_type, out_proj_init="zero"
    )
    with pytest.raises(CompatibilityError) as scheduled_error:
        ModelConfig(
            variant="longshot",
            cache_processor=scheduled,
            diffusion_processor=DiffusionProcessorConfig(
                training_method="scheduled"
            ),
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
