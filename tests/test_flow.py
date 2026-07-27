from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gcw.checkpoint import load_checkpoint, save_checkpoint
from gcw.config import CacheProcessorConfig, FlowProcessorConfig, ModelConfig
from gcw.flops import processor_forward_flops_per_sequence
from gcw.models import LatentFlow, build_model, set_flow_solver_steps
from gcw.params import processor_params


FLOW_VARIANTS = ("bgt", "icarus", "longshot")


def tiny_config(
    variant: str,
    *,
    flow: FlowProcessorConfig | None = None,
    iterations: int = 1,
    iteration_mode: str = "internal",
) -> ModelConfig:
    processor = CacheProcessorConfig(
        process_every=4,
        topk=2,
        topk_heads=1,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        encoder_seq_len=4,
        iterations=iterations,
        iteration_mode=iteration_mode,
    )
    return ModelConfig(
        variant=variant,  # type: ignore[arg-type]
        sequence_len=8,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        rotary_style="llama",
        cache_processor=processor,
        flow_processor=flow,
    )


@pytest.mark.parametrize("variant", FLOW_VARIANTS)
def test_flow_models_forward_backward_and_checkpoint_roundtrip(variant: str):
    config = tiny_config(variant, flow=FlowProcessorConfig(num_steps=2))
    model = build_model(config)
    model.init_weights()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    loss = model(input_ids, input_ids)
    loss.backward()

    flows = [module for module in model.modules() if isinstance(module, LatentFlow)]
    assert len(flows) == 1
    assert flows[0].flow_gate.grad is not None
    assert torch.isfinite(flows[0].flow_gate.grad).all()
    processor_gradients = [
        parameter.grad
        for parameter in model.cache_processor.parameters()
        if parameter.requires_grad
    ]
    assert processor_gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in processor_gradients)

    clone = build_model(config)
    clone.load_state_dict(model.state_dict(), strict=True)


@pytest.mark.parametrize("variant", FLOW_VARIANTS)
def test_flow_adds_one_parameter_without_changing_base_variant(variant: str):
    base_config = tiny_config(variant)
    flow_config = replace(base_config, flow_processor=FlowProcessorConfig(num_steps=2))
    base = build_model(base_config)
    flow = build_model(flow_config)

    assert type(flow) is type(base)
    assert sum(parameter.numel() for parameter in flow.parameters()) == sum(
        parameter.numel() for parameter in base.parameters()
    ) + 1
    assert processor_params(flow_config) == sum(parameter.numel() for parameter in flow.cache_processor.parameters())


@pytest.mark.parametrize("variant", FLOW_VARIANTS)
def test_flow_steps_increase_compute_not_parameters(variant: str):
    one = tiny_config(variant, flow=FlowProcessorConfig(num_steps=1))
    three = replace(one, flow_processor=FlowProcessorConfig(num_steps=3))

    assert processor_params(one) == processor_params(three)
    assert processor_forward_flops_per_sequence(three, 8) > processor_forward_flops_per_sequence(one, 8)


@pytest.mark.parametrize("variant", ("bgt", "icarus"))
def test_each_additional_flow_step_adds_one_field_evaluation(variant: str):
    one = tiny_config(variant, flow=FlowProcessorConfig(num_steps=1))
    two = replace(one, flow_processor=FlowProcessorConfig(num_steps=2))
    three = replace(one, flow_processor=FlowProcessorConfig(num_steps=3))
    one_flops = processor_forward_flops_per_sequence(one, 8)
    two_flops = processor_forward_flops_per_sequence(two, 8)
    three_flops = processor_forward_flops_per_sequence(three, 8)

    assert two_flops - one_flops == three_flops - two_flops


def test_latent_flow_integrates_shared_field_and_returns_displacement():
    flow = LatentFlow(4, FlowProcessorConfig(num_steps=4, time_conditioning=False, eta_max=1.0, flow_gate_init=0.0))
    initial = torch.zeros(2, 3, 4)
    calls = []

    def field(state: torch.Tensor, time: float) -> torch.Tensor:
        calls.append((state.shape, time))
        return torch.ones_like(state)

    displacement = flow(initial, field)

    assert [time for _, time in calls] == [0.0, 0.25, 0.5, 0.75]
    assert torch.allclose(displacement, torch.full_like(initial, 0.5))


def test_latent_flow_rejects_field_shape_change():
    flow = LatentFlow(4, FlowProcessorConfig())
    with pytest.raises(ValueError, match="changed state shape"):
        flow(torch.zeros(2, 4), lambda state, _time: state[:, :2])


def test_solver_step_override_updates_every_flow_without_adding_parameters():
    config = tiny_config("icarus", flow=FlowProcessorConfig(num_steps=1))
    model = build_model(config)
    before = sum(parameter.numel() for parameter in model.parameters())
    before_flops = processor_forward_flops_per_sequence(config, 8)

    set_flow_solver_steps(model, 5)

    flow = next(module for module in model.modules() if isinstance(module, LatentFlow))
    assert flow.config.num_steps == 5
    assert model.config.flow_processor is not None
    assert model.config.flow_processor.num_steps == 5
    assert config.flow_processor is not None
    assert config.flow_processor.num_steps == 5
    assert processor_forward_flops_per_sequence(config, 8) > before_flops
    assert sum(parameter.numel() for parameter in model.parameters()) == before


def test_solver_step_override_persists_through_disk_checkpoint(tmp_path):
    config = tiny_config("bgt", flow=FlowProcessorConfig(num_steps=1))
    model = build_model(config)
    set_flow_solver_steps(model, 4)

    save_checkpoint(tmp_path, model, config)
    loaded_model, loaded_config, _ = load_checkpoint(tmp_path)

    assert loaded_config.flow_processor is not None
    assert loaded_config.flow_processor.num_steps == 4
    loaded_flow = next(module for module in loaded_model.modules() if isinstance(module, LatentFlow))
    assert loaded_flow.config.num_steps == 4


def test_checkpoint_rejects_stale_solver_config(tmp_path):
    config = tiny_config("bgt", flow=FlowProcessorConfig(num_steps=1))
    model = build_model(config)
    set_flow_solver_steps(model, 4)
    stale = replace(config, flow_processor=FlowProcessorConfig(num_steps=1))

    with pytest.raises(ValueError, match="runtime config"):
        save_checkpoint(tmp_path, model, stale)


def test_solver_step_override_rejects_non_flow_model():
    model = build_model(tiny_config("bgt"))
    with pytest.raises(ValueError, match="flow-enabled"):
        set_flow_solver_steps(model, 2)


@pytest.mark.parametrize("variant", FLOW_VARIANTS)
def test_flow_uses_default_cache_processor_configuration(variant: str):
    config = ModelConfig(variant=variant, flow_processor=FlowProcessorConfig())  # type: ignore[arg-type]
    assert isinstance(config.cache_processor, CacheProcessorConfig)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("eta_max", True),
        ("eta_max", "1.0"),
        ("eta_max", 0.0),
        ("eta_max", float("nan")),
        ("eta_max", float("inf")),
        ("eta_max", 1e-46),
        ("eta_max", 1e39),
        ("eta_max", 10**1000),
        ("flow_gate_init", False),
        ("flow_gate_init", "-4"),
        ("flow_gate_init", float("nan")),
        ("flow_gate_init", float("inf")),
        ("flow_gate_init", 1e39),
    ),
)
def test_flow_config_rejects_invalid_float32_scalars(field: str, value: object):
    with pytest.raises(ValueError):
        FlowProcessorConfig(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("variant", ("tf", "bt"))
def test_flow_rejects_non_bgt_family_variants(variant: str):
    with pytest.raises(ValueError, match="flow_processor requires variant"):
        tiny_config(variant, flow=FlowProcessorConfig())


def test_flow_longshot_rejects_internal_recurrence_but_allows_external_recurrence():
    with pytest.raises(ValueError, match="internal"):
        tiny_config("longshot", flow=FlowProcessorConfig(), iterations=2)

    config = tiny_config(
        "longshot",
        flow=FlowProcessorConfig(num_steps=2),
        iterations=2,
        iteration_mode="external",
    )
    model = build_model(config)
    model.init_weights()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    loss = model(input_ids, input_ids)
    loss.backward()

    assert torch.isfinite(loss)
    assert config.architecture_name == "flow_longshot"
    flow = next(module for module in model.modules() if isinstance(module, LatentFlow))
    assert flow.flow_gate.grad is not None
    assert torch.isfinite(flow.flow_gate.grad).all()



@pytest.mark.parametrize("variant", FLOW_VARIANTS)
def test_flow_gradient_checkpointing_forward_backward(variant: str):
    config = tiny_config(variant, flow=FlowProcessorConfig(num_steps=3))
    assert config.cache_processor is not None
    config.cache_processor.gradient_checkpointing = True
    model = build_model(config)
    model.init_weights()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    loss = model(input_ids, input_ids)
    loss.backward()

    flow = next(module for module in model.modules() if isinstance(module, LatentFlow))
    assert flow.flow_gate.grad is not None
    assert torch.isfinite(flow.flow_gate.grad).all()


@pytest.mark.parametrize("variant", FLOW_VARIANTS)
def test_flow_generate_fast_matches_reference(variant: str):
    torch.manual_seed(7)
    config = tiny_config(variant, flow=FlowProcessorConfig(num_steps=2))
    model = build_model(config).eval()
    model.init_weights()
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    with torch.no_grad():
        reference = model.generate(input_ids, max_new_tokens=3)
        fast = model.generate_fast(input_ids, max_new_tokens=3)

    assert torch.equal(fast, reference)


@pytest.mark.parametrize("variant", FLOW_VARIANTS)
def test_flow_models_support_bf16_autocast(variant: str):
    config = tiny_config(variant, flow=FlowProcessorConfig(num_steps=2))
    model = build_model(config)
    model.init_weights()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = model(input_ids, input_ids)
    loss.backward()

    flow = next(module for module in model.modules() if isinstance(module, LatentFlow))
    assert flow.flow_gate.grad is not None
    assert torch.isfinite(loss)


@pytest.mark.parametrize("variant", FLOW_VARIANTS)
def test_flow_supports_value_only_writeback(variant: str):
    config = tiny_config(variant, flow=FlowProcessorConfig(num_steps=2))
    assert config.cache_processor is not None
    config.cache_processor.update_stream = "v"
    model = build_model(config)
    model.init_weights()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    loss = model(input_ids, input_ids)
    loss.backward()

    assert torch.isfinite(loss)


@pytest.mark.parametrize("processor_mode", ("workspace_broadcast", "head_mask_broadcast"))
def test_flow_bgt_supports_broadcast_processor_modes(processor_mode: str):
    config = tiny_config("bgt", flow=FlowProcessorConfig(num_steps=2))
    assert config.cache_processor is not None
    config.cache_processor.processor_mode = processor_mode  # type: ignore[assignment]
    config.cache_processor.score = "kv_norm"
    config.cache_processor.update_stream = "v" if processor_mode == "workspace_broadcast" else "kv"
    model = build_model(config)
    model.init_weights()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    loss = model(input_ids, input_ids)
    loss.backward()

    assert torch.isfinite(loss)

def test_flow_bgt_variant_has_no_legacy_compatibility():
    config = tiny_config("bgt")
    config.variant = "flow_bgt"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="unknown model variant"):
        build_model(config)
