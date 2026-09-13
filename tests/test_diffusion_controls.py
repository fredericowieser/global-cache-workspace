from __future__ import annotations

from dataclasses import replace
import math
from unittest.mock import Mock

import pytest
import torch

from gcw.architecture import resolve_experiment, resolve_model
from gcw.architecture.contracts import _config_payload
from gcw.cli import build_parser
from gcw.config import (
    CacheProcessorConfig,
    DiffusionProcessorConfig,
    EvalConfig,
    LongshotSelectorTrainingConfig,
    ModelConfig,
    TrainConfig,
    to_dict,
)
from gcw.flops import processor_forward_flops_per_sequence
from gcw.models import build_model
from gcw.models.cache_processors.diffusion import (
    TaskDirectedDiffusion,
    pop_diffusion_field_evaluation_count,
    set_diffusion_eval_solver_steps,
    sinusoidal_time_embedding,
)
from gcw.models.cache_processors.longshot_processor import CacheProcessor
from gcw.params import processor_params
from gcw.performance import NoisyDiffusionBackendConfig, PerformanceConfig
from gcw.performance.backends.diffusion import (
    configure_noisy_diffusion_recurrence_backend,
    noisy_diffusion_recurrence_backend_metadata,
)
from gcw.performance.telemetry import cache_processor_metrics


def scheduled_config(**kwargs) -> DiffusionProcessorConfig:
    return DiffusionProcessorConfig(
        **{
            "training_method": "scheduled",
            "schedule_horizon": 50,
            "training_solver_steps": 4,
            "noise_scale": 0.1,
            **kwargs,
        }
    )


def model_config(variant="longshot", **processor_kwargs) -> ModelConfig:
    return ModelConfig(
        variant=variant,
        family="llama",
        sequence_len=8,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        cache_processor=CacheProcessorConfig(
            **{
                "process_every": 4,
                "topk": 2,
                "topk_heads": 1,
                "score": "kv_norm",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_layers": 1,
                "num_heads": 1,
                "num_kv_heads": 1,
                "encoder_method": "query_attention",
                "encoder_query_source": "final_token",
                "out_proj_init": "zero",
                **processor_kwargs,
            }
        ),
        diffusion_processor=scheduled_config(),
    )


@pytest.mark.parametrize("steps", [1, 4, 8])
@pytest.mark.parametrize("level", [1, 25, 50])
def test_exact_euler_depth_and_dt_scaling(steps, level):
    diffusion = TaskDirectedDiffusion(
        4, scheduled_config(training_solver_steps=steps, eta_max=0.7)
    ).train()
    state = torch.zeros(1, 2, 4)
    times = []

    def field(value, time):
        times.append(time)
        return torch.ones_like(value)

    output = diffusion(state, field, noise=state, start_level=level)
    torch.testing.assert_close(output, torch.full_like(state, 0.7 * level / 50))
    assert times == pytest.approx(
        [
            math.log(0.1 * level / 50 * (1 - step / steps))
            for step in range(steps)
        ]
    )
    assert diffusion.pop_field_evaluation_count() == steps
    assert diffusion.pop_field_evaluation_count() == 0
    assert diffusion.refinement_gate is None


def test_coupled_schedule_retains_integer_level_steps_and_metrics():
    diffusion = TaskDirectedDiffusion(
        4,
        DiffusionProcessorConfig(training_method="scheduled", noise_scale=0.1),
    )
    state = torch.zeros(1, 2, 4)
    times = []

    def field(value, time):
        times.append(time)
        return torch.ones_like(value)

    result = diffusion(state, field, noise=state, start_level=3)
    torch.testing.assert_close(result, torch.full_like(state, 0.75))
    assert times == pytest.approx([math.log(s) for s in (0.075, 0.05, 0.025)])
    assert diffusion.pop_scheduled_metrics() == pytest.approx(
        {
            "cache_processor/diffusion_start_level_mean": 3,
            "cache_processor/diffusion_start_time_mean": 0.75,
            "cache_processor/diffusion_sigma_mean": 0.075,
            "cache_processor/diffusion_start_level_1_fraction": 0,
            "cache_processor/diffusion_start_level_2_fraction": 0,
            "cache_processor/diffusion_start_level_3_fraction": 1,
            "cache_processor/diffusion_start_level_4_fraction": 0,
        }
    )
    assert diffusion.pop_scheduled_metrics() == {}
    diffusion.eval()
    times.clear()
    torch.testing.assert_close(diffusion(state, field), torch.ones_like(state))
    assert len(times) == 4


def test_schedule_draws_level_and_noise_once_and_eval_draws_neither(
    monkeypatch,
):
    diffusion = TaskDirectedDiffusion(
        4, scheduled_config(inference_solver_steps=6)
    )
    state = torch.ones(1, 2, 4)
    randint = Mock(return_value=torch.tensor(25))
    randn = Mock(return_value=torch.ones_like(state))
    monkeypatch.setattr(torch, "randint", randint)
    monkeypatch.setattr(torch, "randn_like", randn)
    diffusion(state, lambda value, _time: value)
    randint.assert_called_once_with(1, 51, ())
    randn.assert_called_once()
    assert diffusion.pop_field_evaluation_count() == 4
    diffusion.eval()
    first = diffusion(state, lambda value, _time: value)
    second = diffusion(state, lambda value, _time: value)
    assert torch.equal(first, second)
    assert diffusion.pop_field_evaluation_count() == 12
    randint.assert_called_once()
    randn.assert_called_once()


def test_perturbation_control_keeps_rng_schedule_and_noise_out_of_displacement():
    noisy = TaskDirectedDiffusion(4, scheduled_config())
    clean = TaskDirectedDiffusion(
        4, scheduled_config(perturbation_enabled=False)
    )
    state = torch.ones(1, 2, 4)
    torch.manual_seed(37)
    first = noisy(state, lambda value, _time: value)
    next_rng = torch.get_rng_state().clone()
    torch.manual_seed(37)
    second = clean(state, lambda value, _time: value)
    assert torch.equal(torch.get_rng_state(), next_rng)
    assert not torch.equal(first, second)
    assert noisy.pop_scheduled_metrics() == clean.pop_scheduled_metrics()
    torch.testing.assert_close(
        noisy(state, lambda value, _time: torch.zeros_like(value)),
        torch.zeros_like(state),
    )


def test_time_conditioning_can_be_disabled_without_disabling_schedule():
    diffusion = TaskDirectedDiffusion(
        4, scheduled_config(time_conditioning=False, perturbation_enabled=False)
    )
    state = torch.ones(1, 2, 4)
    observed = []

    def field(value, time):
        observed.append((value.clone(), time))
        return torch.zeros_like(value)

    diffusion(state, field, start_level=25)
    assert len(observed) == 4
    for value, _ in observed:
        torch.testing.assert_close(value, 2 * state)
    assert observed[0][1] == pytest.approx(math.log(0.05))


def test_fixed_default_serialization_state_and_rng_match_legacy_recurrence():
    config = DiffusionProcessorConfig()
    expected = {
        "num_steps": 4,
        "time_conditioning": True,
        "eta_max": 1.0,
        "refinement_gate_init": -4.0,
        "noise_scale": 0.15,
        "conditioning_scale": 1.0,
        "inference_mode": "deterministic",
    }
    assert to_dict(config) == expected
    assert _config_payload(config) == expected
    assert "output_scale_mode" not in to_dict(CacheProcessorConfig())
    assert "output_scale_mode" not in _config_payload(CacheProcessorConfig())
    diffusion = TaskDirectedDiffusion(4, config)
    assert set(diffusion.state_dict()) == {"refinement_gate"}
    state = torch.ones(1, 2, 4)
    torch.manual_seed(47)
    actual = diffusion(state, lambda value, _time: value.tanh())
    actual_rng = torch.get_rng_state().clone()
    torch.manual_seed(47)
    initial = state + config.noise_scale * torch.randn_like(state)
    refined = initial
    step_size = (config.eta_max * torch.sigmoid(diffusion.refinement_gate)) / 4
    for step in range(4):
        time = step / 4
        value = refined + config.conditioning_scale * state
        value = value + sinusoidal_time_embedding(time, 4, refined)
        refined = refined + step_size * value.tanh()
    assert torch.equal(actual, refined - initial)
    assert torch.equal(actual_rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "key",
    ["schedule_horizon", "training_solver_steps", "inference_solver_steps"],
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_solver_controls_require_positive_integers(key, value):
    with pytest.raises(ValueError, match="positive integer"):
        scheduled_config(**{key: value})


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"schedule_horizon": None}, "schedule_horizon"),
        ({"training_solver_steps": None}, "training_solver_steps"),
        ({"training_method": "fixed"}, "require training_method"),
        ({"training_method": "other"}, "training_method"),
        ({"noise_scale": 0.0}, "noise_scale>0"),
        ({"time_conditioning": False}, "time_conditioning=true"),
        ({"perturbation_enabled": 1}, "perturbation_enabled"),
    ],
)
def test_invalid_scheduled_config(kwargs, match):
    with pytest.raises(ValueError, match=match):
        scheduled_config(**kwargs)


def test_config_defaults_and_admission():
    cfg = scheduled_config()
    assert cfg.scheduled_inference_solver_steps == 4
    assert DiffusionProcessorConfig(**to_dict(cfg)) == cfg
    with pytest.raises(ValueError, match="out_proj_init='zero'"):
        model_config(out_proj_init="normal")
    with pytest.raises(ValueError, match="internal.*iterations"):
        model_config(iterations=2)
    assert model_config(iterations=2, iteration_mode="external")
    for variant in ("bt", "icarus"):
        with pytest.raises(ValueError, match="fixed-one.*variant"):
            ModelConfig(
                variant=variant,
                cache_processor=CacheProcessorConfig(
                    output_scale_mode="fixed_one", out_proj_init="zero"
                ),
            )
    with pytest.raises(ValueError, match="output_scale_mode"):
        CacheProcessorConfig(output_scale_mode="other")


def test_eval_override_is_atomic_runtime_only_and_training_independent():
    diffusion = TaskDirectedDiffusion(
        4, scheduled_config(inference_solver_steps=6)
    )
    model = torch.nn.Sequential(diffusion)
    state = torch.zeros(1, 2, 4)
    before_config = to_dict(diffusion.config)
    before_state = dict(diffusion.state_dict())
    assert set_diffusion_eval_solver_steps(model, 8) == 1
    diffusion(state, lambda value, _time: value, noise=state, start_level=1)
    assert pop_diffusion_field_evaluation_count(model) == 4
    diffusion.eval()
    diffusion(state, lambda value, _time: value)
    assert pop_diffusion_field_evaluation_count(model) == 8
    assert pop_diffusion_field_evaluation_count(model) == 0
    set_diffusion_eval_solver_steps(model, None)
    diffusion(state, lambda value, _time: value)
    assert pop_diffusion_field_evaluation_count(model) == 6
    assert to_dict(diffusion.config) == before_config
    assert dict(diffusion.state_dict()) == before_state
    fixed = TaskDirectedDiffusion(4, DiffusionProcessorConfig())
    with pytest.raises(ValueError, match="scheduled"):
        set_diffusion_eval_solver_steps(
            torch.nn.ModuleList([diffusion, fixed]), 2
        )
    assert diffusion.eval_solver_steps_override is None
    with pytest.raises(ValueError, match="diffusion processor"):
        set_diffusion_eval_solver_steps(torch.nn.Linear(2, 2), 2)
    coupled = TaskDirectedDiffusion(
        4, DiffusionProcessorConfig(training_method="scheduled")
    )
    with pytest.raises(ValueError, match="decoupled"):
        coupled.set_eval_solver_steps(2)
    for steps in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match="positive"):
            diffusion.set_eval_solver_steps(steps)


def test_start_level_noise_shape_and_compile_guards():
    diffusion = TaskDirectedDiffusion(4, scheduled_config())
    state = torch.zeros(1, 2, 4)
    for level in (0, 51, True, 2.5):
        with pytest.raises(ValueError, match="start_level"):
            diffusion(state, lambda value, _time: value, start_level=level)
    with pytest.raises(ValueError, match="noise must match"):
        diffusion(state, lambda value, _time: value, noise=torch.zeros(1, 4))
    diffusion.eval()
    with pytest.raises(ValueError, match="explicit diffusion noise"):
        diffusion(state, lambda value, _time: value, noise=state)
    diffusion.recurrence_backend = "compile"
    with pytest.raises(RuntimeError, match="eager"):
        diffusion(state, lambda value, _time: value)
    with pytest.raises(ValueError, match="eager"):
        diffusion.set_eval_solver_steps(8)
    config = model_config("bgt")
    policy = NoisyDiffusionBackendConfig(recurrence_backend="compile")
    with pytest.raises(ValueError, match="scheduled diffusion"):
        TrainConfig(
            model=config, performance=PerformanceConfig(noisy_diffusion=policy)
        )
    with pytest.raises(RuntimeError, match="scheduled diffusion"):
        configure_noisy_diffusion_recurrence_backend(diffusion, config, policy)


@pytest.mark.parametrize("variant", ["bgt", "longshot"])
@pytest.mark.parametrize("output_scale_mode", ["learned", "fixed_one"])
def test_scheduled_gates_noop_gradient_accounting_and_telemetry(
    variant, output_scale_mode
):
    config = model_config(variant, output_scale_mode=output_scale_mode)
    model = build_model(config)
    model.init_weights()
    block = model.cache_processor.block
    assert block.gate is None
    assert block.diffusion.refinement_gate is None
    assert torch.count_nonzero(block.out_proj.weight) == 0
    assert processor_params(config) == sum(
        p.numel() for p in model.cache_processor.parameters()
    )
    optimizer = torch.optim.SGD(model.cache_processor.parameters(), lr=0.1)
    tokens = torch.randint(0, config.vocab_size, (2, config.sequence_len))
    loss = model(tokens, tokens)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(block.out_proj.weight.grad)
    metrics = cache_processor_metrics(model)
    assert metrics["cache_processor/gate"] == 1.0
    assert 0 < metrics["cache_processor/diffusion_start_time_mean"] <= 1
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(tokens, tokens).backward()
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad)
        for p in block.layers.parameters()
    )


@pytest.mark.parametrize("variant", ["bgt", "longshot"])
def test_fixed_one_without_diffusion_parameter_accounting(variant):
    config = replace(
        model_config(variant, output_scale_mode="fixed_one"),
        diffusion_processor=None,
    )
    model = build_model(config)
    assert model.cache_processor.block.gate is None
    assert processor_params(config) == sum(
        p.numel() for p in model.cache_processor.parameters()
    )


def test_longshot_uses_initial_routing_gates_for_all_fields_and_keeps_tuple_abi(
    monkeypatch,
):
    torch.manual_seed(57)
    cfg = model_config(head_selection_method="learned_contextual")
    block = CacheProcessor(
        16,
        4,
        cfg.cache_processor,
        encoder_method="query_attention",
        diffusion_config=cfg.diffusion_processor,
    )
    with torch.no_grad():
        block.out_proj.weight.normal_(std=0.02)
    block.head_selector.set_training_progress(
        LongshotSelectorTrainingConfig(), 0.10
    )
    calls = []
    original = block._run_flow_field

    def field(state, selected, routing_gates=None):
        calls.append((selected, routing_gates))
        return original(state, selected, routing_gates)

    monkeypatch.setattr(block, "_run_flow_field", field)
    route = Mock(wraps=block._route_heads)
    monkeypatch.setattr(block, "_route_heads", route)
    updated, linear_delta, selected = block(
        torch.randn(1, 4, 3, 16), topk_heads=2
    )
    assert updated.shape == (1, 4, 16)
    assert linear_delta is None
    assert len(calls) == 4
    route.assert_called_once()
    assert calls[0][1] is not None
    assert all(ids is selected and gates is calls[0][1] for ids, gates in calls)
    updated.square().sum().backward()
    gradient = block.head_selector.context_proj.weight.grad
    assert gradient is not None and torch.count_nonzero(gradient)
    assert processor_params(cfg) == sum(
        p.numel() for p in build_model(cfg).cache_processor.parameters()
    )


def test_flops_and_receipts_use_solver_depth_not_horizon_or_num_steps():
    cfg = model_config()
    cost = processor_forward_flops_per_sequence(cfg, 8)
    for diffusion in (
        scheduled_config(schedule_horizon=100),
        scheduled_config(num_steps=12),
        scheduled_config(inference_solver_steps=8),
    ):
        assert (
            processor_forward_flops_per_sequence(
                replace(cfg, diffusion_processor=diffusion), 8
            )
            == cost
        )
    assert (
        processor_forward_flops_per_sequence(
            replace(
                cfg,
                diffusion_processor=scheduled_config(training_solver_steps=8),
            ),
            8,
        )
        > cost
    )
    coupled = DiffusionProcessorConfig(training_method="scheduled", num_steps=7)
    assert (
        processor_forward_flops_per_sequence(
            replace(cfg, diffusion_processor=coupled), 8
        )
        == cost
    )
    train = TrainConfig(model=cfg)
    receipt = resolve_experiment(train, resolve_model(cfg).receipt).receipt
    assert "horizon=50:train_solver=4:eval_solver=4" in receipt.flow_plan
    assert "diffusion_training_solver_steps:4" in receipt.intervention_schedule
    payload = _config_payload(train)
    assert "truncated_bptt_windows" not in payload
    assert payload["longshot_selector_training"] is None
    metadata = noisy_diffusion_recurrence_backend_metadata(
        cfg, NoisyDiffusionBackendConfig()
    )
    assert metadata["training_solver_steps"] == 4
    assert metadata["region"].endswith("scheduled_recurrence")


@pytest.mark.parametrize("suite", ["all", "math", "general", "tasks", "chat"])
def test_eval_cli_accepts_independent_solver_depth(suite):
    extra = ["--tasks", "gsm8k"] if suite in {"tasks", "chat"} else []
    args = build_parser().parse_args(
        [
            "eval",
            suite,
            "--checkpoint",
            "/unused",
            "--diffusion-eval-solver-steps",
            "8",
            *extra,
        ]
    )
    assert args.diffusion_eval_solver_steps == 8


@pytest.mark.parametrize(
    "hf_model,enabled,steps,match",
    [
        ("unused", True, 4, "HF targets"),
        (None, False, 4, "processor=on"),
        (None, True, 0, "positive"),
        (None, True, True, "positive"),
    ],
)
def test_eval_override_rejects_invalid_targets_before_loading(
    monkeypatch, hf_model, enabled, steps, match
):
    from gcw.evaluation import lm_eval_adapter

    load = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr(lm_eval_adapter, "load_checkpoint", load)
    with pytest.raises(ValueError, match=match):
        lm_eval_adapter._load_eval_target(
            checkpoint=None if hf_model else "/unused",
            hf_model=hf_model,
            processor_enabled=enabled,
            device=torch.device("cpu"),
            diffusion_eval_solver_steps=steps,
        )
    load.assert_not_called()


def test_eval_loader_applies_depth_after_loading_and_keeps_upstream_overrides(
    monkeypatch,
):
    from gcw.evaluation import lm_eval_adapter

    cfg = model_config()
    model = build_model(cfg)
    original_diffusion = to_dict(cfg.diffusion_processor)
    load = Mock(return_value=(model, cfg, {}))
    monkeypatch.setattr(lm_eval_adapter, "load_checkpoint", load)
    tokenizer = object()
    monkeypatch.setattr(
        lm_eval_adapter, "tokenizer_for_checkpoint", lambda *_: tokenizer
    )
    actual, actual_tokenizer = lm_eval_adapter._load_eval_target(
        checkpoint="/unused",
        hf_model=None,
        processor_enabled=True,
        device=torch.device("cpu"),
        diffusion_eval_solver_steps=8,
        generation_process_every=2,
        head_selection_method="random",
    )
    assert actual is model and actual_tokenizer is tokenizer
    load.assert_called_once()
    assert model.cache_processor.block.diffusion.eval_solver_steps_override == 8
    assert to_dict(cfg.diffusion_processor) == original_diffusion
    assert cfg.cache_processor.generation_process_every == 2
    assert cfg.cache_processor.head_selection_method == "random"
    assert cfg.cache_processor.internal_iteration_score == "random"


@pytest.mark.parametrize(
    "hf_model,enabled,steps,match",
    [
        ("unused", True, 4, "HF targets"),
        (None, False, 4, "processor=on"),
        (None, True, 0, "positive"),
    ],
)
def test_chat_override_validation_precedes_cuda_and_loading(
    hf_model, enabled, steps, match
):
    from gcw.evaluation.chat_benchmarks import run_chat_benchmarks

    with pytest.raises(ValueError, match=match):
        run_chat_benchmarks(
            tasks=["gsm8k_openmath"],
            config=EvalConfig(batch_size=1, apply_chat_template=True),
            checkpoint=None if hf_model else "/unused",
            hf_model=hf_model,
            processor_enabled=enabled,
            diffusion_eval_solver_steps=steps,
        )
