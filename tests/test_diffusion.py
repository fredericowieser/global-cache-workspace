from __future__ import annotations

import torch

from gcw.config import CacheProcessorConfig, DiffusionProcessorConfig, ModelConfig
from gcw.models import build_model
from gcw.models.cache_processors.diffusion import TaskDirectedDiffusion
from gcw.params import processor_params


def tiny_diffusion_config() -> ModelConfig:
    return ModelConfig(
        variant="bgt",
        family="llama",
        sequence_len=8,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        rotary_style="llama",
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=2,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
        ),
        diffusion_processor=DiffusionProcessorConfig(num_steps=4, noise_scale=0.15),
        processor_generation_mode="post_prefill_relative",
    )


def test_diffusion_is_stochastic_during_training_and_deterministic_for_eval():
    processor = TaskDirectedDiffusion(4, DiffusionProcessorConfig(num_steps=2, noise_scale=0.25))
    state = torch.zeros(1, 2, 4)

    def field(value: torch.Tensor, _time: float) -> torch.Tensor:
        return torch.ones_like(value)

    processor.train()
    first = processor(state, field, noise=torch.zeros_like(state))
    second = processor(state, field, noise=torch.ones_like(state))
    assert not torch.equal(first, second)

    processor.eval()
    assert torch.equal(processor(state, field), processor(state, field))


def test_diffusion_bgt_forward_backward_and_accounting():
    config = tiny_diffusion_config()
    model = build_model(config)
    model.init_weights()
    tokens = torch.randint(0, config.vocab_size, (2, config.sequence_len))

    loss = model(tokens, tokens)
    loss.backward()

    diffusion = next(module for module in model.modules() if isinstance(module, TaskDirectedDiffusion))
    assert diffusion.refinement_gate.grad is not None
    assert torch.isfinite(loss)
    assert config.architecture_name == "diffusion_bgt"
    assert processor_params(config) == sum(parameter.numel() for parameter in model.cache_processor.parameters())


def test_flow_and_diffusion_are_mutually_exclusive():
    from gcw.config import FlowProcessorConfig

    try:
        ModelConfig(
            variant="bgt",
            flow_processor=FlowProcessorConfig(),
            diffusion_processor=DiffusionProcessorConfig(),
        )
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("flow and diffusion were admitted together")
