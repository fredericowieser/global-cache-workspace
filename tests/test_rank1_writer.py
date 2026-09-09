from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from gcw.architecture.contracts import _config_payload
from gcw.config import (
    CacheProcessorConfig,
    CacheUpdateGateType,
    DiffusionProcessorConfig,
    ModelConfig,
    to_dict,
)
from gcw.models import build_model


def config(
    *,
    live: bool = True,
    update_gate_type: CacheUpdateGateType = "static_scalar",
) -> ModelConfig:
    return ModelConfig(
        variant="longshot",
        family="qwen3_5",
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=32,
        tie_embeddings=False,
        qk_norm=True,
        rotary_style="llama",
        layer_types=["linear_attention", "full_attention"] * 2,
        processor_layer_indices=[1, 3],
        partial_rotary_factor=0.5,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=2,
            topk_heads=2,
            hidden_size=8,
            intermediate_size=16,
            num_heads=1,
            num_kv_heads=1,
            encoder_method="query_attention",
            encoder_query_source="final_token",
            linear_state_cache=True,
            out_proj_init="zero",
            live_rank1_linear_writer=live,
            update_gate_type=update_gate_type,
        ),
    )


def test_disabled_writer_preserves_serialization() -> None:
    implicit = CacheProcessorConfig()
    explicit = CacheProcessorConfig(live_rank1_linear_writer=False)
    for serialize in (to_dict, _config_payload):
        assert serialize(implicit) == serialize(explicit)
        assert "live_rank1_linear_writer" not in serialize(implicit)
        assert (
            serialize(config().cache_processor)["live_rank1_linear_writer"]
            is True
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"live_rank1_linear_writer": "true"},
        {"linear_state_cache": False},
        {"linear_state_update_mode": "gated_delta"},
        {"out_proj_init": "normal"},
    ],
)
def test_writer_rejects_incompatible_configuration(changes: dict) -> None:
    with pytest.raises(ValueError, match="live_rank1_linear_writer"):
        replace(config().cache_processor, **changes)


def test_writer_preserves_rng_shapes_and_zero_initial_logits() -> None:
    torch.manual_seed(991)
    disabled = build_model(config(live=False))
    disabled.init_weights()
    before = torch.random.get_rng_state()
    torch.manual_seed(991)
    enabled = build_model(config())
    enabled.init_weights()
    assert torch.equal(before, torch.random.get_rng_state())
    old, new = disabled.state_dict(), enabled.state_dict()
    assert old.keys() == new.keys()
    changed = "cache_processor.block.linear_state_adapter.output_proj.weight"
    for name in old:
        assert old[name].shape == new[name].shape
        if name != changed:
            assert torch.equal(old[name], new[name]), name
    adapter = enabled.cache_processor.block.linear_state_adapter
    assert torch.count_nonzero(old[changed]) == 0
    assert torch.count_nonzero(new[changed][: adapter.linear_key_dim]) > 0
    assert torch.count_nonzero(new[changed][adapter.linear_key_dim :]) == 0
    assert torch.count_nonzero(adapter.output_proj.bias) == 0
    tokens = torch.arange(9).unsqueeze(0)
    with torch.no_grad():
        assert torch.equal(disabled.eval()(tokens), enabled.eval()(tokens))
    with torch.device("meta"):
        meta = build_model(config())
    assert all(p.is_meta for p in meta.parameters())


def test_zero_update_has_live_value_then_key_gradients() -> None:
    model = build_model(config())
    adapter = model.cache_processor.block.linear_state_adapter
    hidden = torch.linspace(-1, 1, 3 * adapter.context_dim).reshape(1, 3, -1)
    optimizer = torch.optim.SGD(adapter.output_proj.parameters(), lr=0.1)
    for step in range(2):
        key, value = adapter.output_proj(hidden).split(
            [adapter.linear_key_dim, adapter.linear_value_dim], dim=-1
        )
        delta = torch.einsum(
            "bhk,bhv->bhkv", F.normalize(key.float(), dim=-1), value.float()
        )
        assert bool(torch.count_nonzero(delta)) == bool(step)
        target = torch.linspace(-0.75, 0.5, delta.numel()).reshape_as(delta)
        (delta * target).sum().backward()
        gradient = adapter.output_proj.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient[adapter.linear_key_dim :]) > 0
        assert bool(
            torch.count_nonzero(gradient[: adapter.linear_key_dim])
        ) == bool(step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize(
    "update_gate_type", ["contextual_scalar", "contextual_vector"]
)
def test_contextual_gate_keeps_rank_one_writer_live(
    update_gate_type: CacheUpdateGateType,
) -> None:
    model_config = config(update_gate_type=update_gate_type)
    model = build_model(model_config)
    model.init_weights()
    adapter = model.cache_processor.block.linear_state_adapter
    weight = adapter.output_proj.weight

    assert torch.count_nonzero(weight[: adapter.linear_key_dim]) > 0
    assert (
        torch.count_nonzero(weight[adapter.linear_key_dim : adapter.output_dim])
        == 0
    )
    assert torch.count_nonzero(weight[adapter.output_dim :]) == 0

    tokens = torch.randint(0, model_config.vocab_size, (2, 9))
    model(tokens, tokens).backward()
    gradient = adapter.output_proj.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert (
        torch.count_nonzero(
            gradient[adapter.linear_key_dim : adapter.output_dim]
        )
        > 0
    )


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_terminal_ce_reaches_rank_one_writer(
    scheduled: bool, dtype: torch.dtype
) -> None:
    torch.manual_seed(993)
    model_config = config()
    if scheduled:
        model_config = replace(
            model_config,
            diffusion_processor=DiffusionProcessorConfig(
                training_method="scheduled",
                num_steps=4,
                noise_scale=0.1,
            ),
        )
    model = build_model(model_config)
    model.init_weights()
    model = model.to(dtype=dtype)
    tokens = torch.randint(0, model_config.vocab_size, (2, 9))
    model(tokens, tokens).backward()
    adapter = model.cache_processor.block.linear_state_adapter
    gradient = adapter.output_proj.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[: adapter.linear_key_dim]) == 0
    assert torch.count_nonzero(gradient[adapter.linear_key_dim :]) > 0
