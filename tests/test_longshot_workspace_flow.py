from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gcw.config import CacheProcessorConfig, ModelConfig, WorkspaceFlowRefinementConfig
from gcw.models import LanguageModel, build_model
from gcw.models.cache_processors.longshot import GlobalCacheProcessor
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.params import processor_params


def _config(
    *,
    refinement: WorkspaceFlowRefinementConfig | None,
    out_proj_init: str = "normal",
) -> ModelConfig:
    return ModelConfig(
        variant="longshot",
        family="smollm",
        sequence_len=16,
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=0,
            topk_heads=4,
            score="kv_norm",
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            encoder_method="query_attention",
            encoder_query_source="final_token",
            update_stream="v",
            out_proj_init=out_proj_init,  # type: ignore[arg-type]
            gate_init=0.0,
        ),
        workspace_flow_refinement=refinement,
    )


def _cache() -> WindowedCache:
    torch.manual_seed(7)
    return WindowedCache(
        keys=torch.randn(2, 2, 2, 4, 8),
        values=torch.randn(2, 2, 2, 4, 8),
        scores=torch.randn(2, 4),
    )


def _workspace(processor: GlobalCacheProcessor, cache: WindowedCache, cycles: int):
    source = processor.encode_source(cache, select_all=True)
    workspace = processor.initialise_workspace(source)
    for _ in range(cycles):
        workspace = processor.deliberate(workspace, source)
    return source, workspace


@pytest.mark.parametrize("solver", ["euler", "midpoint"])
@pytest.mark.parametrize("solver_steps", [1, 2, 4])
@pytest.mark.parametrize("cycles", [0, 1, 3])
def test_workspace_flow_preserves_k_and_solver_step_shapes(
    solver: str,
    solver_steps: int,
    cycles: int,
) -> None:
    config = _config(
        refinement=WorkspaceFlowRefinementConfig(
            solver=solver,  # type: ignore[arg-type]
            solver_steps=solver_steps,
            max_velocity=0.2,
        )
    )
    processor = GlobalCacheProcessor(config)
    cache = _cache()
    source, workspace = _workspace(processor, cache, cycles)

    refined = processor.refine_workspace(workspace, source)
    branch = processor.write_for_readout(cache, source, workspace)

    assert workspace.shape == refined.shape == (2, 4, 8)
    assert branch.keys.shape == cache.keys.shape
    assert branch.values.shape == cache.values.shape


@pytest.mark.parametrize("solver", ["euler", "midpoint"])
@pytest.mark.parametrize("solver_steps", [1, 2, 4, 8])
def test_zero_output_workspace_field_is_exact_identity(solver: str, solver_steps: int) -> None:
    config = _config(
        refinement=WorkspaceFlowRefinementConfig(
            solver=solver,  # type: ignore[arg-type]
            solver_steps=solver_steps,
            max_velocity=0.5,
        ),
        out_proj_init="zero",
    )
    processor = GlobalCacheProcessor(config)
    cache = _cache()
    source, workspace = _workspace(processor, cache, cycles=0)

    refined = processor.refine_workspace(workspace, source)
    branch = processor.write_for_readout(cache, source, workspace)

    assert torch.equal(refined, workspace)
    assert torch.equal(branch.keys, cache.keys)
    assert torch.equal(branch.values, cache.values)


def test_workspace_flow_is_bounded_and_backpropagates() -> None:
    max_velocity = 0.25
    config = _config(
        refinement=WorkspaceFlowRefinementConfig(
            solver="midpoint",
            solver_steps=4,
            max_velocity=max_velocity,
        )
    )
    processor = GlobalCacheProcessor(config)
    source, initial = _workspace(processor, _cache(), cycles=2)
    source = replace(source, merged=source.merged.detach().requires_grad_(True))
    workspace = initial.detach().requires_grad_(True)

    refined = processor.refine_workspace(workspace, source)
    refined.square().mean().backward()

    assert (refined - workspace).abs().max().item() <= max_velocity + 1e-6
    assert workspace.grad is not None and torch.count_nonzero(workspace.grad).item() > 0
    assert source.merged.grad is not None and torch.count_nonzero(source.merged.grad).item() > 0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in processor.parameters()
    )


def test_workspace_flow_keeps_frozen_backbone_frozen_and_processor_trainable() -> None:
    model = build_model(
        _config(
            refinement=WorkspaceFlowRefinementConfig(
                solver="euler",
                solver_steps=2,
                max_velocity=0.25,
            )
        )
    )
    assert isinstance(model, LanguageModel)
    model.init_weights()
    model.backbone.requires_grad_(False)
    processor = model.cache_processor
    assert isinstance(processor, GlobalCacheProcessor)
    cache = _cache()
    source, workspace = _workspace(processor, cache, cycles=1)

    processor.write_for_readout(cache, source, workspace).values.square().mean().backward()

    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in processor.parameters()
    )


def test_workspace_flow_parameter_count_is_invariant_to_solver_depth() -> None:
    one_step = _config(
        refinement=WorkspaceFlowRefinementConfig(
            solver="euler",
            solver_steps=1,
            max_velocity=0.25,
        )
    )
    eight_steps = replace(
        one_step,
        workspace_flow_refinement=WorkspaceFlowRefinementConfig(
            solver="midpoint",
            solver_steps=8,
            max_velocity=0.25,
        ),
    )
    one_processor = GlobalCacheProcessor(one_step)
    eight_processor = GlobalCacheProcessor(eight_steps)
    one_actual = sum(parameter.numel() for parameter in one_processor.parameters())
    eight_actual = sum(parameter.numel() for parameter in eight_processor.parameters())

    assert one_actual == eight_actual
    assert processor_params(one_step) == one_actual
    assert processor_params(eight_steps) == eight_actual


def test_workspace_flow_opt_out_preserves_existing_readout_semantics() -> None:
    processor = GlobalCacheProcessor(_config(refinement=None))
    cache = _cache()
    source, workspace = _workspace(processor, cache, cycles=2)

    assert processor.refine_workspace(workspace, source) is workspace
    branch = processor.write_for_readout(cache, source, workspace)
    expected_values = cache.values.clone()
    expected_values[:, :, :, -1] = workspace.view(2, 2, 2, 8).permute(1, 0, 2, 3)

    assert torch.equal(branch.keys, cache.keys)
    assert torch.equal(branch.scores, cache.scores)
    assert torch.equal(branch.values, expected_values)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"solver": "rk4"}, "solver"),
        ({"solver_steps": 0}, "solver_steps"),
        ({"solver_steps": True}, "solver_steps"),
        ({"max_velocity": 0.0}, "max_velocity"),
        ({"max_velocity": float("inf")}, "max_velocity"),
    ],
)
def test_workspace_flow_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorkspaceFlowRefinementConfig(**kwargs)  # type: ignore[arg-type]


def test_workspace_flow_config_is_longshot_value_only() -> None:
    refinement = WorkspaceFlowRefinementConfig()
    with pytest.raises(ValueError, match="variant 'longshot'"):
        ModelConfig(variant="tf", workspace_flow_refinement=refinement)
    with pytest.raises(ValueError, match="value-only"):
        ModelConfig(variant="longshot", workspace_flow_refinement=refinement)


def test_workspace_flow_config_builds_from_experimental_mapping() -> None:
    config = replace(
        _config(refinement=None),
        workspace_flow_refinement={
            "solver": "midpoint",
            "solver_steps": 4,
            "max_velocity": 0.3,
        },  # type: ignore[arg-type]
    )

    assert config.workspace_flow_refinement == WorkspaceFlowRefinementConfig(
        solver="midpoint",
        solver_steps=4,
        max_velocity=0.3,
    )
