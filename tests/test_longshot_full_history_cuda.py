from __future__ import annotations

from collections.abc import Callable
import copy
import os
from types import MethodType

import pytest
import torch

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.data.training import RandomCursor
from gcw.models import build_model
from gcw.models.cache_processors.longshot import GlobalCacheProcessor
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.performance.config import LongshotFullHistoryConfig
from gcw.performance.longshot_cache import (
    configure_longshot_full_history_execution,
    pop_longshot_full_history_metrics,
    longshot_full_history_execution_metadata,
)
from gcw.training.checkpoint import ResumeState, TrainingCheckpoint, save_training_checkpoint
from gcw.training.objectives import row_loss_sums_and_counts

_RUN_CUDA_PARITY = os.environ.get("GCW_RUN_LONGSHOT_CUDA_PARITY") == "1"
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _RUN_CUDA_PARITY or not torch.cuda.is_available(),
        reason="set GCW_RUN_LONGSHOT_CUDA_PARITY=1 on a CUDA host",
    ),
]


def _processor(*, layers: int = 2, topk: int = 8, unrope_keys: bool = False) -> GlobalCacheProcessor:
    config = ModelConfig(
        variant="longshot",
        family="smollm",
        sequence_len=4096,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_layers=layers,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        cache_processor=CacheProcessorConfig(
            process_every=16,
            topk=topk,
            topk_heads=min(4, layers * 2),
            score="kv_norm",
            hidden_size=32,
            intermediate_size=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            encoder_method="query_attention",
            encoder_query_source="final_token",
            iterations=2,
            iteration_mode="internal",
            internal_iteration_score="kv_norm",
            gradient_checkpointing=True,
            attention_dropout=0.0,
            unrope_keys=unrope_keys,
            gate_init=0.5,
        ),
    )
    return GlobalCacheProcessor(config).cuda().train()


def _cache(seed: int, *, layers: int = 2, sequence: int = 32) -> WindowedCache:
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn(layers, 4, 2, sequence, 16, generator=generator, dtype=torch.bfloat16).cuda()
    values = torch.randn(layers, 4, 2, sequence, 16, generator=generator, dtype=torch.bfloat16).cuda()
    scores = torch.randn(4, sequence, generator=generator).cuda()
    return WindowedCache(keys=keys.contiguous(), values=values.contiguous(), scores=scores)


def _trajectory_step(
    processor: GlobalCacheProcessor,
    optimizer: torch.optim.Optimizer,
    cache: WindowedCache,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor | None]]:
    optimizer.zero_grad(set_to_none=True)
    selected = processor._select_kv_norm_history(cache)[2]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = processor(cache)
        row_losses = output.keys.float().square().mean(dim=(0, 2, 3, 4))
        row_losses = row_losses + output.values.float().square().mean(dim=(0, 2, 3, 4))
        pair_losses = row_losses.reshape(-1, 2).mean(dim=1)
        loss = pair_losses.sum()
    loss.backward()
    gradients = [
        None if parameter.grad is None else parameter.grad.detach().clone() for parameter in processor.parameters()
    ]
    optimizer.step()
    return selected, output.keys.detach(), output.values.detach(), pair_losses.detach(), gradients


@pytest.mark.parametrize("unrope_keys", [False, True])
def test_cuda_bf16_checkpointed_trajectory_and_resume_identity(unrope_keys: bool) -> None:
    torch.manual_seed(211)
    reference = _processor(unrope_keys=unrope_keys)
    optimized = copy.deepcopy(reference)
    configure_longshot_full_history_execution(optimized, LongshotFullHistoryConfig(enabled=True, max_temp_bytes=4096))
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4)
    optimized_optimizer = torch.optim.AdamW(optimized.parameters(), lr=3e-4)

    for seed in (223, 227):
        reference_result = _trajectory_step(reference, reference_optimizer, _cache(seed))
        optimized_result = _trajectory_step(optimized, optimized_optimizer, _cache(seed))
        for actual, expected in zip(optimized_result[:4], reference_result[:4], strict=True):
            assert torch.equal(actual, expected)
        for actual, expected in zip(optimized_result[4], reference_result[4], strict=True):
            assert (actual is None) == (expected is None)
            if actual is not None:
                assert expected is not None
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        for actual, expected in zip(optimized.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    saved_model = copy.deepcopy(optimized.state_dict())
    saved_optimizer = copy.deepcopy(optimized_optimizer.state_dict())
    saved_rng = torch.cuda.get_rng_state()
    uninterrupted = _trajectory_step(optimized, optimized_optimizer, _cache(229))
    uninterrupted_state = copy.deepcopy(optimized.state_dict())

    resumed = _processor(unrope_keys=unrope_keys)
    configure_longshot_full_history_execution(resumed, LongshotFullHistoryConfig(enabled=True, max_temp_bytes=4096))
    resumed.load_state_dict(saved_model)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=3e-4)
    resumed_optimizer.load_state_dict(saved_optimizer)
    torch.cuda.set_rng_state(saved_rng)
    resumed_result = _trajectory_step(resumed, resumed_optimizer, _cache(229))

    for actual, expected in zip(resumed_result[:4], uninterrupted[:4], strict=True):
        assert torch.equal(actual, expected)
    for name, expected in uninterrupted_state.items():
        assert torch.equal(resumed.state_dict()[name], expected)


def _language_model_config(*, unrope_keys: bool = False) -> ModelConfig:
    config = ModelConfig(
        variant="longshot",
        family="qwen3",
        sequence_len=16,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=2,
            topk_heads=2,
            score="kv_norm",
            hidden_size=8,
            intermediate_size=16,
            num_layers=2,
            num_heads=1,
            num_kv_heads=1,
            encoder_method="query_attention",
            encoder_query_source="final_token",
            iterations=2,
            iteration_mode="internal",
            internal_iteration_score="kv_norm",
            gradient_checkpointing=True,
            attention_dropout=0.0,
            unrope_keys=unrope_keys,
        ),
    )
    return config


def _language_model(*, unrope_keys: bool = False) -> torch.nn.Module:
    model = build_model(_language_model_config(unrope_keys=unrope_keys))
    model.init_weights()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("cache_processor."))
    return model.cuda().train()


def _install_position_trace(model: torch.nn.Module) -> list[torch.Tensor]:
    processor = model.cache_processor
    original = processor._select_kv_norm_history
    trace: list[torch.Tensor] = []

    def traced(_self: GlobalCacheProcessor, cache: WindowedCache):
        result = original(cache)
        trace.append(result[2].detach().clone())
        return result

    object.__setattr__(processor, "_select_kv_norm_history", MethodType(traced, processor))
    return trace


def _language_model_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    trace: list[torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
    optimizer.zero_grad(set_to_none=True)
    trace.clear()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        row_sums, row_counts = row_loss_sums_and_counts(
            model,
            inputs,
            targets,
            activation_checkpoint_windows=2,
            longshot_gradient_checkpointing=True,
        )
        pair_losses = row_sums.reshape(-1, 2).sum(dim=1) / row_counts.reshape(-1, 2).sum(dim=1)
        pair_losses.mean().backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.cache_processor.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()
    metrics = pop_longshot_full_history_metrics(model.cache_processor)
    return pair_losses.detach(), list(trace), gradients, metrics


@pytest.mark.parametrize("unrope_keys", [False, True])
def test_cuda_bf16_segmented_language_model_pair_loss_trajectory(unrope_keys: bool) -> None:
    torch.manual_seed(241)
    reference = _language_model(unrope_keys=unrope_keys)
    optimized = copy.deepcopy(reference)
    configure_longshot_full_history_execution(optimized, LongshotFullHistoryConfig(enabled=True, max_temp_bytes=4096))
    reference_trace = _install_position_trace(reference)
    optimized_trace = _install_position_trace(optimized)
    reference_optimizer = torch.optim.AdamW(reference.cache_processor.parameters(), lr=3e-4)
    optimized_optimizer = torch.optim.AdamW(optimized.cache_processor.parameters(), lr=3e-4)

    generator = torch.Generator().manual_seed(251)
    for _ in range(2):
        inputs = torch.randint(0, 64, (4, 12), generator=generator).cuda()
        targets = torch.randint(0, 64, (4, 12), generator=generator).cuda()
        expected_losses, expected_trace, expected_gradients, expected_metrics = _language_model_step(
            reference, reference_optimizer, reference_trace, inputs, targets
        )
        actual_losses, actual_trace, actual_gradients, actual_metrics = _language_model_step(
            optimized, optimized_optimizer, optimized_trace, inputs, targets
        )

        assert torch.equal(actual_losses, expected_losses)
        assert len(actual_trace) == len(expected_trace)
        for actual, expected in zip(actual_trace, expected_trace, strict=True):
            assert torch.equal(actual, expected)
        assert actual_gradients.keys() == expected_gradients.keys()
        for name, expected in expected_gradients.items():
            torch.testing.assert_close(actual_gradients[name], expected, rtol=0.0, atol=0.0)
        for actual, expected in zip(
            optimized.cache_processor.parameters(),
            reference.cache_processor.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert expected_metrics == {}
        assert actual_metrics["performance/longshot_full_history_events"] == 2.0


@pytest.mark.parametrize("unrope_keys", [False, True])
def test_cuda_peak_memory_and_counters_prove_selected_only_materialization(
    record_property: Callable[[str, object], None],
    unrope_keys: bool,
) -> None:
    torch.manual_seed(233)
    reference = _processor(layers=4, topk=16, unrope_keys=unrope_keys)
    optimized = copy.deepcopy(reference)
    configure_longshot_full_history_execution(
        optimized, LongshotFullHistoryConfig(enabled=True, max_temp_bytes=2 * 1024 * 1024)
    )
    cache = _cache(239, layers=4, sequence=2048)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    reference_result = reference._select_kv_norm_history(cache)
    torch.cuda.synchronize()
    reference_peak = torch.cuda.max_memory_allocated()
    reference_cpu = tuple(tensor.cpu() for tensor in reference_result)
    del reference_result
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    optimized_result = optimized._select_kv_norm_history(cache)
    torch.cuda.synchronize()
    optimized_peak = torch.cuda.max_memory_allocated()
    metrics = pop_longshot_full_history_metrics(optimized)

    for actual, expected in zip(optimized_result, reference_cpu, strict=True):
        assert torch.equal(actual.cpu(), expected)

    record_property("reference_peak_allocated_bytes", reference_peak)
    record_property("optimized_peak_allocated_bytes", optimized_peak)
    record_property("profile_counters", metrics)
    assert optimized_peak < reference_peak
    assert (
        metrics["performance/longshot_full_history_selected_history_logical_payload_bytes"]
        < metrics["performance/longshot_full_history_reference_full_history_logical_payload_bytes"]
    )
    assert (
        metrics[
            "performance/longshot_full_history_optimized_score_estimated_fp32_working_set_bound_bytes"
        ]
        < metrics[
            "performance/longshot_full_history_reference_score_estimated_fp32_working_set_bound_bytes"
        ]
    )
    if unrope_keys:
        assert (
            metrics[
                "performance/longshot_full_history_optimized_selected_unrope_estimated_live_bytes"
            ]
            < metrics[
                "performance/longshot_full_history_reference_unrope_estimated_live_bytes"
            ]
        )


def test_h200_production_qwen3_unrope_peak_memory_and_exact_outputs(
    record_property: Callable[[str, object], None],
) -> None:
    if torch.cuda.get_device_properties(0).total_memory < 70 * 1024**3:
        pytest.skip("production-sized Longshot memory acceptance requires an H200-class GPU")
    config = ModelConfig(
        variant="longshot",
        family="qwen3",
        sequence_len=8192,
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
        rope_theta=1_000_000.0,
        qk_norm=True,
        cache_processor=CacheProcessorConfig(
            process_every=64,
            topk=32,
            topk_heads=32,
            score="kv_norm",
            hidden_size=1152,
            intermediate_size=4608,
            num_layers=1,
            num_heads=16,
            num_kv_heads=16,
            encoder_method="query_attention",
            encoder_query_source="final_token",
            gradient_checkpointing=True,
            update_stream="v",
            unrope_keys=True,
        ),
    )
    reference = GlobalCacheProcessor(config).cuda().eval()
    optimized = copy.deepcopy(reference)
    configure_longshot_full_history_execution(
        optimized, LongshotFullHistoryConfig(enabled=True, max_temp_bytes=2 * 1024 * 1024)
    )
    generator = torch.Generator().manual_seed(271)
    shape = (28, 2, 8, 8192, 128)
    cache = WindowedCache(
        keys=torch.randn(shape, generator=generator, dtype=torch.bfloat16).cuda(),
        values=torch.randn(shape, generator=generator, dtype=torch.bfloat16).cuda(),
        scores=torch.randn(2, 8192, generator=generator).cuda(),
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    expected = reference._select_kv_norm_history(cache)
    torch.cuda.synchronize()
    reference_peak = torch.cuda.max_memory_allocated()
    expected_cpu = tuple(tensor.cpu() for tensor in expected)
    del expected
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    actual = optimized._select_kv_norm_history(cache)
    torch.cuda.synchronize()
    optimized_peak = torch.cuda.max_memory_allocated()
    for actual_tensor, expected_tensor in zip(actual, expected_cpu, strict=True):
        assert torch.equal(actual_tensor.cpu(), expected_tensor)
    record_property("production_reference_peak_allocated_bytes", reference_peak)
    record_property("production_optimized_peak_allocated_bytes", optimized_peak)
    assert optimized_peak < reference_peak


def test_cuda_bf16_unrope_full_checkpoint_resume_replays_exactly(tmp_path) -> None:
    torch.manual_seed(263)
    history = LongshotFullHistoryConfig(enabled=True, max_temp_bytes=4096)
    config = _language_model_config(unrope_keys=True)
    model = _language_model(unrope_keys=True)
    configure_longshot_full_history_execution(model, history)
    optimizer = torch.optim.AdamW(model.cache_processor.parameters(), lr=3e-4)
    trace = _install_position_trace(model)
    inputs = torch.randint(0, 64, (4, 12), device="cuda")
    targets = torch.randint(0, 64, (4, 12), device="cuda")
    _language_model_step(model, optimizer, trace, inputs, targets)
    fingerprint = {
        "longshot_full_history_execution": longshot_full_history_execution_metadata(
            history, unrope_keys=True
        )
    }
    resume = ResumeState(
        step=1,
        smooth_loss=0.5,
        loader_cursor=RandomCursor(rng_state=(1, 2, 3)),
        saved_world_size=1,
        exact_topology=True,
    )
    checkpoint_path = tmp_path / "checkpoint"
    save_training_checkpoint(checkpoint_path, model, config, optimizer, resume, fingerprint)

    restored = _language_model(unrope_keys=True)
    configure_longshot_full_history_execution(restored, history)
    restored_optimizer = torch.optim.AdamW(restored.cache_processor.parameters(), lr=3e-4)
    checkpoint = TrainingCheckpoint.open(
        checkpoint_path,
        expected_fingerprint=fingerprint,
        topology="same",
        current_world_size=1,
    )
    checkpoint.restore_model(restored)
    checkpoint.restore_optimizer(restored_optimizer, device="cuda")
    assert checkpoint.resume_state == resume

    expected_trace = _install_position_trace(model)
    actual_trace = _install_position_trace(restored)
    next_inputs = torch.randint(0, 64, (4, 12), device="cuda")
    next_targets = torch.randint(0, 64, (4, 12), device="cuda")
    expected = _language_model_step(model, optimizer, expected_trace, next_inputs, next_targets)
    actual = _language_model_step(
        restored, restored_optimizer, actual_trace, next_inputs, next_targets
    )
    assert torch.equal(actual[0], expected[0])
    for actual_indices, expected_indices in zip(actual[1], expected[1], strict=True):
        assert torch.equal(actual_indices, expected_indices)
    assert actual[2].keys() == expected[2].keys()
    for name, expected_gradient in expected[2].items():
        assert torch.equal(actual[2][name], expected_gradient)
    for name, expected_value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], expected_value)
