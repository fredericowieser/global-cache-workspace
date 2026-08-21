"""Matched distinct-pair timing for the benchmark-only pair-wave executor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Callable

import torch

from gcw.config import TrainConfig, load_train_config
from gcw.data.training import TrainingBatch, build_training_stream
from gcw.performance.adjacent_pairs import PAIRS_PER_STEP
from gcw.performance.pair_waves import PairWaveRuntime
from gcw.training.model_setup import _apply_trainable_policy
from gcw.training.objectives import model_loss as _model_loss
from scripts.benchmarks.harness import _build_configured_model, device_identity


def _validate(config: TrainConfig) -> None:
    if (
        config.distributed != "none"
        or config.batch_size != 2
        or config.global_batch_rows != 64
    ):
        raise ValueError(
            "pair-wave timing requires single-device fixed batch-2 and exactly 64 global rows"
        )
    if config.model.family != "qwen3" or config.model.variant not in {
        "bt",
        "bgt",
        "longshot",
        "icarus",
    }:
        raise ValueError(
            "pair-wave timing supports Qwen3 BT/BGT/Longshot/Icarus"
        )
    if (
        config.performance.adaptive_batching.enabled
        or config.logical_pair_grouping
    ):
        raise ValueError("pair-wave timing rejects grouped execution")
    if config.model.diffusion_processor is not None:
        raise ValueError("pair-wave timing does not admit diffusion RNG")


def _reserve(
    config: TrainConfig, device: torch.device
) -> tuple[TrainingBatch, ...]:
    stream = build_training_stream(config, device, rank=0, world_size=1)
    try:
        pairs = tuple(next(stream) for _ in range(PAIRS_PER_STEP))
    finally:
        stream.close()
    if any(pair.x.shape[0] != 2 for pair in pairs):
        raise RuntimeError("reserved pair changed physical batch shape")
    return pairs


def _pair_loss(
    model: torch.nn.Module, config: TrainConfig, pair: TrainingBatch
) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.bf16):
        return _model_loss(
            model,
            pair.x,
            pair.y,
            loss_backend=config.loss_backend,
            attention_mask=pair.attention_mask,
            activation_checkpoint_windows=config.activation_checkpoint_windows,
            activation_checkpoint_min_padded_sequence_length=(
                config.activation_checkpoint_min_padded_sequence_length
            ),
        )


def _serial(
    model: torch.nn.Module,
    config: TrainConfig,
    pairs: tuple[TrainingBatch, ...],
) -> tuple[torch.Tensor, ...]:
    losses = []
    for pair in pairs:
        loss = _pair_loss(model, config, pair)
        losses.append(loss.detach())
        (loss / PAIRS_PER_STEP).backward()
    return tuple(losses)


def _gradient_clones(
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor, ...]:
    if any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("pair-wave timing found a missing gradient")
    return tuple(
        parameter.grad.detach().clone()
        for parameter in parameters
        if parameter.grad is not None
    )


def _measure(
    operation: Callable[[], None], *, warmup: int, steps: int
) -> dict[str, object]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "pair-wave timing requires exactly one visible CUDA GPU"
        )
    config = load_train_config(args.train_config)
    _validate(config)
    device = torch.device("cuda")
    torch.manual_seed(config.seed)
    model = _build_configured_model(config, device)
    _apply_trainable_policy(model, config.trainable)
    model.train()
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    pairs = _reserve(config, device)
    runtime = PairWaveRuntime(device, args.wave_size)

    model.zero_grad(set_to_none=True)
    serial_losses = _serial(model, config, pairs)
    serial_gradients = _gradient_clones(parameters)
    model.zero_grad(set_to_none=True)
    wave_result = runtime.execute(
        pairs,
        lambda _index, pair: _pair_loss(model, config, pair),
        parameters,
    )
    torch.cuda.synchronize()
    wave_gradients = _gradient_clones(parameters)
    torch.testing.assert_close(
        torch.stack(wave_result.pair_losses), torch.stack(serial_losses)
    )
    if len(serial_gradients) != len(wave_gradients) or any(
        not torch.equal(reference, candidate)
        for reference, candidate in zip(serial_gradients, wave_gradients)
    ):
        raise RuntimeError(
            "pair-wave gradients differ bitwise from serial execution"
        )

    def serial_step() -> None:
        model.zero_grad(set_to_none=True)
        _serial(model, config, pairs)
        torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)

    def wave_step() -> None:
        model.zero_grad(set_to_none=True)
        runtime.execute(
            pairs,
            lambda _index, pair: _pair_loss(model, config, pair),
            parameters,
        )
        torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)

    reference = _measure(serial_step, warmup=args.warmup, steps=args.steps)
    candidate = _measure(wave_step, warmup=args.warmup, steps=args.steps)
    runtime.close()
    reference_median = float(reference["median_seconds"])
    candidate_median = float(candidate["median_seconds"])
    return {
        "schema": "gcw.pair_wave_timing.v1",
        "device": device_identity(device),
        "config": str(args.train_config.resolve()),
        "variant": config.model.variant,
        "wave_size": args.wave_size,
        "pairs": PAIRS_PER_STEP,
        "physical_rows": 2,
        "pair_weight": "1/32",
        "correctness": "bitwise_pair_losses_and_raw_gradients",
        "reference": reference,
        "candidate": candidate,
        "speedup": reference_median / candidate_median,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--wave-size", type=int, choices=(2, 4), required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps({"output": str(args.output), "speedup": result["speedup"]})
    )


if __name__ == "__main__":
    main()
