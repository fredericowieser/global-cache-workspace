"""Benchmark structured reference and fused Liger row-sum loss boundaries."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Callable

import torch
import torch.nn.functional as F

from gcw.performance.config import ReferenceLossConfig
from gcw.performance.losses import bounded_reference_loss, liger_fused_linear_cross_entropy_row_sum

LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _reference(weight: torch.Tensor, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return bounded_reference_loss(
        lambda values: F.linear(values, weight),
        hidden,
        targets,
        "row_sum",
        ReferenceLossConfig(mode="bounded", max_projected_positions=1024),
    )


def _liger(weight: torch.Tensor, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return liger_fused_linear_cross_entropy_row_sum(weight, hidden, targets)


def _run_once(
    fn: LossFn,
    weight: torch.Tensor,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    upstream: torch.Tensor,
) -> None:
    weight.grad = None
    hidden.grad = None
    fn(weight, hidden, targets).backward(upstream)


def _measure(
    fn: LossFn,
    weight: torch.Tensor,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    upstream: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    for _ in range(warmup):
        _run_once(fn, weight, hidden, targets, upstream)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        weight.grad = None
        hidden.grad = None
        start_event.record()
        fn(weight, hidden, targets).backward(upstream)
        end_event.record()
        end_event.synchronize()
        samples.append(float(start_event.elapsed_time(end_event)) / 1000.0)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("fused row-loss benchmark requires BF16 CUDA")
    torch.manual_seed(1129)
    weight = torch.randn(args.vocab, args.hidden, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    hidden = torch.randn(args.rows, args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    targets = torch.randint(0, args.vocab, (args.rows, args.tokens), device="cuda")
    targets[:, ::17] = -100
    upstream = torch.linspace(0.75, 1.25, args.rows, device="cuda")

    started = time.time()
    reference = _measure(
        _reference,
        weight,
        hidden,
        targets,
        upstream,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    fused = _measure(
        _liger,
        weight,
        hidden,
        targets,
        upstream,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    reference_median = float(reference["median_seconds"])
    fused_median = float(fused["median_seconds"])
    print(
        json.dumps(
            {
                "schema": "gcw.fused_row_loss_benchmark.v1",
                "device": torch.cuda.get_device_name(),
                "torch_version": torch.__version__,
                "shape": {"rows": args.rows, "tokens": args.tokens, "hidden": args.hidden, "vocab": args.vocab},
                "warmup": args.warmup,
                "iterations": args.iterations,
                "reference": reference,
                "fused": fused,
                "speedup": reference_median / fused_median,
                "peak_allocation_ratio": int(fused["peak_allocated_bytes"]) / int(reference["peak_allocated_bytes"]),
                "wall_seconds": time.time() - started,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
