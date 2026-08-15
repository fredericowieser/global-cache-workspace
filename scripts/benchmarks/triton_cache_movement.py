"""Matched CUDA microbenchmark for the exact per-cache-head Triton gather."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Callable

import torch

from gcw.models.cache_processors.ops import gather_per_cache_head_kv as eager_gather
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.performance.triton import gather_per_cache_head_kv, require_triton_runtime
from scripts.benchmarks.harness import device_identity, timing_summary


def _indices(batch: int, cache_heads: int, selected: int, history: int, device: torch.device) -> torch.Tensor:
    if selected <= 0 or selected > history:
        raise ValueError("selected must be in [1, history]")
    base = torch.linspace(0, history - 1, selected, device=device).round().long().unique()
    if base.numel() != selected:
        base = torch.arange(selected, device=device) + history - selected
    return base.view(1, 1, selected).expand(batch, cache_heads, -1).contiguous()


def _time_operation(operation: Callable[[], None], *, warmup: int, steps: int) -> dict[str, object]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    wall_samples: list[float] = []
    device_samples: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(steps):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        start_event.record()
        operation()
        end_event.record()
        end_event.synchronize()
        wall_samples.append(time.perf_counter() - start)
        device_samples.append(start_event.elapsed_time(end_event) / 1000.0)
    return {
        "wall": {**timing_summary(wall_samples), "samples": wall_samples},
        "device": {**timing_summary(device_samples), "samples": device_samples},
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Triton cache benchmark requires exactly one visible CUDA GPU")
    device = torch.device("cuda")
    runtime = require_triton_runtime(device)
    expected_name = {"h100": "H100", "h200": "H200"}[args.expected_accelerator]
    if expected_name not in runtime.device_name:
        raise RuntimeError(f"expected {args.expected_accelerator.upper()} but found {runtime.device_name}")

    torch.manual_seed(args.seed)
    shape = (args.layers, 2, args.heads, args.history, args.head_dim)
    reference_keys = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
    reference_values = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
    candidate_keys = reference_keys.detach().clone().requires_grad_(True)
    candidate_values = reference_values.detach().clone().requires_grad_(True)
    indices = _indices(2, args.layers * args.heads, args.selected, args.history, device)
    scores = torch.empty(2, 0, device=device)

    reference_output = eager_gather(WindowedCache(reference_keys, reference_values, scores), indices)
    candidate_output = gather_per_cache_head_kv(candidate_keys, candidate_values, indices, validate_indices=True)
    if not torch.equal(reference_output.view(torch.int16), candidate_output.view(torch.int16)):
        raise RuntimeError("Triton gather output differs from the eager raw payload")
    upstream = torch.randn_like(reference_output)
    reference_output.backward(upstream)
    candidate_output.backward(upstream)
    if reference_keys.grad is None or candidate_keys.grad is None:
        raise RuntimeError("Triton gather did not produce key gradients")
    if reference_values.grad is None or candidate_values.grad is None:
        raise RuntimeError("Triton gather did not produce value gradients")
    if not torch.equal(reference_keys.grad.view(torch.int16), candidate_keys.grad.view(torch.int16)):
        raise RuntimeError("Triton key gradients differ from eager")
    if not torch.equal(reference_values.grad.view(torch.int16), candidate_values.grad.view(torch.int16)):
        raise RuntimeError("Triton value gradients differ from eager")

    def reference_step() -> None:
        reference_keys.grad = None
        reference_values.grad = None
        output = eager_gather(WindowedCache(reference_keys, reference_values, scores), indices)
        output.backward(upstream)

    def candidate_step() -> None:
        candidate_keys.grad = None
        candidate_values.grad = None
        output = gather_per_cache_head_kv(
            candidate_keys,
            candidate_values,
            indices,
            validate_indices=False,
        )
        output.backward(upstream)

    reference = _time_operation(reference_step, warmup=args.warmup, steps=args.steps)
    candidate = _time_operation(candidate_step, warmup=args.warmup, steps=args.steps)
    reference_median = float(reference["wall"]["median_seconds"])  # type: ignore[index]
    candidate_median = float(candidate["wall"]["median_seconds"])  # type: ignore[index]
    return {
        "schema": "gcw.triton_cache_movement_benchmark.v1",
        "device": device_identity(device),
        "runtime": {
            "torch": runtime.torch_version,
            "cuda": runtime.cuda_runtime,
            "triton": runtime.triton_version,
            "compute_capability": list(runtime.compute_capability),
        },
        "shape": {
            "layers": args.layers,
            "batch": 2,
            "heads": args.heads,
            "history": args.history,
            "head_dim": args.head_dim,
            "selected": args.selected,
        },
        "correctness": "bitwise_forward_and_backward",
        "reference": reference,
        "candidate": candidate,
        "wall_speedup": reference_median / candidate_median,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-accelerator", choices=("h100", "h200"), required=True)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--history", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--selected", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"wall_speedup": payload["wall_speedup"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
