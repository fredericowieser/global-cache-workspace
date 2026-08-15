"""Matched fixed-shape timing for fused cache-processor pointwise kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

from gcw.performance.triton import fused_processor_silu_mul, require_triton_runtime
from scripts.benchmarks.harness import device_identity


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(torch.finfo(torch.float32).tiny)
    return float((torch.linalg.vector_norm((actual - expected).float()) / denominator).cpu())


def _measure(operation, *, warmup: int, steps: int) -> dict[str, object]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(steps):
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) / 1000.0)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Triton processor benchmark requires one visible CUDA GPU")
    runtime = require_triton_runtime(torch.device("cuda"))
    expected = {"h100": "H100", "h200": "H200"}[args.expected_accelerator]
    if expected not in runtime.device_name:
        raise RuntimeError(f"expected {expected} but found {runtime.device_name}")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    shape = (args.layers, 2, args.tokens, args.intermediate)
    reference_gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator).requires_grad_(True)
    reference_up = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator).requires_grad_(True)
    candidate_gate = reference_gate.detach().clone().requires_grad_(True)
    candidate_up = reference_up.detach().clone().requires_grad_(True)
    upstream = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    reference_output = F.silu(reference_gate) * reference_up
    candidate_output = fused_processor_silu_mul(candidate_gate, candidate_up)
    reference_output.backward(upstream)
    candidate_output.backward(upstream)
    assert reference_gate.grad is not None and candidate_gate.grad is not None
    assert reference_up.grad is not None and candidate_up.grad is not None
    metrics = {
        "output_relative_l2": _relative_l2(candidate_output, reference_output),
        "gate_gradient_relative_l2": _relative_l2(candidate_gate.grad, reference_gate.grad),
        "up_gradient_relative_l2": _relative_l2(candidate_up.grad, reference_up.grad),
    }
    if max(metrics.values()) > 5e-4:
        raise RuntimeError(f"fused processor pointwise numerical gate failed: {metrics}")

    def reference_step() -> None:
        reference_gate.grad = None
        reference_up.grad = None
        (F.silu(reference_gate) * reference_up).backward(upstream)

    def candidate_step() -> None:
        candidate_gate.grad = None
        candidate_up.grad = None
        fused_processor_silu_mul(candidate_gate, candidate_up).backward(upstream)

    reference = _measure(reference_step, warmup=args.warmup, steps=args.steps)
    candidate = _measure(candidate_step, warmup=args.warmup, steps=args.steps)
    baseline = float(reference["median_seconds"])
    optimized = float(candidate["median_seconds"])
    return {
        "schema": "gcw.triton_processor_pointwise_benchmark.v1",
        "device": device_identity(torch.device("cuda")),
        "runtime": {
            "torch": runtime.torch_version,
            "cuda": runtime.cuda_runtime,
            "triton": runtime.triton_version,
        },
        "shape": {
            "layers": args.layers,
            "rows": 2,
            "tokens": args.tokens,
            "intermediate": args.intermediate,
        },
        "numerics": metrics,
        "reference": reference,
        "candidate": candidate,
        "speedup": baseline / optimized,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-accelerator", choices=("h100", "h200"), required=True)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--intermediate", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "speedup": result["speedup"]}))


if __name__ == "__main__":
    main()
