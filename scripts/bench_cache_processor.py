#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
import types
from dataclasses import replace
from pathlib import Path

import torch

from gcw.compile import maybe_compile
from gcw.config import CacheProcessorConfig, ModelConfig, load_train_config
from gcw.models import build_model
from gcw.models.cache_processor import _merge_kv, _split_kv
from gcw.models.cache_processor_ops import (
    merge_global_values,
    merge_layer_values,
    split_global_values,
    split_layer_values,
)
from gcw.models.cache_state import CacheState


def main() -> None:
    args = parse_args()
    if args.compile is not None:
        os.environ["GCW_COMPILE"] = args.compile
    if args.strict_flash:
        os.environ["GCW_STRICT_FLASH"] = "1"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    device = torch.device(args.device)
    results = []
    for case in benchmark_cases(args):
        result = bench_case(case, args, device)
        results.append(result)
        print_result(result)
        if args.profile:
            profile_case(case, args, device)
    print_ratios(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GCW TF/BT/BGT/Icarus/Longshot forward+backward throughput."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["tf", "bt", "bgt", "icarus", "longshot"],
        choices=["tf", "bt", "bgt", "icarus", "longshot"],
    )
    parser.add_argument(
        "--train-config",
        default=None,
        help="Use one train config as a model template for --variants.",
    )
    parser.add_argument(
        "--train-configs",
        nargs="*",
        default=None,
        help="Benchmark these exact train configs.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--reference-loop",
        action="store_true",
        help="Benchmark the pre-optimization CP/list loop.",
    )
    parser.add_argument(
        "--strict-flash",
        action="store_true",
        help="Fail if CUDA SDPA cannot use FlashAttention.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print a one-step profiler table after each result.",
    )
    parser.add_argument("--profile-rows", type=int, default=25)
    parser.add_argument(
        "--compile", choices=["0", "default", "reduce-overhead"], default=None
    )
    return parser.parse_args()


def benchmark_cases(args: argparse.Namespace) -> list[dict]:
    if args.train_configs:
        cases = []
        for path in args.train_configs:
            train = load_train_config(path)
            seq_len = args.seq_len or train.sequence_len
            cases.append(
                {
                    "name": Path(path).stem,
                    "config": replace(train.model, sequence_len=seq_len),
                    "batch_size": args.batch_size or train.batch_size,
                    "seq_len": seq_len,
                }
            )
        return cases

    return [
        {
            "name": variant,
            "config": model_config(args, variant),
            "batch_size": args.batch_size or template_batch_size(args),
            "seq_len": args.seq_len or template_seq_len(args),
        }
        for variant in args.variants
    ]


def template_batch_size(args: argparse.Namespace) -> int:
    if args.train_config:
        return load_train_config(args.train_config).batch_size
    return 2


def template_seq_len(args: argparse.Namespace) -> int:
    if args.train_config:
        return load_train_config(args.train_config).sequence_len
    return 128


def model_config(args: argparse.Namespace, variant: str) -> ModelConfig:
    seq_len = args.seq_len or template_seq_len(args)
    if args.train_config:
        base = load_train_config(args.train_config).model
        cp = base.cache_processor
        if variant == "tf":
            return replace(
                base, variant="tf", cache_processor=None, sequence_len=seq_len
            )
        if cp is None:
            cp = CacheProcessorConfig()
        return replace(
            base, variant=variant, cache_processor=cp, sequence_len=seq_len
        )
    cp = None
    if variant in {"bt", "bgt", "icarus", "longshot"}:
        cp = CacheProcessorConfig(
            process_every=16,
            topk=8,
            hidden_size=64,
            intermediate_size=256,
            num_heads=1,
            num_kv_heads=1,
        )
    return ModelConfig(
        variant=variant,
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=768,
        num_layers=6,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        sequence_len=seq_len,
        tie_embeddings=True,
        cache_processor=cp,
    )


def bench_case(
    case: dict, args: argparse.Namespace, device: torch.device
) -> dict[str, float | str]:
    cfg = case["config"]
    batch_size = case["batch_size"]
    seq_len = case["seq_len"]
    torch.manual_seed(123)
    model = build_model(cfg).to(device)
    model.init_weights()
    if args.reference_loop and cfg.variant in {"bt", "bgt"}:
        use_reference_loop(model)
    model = maybe_compile(model)
    timer = SectionTimer(device)
    if not compile_enabled():
        patch_timers(model, timer)
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    y = torch.roll(x, shifts=-1, dims=1)

    for _ in range(args.warmup):
        run_step(model, x, y, device)
    timer.reset()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        run_step(model, x, y, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    tokens = batch_size * seq_len * args.steps
    timings = timer.seconds_by_label()
    return {
        "name": case["name"],
        "variant": cfg.variant,
        "mode": (
            "ref"
            if args.reference_loop and cfg.variant in {"bt", "bgt"}
            else "fast"
        ),
        "step_seconds": elapsed / args.steps,
        "tokens_per_second": tokens / max(elapsed, 1e-12),
        "processor_seconds": timings.get("processor", 0.0) / args.steps,
        "chunk_seconds": timings.get("chunk", 0.0) / args.steps,
        "loss_seconds": timings.get("loss", 0.0) / args.steps,
        "processor_calls": timer.calls.get("processor", 0) / max(args.steps, 1),
        "chunk_calls": timer.calls.get("chunk", 0) / max(args.steps, 1),
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated(device) / 1024 / 1024
            if device.type == "cuda"
            else 0.0
        ),
    }


def patch_timers(model: torch.nn.Module, timer: "SectionTimer") -> None:
    backbone = model.backbone if hasattr(model, "backbone") else model
    timer.patch(backbone, "forward_chunk", "chunk")
    timer.patch(backbone, "loss_from_hidden", "loss")
    processor = getattr(model, "cache_processor", None)
    if processor is not None:
        timer.patch(processor, "forward", "processor")


def compile_enabled() -> bool:
    return os.environ.get("GCW_COMPILE", "0").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "none",
        "off",
    }


def run_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> None:
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        loss = model(x, y)
    loss.backward()


def profile_case(
    case: dict, args: argparse.Namespace, device: torch.device
) -> None:
    if device.type != "cuda":
        print("Profiler is only enabled for CUDA runs.")
        return
    cfg = case["config"]
    torch.manual_seed(123)
    model = build_model(cfg).to(device)
    model.init_weights()
    if args.reference_loop and cfg.variant in {"bt", "bgt"}:
        use_reference_loop(model)
    model = maybe_compile(model)
    x = torch.randint(
        0, cfg.vocab_size, (case["batch_size"], case["seq_len"]), device=device
    )
    y = torch.roll(x, shifts=-1, dims=1)
    for _ in range(max(1, args.warmup)):
        run_step(model, x, y, device)
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=False,
    ) as prof:
        run_step(model, x, y, device)
    torch.cuda.synchronize(device)
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=args.profile_rows
        )
    )


def use_reference_loop(model: torch.nn.Module) -> None:
    processor = model.cache_processor

    if model.config.variant == "bt":

        def forward(cache, *, select_all: bool = False):
            indices = (
                cache.all_indices()
                if select_all
                else cache.select_indices(
                    processor.config.process_every, processor.config.topk
                )
            )
            if processor.config.update_stream == "v":
                values = torch.stack(
                    [
                        cache.gather_layer(layer_idx, indices)[1]
                        for layer_idx in range(len(processor.blocks))
                    ],
                    dim=0,
                )
                updated = run_reference_layer_processors(
                    processor.blocks, merge_layer_values(values), indices
                )
                new_values = split_layer_values(
                    updated, values.shape[2], values.shape[4]
                )
                scatter_reference_values(cache, indices, new_values)
                return cache
            for layer_idx, block in enumerate(processor.blocks):
                keys, values = cache.gather_layer(layer_idx, indices)
                updated = block(_merge_kv(keys, values), indices)
                new_keys, new_values = _split_kv(
                    updated, keys.shape[1], keys.shape[3]
                )
                cache.scatter_layer(layer_idx, indices, new_keys, new_values)
            return cache

    else:

        def forward(cache, *, select_all: bool = False):
            indices = (
                cache.all_indices()
                if select_all
                else cache.select_indices(
                    processor.config.process_every, processor.config.topk
                )
            )
            if processor.config.update_stream == "v":
                values = torch.stack(
                    [
                        cache.gather_layer(layer_idx, indices)[1]
                        for layer_idx in range(processor.num_layers)
                    ],
                    dim=0,
                )
                updated = processor.block(merge_global_values(values), indices)
                new_values = split_global_values(
                    updated,
                    processor.num_layers,
                    processor.num_kv_heads,
                    processor.head_dim,
                )
                scatter_reference_values(cache, indices, new_values)
                return cache
            parts = []
            shapes = []
            for layer_idx in range(processor.num_layers):
                keys, values = cache.gather_layer(layer_idx, indices)
                shapes.append((keys.shape[1], keys.shape[3]))
                parts.append(_merge_kv(keys, values))
            updated = processor.block(torch.cat(parts, dim=-1), indices)
            for layer_idx, part in enumerate(
                torch.split(updated, [p.shape[-1] for p in parts], dim=-1)
            ):
                keys, values = _split_kv(part, *shapes[layer_idx])
                cache.scatter_layer(layer_idx, indices, keys, values)
            return cache

    processor.forward = forward
    if hasattr(model, "backbone"):
        model.backbone.forward_windowed_loss = types.MethodType(
            reference_windowed_loss, model.backbone
        )


def run_reference_layer_processors(
    blocks, x: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    updated = []
    for layer_idx, block in enumerate(blocks):
        updated.append(block(x[layer_idx], indices))
    return torch.stack(updated, dim=0)


def scatter_reference_values(
    cache: CacheState, indices: torch.Tensor, values: torch.Tensor
) -> None:
    for layer_idx, new_values in enumerate(values):
        old_values = cache.values[layer_idx]
        index = indices[:, None, :, None].expand(
            -1, old_values.shape[1], -1, old_values.shape[3]
        )
        cache.values[layer_idx] = old_values.scatter(
            2, index, new_values.to(old_values.dtype)
        )


def reference_windowed_loss(
    self,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    processor,
    process_every: int,
    loss_reduction: str = "mean",
) -> torch.Tensor:
    cache = CacheState(
        [], [], input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32)
    )
    loss_sum = input_ids.new_zeros((), dtype=torch.float32)
    token_count = input_ids.new_zeros((), dtype=torch.float32)
    for start in range(0, input_ids.shape[1], process_every):
        if start > 0:
            cache = processor(cache)
        end = start + process_every
        chunk_logits, cache = self.forward_chunk(
            input_ids[:, start:end], start, cache, logits=True
        )
        chunk_targets = targets[:, start:end].reshape(-1).clone()
        valid = chunk_targets >= 0
        chunk_targets[~valid] = -100
        loss_sum = loss_sum + torch.nn.functional.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.shape[-1]),
            chunk_targets,
            ignore_index=-100,
            reduction="sum",
        )
        token_count = token_count + valid.sum().to(torch.float32)
    if loss_reduction == "sum":
        return loss_sum
    if loss_reduction == "none":
        raise ValueError("windowed loss does not support reduction='none'")
    return loss_sum / token_count.clamp_min(1.0)


class SectionTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.calls: dict[str, int] = {}
        self.cpu_seconds: dict[str, float] = {}
        self.cuda_events: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = {}

    def patch(self, obj: object, method_name: str, label: str) -> None:
        original = getattr(obj, method_name)

        def timed(*args, **kwargs):
            self.calls[label] = self.calls.get(label, 0) + 1
            if self.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = original(*args, **kwargs)
                end.record()
                self.cuda_events.setdefault(label, []).append((start, end))
                return out
            start_time = time.perf_counter()
            out = original(*args, **kwargs)
            self.cpu_seconds[label] = (
                self.cpu_seconds.get(label, 0.0)
                + time.perf_counter()
                - start_time
            )
            return out

        setattr(obj, method_name, timed)

    def reset(self) -> None:
        self.calls.clear()
        self.cpu_seconds.clear()
        self.cuda_events.clear()

    def seconds_by_label(self) -> dict[str, float]:
        if self.device.type != "cuda":
            return dict(self.cpu_seconds)
        torch.cuda.synchronize(self.device)
        timings = {}
        for label, events in self.cuda_events.items():
            total_ms = 0.0
            for start, end in events:
                try:
                    total_ms += start.elapsed_time(end)
                except ValueError:
                    pass
            timings[label] = total_ms / 1000.0
        return timings


def print_result(result: dict[str, float | str]) -> None:
    non_cp = max(
        0.0, float(result["step_seconds"]) - float(result["processor_seconds"])
    )
    print(
        f"{result['name']:>36} {result['variant']:>3} {result['mode']:>4} "
        f"step={float(result['step_seconds']):.4f}s "
        f"tokens/s={float(result['tokens_per_second']):.1f} "
        f"cp={float(result['processor_seconds']):.4f}s "
        f"chunk={float(result['chunk_seconds']):.4f}s "
        f"loss={float(result['loss_seconds']):.4f}s "
        f"non_cp={non_cp:.4f}s "
        f"cp_calls={float(result['processor_calls']):.1f} "
        f"chunk_calls={float(result['chunk_calls']):.1f} "
        f"peak_vram={float(result['peak_vram_mib']):.0f}MiB"
    )


def print_ratios(results: list[dict[str, float | str]]) -> None:
    tf = next((r for r in results if r["variant"] == "tf"), None)
    if tf is None:
        return
    tf_step = float(tf["step_seconds"])
    print("Ratios vs TF:")
    for result in results:
        if result is tf:
            continue
        print(
            f"  {result['name']}: "
            f"step={float(result['step_seconds']) / tf_step:.2f}x "
            f"cp_only={float(result['processor_seconds']):.4f}s"
        )


if __name__ == "__main__":
    main()
