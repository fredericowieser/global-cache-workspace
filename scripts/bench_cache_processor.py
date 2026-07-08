#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch

from gcw.config import CacheProcessorConfig, ModelConfig, load_train_config
from gcw.models import build_model
from gcw.models.cache_processor import _merge_kv, _split_kv


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    device = torch.device(args.device)
    for variant in args.variants:
        cfg = model_config(args, variant)
        result = bench_variant(cfg, args.batch_size, args.seq_len, args.warmup, args.steps, device, args.reference_loop)
        mode = "ref" if args.reference_loop and variant in {"bt", "bgt"} else "fast"
        print(
            f"{variant:>3} {mode:>4} "
            f"step={result['step_seconds']:.4f}s "
            f"tokens/s={result['tokens_per_second']:.1f} "
            f"cp={result['processor_seconds']:.4f}s "
            f"cp_calls={int(result['processor_calls'])} "
            f"peak_vram={result['peak_vram_mib']:.0f}MiB"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GCW TF/BT/BGT forward+backward throughput.")
    parser.add_argument("--variants", nargs="+", default=["tf", "bt", "bgt"], choices=["tf", "bt", "bgt"])
    parser.add_argument("--train-config", default=None, help="Optional train config to use as the model template.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--reference-loop", action="store_true", help="Benchmark the pre-optimization CP loop.")
    return parser.parse_args()


def model_config(args: argparse.Namespace, variant: str) -> ModelConfig:
    if args.train_config:
        base = load_train_config(args.train_config).model
        cp = base.cache_processor
        if variant == "tf":
            return replace(base, variant="tf", cache_processor=None, sequence_len=args.seq_len)
        if cp is None:
            cp = CacheProcessorConfig()
        return replace(base, variant=variant, cache_processor=cp, sequence_len=args.seq_len)
    cp = None
    if variant in {"bt", "bgt"}:
        cp = CacheProcessorConfig(process_every=16, topk=8, hidden_size=64, intermediate_size=256, num_heads=1, num_kv_heads=1)
    return ModelConfig(
        variant=variant,
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=768,
        num_layers=6,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        sequence_len=args.seq_len,
        tie_embeddings=True,
        cache_processor=cp,
    )


def bench_variant(
    cfg: ModelConfig,
    batch_size: int,
    seq_len: int,
    warmup: int,
    steps: int,
    device: torch.device,
    reference_loop: bool,
) -> dict[str, float]:
    torch.manual_seed(123)
    model = build_model(cfg).to(device)
    model.init_weights()
    if reference_loop and cfg.variant in {"bt", "bgt"}:
        use_reference_loop(model)
    timer = ProcessorTimer(model, device)
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    y = torch.roll(x, shifts=-1, dims=1)

    for _ in range(warmup):
        run_step(model, x, y, device)
    timer.reset()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(steps):
        run_step(model, x, y, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    tokens = batch_size * seq_len * steps
    return {
        "step_seconds": elapsed / steps,
        "tokens_per_second": tokens / max(elapsed, 1e-12),
        "processor_seconds": timer.seconds / steps,
        "processor_calls": timer.calls / max(steps, 1),
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else 0.0,
    }


def run_step(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> None:
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        loss = model(x, y)
    loss.backward()


def use_reference_loop(model: torch.nn.Module) -> None:
    processor = model.cache_processor

    if model.config.variant == "bt":
        def forward(cache, *, select_all: bool = False):
            indices = cache.all_indices() if select_all else cache.select_indices(processor.config.process_every, processor.config.topk)
            for layer_idx, block in enumerate(processor.blocks):
                keys, values = cache.gather_layer(layer_idx, indices)
                updated = block(_merge_kv(keys, values), indices)
                new_keys, new_values = _split_kv(updated, keys.shape[1], keys.shape[3])
                cache.scatter_layer(layer_idx, indices, new_keys, new_values)
            return cache
    else:
        def forward(cache, *, select_all: bool = False):
            indices = cache.all_indices() if select_all else cache.select_indices(processor.config.process_every, processor.config.topk)
            parts = []
            shapes = []
            for layer_idx in range(processor.num_layers):
                keys, values = cache.gather_layer(layer_idx, indices)
                shapes.append((keys.shape[1], keys.shape[3]))
                parts.append(_merge_kv(keys, values))
            updated = processor.block(torch.cat(parts, dim=-1), indices)
            for layer_idx, part in enumerate(torch.split(updated, [p.shape[-1] for p in parts], dim=-1)):
                keys, values = _split_kv(part, *shapes[layer_idx])
                cache.scatter_layer(layer_idx, indices, keys, values)
            return cache

    processor.forward = forward


class ProcessorTimer:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.seconds = 0.0
        self.calls = 0
        self.device = device
        processor = getattr(model, "cache_processor", None)
        if processor is None:
            return
        original = processor.forward

        def timed_forward(*args, **kwargs):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            out = original(*args, **kwargs)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.seconds += time.perf_counter() - start
            self.calls += 1
            return out

        processor.forward = timed_forward

    def reset(self) -> None:
        self.seconds = 0.0
        self.calls = 0


if __name__ == "__main__":
    main()
