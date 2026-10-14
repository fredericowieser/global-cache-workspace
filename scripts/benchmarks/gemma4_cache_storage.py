"""Small synthetic cache-storage measurements; no checkpoint or dataset loading."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models import build_model
from gcw.models.cache_processors.ops import (
    gather_per_cache_head_kv,
    scatter_latest,
    select_per_cache_head_kv_norm_indices_from_scores,
)
from gcw.models.cache_processors.state import CacheState
from gcw.performance.longshot_scores import (
    chunked_fp32_squared_l2,
    native_cache_squared_l2,
)


def measure(operation, device, warmup, repeats):
    for _ in range(warmup):
        operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(1000 * (time.perf_counter() - start))
    return {
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "extra_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) - baseline
            if device.type == "cuda"
            else None
        ),
    }


@torch.no_grad()
def run_case(device, length, subset, warmup, repeats):
    config = ModelConfig(
        family="gemma4",
        variant="longshot",
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        global_head_dim=64,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        sequence_len=32768,
        sliding_window=128,
        layer_types=["sliding_attention", "full_attention"] * 2,
        processor_layer_indices=subset,
        cache_processor=CacheProcessorConfig(
            process_every=8,
            topk=8,
            topk_heads=2,
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            encoder_method="query_attention",
            score="kv_norm",
            head_score_normalization="rms",
            unrope_keys=False,
        ),
    )
    model = build_model(config).to(device=device, dtype=torch.bfloat16).eval()
    all_keys = [
        torch.randn(
            2,
            config.kv_heads_for_layer(i),
            length,
            config.head_dim_for_layer(i),
            device=device,
            dtype=torch.bfloat16,
        )
        for i in range(4)
    ]
    all_values = [torch.randn_like(k) for k in all_keys]
    selected = config.cache_layer_indices
    scores = torch.zeros(2, length, device=device)
    cache = CacheState(
        [all_keys[i] for i in selected],
        [all_values[i] for i in selected],
        scores,
        attention_states={
            i: (all_keys[i], all_values[i])
            for i in range(4)
            if i not in selected
        },
    )
    static = model.backbone.allocate_generation_cache(2, length + 1, device)
    for i in range(static.num_layers):
        static.write_layer(i, 0, *cache.layer(i))
    for i, state in static.attention_states.items():
        state.append(0, all_keys[i], all_values[i])
    static.write_scores(0, scores)
    new_keys = [k[:, :, -1:] for k in all_keys]
    new_values = [v[:, :, -1:] for v in all_values]

    def dynamic_append():
        return [
            (torch.cat((k, nk), 2), torch.cat((v, nv), 2))
            for k, v, nk, nv in zip(
                all_keys, all_values, new_keys, new_values, strict=True
            )
        ]

    def reset_static():
        static.length = length
        for state in static.attention_states.values():
            state.length = length

    def static_append():
        reset_static()
        for i, layer in enumerate(selected):
            static.write_layer(i, length, new_keys[layer], new_values[layer])
        for i, state in static.attention_states.items():
            state.append(length, new_keys[i], new_values[i])
        static.write_scores(length, scores[:, :1])

    # A single native shape isolates score/gather costs from processor adapters.
    group = CacheState([cache.keys[0]], [cache.values[0]], scores)
    all_indices = group.all_indices()[:, None].expand(
        -1, group.keys[0].shape[1], -1
    )
    indices = select_per_cache_head_kv_norm_indices_from_scores(
        native_cache_squared_l2(group), 8, 8
    )

    def reference_gather(index):
        k, v = torch.stack(group.keys), torch.stack(group.values)
        shape = (k.shape[1], k.shape[0] * k.shape[2], k.shape[3], k.shape[4])
        k = k.permute(1, 0, 2, 3, 4).reshape(shape)
        v = v.permute(1, 0, 2, 3, 4).reshape(shape)
        expanded = index[..., None].expand(-1, -1, -1, k.shape[-1])
        return torch.cat((k.gather(2, expanded), v.gather(2, expanded)), -1)

    latest = torch.full((2, 1), length - 1, device=device, dtype=torch.long)
    update_keys = [k[:, :, -1:].clone() for k in cache.keys]
    update_values = [v[:, :, -1:].clone() for v in cache.values]

    def dynamic_writeback():
        current = CacheState(list(cache.keys), list(cache.values), scores)
        return scatter_latest(current, latest, update_keys, update_values)

    def static_writeback():
        reset_static()
        return scatter_latest(static, latest, update_keys, update_values)

    token = torch.ones(2, 1, device=device, dtype=torch.long)

    def dynamic_step():
        current = CacheState(
            list(cache.keys),
            list(cache.values),
            scores,
            attention_states=cache.attention_states,
        )
        current = model.cache_processor(current)
        return model.backbone.forward_chunk(token, length, current)

    def static_step():
        reset_static()
        for i, (k, v) in enumerate(
            zip(update_keys, update_values, strict=True)
        ):
            static.scatter_layer(i, latest, k, v)
        model.cache_processor(static)
        return model.backbone.forward_chunk_static(token, length, static)

    operations = {
        "append_dynamic": dynamic_append,
        "append_static": static_append,
        "score_full_merge": lambda: chunked_fp32_squared_l2(
            reference_gather(all_indices)
        ),
        "score_native": lambda: native_cache_squared_l2(group),
        "gather_full_pack": lambda: reference_gather(indices),
        "gather_native": lambda: gather_per_cache_head_kv(group, indices),
        "writeback_dynamic": dynamic_writeback,
        "writeback_static": static_writeback,
        "processor_decode_dynamic": dynamic_step,
        "processor_decode_static": static_step,
    }
    results = {
        name: measure(operation, device, warmup, repeats)
        for name, operation in operations.items()
    }
    dynamic_bytes = sum(
        t.numel() * t.element_size() for t in all_keys + all_values
    )
    static_bytes = sum(
        t.numel() * t.element_size() for t in static.keys + static.values
    )
    static_bytes += sum(
        t.numel() * t.element_size()
        for state in static.attention_states.values()
        for t in (state.keys, state.values)
    )
    return {
        "length": length,
        "selected_layers": selected,
        "dynamic_kv_bytes": dynamic_bytes,
        "static_kv_bytes": static_bytes,
        "operations": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[1024, 4096, 16384]
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.repeats < 1 or args.warmup < 0 or any(n < 16 for n in args.lengths):
        parser.error(
            "require positive repeats, non-negative warmup, and lengths >= 16"
        )
    torch.manual_seed(93)
    results = []
    for subset in (None, [1, 3]):
        for length in args.lengths:
            results.append(
                run_case(device, length, subset, args.warmup, args.repeats)
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    payload = {
        "scope": "synthetic storage diagnostic, batch 2, BF16, four layers; not 31B throughput evidence",
        "torch": torch.__version__,
        "device": str(device),
        "accelerator": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
