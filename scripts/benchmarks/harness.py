from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any, Callable

import torch

from gcw.checkpoint import load_backbone_checkpoint
from gcw.config import TrainConfig, load_train_config, to_dict
from gcw.data.training import build_training_stream
from gcw.models import build_model
from gcw.models.cache_processors.ops import (
    gather_cache_layers,
    merge_global_kv,
    merge_global_values,
    merge_layer_kv,
    merge_layer_values,
    scatter_cache_layers,
    scatter_cache_values,
    split_global_kv,
    split_global_values,
    split_layer_kv,
    split_layer_values,
)
from gcw.models.cache_processors.state import CacheState
from gcw.performance.prepared_bt import run_batched_cache_processors
from gcw.performance.losses import loss_backend_metadata, require_loss_backend
from gcw.qualification.integrity import (
    integrity_document as _integrity_document,
    load_integrity_document,
    verify_integrity_document as _verify_integrity_document,
    write_integrity_document,
)
from gcw.provenance import (
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    find_repository_root,
    repository_source_identity,
    tensor_state_sha256,
)
from gcw.performance.whole_model import maybe_compile_whole_model
from gcw.training.model_setup import (
    _apply_trainable_policy,
    _load_initial_model,
)
from gcw.training.objectives import model_loss as _model_loss
from gcw.training.optimizer import (
    _build_optimizer,
    _clip_grad_norm,
    _weight_decay,
    _wsd_lr,
)

INTEGRITY_SCHEMA = "gcw.cache_processor_benchmark.integrity.v2"
EVIDENCE_KIND = "gcw.cache_processor_benchmark"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize empty timings")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be in [0, 1]")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def timing_summary(raw_step_seconds: list[float]) -> dict[str, float]:
    return {
        "median_seconds": percentile(raw_step_seconds, 0.5),
        "p10_seconds": percentile(raw_step_seconds, 0.1),
        "p90_seconds": percentile(raw_step_seconds, 0.9),
    }


def integrity_document(payload: dict[str, Any]) -> dict[str, Any]:
    return _integrity_document(
        payload, schema=INTEGRITY_SCHEMA, kind=EVIDENCE_KIND
    )


def verify_integrity_document(document: dict[str, Any]) -> dict[str, Any]:
    return _verify_integrity_document(
        document,
        expected_schema=INTEGRITY_SCHEMA,
        expected_kind=EVIDENCE_KIND,
    )


def write_integrity_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    document = integrity_document(payload)
    write_integrity_document(path, document, exclusive=True)
    return document


def load_integrity_json(path: Path) -> dict[str, Any]:
    return load_integrity_document(
        path,
        expected_schema=INTEGRITY_SCHEMA,
        expected_kind=EVIDENCE_KIND,
    )


def device_identity(device: torch.device) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "device": str(device),
        "device_type": device.type,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
        identity.update(
            {
                "device_index": index,
                "device_name": properties.name,
                "compute_capability": list(
                    torch.cuda.get_device_capability(index)
                ),
                "total_memory_bytes": properties.total_memory,
            }
        )
    else:
        identity.update(
            {"device_index": None, "device_name": platform.processor() or "cpu"}
        )
    return identity


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def _time_forward_operation(
    name: str,
    operation: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    steps: int,
    variant: str,
) -> dict[str, Any]:
    with torch.no_grad():
        for _ in range(warmup):
            operation()
    _synchronize(device)
    _reset_peak_memory(device)
    raw: list[float] = []
    with torch.no_grad():
        for _ in range(steps):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                operation()
                end.record()
                end.synchronize()
                raw.append(start.elapsed_time(end) / 1000.0)
            else:
                start_time = time.perf_counter()
                operation()
                raw.append(time.perf_counter() - start_time)
    return {
        "name": name,
        "kind": "cache_processor_micro_operation",
        "processor_variant": variant,
        "warmup": warmup,
        "measured_steps": steps,
        "raw_step_seconds": raw,
        **timing_summary(raw),
        "peak_memory_bytes": _peak_memory(device),
    }


def _build_configured_model(
    config: TrainConfig, device: torch.device
) -> torch.nn.Module:
    model = build_model(config.model).to(device)
    model.init_weights()
    if config.init_checkpoint:
        _load_initial_model(model, config.init_checkpoint)
    elif config.backbone_checkpoint:
        load_backbone_checkpoint(model, config.backbone_checkpoint)
    return model


def _cache_fixture(config: TrainConfig, device: torch.device) -> CacheState:
    processor = config.model.cache_processor
    if processor is None:
        raise ValueError(
            "cache-processor benchmarks require model.cache_processor"
        )
    cache_len = min(
        config.sequence_len,
        max(
            processor.process_every * 2,
            processor.process_every + processor.topk,
        ),
    )
    if cache_len <= processor.process_every:
        raise ValueError(
            "sequence_len must exceed cache_processor.process_every for a representative fixture"
        )
    num_heads, head_dim = config.model.stored_cache_shape
    shape = (config.batch_size, num_heads, cache_len, head_dim)
    keys = [
        torch.randn(shape, device=device)
        for _ in range(config.model.cache_num_layers)
    ]
    values = [
        torch.randn(shape, device=device)
        for _ in range(config.model.cache_num_layers)
    ]
    scores = torch.randn(
        config.batch_size, cache_len, device=device, dtype=torch.float32
    )
    return CacheState(keys, values, scores)


def benchmark_processor_operations(
    config: TrainConfig,
    device: torch.device,
    *,
    warmup: int,
    steps: int,
) -> list[dict[str, Any]]:
    variant = config.model.variant
    if variant not in {"bt", "bgt"}:
        raise ValueError(
            "maintained processor microbenchmarks support BT and BGT"
        )
    torch.manual_seed(config.seed)
    model = _build_configured_model(config, device)
    model.eval()
    processor = model.cache_processor
    cache = _cache_fixture(config, device)
    cp_config = config.model.cache_processor
    assert cp_config is not None

    def select() -> torch.Tensor:
        if variant == "bgt":
            return processor._select_indices(cache, False)
        return cache.select_indices(cp_config.process_every, cp_config.topk)

    indices = select()
    keys, values = gather_cache_layers(cache, indices)
    if cp_config.update_stream == "v":
        if variant == "bt":
            processor_input = merge_layer_values(values)
            transformed = run_batched_cache_processors(
                processor.blocks, processor_input, indices
            )
            scattered_values = split_layer_values(
                transformed, values.shape[2], values.shape[4]
            )
        else:
            processor_input = merge_global_values(values)
            transformed = processor.block(processor_input, indices)
            scattered_values = split_global_values(
                transformed,
                processor.num_layers,
                processor.num_kv_heads,
                processor.head_dim,
            )

        def transform() -> torch.Tensor:
            if variant == "bt":
                return run_batched_cache_processors(
                    processor.blocks, processor_input, indices
                )
            return processor.block(processor_input, indices)

        def scatter() -> CacheState:
            return scatter_cache_values(cache, indices, scattered_values)

    else:
        if variant == "bt":
            processor_input = merge_layer_kv(keys, values)
            transformed = run_batched_cache_processors(
                processor.blocks, processor_input, indices
            )
            scattered_keys, scattered_values = split_layer_kv(
                transformed, keys.shape[2], keys.shape[4]
            )
        else:
            processor_input = merge_global_kv(keys, values)
            transformed = processor.block(processor_input, indices)
            scattered_keys, scattered_values = split_global_kv(
                transformed,
                processor.num_layers,
                processor.num_kv_heads,
                processor.head_dim,
            )

        def transform() -> torch.Tensor:
            if variant == "bt":
                return run_batched_cache_processors(
                    processor.blocks, processor_input, indices
                )
            return processor.block(processor_input, indices)

        def scatter() -> CacheState:
            return scatter_cache_layers(
                cache, indices, scattered_keys, scattered_values
            )

    operations: list[tuple[str, Callable[[], object]]] = [
        ("select_indices", select),
        ("gather_cache", lambda: gather_cache_layers(cache, indices)),
        ("processor_transform", transform),
        ("scatter_cache", scatter),
        ("processor_end_to_end", lambda: processor(cache)),
    ]
    return [
        _time_forward_operation(
            name,
            operation,
            device=device,
            warmup=warmup,
            steps=steps,
            variant=variant,
        )
        for name, operation in operations
    ]


def _gradient_accumulation_steps(config: TrainConfig) -> int:
    micro_tokens = config.batch_size * config.sequence_len
    if config.global_batch_rows:
        if config.global_batch_rows % config.batch_size:
            raise ValueError(
                "global_batch_rows must be divisible by batch_size"
            )
        return config.global_batch_rows // config.batch_size
    tokens = config.total_batch_size or micro_tokens
    if tokens % micro_tokens:
        raise ValueError(
            "total_batch_size must be divisible by batch_size * sequence_len"
        )
    return tokens // micro_tokens


def _training_fixture(
    config: TrainConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    stream = build_training_stream(config, device, rank=0, world_size=1)
    try:
        batch = next(stream)
        return batch.x, batch.y, batch.attention_mask
    finally:
        stream.close()


def _optimizer_tensor_state(
    optimizer: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    parameter_indices = {
        id(parameter): index
        for index, parameter in enumerate(
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
    }
    for parameter, state in optimizer.state.items():
        index = parameter_indices[id(parameter)]
        for name, value in sorted(state.items()):
            if isinstance(value, torch.Tensor):
                tensors[f"parameter.{index}.{name}"] = value
    return tensors


def benchmark_optimizer_step(
    config: TrainConfig,
    device: torch.device,
    *,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    if config.resume is not None:
        raise ValueError(
            "optimizer-step benchmarks do not restore resume state"
        )
    torch.manual_seed(config.seed)
    model = _build_configured_model(config, device)
    _apply_trainable_policy(model, config.trainable)
    model = maybe_compile_whole_model(model)
    model.train()
    params = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = _build_optimizer(params, config)
    initial_trainable_parameters_sha256 = tensor_state_sha256(
        {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
    )
    x, y, attention_mask = _training_fixture(config, device)
    accumulation = _gradient_accumulation_steps(config)
    benchmark_step = 0

    def optimizer_step() -> None:
        nonlocal benchmark_step
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=config.bf16 and device.type == "cuda",
            ):
                loss = _model_loss(
                    model,
                    x,
                    y,
                    loss_backend=config.loss_backend,
                    attention_mask=attention_mask,
                )
            (loss / accumulation).backward()
        _clip_grad_norm(model, params, config.grad_clip)
        schedule_step = benchmark_step % config.max_steps
        lr_scale = _wsd_lr(
            schedule_step,
            config.max_steps,
            config.warmup_steps,
            config.warmdown_ratio,
            config.final_lr_frac,
        )
        weight_decay = _weight_decay(
            schedule_step, config.max_steps, config.weight_decay
        )
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_scale
            if group.get("decay", False):
                group["weight_decay"] = weight_decay
        optimizer.step()
        benchmark_step += 1

    for _ in range(warmup):
        optimizer_step()
    _synchronize(device)
    _reset_peak_memory(device)
    raw: list[float] = []
    for _ in range(steps):
        _synchronize(device)
        start = time.perf_counter()
        optimizer_step()
        _synchronize(device)
        raw.append(time.perf_counter() - start)

    non_padding_per_microbatch = int(
        x.numel() if attention_mask is None else attention_mask.sum().item()
    )
    supervised_per_microbatch = int((y != -100).sum().item())
    batch_state = {"input_ids": x, "targets": y}
    if attention_mask is not None:
        batch_state["attention_mask"] = attention_mask
    trainable_state = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    gradient_state = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    optimizer_tensors = _optimizer_tensor_state(optimizer)
    return {
        "name": "optimizer_step",
        "kind": "complete_training_optimizer_step",
        "processor_variant": config.model.variant,
        "warmup": warmup,
        "measured_steps": steps,
        "gradient_accumulation_steps": accumulation,
        "training_batch_sha256": tensor_state_sha256(batch_state),
        "initial_trainable_parameters_sha256": initial_trainable_parameters_sha256,
        "final_trainable_parameters_sha256": tensor_state_sha256(
            trainable_state
        ),
        "final_gradients_sha256": tensor_state_sha256(gradient_state),
        "final_optimizer_tensors_sha256": tensor_state_sha256(
            optimizer_tensors
        ),
        "raw_step_seconds": raw,
        **timing_summary(raw),
        "peak_memory_bytes": _peak_memory(device),
        "real_non_padding_tokens_per_step": non_padding_per_microbatch
        * accumulation,
        "real_supervised_tokens_per_step": supervised_per_microbatch
        * accumulation,
        "real_non_padding_tokens_total": non_padding_per_microbatch
        * accumulation
        * steps,
        "real_supervised_tokens_total": supervised_per_microbatch
        * accumulation
        * steps,
    }


def run_benchmarks(
    config: TrainConfig,
    config_path: Path,
    device: torch.device,
    *,
    suite: str,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    if warmup < 0 or steps <= 0:
        raise ValueError(
            "warmup must be non-negative and steps must be positive"
        )
    if config.distributed != "none":
        raise ValueError("benchmark harness requires distributed='none'")
    require_loss_backend(config.loss_backend, device)
    repository = find_repository_root(Path(__file__))
    source = repository_source_identity(
        repository, executed_files=(Path(__file__),), require_git=True
    )
    results: list[dict[str, Any]] = []
    if suite in {"processor", "all"}:
        results.extend(
            benchmark_processor_operations(
                config, device, warmup=warmup, steps=steps
            )
        )
    if suite in {"train-step", "all"}:
        results.append(
            benchmark_optimizer_step(config, device, warmup=warmup, steps=steps)
        )
    resolved_config = to_dict(config)
    return {
        "source_commit": source["git_commit"],
        "source_dirty": source["git_dirty"],
        "source_tree_sha256": source["source_tree_sha256"],
        "config_path": str(config_path.resolve()),
        "config_file_sha256": file_sha256(config_path),
        "config_sha256": canonical_json_sha256(resolved_config),
        "config": resolved_config,
        "processor_variant": config.model.variant,
        "loss_backend": loss_backend_metadata(config.loss_backend),
        "environment": device_identity(device),
        "suite": suite,
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark GCW cache processors and complete optimizer steps"
    )
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--suite", choices=("processor", "train-step", "all"), default="all"
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    config = load_train_config(args.train_config)
    payload = run_benchmarks(
        config,
        args.train_config,
        torch.device(args.device),
        suite=args.suite,
        warmup=args.warmup,
        steps=args.steps,
    )
    document = write_integrity_json(args.output, payload)
    print(canonical_json_bytes(document).decode("utf-8"))


if __name__ == "__main__":
    main()
