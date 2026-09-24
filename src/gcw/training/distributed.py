from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import Any, cast
import torch
import torch.distributed as dist
import torch.nn.functional as F

from gcw.config import TrainConfig
from gcw.data.training import TrainingBatch
from gcw.performance.checkpointing import (
    transformer_layer_classes as _transformer_layer_classes,
)


def _fsdp_active(config: TrainConfig) -> bool:
    return config.distributed == "fsdp" and dist.is_initialized()


def _align_fsdp_batch(batch: TrainingBatch) -> TrainingBatch:
    """Keep windowed module traversal identical across FSDP ranks."""
    width = batch.x.new_tensor(batch.x.shape[1])
    dist.all_reduce(width, op=dist.ReduceOp.MAX)
    padding = int(width.item()) - batch.x.shape[1]
    if padding == 0:
        return batch
    attention_mask = batch.attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(batch.x, dtype=torch.bool)
    return TrainingBatch(
        x=F.pad(batch.x, (0, padding), value=0),
        y=F.pad(batch.y, (0, padding), value=-100),
        attention_mask=F.pad(attention_mask, (0, padding), value=0),
    )


def _fsdp_wrap_classes(family: str) -> set[type[torch.nn.Module]]:
    classes = _transformer_layer_classes(family)
    if family in {"qwen3_5", "qwen3_5_moe"}:
        from gcw.models.backbones.qwen3 import Qwen35GatedDeltaDynamics

        classes.add(Qwen35GatedDeltaDynamics)
    return classes


def _wrap_fsdp(
    model: torch.nn.Module, config: TrainConfig, device: torch.device
) -> torch.nn.Module:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    layer_classes = _fsdp_wrap_classes(config.model.family)
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls=layer_classes
    )
    mixed_precision = (
        MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        if config.precision in {"bf16_mixed", "bf16"}
        else None
    )
    kwargs: dict[str, Any] = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "use_orig_params": True,
        "sync_module_states": True,
        "param_init_fn": _fsdp_param_init_fn(device),
    }
    if device.type == "cuda":
        kwargs["device_id"] = device
    wrapped = FSDP(model, **kwargs)
    refresh_dynamics = getattr(
        _unwrap(wrapped), "refresh_linear_state_dynamics", None
    )
    if refresh_dynamics is not None:
        refresh_dynamics()
    return wrapped


def _fsdp_param_init_fn(device: torch.device):
    def init_fn(module: torch.nn.Module) -> None:
        has_meta_param = any(
            param.is_meta for param in module.parameters(recurse=False)
        )
        has_meta_buffer = any(
            buffer.is_meta for buffer in module.buffers(recurse=False)
        )
        if has_meta_param or has_meta_buffer:
            module.to_empty(device=device, recurse=False)

    return init_fn


def _no_sync(model: torch.nn.Module, *, enabled: bool, use_fsdp: bool = False):
    no_sync = getattr(model, "no_sync", None)
    if enabled and not use_fsdp and callable(no_sync):
        return no_sync()
    return nullcontext()


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return cast(torch.nn.Module, getattr(model, "module", model))
