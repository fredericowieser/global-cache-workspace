from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gcw.checkpoint import load_backbone_checkpoint, save_checkpoint
from gcw.common import cleanup_distributed, dist_info, init_distributed, print0
from gcw.compile import maybe_compile
from gcw.config import TrainConfig, load_train_config, to_dict
from gcw.data.shards import data_loader
from gcw.data.sft import iter_jsonl, sft_batch
from gcw.data.tokenizers import get_tokenizer
from gcw.flops import total_training_flops, training_flops_per_token
from gcw.models import build_model
from gcw.models.transformer import Block
from gcw.params import processor_params, total_non_embedding_params


def train_from_config(path: str | Path) -> None:
    train(load_train_config(path))


def train(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = init_distributed(config.distributed)
    _validate_precision(config, device)
    _, rank, _, world_size = dist_info()
    use_fsdp = _fsdp_active(config)
    model = _build_training_model(config, device, rank=rank, use_fsdp=use_fsdp)
    if config.freeze_backbone and hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    if use_fsdp:
        _apply_activation_checkpointing(model, family=config.model.family)
    if not use_fsdp:
        model = maybe_compile(model)
    if config.distributed == "ddp" and torch.distributed.is_initialized():
        model = DDP(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    elif use_fsdp:
        model = _wrap_fsdp(model, config, device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = _build_optimizer(params, config)
    loader = _loader(config, device)
    micro_tokens = config.batch_size * config.sequence_len * world_size
    tokens_per_step = config.total_batch_size or micro_tokens
    if tokens_per_step % micro_tokens != 0:
        raise ValueError(
            f"total_batch_size {tokens_per_step} must be divisible by micro batch tokens {micro_tokens}"
        )
    grad_accum_steps = tokens_per_step // micro_tokens
    target_tokens = tokens_per_step * config.max_steps
    flops_per_tok = training_flops_per_token(config.model, config.sequence_len)
    run_meta = {
        "step": 0,
        "batch_size": config.batch_size,
        "sequence_len": config.sequence_len,
        "micro_tokens_per_step": micro_tokens,
        "grad_accum_steps": grad_accum_steps,
        "tokens_per_step": tokens_per_step,
        "total_training_tokens": target_tokens,
        "training_flops_per_token": flops_per_tok,
        "total_training_flops": total_training_flops(
            config.model, target_tokens, config.sequence_len
        ),
        "non_embedding_params": total_non_embedding_params(config.model),
        "processor_params": processor_params(config.model),
        "world_size": world_size,
        "model_runtime_backends": _model_runtime_backends(model, device),
    }
    checkpoint_meta = _checkpoint_metadata(config, run_meta)
    print0(
        "training budget: "
        f"tokens={target_tokens:,} flops/token={flops_per_tok:.6e} "
        f"total_flops={run_meta['total_training_flops']:.6e} "
        f"non_embedding_params={run_meta['non_embedding_params']:,} "
        f"micro_tokens={micro_tokens:,} grad_accum={grad_accum_steps}"
    )
    wandb_run = _init_wandb(config, run_meta) if rank == 0 else None
    if config.eval_on_start and config.eval_every:
        val = _eval_loss(model, config, device)
        if rank == 0:
            print0(f"step 0/{config.max_steps} val_loss={val['val/loss']:.4f}")
            _wandb_log(wandb_run, {"train/step": 0, **val}, step=0)
    retained_checkpoint_every = _retained_checkpoint_every()
    model.train()
    smooth_loss = 0.0
    last_state: dict[str, Any] = {"kind": "uninitialized"}
    last_checkpoint_step = 0
    for step in range(config.max_steps):
        step_start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        for micro_step in range(grad_accum_steps):
            x, y, last_state = next(loader)
            if step == 0 and micro_step == 0:
                _validate_train_batch(x, y, config)
                print0(
                    "first microbatch: "
                    f"x_shape={tuple(x.shape)} y_shape={tuple(y.shape)} "
                    f"configured_batch_size={config.batch_size} configured_sequence_len={config.sequence_len}"
                )
            sync_context = _no_sync(
                model,
                enabled=micro_step < grad_accum_steps - 1,
                use_fsdp=use_fsdp,
            )
            with sync_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=_uses_bf16_autocast(config, device),
                ):
                    loss = _model_loss(
                        model,
                        x,
                        y,
                        attention_mask=_attention_mask_from_state(last_state),
                    )
                loss_total += float(loss.detach().cpu())
                (loss / grad_accum_steps).backward()
        grad_norm = None
        if config.grad_clip:
            grad_norm = _clip_grad_norm(
                model, params, config.grad_clip, use_fsdp=use_fsdp
            )
        lr_scale = _wsd_lr(
            step,
            config.max_steps,
            config.warmup_steps,
            config.warmdown_ratio,
            config.final_lr_frac,
        )
        weight_decay = _weight_decay(
            step, config.max_steps, config.weight_decay
        )
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lr_scale
            if group.get("decay", False):
                group["weight_decay"] = weight_decay
        opt.step()
        step_num = step + 1
        step_seconds = time.perf_counter() - step_start
        tokens_seen = tokens_per_step * step_num
        flops_seen = flops_per_tok * tokens_seen
        train_loss = loss_total / grad_accum_steps
        smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
        debiased_loss = smooth_loss / (1.0 - 0.9**step_num)
        train_metrics = {
            "train/step": step_num,
            "train/loss": train_loss,
            "train/loss_smooth": debiased_loss,
            "train/perplexity": math.exp(min(20.0, train_loss)),
            "train/lr": config.learning_rate * lr_scale,
            "train/lr_scale": lr_scale,
            "train/tokens": tokens_seen,
            "train/flops": flops_seen,
            "train/progress": step_num / config.max_steps,
            "train/epoch": (
                last_state.get("epoch")
                if isinstance(last_state, dict)
                else None
            ),
            "train/grad_accum_steps": grad_accum_steps,
            "train/micro_tokens_per_step": micro_tokens,
            "train/tokens_per_step": tokens_per_step,
            "timing/step_seconds": step_seconds,
            "timing/tokens_per_second": tokens_per_step
            / max(step_seconds, 1e-12),
            "timing/flops_per_second": (flops_per_tok * tokens_per_step)
            / max(step_seconds, 1e-12),
            "optim/grad_norm": _maybe_float(grad_norm),
            "optim/weight_decay": weight_decay,
        }
        if device.type == "cuda":
            train_metrics["system/vram_allocated_mib"] = (
                torch.cuda.max_memory_allocated(device) / 1024 / 1024
            )
            train_metrics["system/vram_reserved_mib"] = (
                torch.cuda.max_memory_reserved(device) / 1024 / 1024
            )
        train_metrics.update(_cache_processor_metrics(model))
        if config.eval_every and step_num % config.eval_every == 0:
            train_metrics.update(_eval_loss(model, config, device))
        if rank == 0:
            msg = f"step {step_num}/{config.max_steps} loss={train_metrics['train/loss_smooth']:.4f}"
            if "val/loss" in train_metrics:
                msg += f" val_loss={train_metrics['val/loss']:.4f}"
            print0(msg)
            if (
                config.wandb.enabled
                and step_num % max(1, config.wandb.log_every) == 0
            ):
                _wandb_log(
                    wandb_run, _compact_metrics(train_metrics), step=step_num
                )
        if config.save_every and step_num % config.save_every == 0:
            checkpoint_state = _checkpoint_state(
                checkpoint_meta, step_num, last_state
            )
            _save_training_checkpoint(
                Path(config.output_dir) / "checkpoint",
                model,
                config.model,
                checkpoint_state,
                use_fsdp=use_fsdp,
                cache_processor_only=config.freeze_backbone,
            )
            last_checkpoint_step = step_num
            if (
                retained_checkpoint_every is not None
                and step_num % retained_checkpoint_every == 0
            ):
                _save_training_checkpoint(
                    Path(config.output_dir) / f"checkpoint-{step_num}",
                    model,
                    config.model,
                    checkpoint_state,
                    use_fsdp=use_fsdp,
                    cache_processor_only=config.freeze_backbone,
                )
    if last_checkpoint_step != config.max_steps:
        _save_training_checkpoint(
            Path(config.output_dir) / "checkpoint",
            model,
            config.model,
            _checkpoint_state(checkpoint_meta, config.max_steps),
            use_fsdp=use_fsdp,
            cache_processor_only=config.freeze_backbone,
        )
    if rank == 0:
        if wandb_run is not None:
            wandb_run.finish()
    cleanup_distributed()


def _checkpoint_metadata(
    config: TrainConfig, run_meta: dict[str, Any]
) -> dict[str, Any]:
    state = {
        **run_meta,
        "data_kind": config.data.kind,
        "tokenizer": config.data.tokenizer,
        "precision": config.precision,
        "freeze_backbone": config.freeze_backbone,
    }
    if config.data.dataset is not None:
        state["dataset"] = config.data.dataset
    if config.data.data_dir is not None:
        state["data_dir"] = config.data.data_dir
    if config.backbone_checkpoint is not None:
        state["backbone_checkpoint"] = config.backbone_checkpoint
    if config.data.tokenizer_source is not None:
        state["source"] = config.data.tokenizer_source
        state["tokenizer_source"] = config.data.tokenizer_source
    return state


def _model_runtime_backends(
    model: torch.nn.Module, device: torch.device
) -> dict[str, str]:
    model = _unwrap(model)
    model = getattr(model, "backbone", model)
    for module in model.modules():
        effective_backends = getattr(module, "effective_backends", None)
        if effective_backends is not None:
            return effective_backends(device)
    return {}


def _checkpoint_state(
    base: dict[str, Any], step: int, loader: dict[str, Any] | None = None
) -> dict[str, Any]:
    state = {**base, "step": step}
    if loader is not None:
        state["loader"] = _json_safe_loader_state(loader)
    return state


def _json_safe_loader_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return None
    if isinstance(value, dict):
        return {
            k: safe
            for k, v in value.items()
            if (safe := _json_safe_loader_state(v)) is not None
        }
    if isinstance(value, list):
        return [
            safe
            for item in value
            if (safe := _json_safe_loader_state(item)) is not None
        ]
    if isinstance(value, tuple):
        return [
            safe
            for item in value
            if (safe := _json_safe_loader_state(item)) is not None
        ]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _retained_checkpoint_every() -> int | None:
    keep = os.environ.get("GCW_KEEP_STEP_CHECKPOINTS", "0").lower()
    if keep not in {"1", "true", "yes", "on"}:
        return None
    raw = os.environ.get("GCW_STEP_CHECKPOINT_EVERY", "1000")
    try:
        every = int(raw)
    except ValueError as exc:
        raise ValueError(
            "GCW_STEP_CHECKPOINT_EVERY must be an integer"
        ) from exc
    if every <= 0:
        raise ValueError("GCW_STEP_CHECKPOINT_EVERY must be positive")
    return every


def _loader(config: TrainConfig, device: torch.device):
    if config.data.kind == "random":
        vocab = config.model.vocab_size
        while True:
            x = torch.randint(
                0,
                vocab,
                (config.batch_size, config.sequence_len),
                device=device,
            )
            y = torch.roll(x, shifts=-1, dims=1)
            yield x, y, {"kind": "random"}
    if config.mode == "sft":
        if config.data.dataset is None:
            raise ValueError("SFT mode requires data.dataset JSONL path")
        tokenizer_name = config.data.tokenizer
        if tokenizer_name == "hf":
            if config.data.tokenizer_source is None:
                raise ValueError(
                    "data.tokenizer_source is required when data.tokenizer='hf'"
                )
            tokenizer_name = f"hf:{config.data.tokenizer_source}"
        tokenizer = get_tokenizer(tokenizer_name)
        rows = list(iter_jsonl(config.data.dataset))
        if config.data.sft_skip_rows:
            if config.data.sft_skip_rows >= len(rows):
                raise ValueError(
                    f"data.sft_skip_rows={config.data.sft_skip_rows} exceeds dataset length {len(rows)}"
                )
            rows = rows[config.data.sft_skip_rows :]
        if config.data.sft_take_rows is not None:
            rows = rows[: config.data.sft_take_rows]
        if not rows:
            raise ValueError(f"empty SFT dataset: {config.data.dataset}")
        i = 0
        while True:
            batch = [
                rows[(i + j) % len(rows)] for j in range(config.batch_size)
            ]
            i += config.batch_size
            x, y, attention_mask = sft_batch(
                batch,
                tokenizer,
                config.sequence_len,
                device=device,
                response_prefix=config.data.response_prefix,
                return_attention_mask=True,
            )
            yield x, y, {
                "kind": "sft",
                "row": (config.data.sft_skip_rows + i) % len(rows),
                "attention_mask": attention_mask,
            }
    yield from data_loader(
        config.batch_size,
        config.sequence_len,
        split="train",
        device=device,
        data_dir=config.data.data_dir,
        infinite=True,
    )


def _attention_mask_from_state(
    state: dict[str, Any] | Any,
) -> torch.Tensor | None:
    if isinstance(state, dict):
        mask = state.get("attention_mask")
        if isinstance(mask, torch.Tensor):
            return mask
    return None


def _validate_train_batch(
    x: torch.Tensor, y: torch.Tensor, config: TrainConfig
) -> None:
    if x.shape != y.shape:
        raise ValueError(
            f"input and target batch shapes must match; got x={tuple(x.shape)} y={tuple(y.shape)}"
        )
    if x.ndim != 2:
        raise ValueError(
            f"training batches must be 2D token tensors; got shape={tuple(x.shape)}"
        )
    if x.shape[0] != config.batch_size:
        raise ValueError(
            f"loader batch size {x.shape[0]} does not match config.batch_size {config.batch_size}"
        )
    if x.shape[1] > config.sequence_len:
        raise ValueError(
            f"loader sequence length {x.shape[1]} exceeds config.sequence_len {config.sequence_len}"
        )


def _model_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_reduction: str = "mean",
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if attention_mask is None:
        return model(input_ids, targets, loss_reduction=loss_reduction)
    return model(
        input_ids,
        targets,
        loss_reduction=loss_reduction,
        attention_mask=attention_mask,
    )


def _build_optimizer(
    params: list[torch.nn.Parameter], config: TrainConfig
) -> torch.optim.AdamW:
    decay = [p for p in params if p.ndim >= 2]
    nodecay = [p for p in params if p.ndim < 2]
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append(
            {
                "params": decay,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
                "decay": True,
            }
        )
    if nodecay:
        groups.append(
            {
                "params": nodecay,
                "lr": config.learning_rate,
                "weight_decay": 0.0,
                "decay": False,
            }
        )
    opt = torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
    return opt


def _wsd_lr(
    step: int,
    max_steps: int,
    warmup_steps: int,
    warmdown_ratio: float,
    final_lr_frac: float,
) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    warmdown = round(warmdown_ratio * max_steps)
    if warmdown <= 0 or step <= max_steps - warmdown:
        return 1.0
    progress = (max_steps - step) / warmdown
    return progress + (1.0 - progress) * final_lr_frac


def _weight_decay(step: int, max_steps: int, base_weight_decay: float) -> float:
    return (
        base_weight_decay
        * 0.5
        * (1.0 + math.cos(math.pi * step / max(1, max_steps)))
    )


def _no_sync(model: torch.nn.Module, *, enabled: bool, use_fsdp: bool = False):
    if enabled and not use_fsdp and hasattr(model, "no_sync"):
        return model.no_sync()
    return nullcontext()


def _clip_grad_norm(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    max_norm: float,
    *,
    use_fsdp: bool = False,
) -> torch.Tensor:
    if use_fsdp:
        return model.clip_grad_norm_(max_norm)
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


@torch.no_grad()
def _eval_loss(
    model: torch.nn.Module, config: TrainConfig, device: torch.device
) -> dict[str, float]:
    if config.mode != "pretrain" or config.data.kind == "random":
        loader = _random_loader(config, device)
    else:
        loader = data_loader(
            config.batch_size,
            config.sequence_len,
            split="val",
            device=device,
            data_dir=config.data.data_dir,
            infinite=True,
        )
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device)
    for _ in range(max(1, config.eval_batches)):
        x, y, _ = next(loader)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=_uses_bf16_autocast(config, device),
        ):
            loss = model(x, y, loss_reduction="sum")
        total_loss += loss.detach().float()
        total_tokens += (y >= 0).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    if was_training:
        model.train()
    val_loss = float((total_loss / total_tokens.clamp_min(1)).cpu())
    return {
        "val/loss": val_loss,
        "val/perplexity": math.exp(min(20.0, val_loss)),
        "val/tokens": float(total_tokens.cpu()),
    }


def _random_loader(config: TrainConfig, device: torch.device):
    vocab = config.model.vocab_size
    while True:
        x = torch.randint(
            0, vocab, (config.batch_size, config.sequence_len), device=device
        )
        y = torch.roll(x, shifts=-1, dims=1)
        yield x, y, {"kind": "random"}


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _fsdp_active(config: TrainConfig) -> bool:
    return config.distributed == "fsdp" and torch.distributed.is_initialized()


def _uses_bf16_autocast(config: TrainConfig, device: torch.device) -> bool:
    return config.precision == "bf16_mixed" and device.type == "cuda"


def _validate_precision(config: TrainConfig, device: torch.device) -> None:
    if (
        config.precision in {"bf16_mixed", "bf16"}
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            f"precision={config.precision!r} requires a CUDA device with BF16 support"
        )


def _build_training_model(
    config: TrainConfig, device: torch.device, *, rank: int, use_fsdp: bool
) -> torch.nn.Module:
    model_dtype = (
        torch.bfloat16 if config.precision == "bf16" else torch.float32
    )
    if use_fsdp:
        if rank == 0:
            model = build_model(config.model)
            model.init_weights()
            if config.backbone_checkpoint:
                load_backbone_checkpoint(model, config.backbone_checkpoint)
            return model.to(dtype=model_dtype)
        with torch.device("meta"):
            model = build_model(config.model)
        return model.to(dtype=model_dtype)

    model = build_model(config.model).to(device)
    model.init_weights()
    if config.backbone_checkpoint:
        load_backbone_checkpoint(model, config.backbone_checkpoint)
    return model.to(dtype=model_dtype)


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
        if config.precision == "bf16_mixed"
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


def _transformer_layer_classes(
    family: str,
) -> set[type[torch.nn.Module]]:
    classes: set[type[torch.nn.Module]] = {Block}
    if family == "gemma4":
        from gcw.models.gemma4 import Gemma4Block

        classes.add(Gemma4Block)
    elif family in {"qwen3_5", "qwen3_5_moe"}:
        from gcw.models.qwen3 import Qwen35Block, Qwen35MoeBlock

        classes.add(Qwen35Block)
        classes.add(Qwen35MoeBlock)
    return classes


def _fsdp_wrap_classes(
    family: str,
) -> set[type[torch.nn.Module]]:
    classes = _transformer_layer_classes(family)
    if family in {"qwen3_5", "qwen3_5_moe"}:
        from gcw.models.qwen3 import Qwen35GatedDeltaDynamics

        classes.add(Qwen35GatedDeltaDynamics)
    return classes


def _apply_activation_checkpointing(
    model: torch.nn.Module, *, family: str
) -> None:
    enabled = os.environ.get(
        "GCW_FSDP_ACTIVATION_CHECKPOINTING", "1"
    ).lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
    except ImportError:
        return
    layer_classes = _transformer_layer_classes(family)
    wrapper = partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: type(module) in layer_classes,
    )


def _save_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_config,
    checkpoint_state: dict[str, Any],
    *,
    use_fsdp: bool = False,
    cache_processor_only: bool = False,
) -> None:
    if use_fsdp:
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state = model.state_dict()
        if dist_info()[1] == 0:
            save_checkpoint(
                path,
                state,
                model_config,
                checkpoint_state,
                cache_processor_only=cache_processor_only,
            )
        return
    if dist.is_available() and dist.is_initialized() and dist_info()[1] != 0:
        return
    unwrapped = _unwrap(model)
    unwrapped = getattr(unwrapped, "_orig_mod", unwrapped)
    save_checkpoint(
        path,
        unwrapped,
        model_config,
        checkpoint_state,
        cache_processor_only=cache_processor_only,
    )


def _cache_processor_metrics(model) -> dict[str, float]:
    model = _unwrap(model)
    model = getattr(model, "_orig_mod", model)
    processor = getattr(model, "cache_processor", None)
    block = getattr(processor, "block", None)
    pop_metrics = getattr(block, "pop_iteration_metrics", None)
    if pop_metrics is None:
        return {}
    return pop_metrics()


def _init_wandb(config: TrainConfig, run_meta: dict[str, Any]):
    if not config.wandb.enabled:
        return None
    import wandb

    name = config.wandb.name or Path(config.output_dir).name
    init_kwargs = {
        "project": config.wandb.project,
        "name": name,
        "group": config.wandb.group,
        "tags": config.wandb.tags,
        "config": {
            **to_dict(config),
            "run_budget": run_meta,
        },
        "resume": "allow",
    }
    if config.wandb.entity:
        init_kwargs["entity"] = config.wandb.entity
    if config.wandb.mode:
        init_kwargs["mode"] = config.wandb.mode
    run = wandb.init(**init_kwargs)
    run.define_metric("train/step")
    run.define_metric("*", step_metric="train/step")
    run.summary["architecture"] = config.model.variant
    run.summary["non_embedding_params"] = run_meta["non_embedding_params"]
    run.summary["processor_params"] = run_meta["processor_params"]
    run.summary["total_training_tokens"] = run_meta["total_training_tokens"]
    run.summary["total_training_flops"] = run_meta["total_training_flops"]
    return run


def _wandb_log(run, metrics: dict[str, Any], step: int) -> None:
    if run is not None:
        run.log(metrics, step=step)


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if v is not None}


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)
