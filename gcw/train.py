from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

from gcw.checkpoint import load_backbone_checkpoint, save_checkpoint
from gcw.common import cleanup_distributed, dist_info, init_distributed, print0
from gcw.compile import maybe_compile
from gcw.config import TrainConfig, load_train_config, to_dict
from gcw.data.shards import data_loader
from gcw.data.training import TrainingBatchStream, build_training_stream, training_data_identity
from gcw.flops import total_training_flops, training_flops_per_token
from gcw.models import build_model
from gcw.params import processor_params, total_non_embedding_params
from gcw.training_checkpoint import ResumeState, TrainingCheckpoint, save_training_checkpoint


def train_from_config(path: str | Path) -> None:
    train(load_train_config(path))


def train(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = init_distributed(config.distributed)
    _, rank, _, world_size = dist_info()
    micro_tokens = config.batch_size * config.sequence_len * world_size
    tokens_per_step = config.total_batch_size or micro_tokens
    if tokens_per_step % micro_tokens != 0:
        raise ValueError(f"total_batch_size {tokens_per_step} must be divisible by micro batch tokens {micro_tokens}")
    grad_accum_steps = tokens_per_step // micro_tokens
    fingerprint = _training_fingerprint(config, world_size, micro_tokens, tokens_per_step, grad_accum_steps)
    checkpoint = None
    start_step = 0
    smooth_loss = 0.0
    cursor = None
    if config.resume is not None:
        checkpoint = TrainingCheckpoint.open(
            config.resume.checkpoint,
            expected_fingerprint=fingerprint,
            topology=config.resume.topology,
            current_world_size=world_size,
        )
        start_step = checkpoint.resume_state.step
        if start_step > config.max_steps:
            raise ValueError(f"checkpoint step {start_step} exceeds max_steps {config.max_steps}")
        if start_step == config.max_steps:
            print0(f"checkpoint already completed max_steps={config.max_steps}")
            cleanup_distributed()
            return
        smooth_loss = checkpoint.resume_state.smooth_loss
        cursor = checkpoint.loader_cursor(rank=rank)

    model = build_model(config.model).to(device)
    model.init_weights()
    if checkpoint is not None:
        checkpoint.restore_model(model)
    elif config.backbone_checkpoint:
        load_backbone_checkpoint(model, config.backbone_checkpoint)
    if config.freeze_backbone and hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    model = maybe_compile(model)
    if config.distributed == "ddp" and torch.distributed.is_initialized():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    elif config.distributed == "fsdp" and torch.distributed.is_initialized():
        model = FSDP(model)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = _build_optimizer(params, config)
    if checkpoint is not None:
        checkpoint.restore_optimizer(opt, device=device)
    stream = build_training_stream(config, device, rank=rank, world_size=world_size, cursor=cursor)
    if checkpoint is not None:
        checkpoint.restore_rng(rank=rank)
    target_tokens = tokens_per_step * config.max_steps
    flops_per_tok = training_flops_per_token(config.model, config.sequence_len)
    run_meta = {
        "step": 0,
        "micro_tokens_per_step": micro_tokens,
        "grad_accum_steps": grad_accum_steps,
        "tokens_per_step": tokens_per_step,
        "total_training_tokens": target_tokens,
        "training_flops_per_token": flops_per_tok,
        "total_training_flops": total_training_flops(config.model, target_tokens, config.sequence_len),
        "non_embedding_params": total_non_embedding_params(config.model),
        "processor_params": processor_params(config.model),
        "world_size": world_size,
        "resumed_from_step": start_step if checkpoint is not None else None,
        "resume_exact_topology": checkpoint.resume_state.exact_topology if checkpoint is not None else True,
        "resume_saved_world_size": checkpoint.resume_state.saved_world_size if checkpoint is not None else world_size,
        "resume_current_world_size": world_size,
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
    if start_step == 0 and config.eval_on_start and config.eval_every:
        val = _eval_loss(model, config, device)
        if rank == 0:
            print0(f"step 0/{config.max_steps} val_loss={val['val/loss']:.4f}")
            _wandb_log(wandb_run, {"train/step": 0, **val}, step=0)
    retained_checkpoint_every = _retained_checkpoint_every()
    model.train()
    for step in range(start_step, config.max_steps):
        step_start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        for micro_step in range(grad_accum_steps):
            batch = next(stream)
            sync_context = _no_sync(model, enabled=micro_step < grad_accum_steps - 1)
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.bf16 and device.type == "cuda"):
                    loss = model(batch.x, batch.y)
                loss_total += float(loss.detach().cpu())
                (loss / grad_accum_steps).backward()
        grad_norm = None
        if config.grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
        lr_scale = _wsd_lr(step, config.max_steps, config.warmup_steps, config.warmdown_ratio, config.final_lr_frac)
        weight_decay = _weight_decay(step, config.max_steps, config.weight_decay)
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
            "train/epoch": getattr(stream.cursor, "epoch", None),
            "train/grad_accum_steps": grad_accum_steps,
            "train/micro_tokens_per_step": micro_tokens,
            "train/tokens_per_step": tokens_per_step,
            "timing/step_seconds": step_seconds,
            "timing/tokens_per_second": tokens_per_step / max(step_seconds, 1e-12),
            "timing/flops_per_second": (flops_per_tok * tokens_per_step) / max(step_seconds, 1e-12),
            "optim/grad_norm": _maybe_float(grad_norm),
            "optim/weight_decay": weight_decay,
        }
        if device.type == "cuda":
            train_metrics["system/vram_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            train_metrics["system/vram_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 1024 / 1024
        if config.eval_every and step_num % config.eval_every == 0:
            train_metrics.update(_eval_loss(model, config, device))
        if rank == 0:
            msg = f"step {step_num}/{config.max_steps} loss={train_metrics['train/loss_smooth']:.4f}"
            if "val/loss" in train_metrics:
                msg += f" val_loss={train_metrics['val/loss']:.4f}"
            print0(msg)
            if config.wandb.enabled and step_num % max(1, config.wandb.log_every) == 0:
                _wandb_log(wandb_run, _compact_metrics(train_metrics), step=step_num)
        if config.save_every and step_num % config.save_every == 0:
            resume_state = _resume_state(step_num, smooth_loss, stream, world_size, checkpoint_meta)
            _save_training_state(
                Path(config.output_dir) / "checkpoint",
                _unwrap(model),
                opt,
                config,
                resume_state,
                fingerprint,
                rank,
            )
            if retained_checkpoint_every is not None and step_num % retained_checkpoint_every == 0:
                _save_training_state(
                    Path(config.output_dir) / f"checkpoint-{step_num}",
                    _unwrap(model),
                    opt,
                    config,
                    resume_state,
                    fingerprint,
                    rank,
                )
    final_state = _resume_state(config.max_steps, smooth_loss, stream, world_size, checkpoint_meta)
    _save_training_state(
        Path(config.output_dir) / "checkpoint",
        _unwrap(model),
        opt,
        config,
        final_state,
        fingerprint,
        rank,
    )
    if rank == 0 and wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed()


def _checkpoint_metadata(config: TrainConfig, run_meta: dict[str, Any]) -> dict[str, Any]:
    state = {
        **run_meta,
        "data_kind": config.data.kind,
        "tokenizer": config.data.tokenizer,
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


def _checkpoint_state(base: dict[str, Any], step: int, loader: dict[str, Any] | None = None) -> dict[str, Any]:
    state = {**base, "step": step}
    if loader is not None:
        state["loader"] = loader
    return state


def _training_fingerprint(
    config: TrainConfig,
    world_size: int,
    micro_tokens: int,
    tokens_per_step: int,
    grad_accum_steps: int,
) -> dict[str, Any]:
    return {
        "model": to_dict(config.model),
        "data": to_dict(config.data),
        "data_identity": training_data_identity(config),
        "data_kind": config.data.kind,
        "mode": config.mode,
        "distributed": config.distributed,
        "freeze_backbone": config.freeze_backbone,
        "loss": config.loss,
        "checkpoint_format": config.checkpoint_format,
        "bf16": config.bf16,
        "seed": config.seed,
        "optimizer": {
            "learning_rate": config.learning_rate,
            "adam_beta1": config.adam_beta1,
            "adam_beta2": config.adam_beta2,
            "adam_eps": config.adam_eps,
            "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip,
        },
        "schedule": {
            "max_steps": config.max_steps,
            "warmup_steps": config.warmup_steps,
            "warmdown_ratio": config.warmdown_ratio,
            "final_lr_frac": config.final_lr_frac,
        },
        "batch": {
            "sequence_len": config.sequence_len,
            "micro_tokens": micro_tokens,
            "tokens_per_step": tokens_per_step,
            "grad_accum_steps": grad_accum_steps,
        },
    }


def _resume_state(
    step: int,
    smooth_loss: float,
    stream: TrainingBatchStream,
    world_size: int,
    metadata: dict[str, Any],
) -> ResumeState:
    return ResumeState(
        step=step,
        smooth_loss=smooth_loss,
        loader_cursor=stream.cursor,
        saved_world_size=world_size,
        exact_topology=True,
        metadata=metadata,
    )


def _save_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    resume_state: ResumeState,
    fingerprint: dict[str, Any],
    rank: int,
) -> None:
    if config.distributed == "fsdp":
        if rank == 0:
            save_checkpoint(path, model, config.model, {**resume_state.metadata, "step": resume_state.step})
        return
    save_training_checkpoint(path, model, config.model, optimizer, resume_state, fingerprint)


def _retained_checkpoint_every() -> int | None:
    keep = os.environ.get("GCW_KEEP_STEP_CHECKPOINTS", "0").lower()
    if keep not in {"1", "true", "yes", "on"}:
        return None
    raw = os.environ.get("GCW_STEP_CHECKPOINT_EVERY", "1000")
    try:
        every = int(raw)
    except ValueError as exc:
        raise ValueError("GCW_STEP_CHECKPOINT_EVERY must be an integer") from exc
    if every <= 0:
        raise ValueError("GCW_STEP_CHECKPOINT_EVERY must be positive")
    return every


def _build_optimizer(params: list[torch.nn.Parameter], config: TrainConfig) -> torch.optim.AdamW:
    decay = [p for p in params if p.ndim >= 2]
    nodecay = [p for p in params if p.ndim < 2]
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "lr": config.learning_rate, "weight_decay": config.weight_decay, "decay": True})
    if nodecay:
        groups.append({"params": nodecay, "lr": config.learning_rate, "weight_decay": 0.0, "decay": False})
    opt = torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
    return opt


def _wsd_lr(step: int, max_steps: int, warmup_steps: int, warmdown_ratio: float, final_lr_frac: float) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    warmdown = round(warmdown_ratio * max_steps)
    if warmdown <= 0 or step <= max_steps - warmdown:
        return 1.0
    progress = (max_steps - step) / warmdown
    return progress + (1.0 - progress) * final_lr_frac


def _weight_decay(step: int, max_steps: int, base_weight_decay: float) -> float:
    return base_weight_decay * 0.5 * (1.0 + math.cos(math.pi * step / max(1, max_steps)))


def _no_sync(model: torch.nn.Module, *, enabled: bool):
    no_sync = getattr(model, "no_sync", None)
    if enabled and callable(no_sync):
        return no_sync()
    return nullcontext()


@torch.no_grad()
def _eval_loss(model: torch.nn.Module, config: TrainConfig, device: torch.device) -> dict[str, float]:
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
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.bf16 and device.type == "cuda"):
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
        x = torch.randint(0, vocab, (config.batch_size, config.sequence_len), device=device)
        y = torch.roll(x, shifts=-1, dims=1)
        yield x, y, {"kind": "random"}


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _init_wandb(config: TrainConfig, run_meta: dict[str, Any]):
    if not config.wandb.enabled:
        return None
    import wandb

    name = config.wandb.name or Path(config.output_dir).name
    init_kwargs: dict[str, Any] = {
        "project": config.wandb.project,
        "name": name,
        "group": config.wandb.group,
        "tags": config.wandb.tags,
        "config": {
            **to_dict(config),
            "run_budget": run_meta,
        },
    }
    if config.wandb.entity:
        init_kwargs["entity"] = config.wandb.entity
    if config.wandb.mode:
        init_kwargs["mode"] = config.wandb.mode
    if config.wandb.id:
        init_kwargs["id"] = config.wandb.id
    if config.wandb.resume:
        init_kwargs["resume"] = config.wandb.resume
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
