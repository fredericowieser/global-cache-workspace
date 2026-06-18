from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

from gcw.checkpoint import save_checkpoint
from gcw.common import cleanup_distributed, dist_info, init_distributed, print0
from gcw.config import TrainConfig, load_train_config, to_dict
from gcw.data.shards import data_loader
from gcw.data.sft import iter_jsonl, sft_batch
from gcw.data.tokenizers import get_tokenizer
from gcw.flops import total_training_flops, training_flops_per_token
from gcw.models import build_model
from gcw.models.transformer import cosine_lr
from gcw.params import processor_params, total_non_embedding_params


def train_from_config(path: str | Path) -> None:
    train(load_train_config(path))


def train(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = init_distributed(config.distributed)
    _, rank, _, world_size = dist_info()
    model = build_model(config.model).to(device)
    model.init_weights()
    if config.freeze_backbone and hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    if config.distributed == "ddp" and torch.distributed.is_initialized():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    elif config.distributed == "fsdp" and torch.distributed.is_initialized():
        model = FSDP(model)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.95))
    loader = _loader(config, device)
    tokens_per_step = config.batch_size * config.sequence_len * world_size
    target_tokens = tokens_per_step * config.max_steps
    flops_per_tok = training_flops_per_token(config.model, config.sequence_len)
    run_meta = {
        "step": 0,
        "tokens_per_step": tokens_per_step,
        "total_training_tokens": target_tokens,
        "training_flops_per_token": flops_per_tok,
        "total_training_flops": total_training_flops(config.model, target_tokens, config.sequence_len),
        "non_embedding_params": total_non_embedding_params(config.model),
        "processor_params": processor_params(config.model),
        "world_size": world_size,
    }
    print0(
        "training budget: "
        f"tokens={target_tokens:,} flops/token={flops_per_tok:.6e} "
        f"total_flops={run_meta['total_training_flops']:.6e} "
        f"non_embedding_params={run_meta['non_embedding_params']:,}"
    )
    wandb_run = _init_wandb(config, run_meta) if rank == 0 else None
    if config.eval_on_start and config.eval_every:
        val = _eval_loss(model, config, device)
        if rank == 0:
            print0(f"step 0/{config.max_steps} val_loss={val['val/loss']:.4f}")
            _wandb_log(wandb_run, {"train/step": 0, **val}, step=0)
    model.train()
    for step in range(config.max_steps):
        step_start = time.perf_counter()
        x, y, state = next(loader)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.bf16 and device.type == "cuda"):
            loss = model(x, y)
        loss.backward()
        grad_norm = None
        if config.grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
        lr_scale = cosine_lr(step, config.max_steps, config.warmup_steps)
        for group in opt.param_groups:
            group["lr"] = config.learning_rate * lr_scale
        opt.step()
        step_num = step + 1
        step_seconds = time.perf_counter() - step_start
        tokens_seen = tokens_per_step * step_num
        flops_seen = flops_per_tok * tokens_seen
        train_metrics = {
            "train/step": step_num,
            "train/loss": float(loss.detach().cpu()),
            "train/perplexity": math.exp(min(20.0, float(loss.detach().cpu()))),
            "train/lr": config.learning_rate * lr_scale,
            "train/lr_scale": lr_scale,
            "train/tokens": tokens_seen,
            "train/flops": flops_seen,
            "train/progress": step_num / config.max_steps,
            "train/epoch": state.get("epoch") if isinstance(state, dict) else None,
            "timing/step_seconds": step_seconds,
            "timing/tokens_per_second": tokens_per_step / max(step_seconds, 1e-12),
            "timing/flops_per_second": (flops_per_tok * tokens_per_step) / max(step_seconds, 1e-12),
            "optim/grad_norm": _maybe_float(grad_norm),
            "optim/weight_decay": config.weight_decay,
        }
        if config.eval_every and step_num % config.eval_every == 0:
            train_metrics.update(_eval_loss(model, config, device))
        if rank == 0:
            msg = f"step {step_num}/{config.max_steps} loss={train_metrics['train/loss']:.4f}"
            if "val/loss" in train_metrics:
                msg += f" val_loss={train_metrics['val/loss']:.4f}"
            print0(msg)
            if config.wandb.enabled and step_num % max(1, config.wandb.log_every) == 0:
                _wandb_log(wandb_run, _compact_metrics(train_metrics), step=step_num)
        if config.save_every and step_num % config.save_every == 0 and rank == 0:
            save_checkpoint(
                Path(config.output_dir) / "checkpoint",
                _unwrap(model),
                config.model,
                {**run_meta, "step": step_num, "loader": state},
            )
    if rank == 0:
        save_checkpoint(Path(config.output_dir) / "checkpoint", _unwrap(model), config.model, {**run_meta, "step": config.max_steps})
        if wandb_run is not None:
            wandb_run.finish()
    cleanup_distributed()


def _loader(config: TrainConfig, device: torch.device):
    if config.data.kind == "random":
        vocab = config.model.vocab_size
        while True:
            x = torch.randint(0, vocab, (config.batch_size, config.sequence_len), device=device)
            y = torch.roll(x, shifts=-1, dims=1)
            yield x, y, {"kind": "random"}
    if config.mode == "sft":
        if config.data.dataset is None:
            raise ValueError("SFT mode requires data.dataset JSONL path")
        tokenizer = get_tokenizer("cosmo2")
        rows = list(iter_jsonl(config.data.dataset))
        if not rows:
            raise ValueError(f"empty SFT dataset: {config.data.dataset}")
        i = 0
        while True:
            batch = [rows[(i + j) % len(rows)] for j in range(config.batch_size)]
            i += config.batch_size
            x, y = sft_batch(batch, tokenizer, config.sequence_len, device=device)
            yield x, y, {"kind": "sft", "row": i % len(rows)}
    return data_loader(
        config.batch_size,
        config.sequence_len,
        split="train",
        device=device,
        data_dir=config.data.data_dir,
        infinite=True,
    )


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
