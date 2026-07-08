from __future__ import annotations

import math
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
from gcw.data.sft import iter_jsonl, sft_batch
from gcw.data.tokenizers import get_tokenizer
from gcw.flops import total_training_flops, training_flops_per_token
from gcw.models import build_model
from gcw.params import processor_params, total_non_embedding_params


def train_from_config(path: str | Path) -> None:
    train(load_train_config(path))


def train(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = init_distributed(config.distributed)
    _, rank, _, world_size = dist_info()
    model = build_model(config.model).to(device)
    model.init_weights()
    if config.backbone_checkpoint:
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
    loader = _loader(config, device)
    micro_tokens = config.batch_size * config.sequence_len * world_size
    tokens_per_step = config.total_batch_size or micro_tokens
    if tokens_per_step % micro_tokens != 0:
        raise ValueError(f"total_batch_size {tokens_per_step} must be divisible by micro batch tokens {micro_tokens}")
    grad_accum_steps = tokens_per_step // micro_tokens
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
    model.train()
    smooth_loss = 0.0
    last_state: dict[str, Any] = {"kind": "uninitialized"}
    for step in range(config.max_steps):
        step_start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        for micro_step in range(grad_accum_steps):
            x, y, last_state = next(loader)
            sync_context = _no_sync(model, enabled=micro_step < grad_accum_steps - 1)
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.bf16 and device.type == "cuda"):
                    loss = model(x, y)
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
            "train/epoch": last_state.get("epoch") if isinstance(last_state, dict) else None,
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
        if config.save_every and step_num % config.save_every == 0 and rank == 0:
            save_checkpoint(
                Path(config.output_dir) / "checkpoint",
                _unwrap(model),
                config.model,
                _checkpoint_state(checkpoint_meta, step_num, last_state),
            )
    if rank == 0:
        save_checkpoint(Path(config.output_dir) / "checkpoint", _unwrap(model), config.model, _checkpoint_state(checkpoint_meta, config.max_steps))
        if wandb_run is not None:
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
        tokenizer_name = config.data.tokenizer
        if tokenizer_name == "hf":
            if config.data.tokenizer_source is None:
                raise ValueError("data.tokenizer_source is required when data.tokenizer='hf'")
            tokenizer_name = f"hf:{config.data.tokenizer_source}"
        tokenizer = get_tokenizer(tokenizer_name)
        rows = list(iter_jsonl(config.data.dataset))
        if config.data.sft_skip_rows:
            if config.data.sft_skip_rows >= len(rows):
                raise ValueError(f"data.sft_skip_rows={config.data.sft_skip_rows} exceeds dataset length {len(rows)}")
            rows = rows[config.data.sft_skip_rows :]
        if config.data.sft_take_rows is not None:
            rows = rows[: config.data.sft_take_rows]
        if not rows:
            raise ValueError(f"empty SFT dataset: {config.data.dataset}")
        i = 0
        while True:
            batch = [rows[(i + j) % len(rows)] for j in range(config.batch_size)]
            i += config.batch_size
            x, y = sft_batch(
                batch,
                tokenizer,
                config.sequence_len,
                device=device,
                response_prefix=config.data.response_prefix,
            )
            yield x, y, {"kind": "sft", "row": (config.data.sft_skip_rows + i) % len(rows)}
    yield from data_loader(
        config.batch_size,
        config.sequence_len,
        split="train",
        device=device,
        data_dir=config.data.data_dir,
        infinite=True,
    )


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
    if enabled and hasattr(model, "no_sync"):
        return model.no_sync()
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
