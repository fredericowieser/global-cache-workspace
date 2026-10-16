"""Training orchestration: admission, model assembly, and the step loop.

This module owns the top-level ``train`` entry point. It validates the
resolved execution plan, assembles the model/optimizer/data stream, and runs
the optimizer step loop with exact resume, checkpointing, and evaluation
hooks. The per-feature execution backends are configured here but implemented
in :mod:`gcw.performance`.
"""

from __future__ import annotations

import math
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist

from gcw.architecture import resolve_execution_plan
from gcw.runtime import (
    cleanup_distributed,
    print0,
)
from gcw.config import TrainConfig, load_train_config
from gcw.data.shards import data_loader
from gcw.data.training import (
    TrainingBatch,
    TrainingBatchStream,
    build_training_stream,
    fineweb_prefix_tokens,
)
from gcw.performance.adjacent_pairs import (
    PAIRS_PER_STEP,
    build_adaptive_pair_step,
    collate_pair_group,
)
from gcw.performance.checkpointing import (
    longshot_gradient_checkpointing_decision as _longshot_gradient_checkpointing_decision,
    longshot_gradient_checkpointing_step_metrics as _longshot_gradient_checkpointing_step_metrics,
)
from gcw.performance.telemetry import (
    cache_processor_metrics as _cache_processor_metrics,
)
from gcw.models.cache_processors.longshot_selection import (
    set_longshot_selector_training_progress,
)
from gcw.training.objectives import (
    adjacent_pair_loss_sum,
    logical_pair_grouped_loss as _logical_pair_grouped_loss,
    model_loss as _model_loss,
    row_loss_sums_and_counts as _row_loss_sums_and_counts,
)
from gcw.training.assembly import (
    TrainingPlan,
    _checkpoint_stop_requested,
    assemble_training_execution,
)
from gcw.training.audit import (
    _audit_training_step,
    _validate_train_batch,
    _write_training_audit,
)
from gcw.training.distributed import _align_fsdp_batch, _fsdp_active, _no_sync
from gcw.training.optimizer import (
    _clip_grad_norm,
    _compact_metrics,
    _maybe_float,
    _weight_decay,
    _wsd_lr,
)
from gcw.training.precision import (
    _autocast_kwargs,
)
from gcw.training.state import (
    _retain_final_checkpoint,
    _retained_checkpoint_every,
    _save_step_checkpoint,
)

# Aliases used by the step loop; kept at module scope for the audit hooks.
_gradient_audit = _audit_training_step
_write_pilot_audit = _write_training_audit


def train_from_config(path: str | Path) -> None:
    try:
        train(load_train_config(path))
    finally:
        # The CLI process must never retain a distributed group after any
        # checkpoint, validation, tracking, or model failure.
        cleanup_distributed()


def train(config: TrainConfig) -> None:
    """Resolve, assemble, and run a complete training run.

    The execution plan is resolved (admission) once, assembled into runnable
    state by :func:`gcw.training.assembly.assemble_training_execution`, then the
    step loop runs. ``None`` from assembly means the run is already complete.
    """
    execution_plan = resolve_execution_plan(config)
    print0(f"execution strategy: {execution_plan.strategy}")
    execution = assemble_training_execution(config, execution_plan)
    if execution is None:
        return
    run_training_loop(execution)


def run_training_loop(plan: TrainingPlan) -> None:
    """Run the optimizer step loop over an assembled :class:`TrainingPlan`.

    Unpacks the plan into locals so the step body reads exactly as it did when
    it lived inside ``train``; mutable progress (smooth loss, step counters) is
    owned here, not on the frozen plan.
    """
    config = plan.config
    model = plan.model
    params = plan.params
    opt = plan.opt
    stream = plan.stream
    device = plan.device
    rank = plan.rank
    world_size = plan.world_size
    use_fsdp = plan.use_fsdp
    adaptive_batching = plan.adaptive_batching
    icarus_execution_scope = plan.icarus_execution_scope
    grad_accum_steps = plan.grad_accum_steps
    micro_tokens = plan.micro_tokens
    micro_rows = plan.micro_rows
    tokens_per_step = plan.tokens_per_step
    flops_per_tok_by_step = plan.flops_per_tok_by_step
    start_step = plan.start_step
    initial_processor_state = plan.initial_processor_state
    fingerprint = plan.fingerprint
    checkpoint_meta = plan.checkpoint_meta
    run_meta = plan.run_meta
    checkpoint_stop_policy = plan.checkpoint_stop_policy
    tracker = plan.tracker
    audit_output = plan.audit_output

    smooth_loss = plan.smooth_loss
    retained_checkpoint_every = _retained_checkpoint_every()
    last_checkpoint_step = start_step
    completed_step = start_step
    stopped_at_checkpoint = False
    audit_gradients = None

    if start_step == 0 and config.eval_on_start and config.eval_every:
        val = _eval_loss(model, config, device)
        if rank == 0:
            message = (
                f"step 0/{config.max_steps} val_loss={val['val/loss']:.4f}"
            )
            print0(message)
            if tracker is not None:
                tracker.log_message(message)
                tracker.log({"train/step": 0, **val}, step=0)

    model.train()
    flops_accumulated = tokens_per_step * sum(
        flops_per_tok_by_step[:start_step]
    )
    for step in range(start_step, config.max_steps):
        if config.longshot_selector_training is not None:
            set_longshot_selector_training_progress(
                model,
                config.longshot_selector_training,
                step / max(1, config.max_steps - 1),
            )
        step_start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        # Per-step accumulators for the metrics block below.
        loss_total = 0.0
        batch_non_padding_tokens = 0
        batch_supervised_tokens = 0
        segmented_window_checkpointed_microbatches = 0
        adaptive_checkpointing_decisions: list[bool] = []
        adaptive_padded_sequence_lengths: list[int] = []
        # Adaptive pair ladder (16/8/4/2) replaces plain gradient accumulation when enabled.
        adaptive_step_plan = (
            build_adaptive_pair_step(stream, adaptive_batching)
            if adaptive_batching.enabled
            else None
        )
        adaptive_physical_batches = (
            tuple(
                collate_pair_group(adaptive_step_plan, group)
                for group in adaptive_step_plan.groups
            )
            if adaptive_step_plan is not None
            else ()
        )
        physical_microsteps = (
            len(adaptive_physical_batches)
            if adaptive_step_plan is not None
            else grad_accum_steps
        )
        # Forward/backward over each micro-batch; the last micro-step syncs grads.
        for micro_step in range(physical_microsteps):
            batch = (
                adaptive_physical_batches[micro_step].batch
                if adaptive_step_plan is not None
                else next(stream)
            )
            if use_fsdp:
                batch = _align_fsdp_batch(batch)
            if adaptive_step_plan is None and step == 0 and micro_step == 0:
                _validate_train_batch(batch.x, batch.y, config)
            batch_non_padding_tokens += int(
                batch.x.numel()
                if batch.attention_mask is None
                else batch.attention_mask.sum().item()
            )
            batch_supervised_tokens += int((batch.y != -100).sum().item())
            checkpoint_length_eligible = (
                config.activation_checkpoint_min_padded_sequence_length == 0
                or batch.x.shape[1]
                >= config.activation_checkpoint_min_padded_sequence_length
            )
            segmented_window_checkpointed_microbatches += int(
                config.activation_checkpoint_windows > 0
                and checkpoint_length_eligible
            )
            longshot_checkpointing = _longshot_gradient_checkpointing_decision(
                config, batch.x
            )
            if longshot_checkpointing is not None:
                adaptive_checkpointing_decisions.append(longshot_checkpointing)
                adaptive_padded_sequence_lengths.append(batch.x.shape[1])
            sync_context = _no_sync(
                model,
                enabled=micro_step < physical_microsteps - 1,
                use_fsdp=use_fsdp,
            )
            with sync_context, icarus_execution_scope(batch.x.shape[0]):
                with torch.autocast(**_autocast_kwargs(config, device)):
                    if adaptive_step_plan is not None:
                        loss = adjacent_pair_loss_sum(
                            model,
                            batch.x,
                            batch.y,
                            loss_backend=config.loss_backend,
                            attention_mask=batch.attention_mask,
                            activation_checkpoint_windows=config.activation_checkpoint_windows,
                            activation_checkpoint_min_padded_sequence_length=(
                                config.activation_checkpoint_min_padded_sequence_length
                            ),
                            truncated_bptt_windows=config.truncated_bptt_windows,
                            longshot_gradient_checkpointing=longshot_checkpointing,
                        )
                    elif config.logical_pair_grouping:
                        loss = _logical_pair_grouped_loss(
                            model,
                            batch.x,
                            batch.y,
                            loss_backend=config.loss_backend,
                            attention_mask=batch.attention_mask,
                            activation_checkpoint_windows=config.activation_checkpoint_windows,
                            activation_checkpoint_min_padded_sequence_length=(
                                config.activation_checkpoint_min_padded_sequence_length
                            ),
                            truncated_bptt_windows=config.truncated_bptt_windows,
                            longshot_gradient_checkpointing=longshot_checkpointing,
                        )
                    else:
                        loss = _model_loss(
                            model,
                            batch.x,
                            batch.y,
                            loss_backend=config.loss_backend,
                            attention_mask=batch.attention_mask,
                            activation_checkpoint_windows=config.activation_checkpoint_windows,
                            activation_checkpoint_min_padded_sequence_length=(
                                config.activation_checkpoint_min_padded_sequence_length
                            ),
                            truncated_bptt_windows=config.truncated_bptt_windows,
                            longshot_gradient_checkpointing=longshot_checkpointing,
                        )
                normalized_loss = (
                    loss / PAIRS_PER_STEP
                    if adaptive_step_plan is not None
                    else loss / grad_accum_steps
                )
                loss_total += float(
                    normalized_loss.detach().cpu()
                    if adaptive_step_plan is not None
                    else loss.detach().cpu()
                )
                normalized_loss.backward()
        if audit_output is not None and step + 1 == config.max_steps:
            audit_gradients = _gradient_audit(model)
        # Gradient clipping, then per-group LR/weight-decay schedule before the step.
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
        completed_step = step_num
        stop_after_step = _checkpoint_stop_requested(
            checkpoint_stop_policy, step_num
        )
        step_seconds = time.perf_counter() - step_start
        tokens_seen = tokens_per_step * step_num
        step_flops_per_tok = flops_per_tok_by_step[step]
        flops_accumulated += tokens_per_step * step_flops_per_tok
        flops_seen = flops_accumulated
        # Reduce the per-step loss across ranks when logical pair grouping is active.
        train_loss = (
            loss_total
            if adaptive_step_plan is not None
            else loss_total / grad_accum_steps
        )
        if (
            config.logical_pair_grouping
            and dist.is_available()
            and dist.is_initialized()
        ):
            metric_totals = torch.tensor(
                [train_loss, batch_non_padding_tokens, batch_supervised_tokens],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(metric_totals, op=dist.ReduceOp.SUM)
            train_loss = float(metric_totals[0].cpu()) / world_size
            batch_non_padding_tokens = int(metric_totals[1].cpu())
            batch_supervised_tokens = int(metric_totals[2].cpu())
        smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
        debiased_loss = smooth_loss / (1.0 - 0.9**step_num)
        train_metrics = {
            "train/step": step_num,
            "train/loss": train_loss,
            "train/loss_smooth": debiased_loss,
            "train/perplexity": math.exp(min(20.0, train_loss)),
            "train/non_padding_tokens": batch_non_padding_tokens,
            "train/supervised_tokens": batch_supervised_tokens,
            "train/lr": config.learning_rate * lr_scale,
            "train/lr_scale": lr_scale,
            "train/tokens": tokens_seen,
            "train/flops": flops_seen,
            "train/progress": step_num / config.max_steps,
            "train/epoch": getattr(stream.cursor, "epoch", None),
            "train/grad_accum_steps": physical_microsteps,
            "train/nominal_grad_accum_steps": grad_accum_steps,
            "train/adaptive_batching_active": float(
                adaptive_step_plan is not None
            ),
            "train/micro_tokens_per_step": micro_tokens,
            "train/tokens_per_step": tokens_per_step,
            "train/rows_per_step": micro_rows * grad_accum_steps,
            "train/logical_pairs_per_step": (
                config.global_batch_rows // 2
                if config.logical_pair_grouping
                else None
            ),
            "train/segmented_window_checkpointing_active": (
                segmented_window_checkpointed_microbatches > 0
            ),
            "train/segmented_window_checkpointed_microbatches": (
                segmented_window_checkpointed_microbatches
            ),
            "train/segmented_window_eager_microbatches": (
                physical_microsteps - segmented_window_checkpointed_microbatches
            ),
            "train/training_method": (
                "truncated_bptt"
                if config.truncated_bptt_windows
                else "full_bptt"
            ),
            "train/truncated_bptt_windows": config.truncated_bptt_windows,
            "train/truncated_bptt_enabled": float(
                config.truncated_bptt_windows > 0
            ),
            "timing/step_seconds": step_seconds,
            "timing/tokens_per_second": tokens_per_step
            / max(step_seconds, 1e-12),
            "timing/flops_per_second": (step_flops_per_tok * tokens_per_step)
            / max(step_seconds, 1e-12),
            "optim/grad_norm": _maybe_float(grad_norm),
            "optim/weight_decay": weight_decay,
        }
        if adaptive_step_plan is not None:
            train_metrics.update(adaptive_step_plan.metrics())
        if device.type == "cuda":
            train_metrics["system/vram_allocated_mib"] = (
                torch.cuda.max_memory_allocated(device) / 1024 / 1024
            )
            train_metrics["system/vram_reserved_mib"] = (
                torch.cuda.max_memory_reserved(device) / 1024 / 1024
            )
        train_metrics.update(_cache_processor_metrics(model))
        train_metrics.update(
            _longshot_gradient_checkpointing_step_metrics(
                adaptive_checkpointing_decisions,
                adaptive_padded_sequence_lengths,
            )
        )
        if (
            not stop_after_step
            and config.eval_every
            and step_num % config.eval_every == 0
        ):
            train_metrics.update(_eval_loss(model, config, device))
        if rank == 0:
            msg = f"step {step_num}/{config.max_steps} loss={train_metrics['train/loss_smooth']:.4f}"
            if "val/loss" in train_metrics:
                msg += f" val_loss={train_metrics['val/loss']:.4f}"
            print0(msg)
            if tracker is not None:
                tracker.log_message(msg)
                tracker.log(_compact_metrics(train_metrics), step=step_num)
            if audit_output is not None and step_num == config.max_steps:
                if audit_gradients is None:
                    raise RuntimeError(
                        "training audit gradients were not captured"
                    )
                _write_training_audit(
                    Path(audit_output),
                    {
                        **audit_gradients,
                        "architecture": config.model.architecture_name,
                        "step": step_num,
                        "tokens_per_second": train_metrics[
                            "timing/tokens_per_second"
                        ],
                        "peak_memory_mib": train_metrics.get(
                            "system/vram_allocated_mib", 0.0
                        ),
                        "source_commit": os.environ.get(
                            "GCW_AUDIT_SOURCE_COMMIT"
                        ),
                        "experiment_manifest_sha256": os.environ.get(
                            "GCW_AUDIT_MANIFEST_SHA256"
                        ),
                    },
                )

        if stop_after_step or (
            config.save_every and step_num % config.save_every == 0
        ):
            _save_step_checkpoint(
                Path(config.output_dir) / "checkpoint",
                model,
                opt,
                config,
                step_num,
                smooth_loss,
                stream,
                world_size,
                checkpoint_meta,
                fingerprint,
                rank,
                use_fsdp=use_fsdp,
                initial_processor_state=initial_processor_state,
            )
            last_checkpoint_step = step_num
            if step_num == config.max_steps:
                if rank == 0:
                    _retain_final_checkpoint(
                        Path(config.output_dir) / "checkpoint", step_num
                    )
            elif (
                retained_checkpoint_every is not None
                and step_num % retained_checkpoint_every == 0
            ):
                _save_step_checkpoint(
                    Path(config.output_dir) / f"checkpoint-{step_num}",
                    model,
                    opt,
                    config,
                    step_num,
                    smooth_loss,
                    stream,
                    world_size,
                    checkpoint_meta,
                    fingerprint,
                    rank,
                    use_fsdp=use_fsdp,
                    initial_processor_state=initial_processor_state,
                )
            if stop_after_step:
                stopped_at_checkpoint = True
                print0(
                    f"stopping after authenticated checkpoint step={step_num}"
                )
                break

    if (
        not stopped_at_checkpoint
        and config.save_final
        and last_checkpoint_step != config.max_steps
    ):
        _save_step_checkpoint(
            Path(config.output_dir) / "checkpoint",
            model,
            opt,
            config,
            config.max_steps,
            smooth_loss,
            stream,
            world_size,
            checkpoint_meta,
            fingerprint,
            rank,
            use_fsdp=use_fsdp,
            initial_processor_state=initial_processor_state,
        )
        if rank == 0:
            _retain_final_checkpoint(
                Path(config.output_dir) / "checkpoint", config.max_steps
            )
    _close_training_stream(stream)
    if tracker is not None:
        tracker.finish(
            summary={
                **run_meta,
                "completed_step": completed_step,
                "planned_max_steps": config.max_steps,
                "stopped_at_checkpoint": stopped_at_checkpoint,
            }
        )
    cleanup_distributed()


@torch.no_grad()
def _eval_loss(
    model: torch.nn.Module, config: TrainConfig, device: torch.device
) -> dict[str, float]:
    validation_stream: TrainingBatchStream | None = None
    if config.mode == "sft" and config.data.validation_dataset is not None:
        validation_config = deepcopy(config)
        validation_config.data.dataset = config.data.validation_dataset
        validation_config.data.sft_skip_rows = 0
        validation_config.data.sft_take_rows = config.batch_size * max(
            1, config.eval_batches
        )
        validation_stream = build_training_stream(
            validation_config, device, rank=0, world_size=1
        )
        loader = _training_batch_loader(validation_stream)
    elif config.mode != "pretrain" or config.data.kind == "random":
        loader = _random_loader(config, device)
    else:
        loader = data_loader(
            config.batch_size,
            config.sequence_len,
            split="val",
            device=device,
            data_dir=config.data.data_dir,
            infinite=True,
            prefix_tokens=fineweb_prefix_tokens(config),
        )
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_weight = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device)
    use_fsdp = _fsdp_active(config)
    for _ in range(max(1, config.eval_batches)):
        x, y, attention_mask = next(loader)
        if use_fsdp:
            batch = _align_fsdp_batch(TrainingBatch(x, y, attention_mask))
            x, y, attention_mask = batch.x, batch.y, batch.attention_mask
        longshot_checkpointing = _longshot_gradient_checkpointing_decision(
            config, x
        )
        with torch.autocast(**_autocast_kwargs(config, device)):
            grouped_attention_mask = (
                attention_mask
                if isinstance(attention_mask, torch.Tensor)
                else None
            )
            if config.logical_pair_grouping:
                row_sums, row_counts = _row_loss_sums_and_counts(
                    model,
                    x,
                    y,
                    loss_backend=config.loss_backend,
                    attention_mask=grouped_attention_mask,
                    activation_checkpoint_windows=config.activation_checkpoint_windows,
                    activation_checkpoint_min_padded_sequence_length=(
                        config.activation_checkpoint_min_padded_sequence_length
                    ),
                    longshot_gradient_checkpointing=longshot_checkpointing,
                )
                pair_counts = row_counts.reshape(-1, 2).sum(dim=1)
                loss = (
                    row_sums.reshape(-1, 2).sum(dim=1)
                    / pair_counts.clamp_min(1.0)
                ).sum()
                weight = x.new_tensor(x.shape[0] // 2, dtype=torch.float32)
            else:
                loss = _model_loss(
                    model,
                    x,
                    y,
                    loss_reduction="sum",
                    loss_backend=config.loss_backend,
                    attention_mask=grouped_attention_mask,
                    activation_checkpoint_windows=config.activation_checkpoint_windows,
                    activation_checkpoint_min_padded_sequence_length=(
                        config.activation_checkpoint_min_padded_sequence_length
                    ),
                    longshot_gradient_checkpointing=longshot_checkpointing,
                )
                weight = (y >= 0).sum()
        total_loss += loss.detach().float()
        total_weight += weight
        total_tokens += (y >= 0).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_weight, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    if validation_stream is not None:
        _close_training_stream(validation_stream)
    if was_training:
        model.train()
    if not config.logical_pair_grouping:
        total_weight = total_tokens
    val_loss = float((total_loss / total_weight.clamp_min(1)).cpu())
    return {
        "val/loss": val_loss,
        "val/perplexity": math.exp(min(20.0, val_loss)),
        "val/tokens": float(total_tokens.cpu()),
    }


def _close_training_stream(stream: TrainingBatchStream) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        close()


def _training_batch_loader(stream: TrainingBatchStream):
    while True:
        batch = next(stream)
        yield batch.x, batch.y, batch.attention_mask


def _random_loader(config: TrainConfig, device: torch.device):
    vocab = config.model.vocab_size
    while True:
        x = torch.randint(
            0, vocab, (config.batch_size, config.sequence_len), device=device
        )
        y = torch.roll(x, shifts=-1, dims=1)
        yield x, y, {"kind": "random"}
