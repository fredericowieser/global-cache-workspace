from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gcw.architecture import admission_receipt_metadata, resolve_experiment, resolve_model
from gcw.checkpoint import load_backbone_checkpoint, load_checkpoint, save_checkpoint
from gcw.experiment import validate_experiment
from gcw.runtime import (
    cleanup_distributed,
    dist_info,
    init_distributed,
    print0,
)
from gcw.config import TrainConfig, load_train_config, to_dict
from gcw.data.shards import data_loader
from gcw.data.training import TrainingBatchStream, build_training_stream, cursor_to_dict, training_data_identity
from gcw.flops import total_training_flops, training_flops_per_token
from gcw.models import build_model
from gcw.performance.adjacent_pairs import (
    PAIRS_PER_STEP,
    adaptive_batching_metadata,
    build_adaptive_pair_step,
    collate_pair_group,
)
from gcw.performance.backends.qwen3 import (
    configure_qwen3_backbone_chunk_backend,
    qwen3_backbone_chunk_backend_metadata,
)
from gcw.performance.checkpointing import (
    apply_fsdp_activation_checkpointing as _apply_fsdp_activation_checkpointing,
    configure_native_activation_checkpointing as _configure_native_activation_checkpointing,
    longshot_gradient_checkpointing_decision as _longshot_gradient_checkpointing_decision,
    longshot_gradient_checkpointing_metadata as _longshot_gradient_checkpointing_runtime,
    longshot_gradient_checkpointing_step_metrics as _longshot_gradient_checkpointing_step_metrics,
    transformer_layer_classes as _transformer_layer_classes,
)
from gcw.models.lora import LoRALinear
from gcw.performance.backends.cache_processors import (
    bt_processor_backend_metadata,
    cache_processor_backend_metadata,
    resolve_cache_processor_backend,
)
from gcw.performance.backends.diffusion import (
    configure_noisy_diffusion_recurrence_backend,
    noisy_diffusion_recurrence_backend_metadata,
)
from gcw.performance.backends.icarus import (
    configure_icarus_training_execution,
    icarus_processor_backend_metadata,
    icarus_training_execution,
    resolve_icarus_processor_backend,
    validate_icarus_training_admission,
    warmup_icarus_training_execution,
)
from gcw.performance.longshot_cache import (
    configure_longshot_full_history_execution,
    longshot_full_history_execution_metadata,
)
from gcw.performance.kv_score_sidecar import (
    KVScoreSidecarExecutionContext,
    configure_kv_score_sidecar_execution,
    kv_score_sidecar_execution_metadata,
)
from gcw.performance.whole_model import maybe_compile_whole_model, whole_model_compile_requested
from gcw.performance.losses import (
    configure_reference_loss_execution,
    loss_backend_metadata,
    reference_loss_execution_metadata,
    require_loss_backend,
)
from gcw.performance.telemetry import (
    cache_processor_metrics as _cache_processor_metrics,
    model_runtime_backends as _model_runtime_backends,
)
from gcw.params import processor_params, total_non_embedding_params
from gcw.tracking import RunTracker
from gcw.training.checkpoint import ResumeState, TrainingCheckpoint, save_training_checkpoint
from gcw.training.objectives import (
    adjacent_pair_loss_sum,
    logical_pair_grouped_loss as _logical_pair_grouped_loss,
    model_loss as _model_loss,
    row_loss_sums_and_counts as _row_loss_sums_and_counts,
)


def train_from_config(path: str | Path) -> None:
    train(load_train_config(path))


def train(config: TrainConfig) -> None:
    validate_experiment(config)
    _, model_receipt = resolve_model(config.model).require()
    resolve_experiment(config, model_receipt).require()
    torch.manual_seed(config.seed)
    device = init_distributed(config.distributed)
    require_loss_backend(config.loss_backend, device)
    _, rank, _, world_size = dist_info()
    adaptive_batching = config.performance.adaptive_batching
    if adaptive_batching.enabled and (
        world_size != 1
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
        or (dist.is_available() and dist.is_initialized())
    ):
        raise ValueError("adaptive batching v1 requires one process and no initialized distributed group")
    micro_tokens = config.batch_size * config.sequence_len * world_size
    micro_rows = config.batch_size * world_size
    if config.global_batch_rows:
        if config.global_batch_rows % micro_rows != 0:
            raise ValueError(
                f"global_batch_rows {config.global_batch_rows} must be divisible by micro batch rows {micro_rows}"
            )
        grad_accum_steps = config.global_batch_rows // micro_rows
        tokens_per_step = grad_accum_steps * micro_tokens
    else:
        tokens_per_step = config.total_batch_size or micro_tokens
        if tokens_per_step % micro_tokens != 0:
            raise ValueError(
                f"total_batch_size {tokens_per_step} must be divisible by micro batch tokens {micro_tokens}"
            )
        grad_accum_steps = tokens_per_step // micro_tokens
    fingerprint = _training_fingerprint(config, world_size, micro_tokens, tokens_per_step, grad_accum_steps)
    audit_output = os.environ.get("GCW_TRAIN_AUDIT_OUT")
    if audit_output is not None and world_size != 1:
        raise ValueError("training audit currently requires the approved one-GPU topology")
    audit_gradients: dict[str, Any] | None = None
    use_fsdp = _fsdp_active(config)
    icarus_backend = resolve_icarus_processor_backend(config.model)
    validate_icarus_training_admission(
        icarus_backend,
        distributed=config.distributed,
        world_size=max(world_size, int(os.environ.get("WORLD_SIZE", "1"))),
        distributed_initialized=dist.is_available() and dist.is_initialized(),
        batch_size=config.batch_size,
        adaptive_batch_sizes=(
            tuple(band.physical_rows for band in adaptive_batching.bands)
            if adaptive_batching.enabled
            else None
        ),
        device=device,
    )
    if use_fsdp and config.resume is not None:
        raise ValueError("exact training resume is not supported with FSDP")

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
            expected_manifest_sha256=config.resume.expected_manifest_sha256,
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

    initial_processor_state: dict[str, torch.Tensor] | None = None
    if use_fsdp:
        model = _build_training_model(config, device, rank=rank, use_fsdp=True)
    else:
        model = build_model(config.model).to(device)
        model.init_weights()
        if checkpoint is not None:
            checkpoint.restore_model(model)
        elif config.init_checkpoint:
            _load_initial_model(model, config.init_checkpoint)
        elif config.backbone_checkpoint:
            load_backbone_checkpoint(model, config.backbone_checkpoint)
        if checkpoint is None:
            initial_processor_state = _processor_state(model)
    _apply_trainable_policy(model, config.trainable)
    activation_checkpointing_runtime = _configure_native_activation_checkpointing(
        model, config.activation_checkpointing
    )
    configure_qwen3_backbone_chunk_backend(model, config.backbone_chunk_backend)
    configure_reference_loss_execution(model, config.performance.reference_loss)
    configure_longshot_full_history_execution(
        model, config.performance.longshot_full_history
    )
    configure_kv_score_sidecar_execution(
        model,
        config.performance.kv_score_sidecar,
        context=KVScoreSidecarExecutionContext(
            model_variant=config.model.variant,
            model_family=config.model.family,
            physical_batch_size=config.batch_size,
            distributed=config.distributed,
            backbone_chunk_backend=config.backbone_chunk_backend,
            adaptive_batching_enabled=config.performance.adaptive_batching.enabled,
            longshot_full_history_enabled=config.performance.longshot_full_history.enabled,
            whole_model_compile_requested=whole_model_compile_requested(),
            activation_checkpointing=config.activation_checkpointing,
            incompatible_refinement_enabled=any(
                value is not None
                for value in (
                    config.model.flow_processor,
                    config.model.diffusion_processor,
                    config.model.workspace_flow_refinement,
                )
            ),
        ),
    )
    noisy_diffusion_backend = configure_noisy_diffusion_recurrence_backend(
        model,
        config.model,
        config.performance.noisy_diffusion,
    )
    if use_fsdp:
        _apply_fsdp_activation_checkpointing(model, family=config.model.family)
        model = _wrap_fsdp(model, config, device)
    else:
        if whole_model_compile_requested() and config.performance.longshot_full_history.enabled:
            raise RuntimeError("Longshot selected-history execution cannot be combined with GCW_COMPILE")
        if whole_model_compile_requested() and config.performance.kv_score_sidecar.enabled:
            raise RuntimeError("KV-score sidecar cannot be combined with GCW_COMPILE")
        if whole_model_compile_requested() and config.performance.reference_loss.mode != "legacy":
            raise RuntimeError("bounded reference-loss execution cannot be combined with GCW_COMPILE")
        regional_backend = resolve_cache_processor_backend(config.model)
        if whole_model_compile_requested() and (
            config.model.variant == "bt"
            or config.backbone_chunk_backend != "eager"
            or regional_backend == "compile"
            or noisy_diffusion_backend == "compile"
            or icarus_backend == "compile"
        ):
            raise RuntimeError(
                "GCW_COMPILE cannot wrap a regionally compiled language model; disable it and use "
                "backbone_chunk_backend=compile or compile_full_windows, "
                "GCW_BT_PROCESSOR_BACKEND=compile, GCW_CACHE_PROCESSOR_BACKEND=compile, "
                "performance.noisy_diffusion.recurrence_backend=compile, and/or "
                "GCW_ICARUS_PROCESSOR_BACKEND=compile"
            )
        if (
            config.model.variant != "bt"
            and config.backbone_chunk_backend == "eager"
            and regional_backend == "eager"
            and noisy_diffusion_backend == "eager"
            and icarus_backend == "eager"
        ):
            model = maybe_compile_whole_model(model)
        if config.distributed == "ddp" and torch.distributed.is_initialized():
            model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    icarus_admitted_batches = (
        tuple(band.physical_rows for band in adaptive_batching.bands)
        if adaptive_batching.enabled
        else (config.batch_size,)
    )
    if icarus_backend == "compile":
        configure_icarus_training_execution(model, icarus_admitted_batches)
        warmup_icarus_training_execution(
            model,
            icarus_admitted_batches,
            sequence_length=config.sequence_len,
            device=device,
            bf16=config.bf16,
        )
    icarus_execution_scope = (
        partial(icarus_training_execution, model) if icarus_backend == "compile" else nullcontext
    )

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
        "batch_size": config.batch_size,
        "sequence_len": config.sequence_len,
        "micro_tokens_per_step": micro_tokens,
        "grad_accum_steps": grad_accum_steps,
        "tokens_per_step": tokens_per_step,
        "rows_per_step": micro_rows * grad_accum_steps,
        "logical_pair_grouping": config.logical_pair_grouping,
        "logical_pairs_per_step": config.global_batch_rows // 2 if config.logical_pair_grouping else None,
        "adaptive_batching": adaptive_batching_metadata(
            adaptive_batching, global_batch_rows=config.global_batch_rows
        ),
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
        "model_runtime_backends": _model_runtime_backends(model, device),
        "bt_processor_backend": bt_processor_backend_metadata() if config.model.variant == "bt" else None,
        "qwen3_backbone_chunk_backend": qwen3_backbone_chunk_backend_metadata(config.backbone_chunk_backend),
        "cache_processor_backend": (
            cache_processor_backend_metadata(config.model) if config.model.variant == "bgt" else None
        ),
        "noisy_diffusion_recurrence_backend": (
            noisy_diffusion_recurrence_backend_metadata(
                config.model, config.performance.noisy_diffusion
            )
            if config.model.diffusion_processor is not None
            else None
        ),
        "icarus_processor_backend": (
            icarus_processor_backend_metadata(config.model)
            if config.model.variant == "icarus"
            else None
        ),
        "activation_checkpointing": activation_checkpointing_runtime,
        "activation_checkpoint_windows": config.activation_checkpoint_windows,
        "activation_checkpoint_min_padded_sequence_length": (
            config.activation_checkpoint_min_padded_sequence_length
        ),
        "longshot_gradient_checkpointing": _longshot_gradient_checkpointing_runtime(config),
        "loss_backend": loss_backend_metadata(config.loss_backend),
        **(
            {
                "longshot_full_history_execution": longshot_full_history_execution_metadata(
                    config.performance.longshot_full_history,
                    unrope_keys=bool(
                        config.model.cache_processor and config.model.cache_processor.unrope_keys
                    ),
                )
            }
            if config.performance.longshot_full_history.enabled
            else {}
        ),
        **(
            {
                "kv_score_sidecar_execution": kv_score_sidecar_execution_metadata(
                    config.performance.kv_score_sidecar,
                    unrope_keys=bool(
                        config.model.cache_processor and config.model.cache_processor.unrope_keys
                    ),
                )
            }
            if config.performance.kv_score_sidecar.enabled
            else {}
        ),
        **(
            {
                "reference_loss_execution": reference_loss_execution_metadata(
                    config.performance.reference_loss
                )
            }
            if config.performance.reference_loss.mode != "legacy"
            else {}
        ),
    }
    checkpoint_meta = _checkpoint_metadata(config, run_meta)
    budget_message = (
        "training budget: "
        f"tokens={target_tokens:,} flops/token={flops_per_tok:.6e} "
        f"total_flops={run_meta['total_training_flops']:.6e} "
        f"non_embedding_params={run_meta['non_embedding_params']:,} "
        f"micro_tokens={micro_tokens:,} rows/step={micro_rows * grad_accum_steps:,} "
        f"grad_accum={grad_accum_steps}"
    )
    print0(budget_message)
    tracker = RunTracker.for_training(config, run_meta, model=model) if rank == 0 else None
    if tracker is not None:
        tracker.log_message(budget_message)
    if start_step == 0 and config.eval_on_start and config.eval_every:
        val = _eval_loss(model, config, device)
        if rank == 0:
            message = f"step 0/{config.max_steps} val_loss={val['val/loss']:.4f}"
            print0(message)
            if tracker is not None:
                tracker.log_message(message)
                tracker.log({"train/step": 0, **val}, step=0)

    retained_checkpoint_every = _retained_checkpoint_every()
    last_checkpoint_step = start_step
    model.train()
    for step in range(start_step, config.max_steps):
        step_start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        batch_non_padding_tokens = 0
        batch_supervised_tokens = 0
        segmented_window_checkpointed_microbatches = 0
        adaptive_checkpointing_decisions: list[bool] = []
        adaptive_padded_sequence_lengths: list[int] = []
        adaptive_step_plan = (
            build_adaptive_pair_step(stream, adaptive_batching) if adaptive_batching.enabled else None
        )
        adaptive_physical_batches = (
            tuple(collate_pair_group(adaptive_step_plan, group) for group in adaptive_step_plan.groups)
            if adaptive_step_plan is not None
            else ()
        )
        physical_microsteps = len(adaptive_physical_batches) if adaptive_step_plan is not None else grad_accum_steps
        for micro_step in range(physical_microsteps):
            batch = (
                adaptive_physical_batches[micro_step].batch
                if adaptive_step_plan is not None
                else next(stream)
            )
            if adaptive_step_plan is None and step == 0 and micro_step == 0:
                _validate_train_batch(batch.x, batch.y, config)
            batch_non_padding_tokens += int(
                batch.x.numel() if batch.attention_mask is None else batch.attention_mask.sum().item()
            )
            batch_supervised_tokens += int((batch.y != -100).sum().item())
            checkpoint_length_eligible = (
                config.activation_checkpoint_min_padded_sequence_length == 0
                or batch.x.shape[1] >= config.activation_checkpoint_min_padded_sequence_length
            )
            segmented_window_checkpointed_microbatches += int(
                config.activation_checkpoint_windows > 0 and checkpoint_length_eligible
            )
            longshot_checkpointing = _longshot_gradient_checkpointing_decision(config, batch.x)
            if longshot_checkpointing is not None:
                adaptive_checkpointing_decisions.append(longshot_checkpointing)
                adaptive_padded_sequence_lengths.append(batch.x.shape[1])
            sync_context = _no_sync(
                model,
                enabled=micro_step < physical_microsteps - 1,
                use_fsdp=use_fsdp,
            )
            with sync_context, icarus_execution_scope(batch.x.shape[0]):
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=config.bf16 and device.type == "cuda",
                ):
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
                            longshot_gradient_checkpointing=longshot_checkpointing,
                        )
                normalized_loss = (
                    loss / PAIRS_PER_STEP if adaptive_step_plan is not None else loss / grad_accum_steps
                )
                loss_total += float(
                    normalized_loss.detach().cpu() if adaptive_step_plan is not None else loss.detach().cpu()
                )
                normalized_loss.backward()
        if audit_output is not None and step + 1 == config.max_steps:
            audit_gradients = _gradient_audit(model)
        grad_norm = None
        if config.grad_clip:
            grad_norm = _clip_grad_norm(model, params, config.grad_clip, use_fsdp=use_fsdp)
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
        train_loss = loss_total if adaptive_step_plan is not None else loss_total / grad_accum_steps
        if config.logical_pair_grouping and dist.is_available() and dist.is_initialized():
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
            "train/adaptive_batching_active": float(adaptive_step_plan is not None),
            "train/micro_tokens_per_step": micro_tokens,
            "train/tokens_per_step": tokens_per_step,
            "train/rows_per_step": micro_rows * grad_accum_steps,
            "train/logical_pairs_per_step": (
                config.global_batch_rows // 2 if config.logical_pair_grouping else None
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
            "timing/step_seconds": step_seconds,
            "timing/tokens_per_second": tokens_per_step / max(step_seconds, 1e-12),
            "timing/flops_per_second": (flops_per_tok * tokens_per_step) / max(step_seconds, 1e-12),
            "optim/grad_norm": _maybe_float(grad_norm),
            "optim/weight_decay": weight_decay,
        }
        if adaptive_step_plan is not None:
            train_metrics.update(adaptive_step_plan.metrics())
        if device.type == "cuda":
            train_metrics["system/vram_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            train_metrics["system/vram_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 1024 / 1024
        train_metrics.update(_cache_processor_metrics(model))
        train_metrics.update(
            _longshot_gradient_checkpointing_step_metrics(
                adaptive_checkpointing_decisions,
                adaptive_padded_sequence_lengths,
            )
        )
        if config.eval_every and step_num % config.eval_every == 0:
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
                    raise RuntimeError("training audit gradients were not captured")
                _write_training_audit(
                    Path(audit_output),
                    {
                        **audit_gradients,
                        "architecture": config.model.architecture_name,
                        "step": step_num,
                        "tokens_per_second": train_metrics["timing/tokens_per_second"],
                        "peak_memory_mib": train_metrics.get("system/vram_allocated_mib", 0.0),
                        "source_commit": os.environ.get("GCW_AUDIT_SOURCE_COMMIT"),
                        "experiment_manifest_sha256": os.environ.get("GCW_AUDIT_MANIFEST_SHA256"),
                    },
                )

        if config.save_every and step_num % config.save_every == 0:
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
            if retained_checkpoint_every is not None and step_num % retained_checkpoint_every == 0:
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

    if config.save_final and last_checkpoint_step != config.max_steps:
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
    if tracker is not None:
        tracker.finish(summary={**run_meta, "completed_step": config.max_steps})
    cleanup_distributed()


def _checkpoint_metadata(config: TrainConfig, run_meta: dict[str, Any]) -> dict[str, Any]:
    state = {
        **run_meta,
        "data_kind": config.data.kind,
        "tokenizer": config.data.tokenizer,
        "dtype": "bfloat16" if config.bf16 else "float32",
        "loss_backend": loss_backend_metadata(config.loss_backend),
    }
    if config.data.dataset is not None:
        state["dataset"] = config.data.dataset
    if config.data.data_dir is not None:
        state["data_dir"] = config.data.data_dir
    if config.backbone_checkpoint is not None:
        state["backbone_checkpoint"] = config.backbone_checkpoint
    if config.init_checkpoint is not None:
        state["init_checkpoint"] = config.init_checkpoint
    if config.data.tokenizer_source is not None:
        state["source"] = config.data.tokenizer_source
        state["tokenizer_source"] = config.data.tokenizer_source
    state["admission"] = admission_receipt_metadata(config)
    return state


def _load_initial_model(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    """Load a complete model state without restoring optimizer or run state."""
    source, source_config, _ = load_checkpoint(checkpoint_path, device="cpu")
    model_config = getattr(model, "config", None)
    if not isinstance(model_config, type(source_config)) or to_dict(model_config) != to_dict(source_config):
        raise ValueError("init_checkpoint model configuration differs from train.model")
    model.load_state_dict(source.state_dict(), strict=True)


def _apply_trainable_policy(model: torch.nn.Module, policy: str) -> None:
    """Select trainable ownership before wrapping or optimizer construction."""
    backbone = getattr(model, "backbone", model)
    processor = getattr(model, "cache_processor", None)
    if policy == "all":
        model.requires_grad_(True)
        return
    if policy == "backbone":
        backbone.requires_grad_(True)
        if isinstance(processor, torch.nn.Module):
            processor.requires_grad_(False)
        return
    if policy == "lora":
        model.requires_grad_(False)
        adapters = [module for module in backbone.modules() if isinstance(module, LoRALinear)]
        if not adapters:
            raise ValueError("trainable='lora' requires attached LoRA projections")
        for adapter in adapters:
            adapter.lora_a.requires_grad_(True)
            adapter.lora_b.requires_grad_(True)
        return
    if policy == "cache_processor":
        if not isinstance(processor, torch.nn.Module):
            raise ValueError("trainable='cache_processor' requires a cache processor")
        backbone.requires_grad_(False)
        processor.requires_grad_(True)
        return
    raise ValueError(f"unsupported trainable policy: {policy}")




def _checkpoint_state(base: dict[str, Any], step: int, loader: dict[str, Any] | None = None) -> dict[str, Any]:
    state = {**base, "step": step}
    if loader is not None:
        state["loader"] = _json_safe_loader_state(loader)
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
        "bt_processor_backend_fingerprint": (
            bt_processor_backend_metadata()["fingerprint_sha256"] if config.model.variant == "bt" else None
        ),
        "qwen3_backbone_chunk_backend_fingerprint": qwen3_backbone_chunk_backend_metadata(
            config.backbone_chunk_backend
        )["fingerprint_sha256"],
        "cache_processor_backend_fingerprint": (
            cache_processor_backend_metadata(config.model)["fingerprint_sha256"]
            if config.model.variant == "bgt"
            else None
        ),
        **(
            {
                "noisy_diffusion_recurrence_backend_fingerprint": (
                    noisy_diffusion_recurrence_backend_metadata(
                        config.model, config.performance.noisy_diffusion
                    )["fingerprint_sha256"]
                )
            }
            if config.performance.noisy_diffusion.recurrence_backend == "compile"
            else {}
        ),
        **(
            {
                "icarus_processor_backend_fingerprint": icarus_processor_backend_metadata(
                    config.model
                )["fingerprint_sha256"]
            }
            if (
                config.model.variant == "icarus"
                and resolve_icarus_processor_backend(config.model) == "compile"
            )
            else {}
        ),
        "initialization": {
            "backbone_checkpoint": config.backbone_checkpoint,
            "init_checkpoint": config.init_checkpoint,
        },
        "trainable": config.trainable,
        "activation_checkpointing": config.activation_checkpointing,
        "activation_checkpoint_windows": config.activation_checkpoint_windows,
        "activation_checkpoint_min_padded_sequence_length": (
            config.activation_checkpoint_min_padded_sequence_length
        ),
        "loss_backend": config.loss_backend,
        **(
            {
                "adaptive_batching": adaptive_batching_metadata(
                    config.performance.adaptive_batching,
                    global_batch_rows=config.global_batch_rows,
                )
            }
            if config.performance.adaptive_batching.enabled
            else {}
        ),
        **(
            {
                "reference_loss_execution": reference_loss_execution_metadata(
                    config.performance.reference_loss
                )
            }
            if config.performance.reference_loss.mode != "legacy"
            else {}
        ),
        **(
            {
                "longshot_full_history_execution": longshot_full_history_execution_metadata(
                    config.performance.longshot_full_history,
                    unrope_keys=bool(
                        config.model.cache_processor and config.model.cache_processor.unrope_keys
                    ),
                )
            }
            if config.performance.longshot_full_history.enabled
            else {}
        ),
        **(
            {
                "kv_score_sidecar_execution": kv_score_sidecar_execution_metadata(
                    config.performance.kv_score_sidecar,
                    unrope_keys=bool(
                        config.model.cache_processor and config.model.cache_processor.unrope_keys
                    ),
                )
            }
            if config.performance.kv_score_sidecar.enabled
            else {}
        ),
        **(
            {
                "longshot_gradient_checkpointing_seq_len_threshold": (
                    config.longshot_gradient_checkpointing_seq_len_threshold
                )
            }
            if config.longshot_gradient_checkpointing_seq_len_threshold is not None
            else {}
        ),
        "objective": (
            "adjacent_pair_token_mean_answer_only_ce"
            if config.logical_pair_grouping
            else f"{config.data.sft_loss_scope}_ce"
            if config.mode == "sft"
            else "next_token_ce"
        ),
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
            "global_batch_rows": config.global_batch_rows,
            "logical_pair_grouping": config.logical_pair_grouping,
            "logical_pairs_per_step": config.global_batch_rows // 2 if config.logical_pair_grouping else None,
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


def _processor_state(model: torch.nn.Module) -> dict[str, torch.Tensor] | None:
    processor = getattr(model, "cache_processor", None)
    if not isinstance(processor, torch.nn.Module):
        return None
    return {name: tensor.detach().cpu().clone() for name, tensor in processor.state_dict().items()}


def _save_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    resume_state: ResumeState,
    fingerprint: dict[str, Any],
    rank: int,
    initial_processor_state: dict[str, torch.Tensor] | None = None,
) -> None:
    if config.distributed == "fsdp":
        if rank == 0:
            save_checkpoint(path, model, config.model, {**resume_state.metadata, "step": resume_state.step})
        return
    checkpoint_kwargs = (
        {"initial_processor_state": initial_processor_state} if initial_processor_state is not None else {}
    )
    save_training_checkpoint(
        path,
        model,
        config.model,
        optimizer,
        resume_state,
        fingerprint,
        **checkpoint_kwargs,
    )


def _save_step_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    step: int,
    smooth_loss: float,
    stream: TrainingBatchStream,
    world_size: int,
    metadata: dict[str, Any],
    fingerprint: dict[str, Any],
    rank: int,
    *,
    use_fsdp: bool,
    initial_processor_state: dict[str, torch.Tensor] | None,
) -> None:
    if use_fsdp:
        state = _checkpoint_state(metadata, step, cursor_to_dict(stream.cursor))
        _save_training_checkpoint(path, model, config.model, state, use_fsdp=True)
        return
    resume_state = _resume_state(step, smooth_loss, stream, world_size, metadata)
    if initial_processor_state is None:
        _save_training_state(
            path,
            _unwrap(model),
            optimizer,
            config,
            resume_state,
            fingerprint,
            rank,
        )
    else:
        _save_training_state(
            path,
            _unwrap(model),
            optimizer,
            config,
            resume_state,
            fingerprint,
            rank,
            initial_processor_state=initial_processor_state,
        )


def _json_safe_loader_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return None
    if isinstance(value, dict):
        return {k: safe for k, v in value.items() if (safe := _json_safe_loader_state(v)) is not None}
    if isinstance(value, list):
        return [safe for item in value if (safe := _json_safe_loader_state(item)) is not None]
    if isinstance(value, tuple):
        return [safe for item in value if (safe := _json_safe_loader_state(item)) is not None]
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
        raise ValueError("GCW_STEP_CHECKPOINT_EVERY must be an integer") from exc
    if every <= 0:
        raise ValueError("GCW_STEP_CHECKPOINT_EVERY must be positive")
    return every


def _validate_train_batch(x: torch.Tensor, y: torch.Tensor, config: TrainConfig) -> None:
    if x.shape != y.shape:
        raise ValueError(f"input and target batch shapes must match; got x={tuple(x.shape)} y={tuple(y.shape)}")
    if x.ndim != 2:
        raise ValueError(f"training batches must be 2D token tensors; got shape={tuple(x.shape)}")
    if x.shape[0] != config.batch_size:
        raise ValueError(f"loader batch size {x.shape[0]} does not match config.batch_size {config.batch_size}")
    if x.shape[1] > config.sequence_len:
        raise ValueError(f"loader sequence length {x.shape[1]} exceeds config.sequence_len {config.sequence_len}")



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
    if warmdown <= 0 or step < max_steps - warmdown:
        return 1.0
    if warmdown == 1:
        return final_lr_frac
    progress = (max_steps - 1 - step) / (warmdown - 1)
    return final_lr_frac + progress * (1.0 - final_lr_frac)


def _weight_decay(step: int, max_steps: int, base_weight_decay: float) -> float:
    return base_weight_decay * 0.5 * (1.0 + math.cos(math.pi * step / max(1, max_steps)))


def _no_sync(model: torch.nn.Module, *, enabled: bool, use_fsdp: bool = False):
    no_sync = getattr(model, "no_sync", None)
    if enabled and not use_fsdp and callable(no_sync):
        return no_sync()
    return nullcontext()


def _clip_grad_norm(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    max_norm: float,
    *,
    use_fsdp: bool = False,
) -> torch.Tensor:
    if use_fsdp:
        clip_grad_norm = getattr(model, "clip_grad_norm_", None)
        if not callable(clip_grad_norm):
            raise TypeError("FSDP model does not expose clip_grad_norm_")
        return clip_grad_norm(max_norm)
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


@torch.no_grad()
def _eval_loss(model: torch.nn.Module, config: TrainConfig, device: torch.device) -> dict[str, float]:
    if config.mode == "sft" and config.data.validation_dataset is not None:
        validation_config = deepcopy(config)
        validation_config.data.dataset = config.data.validation_dataset
        validation_config.data.sft_skip_rows = 0
        validation_config.data.sft_take_rows = config.batch_size * max(1, config.eval_batches)
        stream = build_training_stream(validation_config, device, rank=0, world_size=1)
        loader = _training_batch_loader(stream)
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
        )
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_weight = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device)
    for _ in range(max(1, config.eval_batches)):
        x, y, attention_mask = next(loader)
        longshot_checkpointing = _longshot_gradient_checkpointing_decision(config, x)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=config.bf16 and device.type == "cuda",
        ):
            grouped_attention_mask = attention_mask if isinstance(attention_mask, torch.Tensor) else None
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
                loss = (row_sums.reshape(-1, 2).sum(dim=1) / pair_counts.clamp_min(1.0)).sum()
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


def _training_batch_loader(stream: TrainingBatchStream):
    while True:
        batch = next(stream)
        yield batch.x, batch.y, batch.attention_mask


def _random_loader(config: TrainConfig, device: torch.device):
    vocab = config.model.vocab_size
    while True:
        x = torch.randint(0, vocab, (config.batch_size, config.sequence_len), device=device)
        y = torch.roll(x, shifts=-1, dims=1)
        yield x, y, {"kind": "random"}


def _gradient_audit(model: torch.nn.Module) -> dict[str, Any]:
    processor_grad_nonzero = False
    refinement_gate_grad_nonzero = False
    backbone_grad_tensors = 0
    processor_grad_tensors = 0
    refinement_gate_grad_tensors = 0
    nonfinite_grad_tensors = 0
    for name, parameter in _unwrap(model).named_parameters():
        gradient = parameter.grad
        is_backbone = name.startswith("backbone.")
        if is_backbone:
            if gradient is not None:
                backbone_grad_tensors += 1
            continue
        if not parameter.requires_grad or gradient is None:
            continue
        processor_grad_tensors += 1
        if not torch.isfinite(gradient).all().item():
            nonfinite_grad_tensors += 1
            continue
        if torch.count_nonzero(gradient).item() > 0:
            processor_grad_nonzero = True
            if name.endswith(("flow_gate", "refinement_gate")):
                refinement_gate_grad_nonzero = True
        if name.endswith(("flow_gate", "refinement_gate")):
            refinement_gate_grad_tensors += 1
    return {
        "processor_grad_nonzero": processor_grad_nonzero,
        "refinement_gate_grad_nonzero": refinement_gate_grad_nonzero,
        "backbone_grad_tensors": backbone_grad_tensors,
        "processor_grad_tensors": processor_grad_tensors,
        "refinement_gate_grad_tensors": refinement_gate_grad_tensors,
        "nonfinite_grad_tensors": nonfinite_grad_tensors,
    }


def _write_training_audit(path: Path, payload: dict[str, Any]) -> None:
    if not isinstance(payload.get("source_commit"), str) or not isinstance(
        payload.get("experiment_manifest_sha256"), str
    ):
        raise ValueError("training audit requires source and manifest identities")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _fsdp_active(config: TrainConfig) -> bool:
    return config.distributed == "fsdp" and torch.distributed.is_initialized()


def _build_training_model(config: TrainConfig, device: torch.device, *, rank: int, use_fsdp: bool) -> torch.nn.Module:
    if use_fsdp:
        if rank == 0:
            model = build_model(config.model)
            model.init_weights()
            if config.init_checkpoint:
                _load_initial_model(model, config.init_checkpoint)
            elif config.backbone_checkpoint:
                load_backbone_checkpoint(model, config.backbone_checkpoint)
            return model
        with torch.device("meta"):
            return build_model(config.model)

    model = build_model(config.model).to(device)
    model.init_weights()
    if config.init_checkpoint:
        _load_initial_model(model, config.init_checkpoint)
    elif config.backbone_checkpoint:
        load_backbone_checkpoint(model, config.backbone_checkpoint)
    return model


def _wrap_fsdp(model: torch.nn.Module, config: TrainConfig, device: torch.device) -> torch.nn.Module:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    layer_classes = _transformer_layer_classes(config.model.family)
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_classes)
    # TODO: Decide whether FSDP should preserve the older DDP precision contract:
    # fp32 parameter/optimizer storage with bf16 autocast compute.
    mixed_precision = (
        MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        if config.bf16
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
    return FSDP(model, **kwargs)


def _fsdp_param_init_fn(device: torch.device):
    def init_fn(module: torch.nn.Module) -> None:
        has_meta_param = any(param.is_meta for param in module.parameters(recurse=False))
        has_meta_buffer = any(buffer.is_meta for buffer in module.buffers(recurse=False))
        if has_meta_param or has_meta_buffer:
            module.to_empty(device=device, recurse=False)

    return init_fn



def _save_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_config,
    checkpoint_state: dict[str, Any],
    *,
    use_fsdp: bool = False,
) -> None:
    if use_fsdp:
        from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state = model.state_dict()
        if dist_info()[1] == 0:
            save_checkpoint(path, state, model_config, checkpoint_state)
        return
    if dist.is_available() and dist.is_initialized() and dist_info()[1] != 0:
        return
    save_checkpoint(path, _unwrap(model), model_config, checkpoint_state)



def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if v is not None}


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)
