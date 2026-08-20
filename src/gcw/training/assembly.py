"""Training assembly: turn a resolved config into a runnable plan.

This module owns the one-time setup that used to live at the top of ``train``:
distributed init, batch/grad-accum arithmetic, resume restore, model build,
per-feature backend configuration, optimizer, and data-stream construction. It
returns a frozen :class:`TrainingPlan` that the step loop consumes.

The admission contract (which execution strategy a config selects and whether
it is admissible) lives in :mod:`gcw.architecture.execution`; this module only
*assembles* an already-admitted plan into runnable state.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
import os
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gcw.architecture import ExecutionPlan, admission_receipt_metadata, resolve_experiment, resolve_model
from gcw.checkpoint import load_backbone_checkpoint
from gcw.config import TrainConfig
from gcw.experiment import validate_experiment
from gcw.runtime import cleanup_distributed, dist_info, init_distributed, print0
from gcw.data.training import TrainingBatchStream, build_training_stream
from gcw.flops import total_training_flops, training_flops_per_token
from gcw.models import build_model
from gcw.performance.adjacent_pairs import adaptive_batching_metadata
from gcw.performance.backends.qwen3 import (
    configure_qwen3_backbone_chunk_backend,
    qwen3_backbone_chunk_backend_metadata,
    validate_qwen3_backbone_chunk_backend,
)
from gcw.performance.checkpointing import (
    apply_fsdp_activation_checkpointing as _apply_fsdp_activation_checkpointing,
    configure_native_activation_checkpointing as _configure_native_activation_checkpointing,
    longshot_gradient_checkpointing_metadata as _longshot_gradient_checkpointing_runtime,
)
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
from gcw.performance.config import AdaptiveBatchingConfig
from gcw.performance.execution_plan import require_optimized_step_admission
from gcw.performance.kv_score_sidecar import (
    KVScoreSidecarExecutionContext,
    configure_kv_score_sidecar_execution,
    kv_score_sidecar_execution_metadata,
)
from gcw.performance.longshot_cache import (
    configure_longshot_full_history_execution,
    longshot_full_history_execution_metadata,
)
from gcw.performance.losses import (
    configure_reference_loss_execution,
    loss_backend_metadata,
    reference_loss_execution_metadata,
    require_loss_backend,
)
from gcw.performance.telemetry import model_runtime_backends as _model_runtime_backends
from gcw.performance.triton.execution import (
    TritonCacheExecutionContext,
    configure_triton_cache_movement,
    triton_cache_movement_metadata,
    validate_triton_cache_movement_request,
)
from gcw.performance.whole_model import maybe_compile_whole_model, whole_model_compile_requested
from gcw.params import processor_params, total_non_embedding_params
from gcw.tracking import RunTracker
from gcw.training.audit import _checkpoint_metadata, _processor_state, _training_fingerprint
from gcw.training.checkpoint import TrainingCheckpoint
from gcw.training.distributed import _fsdp_active, _wrap_fsdp
from gcw.training.model_setup import (
    _apply_trainable_policy,
    _build_training_model,
    _load_initial_model,
    _model_dtype,
)
from gcw.training.optimizer import _build_optimizer
from gcw.training.precision import _precision_summary, _validate_precision


@dataclass(frozen=True)
class TrainingPlan:
    """Frozen hand-off from assembly to the step loop.

    Assembly produces every value the loop reads; the loop owns its own mutable
    progress (``smooth_loss``, step counters) as locals so this contract stays
    pure-input.
    """

    config: TrainConfig
    execution_plan: ExecutionPlan
    device: torch.device
    rank: int
    world_size: int
    model: torch.nn.Module
    params: list[torch.nn.Parameter]
    opt: torch.optim.Optimizer
    stream: TrainingBatchStream
    icarus_execution_scope: Callable[[int], Any]
    use_fsdp: bool
    adaptive_batching: AdaptiveBatchingConfig
    grad_accum_steps: int
    micro_tokens: int
    micro_rows: int
    tokens_per_step: int
    flops_per_tok: float
    start_step: int
    smooth_loss: float
    initial_processor_state: dict[str, torch.Tensor] | None
    fingerprint: dict[str, Any]
    checkpoint_meta: dict[str, Any]
    run_meta: dict[str, Any]
    checkpoint_stop_policy: tuple[int | None, datetime | None] | None
    tracker: RunTracker | None
    audit_output: str | None


def _checkpoint_stop_policy(config: TrainConfig) -> tuple[int | None, datetime | None] | None:
    step_text = os.environ.get("GCW_STOP_AFTER_CHECKPOINT_STEP")
    utc_text = os.environ.get("GCW_STOP_AFTER_CHECKPOINT_UTC")
    if step_text is None and utc_text is None:
        return None
    if config.save_every <= 0:
        raise ValueError("checkpoint-boundary stopping requires save_every > 0")
    stop_step: int | None = None
    if step_text is not None:
        try:
            stop_step = int(step_text)
        except ValueError as error:
            raise ValueError("GCW_STOP_AFTER_CHECKPOINT_STEP must be an integer") from error
        if not 1 <= stop_step <= config.max_steps:
            raise ValueError("GCW_STOP_AFTER_CHECKPOINT_STEP must be within the configured run")
    stop_utc: datetime | None = None
    if utc_text is not None:
        try:
            stop_utc = datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("GCW_STOP_AFTER_CHECKPOINT_UTC must be ISO-8601") from error
        if stop_utc.tzinfo is None or stop_utc.utcoffset() is None:
            raise ValueError("GCW_STOP_AFTER_CHECKPOINT_UTC must include a timezone")
        stop_utc = stop_utc.astimezone(timezone.utc)
    return stop_step, stop_utc


def _checkpoint_stop_requested(
    policy: tuple[int | None, datetime | None] | None,
    step: int,
    *,
    now: datetime | None = None,
) -> bool:
    if policy is None:
        return False
    stop_step, stop_utc = policy
    if stop_step is not None and step >= stop_step:
        return True
    observed = datetime.now(timezone.utc) if now is None else now.astimezone(timezone.utc)
    return stop_utc is not None and observed >= stop_utc


def assemble_training_execution(
    config: TrainConfig,
    execution_plan: ExecutionPlan,
) -> TrainingPlan | None:
    """Assemble a runnable plan from an already-admitted config.

    Returns ``None`` (after releasing the distributed group) when the run has
    nothing left to do — e.g. a resumed checkpoint already at ``max_steps`` or
    a checkpoint stop boundary already reached.
    """
    require_optimized_step_admission(config.performance.optimized_step)
    validate_qwen3_backbone_chunk_backend(config.backbone_chunk_backend)
    validate_experiment(config)
    checkpoint_stop_policy = _checkpoint_stop_policy(config)
    _, model_receipt = resolve_model(config.model).require()
    _, composition_receipt = resolve_experiment(config, model_receipt).require()
    torch.manual_seed(config.seed)
    device = init_distributed(config.distributed)
    require_loss_backend(config.loss_backend, device)
    validate_triton_cache_movement_request(config.performance.triton_cache_movement, device)
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
    _validate_precision(config, device)
    use_fsdp = _fsdp_active(config)
    icarus_backend = resolve_icarus_processor_backend(config.model)
    validate_icarus_training_admission(
        icarus_backend,
        distributed=config.distributed,
        world_size=max(world_size, int(os.environ.get("WORLD_SIZE", "1"))),
        distributed_initialized=dist.is_available() and dist.is_initialized(),
        batch_size=config.batch_size,
        adaptive_batch_sizes=(
            tuple(band.physical_rows for band in adaptive_batching.bands) if adaptive_batching.enabled else None
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
            return None
        if _checkpoint_stop_requested(checkpoint_stop_policy, start_step):
            print0(f"checkpoint stop boundary already reached step={start_step}")
            cleanup_distributed()
            return None
        smooth_loss = checkpoint.resume_state.smooth_loss
        cursor = checkpoint.loader_cursor(rank=rank)
    elif _checkpoint_stop_requested(checkpoint_stop_policy, start_step):
        cleanup_distributed()
        raise RuntimeError("checkpoint deadline elapsed before the first authenticated checkpoint")

    initial_processor_state: dict[str, torch.Tensor] | None = None
    if use_fsdp:
        model = _build_training_model(config, device, rank=rank, use_fsdp=True)
    else:
        model = build_model(config.model).to(device)
        model.init_weights()
        if checkpoint is not None:
            checkpoint.restore_model(model)
            if config.trainable == "cache_processor":
                initial_processor_state = checkpoint.load_initial_processor_state(device="cpu")
        elif config.init_checkpoint:
            _load_initial_model(model, config.init_checkpoint)
        elif config.backbone_checkpoint:
            load_backbone_checkpoint(model, config.backbone_checkpoint)
        model = model.to(dtype=_model_dtype(config))
        if checkpoint is None:
            initial_processor_state = _processor_state(model)
    _apply_trainable_policy(model, config.trainable)
    activation_checkpointing_runtime = _configure_native_activation_checkpointing(
        model, config.activation_checkpointing
    )
    configure_qwen3_backbone_chunk_backend(model, config.backbone_chunk_backend)
    configure_reference_loss_execution(model, config.performance.reference_loss)
    configure_longshot_full_history_execution(model, config.performance.longshot_full_history)
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
    configure_triton_cache_movement(
        model,
        config.performance.triton_cache_movement,
        context=TritonCacheExecutionContext(
            model_variant=config.model.variant,
            model_family=config.model.family,
            physical_batch_size=config.batch_size,
            distributed=config.distributed,
            trainable=config.trainable,
            bf16=config.bf16,
            backbone_chunk_backend=config.backbone_chunk_backend,
            adaptive_batching_enabled=config.performance.adaptive_batching.enabled,
            whole_model_compile_requested=whole_model_compile_requested(),
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
    icarus_execution_scope = partial(icarus_training_execution, model) if icarus_backend == "compile" else nullcontext

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    opt = _build_optimizer(params, config)
    if checkpoint is not None:
        checkpoint.restore_optimizer(opt, device=device)
    stream = build_training_stream(config, device, rank=rank, world_size=world_size, cursor=cursor)

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
        "adaptive_batching": adaptive_batching_metadata(adaptive_batching, global_batch_rows=config.global_batch_rows),
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
        "checkpoint_stop_policy": (
            None
            if checkpoint_stop_policy is None
            else {
                "step": checkpoint_stop_policy[0],
                "utc": (None if checkpoint_stop_policy[1] is None else checkpoint_stop_policy[1].isoformat()),
            }
        ),
        "model_runtime_backends": _model_runtime_backends(model, device),
        "bt_processor_backend": bt_processor_backend_metadata() if config.model.variant == "bt" else None,
        "qwen3_backbone_chunk_backend": qwen3_backbone_chunk_backend_metadata(config.backbone_chunk_backend),
        "cache_processor_backend": (
            cache_processor_backend_metadata(config.model) if config.model.variant == "bgt" else None
        ),
        "noisy_diffusion_recurrence_backend": (
            noisy_diffusion_recurrence_backend_metadata(config.model, config.performance.noisy_diffusion)
            if config.model.diffusion_processor is not None
            else None
        ),
        "icarus_processor_backend": (
            icarus_processor_backend_metadata(config.model) if config.model.variant == "icarus" else None
        ),
        "activation_checkpointing": activation_checkpointing_runtime,
        "activation_checkpoint_windows": config.activation_checkpoint_windows,
        **(
            {"triton_cache_movement": triton_cache_movement_metadata(config.performance.triton_cache_movement)}
            if config.performance.triton_cache_movement.enabled
            else {}
        ),
        "activation_checkpoint_min_padded_sequence_length": (config.activation_checkpoint_min_padded_sequence_length),
        "longshot_gradient_checkpointing": _longshot_gradient_checkpointing_runtime(config),
        "loss_backend": loss_backend_metadata(config.loss_backend),
        **(
            {
                "longshot_full_history_execution": longshot_full_history_execution_metadata(
                    config.performance.longshot_full_history,
                    unrope_keys=bool(config.model.cache_processor and config.model.cache_processor.unrope_keys),
                )
            }
            if config.performance.longshot_full_history.enabled
            else {}
        ),
        **(
            {
                "kv_score_sidecar_execution": kv_score_sidecar_execution_metadata(
                    config.performance.kv_score_sidecar,
                    unrope_keys=bool(config.model.cache_processor and config.model.cache_processor.unrope_keys),
                )
            }
            if config.performance.kv_score_sidecar.enabled
            else {}
        ),
        **(
            {"reference_loss_execution": reference_loss_execution_metadata(config.performance.reference_loss)}
            if config.performance.reference_loss.mode != "legacy"
            else {}
        ),
    }
    # Reuse the receipts already resolved above so the checkpoint metadata does not
    # re-serialize + re-hash the config (``admission_receipt_metadata`` would).
    admission = admission_receipt_metadata(
        config,
        model_receipt=model_receipt,
        composition_receipt=composition_receipt,
    )
    checkpoint_meta = _checkpoint_metadata(config, run_meta, admission=admission)
    budget_message = (
        "training budget: "
        f"tokens={target_tokens:,} flops/token={flops_per_tok:.6e} "
        f"total_flops={run_meta['total_training_flops']:.6e} "
        f"non_embedding_params={run_meta['non_embedding_params']:,} "
        f"micro_tokens={micro_tokens:,} rows/step={micro_rows * grad_accum_steps:,} "
        f"grad_accum={grad_accum_steps}"
    )
    print0(budget_message)
    print0(_precision_summary(config, device))
    tracker = RunTracker.for_training(config, run_meta, model=model) if rank == 0 else None
    if tracker is not None:
        tracker.log_message(budget_message)
    if checkpoint is not None:
        # W&B/run-record construction is outside the scientific trajectory. Restore
        # the authenticated state immediately before resumed model execution.
        checkpoint.restore_rng(rank=rank)

    return TrainingPlan(
        config=config,
        execution_plan=execution_plan,
        device=device,
        rank=rank,
        world_size=world_size,
        model=model,
        params=params,
        opt=opt,
        stream=stream,
        icarus_execution_scope=icarus_execution_scope,
        use_fsdp=use_fsdp,
        adaptive_batching=adaptive_batching,
        grad_accum_steps=grad_accum_steps,
        micro_tokens=micro_tokens,
        micro_rows=micro_rows,
        tokens_per_step=tokens_per_step,
        flops_per_tok=flops_per_tok,
        start_step=start_step,
        smooth_loss=smooth_loss,
        initial_processor_state=initial_processor_state,
        fingerprint=fingerprint,
        checkpoint_meta=checkpoint_meta,
        run_meta=run_meta,
        checkpoint_stop_policy=checkpoint_stop_policy,
        tracker=tracker,
        audit_output=audit_output,
    )
