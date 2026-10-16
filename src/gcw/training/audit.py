from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import torch

from gcw.architecture import admission_receipt_metadata
from gcw.checkpoint import initialization_backbone_checkpoint
from gcw.config import TrainConfig, to_dict
from gcw.data.training import (
    TrainingBatchStream,
    training_data_identity,
    fineweb_prefix_tokens,
)
from gcw.performance.adjacent_pairs import adaptive_batching_metadata
from gcw.performance.backends.cache_processors import (
    bt_processor_backend_metadata,
    cache_processor_backend_metadata,
)
from gcw.performance.backends.diffusion import (
    noisy_diffusion_recurrence_backend_metadata,
)
from gcw.performance.backends.icarus import (
    icarus_processor_backend_metadata,
    resolve_icarus_processor_backend,
)
from gcw.performance.backends.qwen3 import qwen3_backbone_chunk_backend_metadata
from gcw.performance.kv_score_sidecar import kv_score_sidecar_execution_metadata
from gcw.performance.longshot_cache import (
    longshot_full_history_execution_metadata,
)
from gcw.performance.losses import (
    loss_backend_metadata,
    reference_loss_execution_metadata,
)
from gcw.performance.triton.execution import triton_cache_movement_metadata
from gcw.training.checkpoint import ResumeState
from gcw.training.distributed import _unwrap


def _checkpoint_metadata(
    config: TrainConfig,
    run_meta: dict[str, Any],
    *,
    admission: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = {
        **run_meta,
        "data_kind": config.data.kind,
        "tokenizer": config.data.tokenizer,
        "precision": config.precision,
        "dtype": "bfloat16" if config.precision == "bf16" else "float32",
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
        if config.freeze_backbone or config.save_cache_processor_only:
            state["backbone_checkpoint"] = initialization_backbone_checkpoint(
                config.init_checkpoint
            )
    if config.data.tokenizer_source is not None:
        state["source"] = config.data.tokenizer_source
        state["tokenizer_source"] = config.data.tokenizer_source
    if config.data.kind == "fineweb":
        identity = training_data_identity(config)
        if "manifest_sha256" in identity:
            state["fineweb_data_identity"] = identity
    state["admission"] = (
        admission
        if admission is not None
        else admission_receipt_metadata(config)
    )
    return state


def _checkpoint_state(
    base: dict[str, Any], step: int, loader: dict[str, Any] | None = None
) -> dict[str, Any]:
    from gcw.training.state import _json_safe_loader_state

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
            bt_processor_backend_metadata()["fingerprint_sha256"]
            if config.model.variant == "bt"
            else None
        ),
        "qwen3_backbone_chunk_backend_fingerprint": qwen3_backbone_chunk_backend_metadata(
            config.backbone_chunk_backend
        )[
            "fingerprint_sha256"
        ],
        **(
            {
                "triton_cache_movement_fingerprint": triton_cache_movement_metadata(
                    config.performance.triton_cache_movement
                )[
                    "fingerprint_sha256"
                ]
            }
            if config.performance.triton_cache_movement.enabled
            else {}
        ),
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
            if config.performance.noisy_diffusion.recurrence_backend
            == "compile"
            else {}
        ),
        **(
            {
                "icarus_processor_backend_fingerprint": icarus_processor_backend_metadata(
                    config.model
                )[
                    "fingerprint_sha256"
                ]
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
        **(
            {
                "longshot_selector_training": to_dict(
                    config.longshot_selector_training
                )
            }
            if config.longshot_selector_training is not None
            else {}
        ),
        **(
            {"truncated_bptt_windows": config.truncated_bptt_windows}
            if config.truncated_bptt_windows
            else {}
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
                        config.model.cache_processor
                        and config.model.cache_processor.unrope_keys
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
                        config.model.cache_processor
                        and config.model.cache_processor.unrope_keys
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
            if config.longshot_gradient_checkpointing_seq_len_threshold
            is not None
            else {}
        ),
        "objective": (
            "adjacent_pair_token_mean_answer_only_ce"
            if config.logical_pair_grouping
            else (
                f"{config.data.sft_loss_scope}_ce"
                if config.mode == "sft"
                else "next_token_ce"
            )
        ),
        "checkpoint_format": config.checkpoint_format,
        "precision": config.precision,
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
            "logical_pairs_per_step": (
                config.global_batch_rows // 2
                if config.logical_pair_grouping
                else None
            ),
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
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in processor.state_dict().items()
    }


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
    sequence_limit = config.sequence_len + len(fineweb_prefix_tokens(config))
    if x.shape[1] > sequence_limit:
        raise ValueError(
            f"loader sequence length {x.shape[1]} exceeds configured input length {sequence_limit}"
        )


def _audit_training_step(model: torch.nn.Module) -> dict[str, Any]:
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
        raise ValueError(
            "training audit requires source and manifest identities"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload, handle, indent=2, sort_keys=True, allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
