"""Execution-plan resolution: the single admission gate for training.

This module owns the cross-feature composition rules that decide which
execution strategy a :class:`~gcw.config.TrainConfig` selects and whether that
strategy is admissible. It is the canonical home for "does feature X compose
with feature Y".

Field-level sanity (numeric ranges, enum membership, dataclass coercion) stays
in the config dataclasses' ``__post_init__``; this module only answers the
composition question and returns a frozen :class:`ExecutionPlan`.

The ``ValueError`` messages raised here are the public admission contract;
they are asserted verbatim by the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from gcw.config import TrainConfig


ExecutionStrategy = Literal[
    "eager_reference",
    "backbone_chunk_compile",
    "longshot_selected_history",
    "kv_score_sidecar",
    "triton_cache_kernel",
    "noisy_diffusion_compile",
    "adaptive_pairs",
]


@dataclass(frozen=True)
class ExecutionPlan:
    """The single resolved execution strategy admitted for a training config."""

    strategy: ExecutionStrategy


def resolve_execution_plan(config: "TrainConfig") -> ExecutionPlan:
    """Validate feature composition and return the selected execution plan.

    Runs every cross-feature admission check and raises ``ValueError`` on the
    first violation, matching the historical inline ``__post_init__`` order.
    """
    _validate_backbone_chunk_backend(config)
    _validate_logical_pair_grouping(config)
    _validate_longshot_full_history(config)
    _validate_kv_score_sidecar(config)
    _validate_triton_cache_movement(config)
    _validate_noisy_diffusion(config)
    _validate_reference_loss(config)
    _validate_adaptive_batching(config)
    return ExecutionPlan(strategy=_select_strategy(config))


def _select_strategy(config: "TrainConfig") -> ExecutionStrategy:
    """Map the enabled execution feature(s) to one canonical strategy name."""
    performance = config.performance
    if performance.adaptive_batching.enabled:
        return "adaptive_pairs"
    if performance.triton_cache_movement.enabled:
        return "triton_cache_kernel"
    if config.backbone_chunk_backend != "eager":
        return "backbone_chunk_compile"
    if performance.longshot_full_history.enabled:
        return "longshot_selected_history"
    if performance.kv_score_sidecar.enabled:
        return "kv_score_sidecar"
    if performance.noisy_diffusion.recurrence_backend == "compile":
        return "noisy_diffusion_compile"
    return "eager_reference"


def _validate_backbone_chunk_backend(config: "TrainConfig") -> None:
    # Eager-vs-compiled windowed-Qwen3 backend switch: non-eager requires a
    # windowed processor trained cache-processor-only.
    if config.backbone_chunk_backend == "eager":
        return
    selected_backend = config.backbone_chunk_backend
    if config.model.family != "qwen3":
        raise ValueError(f"backbone_chunk_backend={selected_backend} requires model.family='qwen3'")
    if config.model.variant == "tf" or config.model.cache_processor is None:
        raise ValueError(f"backbone_chunk_backend={selected_backend} requires a windowed cache processor")
    if config.trainable != "cache_processor":
        raise ValueError(f"backbone_chunk_backend={selected_backend} requires trainable='cache_processor'")
    if config.model.cache_processor.process_every != 64:
        raise ValueError(f"backbone_chunk_backend={selected_backend} requires cache_processor.process_every=64")
    if config.sequence_len < 128:
        raise ValueError(f"backbone_chunk_backend={selected_backend} requires sequence_len >= 128")
    if config.distributed == "fsdp":
        raise ValueError(f"backbone_chunk_backend={selected_backend} is not supported with FSDP")


def _validate_logical_pair_grouping(config: "TrainConfig") -> None:
    # Adjacent-pair SFT grouping: answer-only pairs that preserve RNG identity.
    if not config.logical_pair_grouping:
        return
    if config.mode != "sft" or config.data.kind != "sft":
        raise ValueError("logical_pair_grouping requires SFT data")
    if config.data.sft_loss_scope != "answer_only":
        raise ValueError("logical_pair_grouping requires answer-only SFT loss")
    if (
        isinstance(config.batch_size, bool)
        or not isinstance(config.batch_size, int)
        or config.batch_size < 2
        or config.batch_size % 2
    ):
        raise ValueError("logical_pair_grouping requires an even physical batch_size of at least 2")
    if (
        isinstance(config.global_batch_rows, bool)
        or not isinstance(config.global_batch_rows, int)
        or config.global_batch_rows <= 0
        or config.global_batch_rows % 2
    ):
        raise ValueError("logical_pair_grouping requires a positive even global_batch_rows")
    if config.global_batch_rows % config.batch_size:
        raise ValueError("logical_pair_grouping requires global_batch_rows divisible by physical batch_size")
    if config.model.router_aux_loss_coef:
        raise ValueError("logical_pair_grouping does not support router auxiliary loss")
    diffusion = config.model.diffusion_processor
    if diffusion is not None and diffusion.noise_scale > 0.0:
        raise ValueError(
            "logical_pair_grouping rejects noisy Diffusion because RNG assignment changes with batching"
        )


def _validate_longshot_full_history(config: "TrainConfig") -> None:
    # Longshot selected-history execution: single-process eager KV-norm replay.
    longshot_history = config.performance.longshot_full_history
    if not longshot_history.enabled:
        return
    processor = config.model.cache_processor
    if config.model.variant != "longshot" or processor is None:
        raise ValueError("performance.longshot_full_history requires a Longshot cache processor")
    if config.trainable != "cache_processor":
        raise ValueError("performance.longshot_full_history requires trainable='cache_processor'")
    if config.distributed != "none":
        raise ValueError("performance.longshot_full_history supports only single-process distributed='none'")
    if config.backbone_chunk_backend != "eager":
        raise ValueError("performance.longshot_full_history requires backbone_chunk_backend='eager'")
    if config.performance.adaptive_batching.enabled:
        raise ValueError("performance.longshot_full_history does not compose with adaptive batching")
    if processor.score != "kv_norm":
        raise ValueError("performance.longshot_full_history requires cache_processor.score='kv_norm'")
    if processor.attention_dropout != 0.0:
        raise ValueError("performance.longshot_full_history requires cache_processor.attention_dropout=0")
    if processor.linear_state_cache:
        raise ValueError("performance.longshot_full_history does not support linear recurrent cache state")
    if config.model.flow_processor is not None or config.model.workspace_flow_refinement is not None:
        raise ValueError("performance.longshot_full_history does not support Flow or workspace refinement")
    if config.activation_checkpointing != "none":
        raise ValueError("performance.longshot_full_history supports Longshot and segmented checkpointing only")
    if (
        processor.iterations > 1
        and processor.iteration_mode == "internal"
        and processor.internal_iteration_score != "kv_norm"
    ):
        raise ValueError(
            "performance.longshot_full_history internal recurrence requires "
            "cache_processor.internal_iteration_score='kv_norm'"
        )
    if config.model.family not in {"smollm", "llama", "qwen3"}:
        raise ValueError("performance.longshot_full_history requires a dense WindowedCache backbone family")
    head_dim = config.model.stored_cache_shape[1]
    if processor.unrope_keys:
        rotary_dim = int(head_dim * config.model.partial_rotary_factor)
        minimum_temp_bytes = (7 * head_dim + 8 * rotary_dim) * 4 + 8
    else:
        minimum_temp_bytes = 2 * head_dim * 4
    if longshot_history.max_temp_bytes < minimum_temp_bytes:
        if processor.unrope_keys:
            raise ValueError(
                "performance.longshot_full_history.max_temp_bytes must cover at least "
                f"one complete un-RoPE rotation and score row ({minimum_temp_bytes} bytes)"
            )
        raise ValueError(
            "performance.longshot_full_history.max_temp_bytes must cover at least "
            f"one FP32 K/V row ({minimum_temp_bytes} bytes)"
        )


def _validate_kv_score_sidecar(config: "TrainConfig") -> None:
    # KV-score sidecar: fixed batch-2 eager KV-norm scoring, no composition.
    sidecar = config.performance.kv_score_sidecar
    if not sidecar.enabled:
        return
    processor = config.model.cache_processor
    if config.model.variant != "longshot" or processor is None:
        raise ValueError("performance.kv_score_sidecar requires a Longshot cache processor")
    if config.performance.longshot_full_history.enabled:
        raise ValueError("performance.kv_score_sidecar does not compose with performance.longshot_full_history")
    if config.batch_size != 2:
        raise ValueError("performance.kv_score_sidecar requires fixed eager batch_size=2")
    if config.distributed != "none":
        raise ValueError("performance.kv_score_sidecar supports only single-process distributed='none'")
    if config.backbone_chunk_backend != "eager":
        raise ValueError("performance.kv_score_sidecar requires backbone_chunk_backend='eager'")
    if config.performance.adaptive_batching.enabled:
        raise ValueError("performance.kv_score_sidecar does not compose with adaptive batching")
    if processor.score != "kv_norm":
        raise ValueError("performance.kv_score_sidecar requires cache_processor.score='kv_norm'")
    if processor.attention_dropout != 0.0:
        raise ValueError("performance.kv_score_sidecar requires cache_processor.attention_dropout=0")
    if processor.linear_state_cache:
        raise ValueError("performance.kv_score_sidecar rejects linear recurrent cache state")
    if (
        config.model.flow_processor is not None
        or config.model.diffusion_processor is not None
        or config.model.workspace_flow_refinement is not None
    ):
        raise ValueError(
            "performance.kv_score_sidecar does not compose with Flow, Diffusion, or workspace refinement"
        )
    if config.activation_checkpointing != "none":
        raise ValueError(
            "performance.kv_score_sidecar supports Longshot native and segmented checkpointing only"
        )
    if config.model.family not in {"smollm", "llama", "qwen3"}:
        raise ValueError("performance.kv_score_sidecar requires a dense WindowedCache-compatible family")


def _validate_triton_cache_movement(config: "TrainConfig") -> None:
    # Handwritten Triton cache movement: fixed batch-2 BF16 CP-only training.
    triton_cache = config.performance.triton_cache_movement
    if not triton_cache.enabled:
        return
    processor = config.model.cache_processor
    if config.model.variant not in {"bt", "bgt", "longshot", "icarus"} or processor is None:
        raise ValueError("performance.triton_cache_movement requires a cache processor")
    if config.model.family != "qwen3":
        raise ValueError("performance.triton_cache_movement currently supports only Qwen3")
    if config.batch_size != 2 or config.distributed != "none":
        raise ValueError("performance.triton_cache_movement requires fixed batch two and distributed='none'")
    if config.trainable != "cache_processor" or not config.bf16:
        raise ValueError("performance.triton_cache_movement requires BF16 cache-processor-only training")
    if config.backbone_chunk_backend != "eager":
        raise ValueError("performance.triton_cache_movement does not compose with compiled backbone execution")
    if config.performance.adaptive_batching.enabled:
        raise ValueError("performance.triton_cache_movement does not compose with adaptive batching")
    if processor.processor_mode != "standard" or processor.update_stream != "kv":
        raise ValueError("performance.triton_cache_movement v1 requires standard K/V cache updates")
    if processor.attention_dropout != 0.0:
        raise ValueError("performance.triton_cache_movement requires zero processor dropout")
    if config.performance.longshot_full_history.enabled:
        raise ValueError("performance.triton_cache_movement does not compose with Longshot selected-history")


def _validate_noisy_diffusion(config: "TrainConfig") -> None:
    # Compiled noisy-Diffusion recurrence: BGT with a diffusion noise schedule.
    noisy_diffusion_backend = config.performance.noisy_diffusion.recurrence_backend
    if noisy_diffusion_backend != "compile":
        return
    diffusion = config.model.diffusion_processor
    if config.model.variant != "bgt" or diffusion is None or diffusion.noise_scale <= 0.0:
        raise ValueError(
            "compiled noisy-Diffusion recurrence requires BGT with diffusion_processor.noise_scale>0"
        )
    processor = config.model.cache_processor
    if processor is not None and processor.attention_dropout != 0.0:
        raise ValueError("compiled noisy-Diffusion recurrence requires cache_processor.attention_dropout=0")
    if config.distributed != "none":
        raise ValueError("compiled noisy-Diffusion recurrence currently supports distributed='none'")
    if config.performance.adaptive_batching.enabled:
        raise ValueError("compiled noisy-Diffusion recurrence does not enable adaptive batching")


def _validate_reference_loss(config: "TrainConfig") -> None:
    reference_loss = config.performance.reference_loss
    if reference_loss.mode != "legacy" and config.loss_backend != "reference":
        raise ValueError("bounded reference-loss execution requires loss_backend='reference'")
    if reference_loss.mode == "bounded":
        if config.distributed != "none":
            raise ValueError("bounded reference-loss execution currently supports only distributed='none'")
        adaptive_batching = config.performance.adaptive_batching
        maximum_physical_rows = (
            max(band.physical_rows for band in adaptive_batching.bands)
            if adaptive_batching.enabled
            else config.batch_size
        )
        if reference_loss.max_projected_positions < maximum_physical_rows:
            raise ValueError(
                "performance.reference_loss.max_projected_positions must cover the maximum physical rows"
            )


def _validate_adaptive_batching(config: "TrainConfig") -> None:
    # Adaptive physical-batch ladder (16/8/4/2): adjacent-pair SFT only.
    adaptive_batching = config.performance.adaptive_batching
    if not adaptive_batching.enabled:
        return
    if not config.logical_pair_grouping:
        raise ValueError("adaptive batching requires logical_pair_grouping=true")
    if config.batch_size != 2:
        raise ValueError("adaptive batching requires batch_size=2 as the adjacent-pair source stream")
    if config.global_batch_rows != 64:
        raise ValueError("adaptive batching v1 requires global_batch_rows=64")
    if config.distributed != "none":
        raise ValueError("adaptive batching v1 supports only single-process distributed='none' execution")
    if config.loss_backend != "reference":
        raise ValueError("adaptive batching requires loss_backend='reference'")
    processor = config.model.cache_processor
    if processor is not None and processor.attention_dropout != 0.0:
        raise ValueError("adaptive batching requires cache_processor.attention_dropout=0")
    if config.model.lora is not None and config.model.lora.dropout != 0.0:
        raise ValueError("adaptive batching requires model.lora.dropout=0")
    if config.sequence_len > 4096:
        raise ValueError("adaptive batching v1 supports sequence_len up to 4096")
    physical_rows = tuple(band.physical_rows for band in adaptive_batching.bands)
    if physical_rows != (16, 8, 4, 2):
        raise ValueError("adaptive batching v1 physical_rows must be exactly 16, 8, 4, 2")
    if adaptive_batching.bands[-1].max_padded_tokens < config.sequence_len:
        raise ValueError("adaptive batching fallback band must cover sequence_len")
