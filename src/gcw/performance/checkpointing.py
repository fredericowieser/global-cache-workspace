"""Activation-checkpoint admission, policy decisions, and telemetry.

Model modules retain the actual checkpoint boundaries because those boundaries
encode cache semantics. This module owns only execution policy and metadata.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import Any

import torch

from gcw.config import TrainConfig
from gcw.models.backbones.transformer import Block
from gcw.models.cache_processors.longshot_layers import (
    longshot_gradient_checkpointing_override,
)


def configure_native_activation_checkpointing(
    model: torch.nn.Module,
    policy: str,
) -> dict[str, Any]:
    """Apply processor-layer checkpointing without changing state-dict names."""
    from gcw.models.cache_processors.standard import CacheProcessor

    processors = [
        module
        for module in model.modules()
        if isinstance(module, CacheProcessor)
    ]
    enabled = policy == "standard_processor_layers"
    if enabled and not processors:
        raise ValueError(
            "activation_checkpointing='standard_processor_layers' found no standard processor layers"
        )
    for processor in processors:
        processor.activation_checkpointing = enabled
    layer_boundaries = sum(len(processor.layers) for processor in processors)
    return {
        "policy": policy,
        "implementation": (
            "torch.utils.checkpoint.checkpoint" if enabled else "eager"
        ),
        "boundary": "standard_processor_layer" if enabled else None,
        "use_reentrant": False if enabled else None,
        "preserve_rng_state": True if enabled else None,
        "processor_blocks": len(processors),
        "layer_boundaries": layer_boundaries if enabled else 0,
    }


def longshot_gradient_checkpointing_decision(
    config: TrainConfig,
    input_ids: torch.Tensor,
) -> bool | None:
    """Choose the Longshot processor-layer path from the physical padded width."""
    threshold = config.longshot_gradient_checkpointing_seq_len_threshold
    if threshold is None:
        return None
    return input_ids.shape[1] >= threshold


def longshot_gradient_checkpointing_context(
    decision: bool | None,
) -> AbstractContextManager[Any]:
    """Apply one physical forward's explicit Longshot checkpoint override."""
    if decision is None:
        return nullcontext()
    return longshot_gradient_checkpointing_override(decision)


def longshot_gradient_checkpointing_metadata(
    config: TrainConfig,
) -> dict[str, Any] | None:
    """Describe the configured Longshot processor-layer checkpoint policy."""
    if config.model.variant != "longshot":
        return None
    processor = config.model.cache_processor
    if processor is None:
        raise RuntimeError(
            "Longshot runtime metadata requires a cache processor"
        )
    threshold = config.longshot_gradient_checkpointing_seq_len_threshold
    adaptive = threshold is not None
    return {
        "policy": (
            "padded_sequence_length_threshold" if adaptive else "model_config"
        ),
        "model_config_gradient_checkpointing": processor.gradient_checkpointing,
        "padded_sequence_length_threshold": threshold,
        "checkpoint_when": (
            "padded_input_sequence_length >= threshold"
            if adaptive
            else "model.cache_processor.gradient_checkpointing"
        ),
        "short_microbatches": "eager" if adaptive else None,
        "long_microbatches": "checkpointed" if adaptive else None,
        "implementation": (
            "torch.utils.checkpoint.checkpoint"
            if adaptive or processor.gradient_checkpointing
            else "eager"
        ),
        "boundary": "longshot_processor_layer",
        "use_reentrant": False,
        "preserve_rng_state": True,
        "scope": "physical_train_and_eval_forward",
        "requires_training_and_grad": True,
        "sequence_length_source": "input_ids.shape[1] after batch padding",
    }


def longshot_gradient_checkpointing_step_metrics(
    decisions: list[bool],
    padded_sequence_lengths: list[int],
) -> dict[str, float]:
    """Summarize physical Longshot checkpoint choices for one optimizer step."""
    if not decisions:
        return {}
    checkpointed = sum(decisions)
    total = len(decisions)
    return {
        "cache_processor/longshot_checkpointed_microbatches": float(
            checkpointed
        ),
        "cache_processor/longshot_eager_microbatches": float(
            total - checkpointed
        ),
        "cache_processor/longshot_checkpointed_microbatch_fraction": checkpointed
        / total,
        "cache_processor/longshot_padded_sequence_length_min": float(
            min(padded_sequence_lengths)
        ),
        "cache_processor/longshot_padded_sequence_length_max": float(
            max(padded_sequence_lengths)
        ),
    }


def transformer_layer_classes(family: str) -> set[type[torch.nn.Module]]:
    """Return the concrete transformer blocks admitted to FSDP checkpoint wrapping."""
    classes: set[type[torch.nn.Module]] = {Block}
    if family == "gemma4":
        from gcw.models.backbones.gemma4 import Gemma4Block

        classes.add(Gemma4Block)
    elif family in {"qwen3_5", "qwen3_5_moe"}:
        from gcw.models.backbones.qwen3 import Qwen35Block, Qwen35MoeBlock

        classes.add(Qwen35Block)
        classes.add(Qwen35MoeBlock)
    return classes


def apply_fsdp_activation_checkpointing(
    model: torch.nn.Module, *, family: str
) -> None:
    """Apply the existing opt-out FSDP layer checkpoint policy."""
    import os

    enabled = os.environ.get(
        "GCW_FSDP_ACTIVATION_CHECKPOINTING", "1"
    ).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
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
    layer_classes = transformer_layer_classes(family)
    wrapper = partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: type(module) in layer_classes,
    )
