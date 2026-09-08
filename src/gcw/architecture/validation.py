from __future__ import annotations

from typing import TYPE_CHECKING
from gcw.architecture.contracts import (
    CompatibilityIssue,
    _SUPPORTED_CHECKPOINT_FORMATS,
    _SUPPORTED_DATA_KINDS,
    _SUPPORTED_FAMILIES,
    _SUPPORTED_VARIANTS,
    _has_fields,
)

if TYPE_CHECKING:
    from gcw.config import ModelConfig, TrainConfig


def _model_issues(config: ModelConfig) -> list[CompatibilityIssue]:
    issues: list[CompatibilityIssue] = []

    def add(code: str, message: str, path: str) -> None:
        issues.append(CompatibilityIssue(code, message, path))

    if config.variant not in _SUPPORTED_VARIANTS:
        add(
            "unknown_variant",
            f"unknown model variant: {config.variant}",
            "model.variant",
        )
    if config.family not in _SUPPORTED_FAMILIES:
        add(
            "unknown_family",
            f"unknown model family: {config.family}",
            "model.family",
        )

    processor = config.cache_processor
    if config.variant != "tf" and processor is None:
        add(
            "processor_required",
            f"{config.variant} requires cache_processor config",
            "model.cache_processor",
        )
    if processor is not None and not _has_fields(
        processor, "processor_mode", "update_stream", "iterations"
    ):
        add(
            "processor_config",
            "cache_processor must be a CacheProcessorConfig",
            "model.cache_processor",
        )
        processor = None

    flow = config.flow_processor
    if flow is not None and not _has_fields(
        flow, "num_steps", "time_conditioning"
    ):
        add(
            "flow_config",
            "flow_processor must be a FlowProcessorConfig",
            "model.flow_processor",
        )
    if flow is not None and config.variant not in {"bgt", "icarus", "longshot"}:
        add(
            "flow_variant",
            "flow_processor requires variant 'bgt', 'icarus', or 'longshot'",
            "model.flow_processor",
        )
    diffusion = config.diffusion_processor
    if diffusion is not None and not _has_fields(
        diffusion, "num_steps", "noise_scale"
    ):
        add(
            "diffusion_config",
            "diffusion_processor must be a DiffusionProcessorConfig",
            "model.diffusion_processor",
        )
    if diffusion is not None and config.variant not in {"bgt", "longshot"}:
        add(
            "diffusion_variant",
            "diffusion_processor requires variant 'bgt' or 'longshot'",
            "model.diffusion_processor",
        )
    if diffusion is not None and flow is not None:
        add(
            "iterative_processor_conflict",
            "flow_processor and diffusion_processor are mutually exclusive",
            "model.diffusion_processor",
        )

    if processor is not None:
        if processor.output_scale_mode == "fixed_one":
            if config.variant in {"bt", "icarus"}:
                add(
                    "fixed_output_scale_variant",
                    "fixed-one cache output scaling requires variant 'bgt' or 'longshot'",
                    "model.cache_processor.output_scale_mode",
                )
            if processor.out_proj_init != "zero":
                add(
                    "fixed_output_scale_readout_init",
                    "fixed-one cache output scaling requires cache_processor.out_proj_init='zero' to preserve the initial no-op",
                    "model.cache_processor.out_proj_init",
                )
        if (
            diffusion is not None
            and diffusion.training_method == "scheduled"
            and processor.out_proj_init != "zero"
        ):
            add(
                "scheduled_diffusion_readout_init",
                "scheduled diffusion requires cache_processor.out_proj_init='zero' because it writes without learned scalar gates",
                "model.cache_processor.out_proj_init",
            )
        if (
            config.variant == "longshot"
            and diffusion is not None
            and processor.iteration_mode == "internal"
            and processor.iterations > 1
        ):
            add(
                "diffusion_recurrence",
                "Diffusion-Longshot does not support internal cache_processor iterations greater than one; use diffusion solver steps",
                "model.cache_processor.iterations",
            )
        if config.variant == "bt":
            if processor.processor_mode != "standard":
                add(
                    "bt_processor_mode",
                    f"cache_processor.processor_mode='{processor.processor_mode}' is currently BGT-only",
                    "model.cache_processor.processor_mode",
                )
            if processor.score != "attn_l2":
                add(
                    "bt_score",
                    "cache_processor.score='kv_norm' is currently BGT-only",
                    "model.cache_processor.score",
                )
        elif config.variant == "bgt":
            if (
                processor.processor_mode == "workspace_broadcast"
                and processor.update_stream != "v"
            ):
                add(
                    "workspace_broadcast_write",
                    "workspace_broadcast requires cache_processor.update_stream='v'",
                    "model.cache_processor.update_stream",
                )
            if (
                processor.processor_mode == "workspace_broadcast"
                and processor.score != "kv_norm"
            ):
                add(
                    "workspace_broadcast_score",
                    "workspace_broadcast requires cache_processor.score='kv_norm'",
                    "model.cache_processor.score",
                )
            if (
                processor.processor_mode == "head_mask_broadcast"
                and processor.update_stream != "kv"
            ):
                add(
                    "head_mask_write",
                    "head_mask_broadcast requires cache_processor.update_stream='kv'",
                    "model.cache_processor.update_stream",
                )
            if (
                processor.processor_mode == "head_mask_broadcast"
                and processor.score != "kv_norm"
            ):
                add(
                    "head_mask_score",
                    "head_mask_broadcast requires cache_processor.score='kv_norm'",
                    "model.cache_processor.score",
                )
        elif (
            config.variant in {"icarus", "longshot"}
            and processor.processor_mode != "standard"
        ):
            add(
                "processor_mode",
                f"{config.variant.capitalize()} supports only cache_processor.processor_mode='standard'",
                "model.cache_processor.processor_mode",
            )
        if (
            config.variant == "longshot"
            and flow is not None
            and processor.iteration_mode == "internal"
            and processor.iterations > 1
        ):
            add(
                "flow_recurrence",
                "Flow-Longshot does not support internal cache_processor iterations greater than one",
                "model.cache_processor.iterations",
            )
        if processor.linear_state_cache:
            if config.variant != "longshot":
                add(
                    "linear_state_variant",
                    "cache_processor.linear_state_cache is supported only for variant='longshot'",
                    "model.cache_processor.linear_state_cache",
                )
            if config.family not in {"qwen3_5", "qwen3_5_moe"}:
                add(
                    "linear_state_family",
                    "cache_processor.linear_state_cache requires family='qwen3_5' or 'qwen3_5_moe'",
                    "model.family",
                )
            if processor.encoder_method != "query_attention":
                add(
                    "linear_state_encoder",
                    "cache_processor.linear_state_cache requires encoder_method='query_attention'",
                    "model.cache_processor.encoder_method",
                )
            if processor.encoder_query_source != "final_token":
                add(
                    "linear_state_query",
                    "cache_processor.linear_state_cache requires encoder_query_source='final_token'",
                    "model.cache_processor.encoder_query_source",
                )
            if processor.update_stream != "kv":
                add(
                    "linear_state_write",
                    "cache_processor.linear_state_cache requires update_stream='kv'",
                    "model.cache_processor.update_stream",
                )
        if (
            processor.linear_state_update_mode == "gated_delta"
            and not processor.linear_state_cache
        ):
            add(
                "linear_state_update_mode",
                "cache_processor.linear_state_update_mode='gated_delta' requires linear_state_cache=true",
                "model.cache_processor.linear_state_update_mode",
            )

    if config.variant != "tf" and config.family == "gemma4":
        try:
            config.ensure_uniform_stored_cache_shape(
                config.variant.capitalize()
            )
        except ValueError as error:
            add("uniform_cache_shape", str(error), "model.layer_types")
    return _unique_issues(issues)


def _experiment_issues(config: TrainConfig) -> list[CompatibilityIssue]:
    issues: list[CompatibilityIssue] = []

    def add(code: str, message: str, path: str) -> None:
        issues.append(CompatibilityIssue(code, message, path))

    if config.data.kind not in _SUPPORTED_DATA_KINDS:
        add(
            "unknown_data_kind",
            f"unsupported data kind: {config.data.kind}",
            "data.kind",
        )
    # Field-level enum/policy sanity (``trainable``, ``precision``, ``loss_backend``,
    # activation checkpointing) is owned by ``TrainConfig.__post_init__``; this module
    # only answers the cross-field and protocol questions the dataclass cannot.
    if config.trainable == "cache_processor" and config.model.variant == "tf":
        add(
            "processor_training_without_processor",
            "trainable='cache_processor' requires a cache processor",
            "trainable",
        )
    if config.checkpoint_format not in _SUPPORTED_CHECKPOINT_FORMATS:
        add(
            "unknown_checkpoint_format",
            f"unsupported checkpoint format: {config.checkpoint_format}",
            "checkpoint_format",
        )
    if config.mode not in {"pretrain", "sft"}:
        add(
            "unknown_training_mode",
            f"unknown training mode: {config.mode}",
            "mode",
        )
    if config.distributed not in {"none", "ddp", "fsdp"}:
        add(
            "unknown_execution",
            f"unknown distributed strategy: {config.distributed}",
            "distributed",
        )
    if (
        config.init_checkpoint is not None
        and config.backbone_checkpoint is not None
    ):
        add(
            "backbone_with_init",
            "backbone_checkpoint and init_checkpoint are mutually exclusive",
            "init_checkpoint",
        )
    if config.init_checkpoint is not None and config.resume is not None:
        add(
            "init_with_resume",
            "init_checkpoint and resume are mutually exclusive",
            "resume",
        )
    if config.mode == "pretrain" and config.data.kind not in {
        "fineweb",
        "random",
    }:
        add(
            "pretrain_data",
            "pretrain mode requires fineweb or random data",
            "data.kind",
        )
    if config.mode == "sft" and config.data.kind != "sft":
        add("sft_data", "sft mode requires data.kind='sft'", "data.kind")
    if (
        config.data.canonical_execution
        and config.resume is not None
        and config.resume.expected_manifest_sha256 is None
    ):
        add(
            "canonical_resume_manifest",
            "canonical training resume requires expected manifest sha256",
            "resume.expected_manifest_sha256",
        )

    return _unique_issues(issues)


def _unique_issues(
    issues: list[CompatibilityIssue],
) -> list[CompatibilityIssue]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[CompatibilityIssue] = []
    for issue in issues:
        key = (issue.code, issue.message, issue.path)
        if key not in seen:
            unique.append(issue)
            seen.add(key)
    return unique
