from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, cast
from gcw.config.model import CacheProcessorConfig, DiffusionProcessorConfig
from gcw.performance.config import PerformanceConfig

if TYPE_CHECKING:
    from gcw.config import ModelConfig, TrainConfig

MODEL_ASSEMBLY_SCHEMA = "gcw.model_assembly.v1"
COMPOSITION_SCHEMA = "gcw.composition.v1"
MODEL_CONFIG_CONTRACT = "gcw.model_config.v1"
TRAIN_CONFIG_CONTRACT = "gcw.training.core_config.v1"

_SUPPORTED_VARIANTS = ("tf", "bt", "bgt", "icarus", "longshot")
_SUPPORTED_FAMILIES = (
    "smollm",
    "llama",
    "qwen3",
    "qwen3_5",
    "qwen3_5_moe",
    "gemma4",
)
_SUPPORTED_DATA_KINDS = (
    "random",
    "fineweb",
    "sft",
)
_SUPPORTED_CHECKPOINT_FORMATS = ("native_safetensors",)


def _config_payload(value: object) -> object:
    from gcw.config.training import TrainConfig

    if isinstance(value, TrainConfig):
        payload = {
            field.name: _config_payload(getattr(value, field.name))
            for field in fields(value)
        }
        if not value.truncated_bptt_windows:
            payload.pop("truncated_bptt_windows")
        return payload
    if isinstance(value, CacheProcessorConfig):
        payload = {
            field.name: _config_payload(getattr(value, field.name))
            for field in fields(value)
        }
        if not value.live_rank1_linear_writer:
            payload.pop("live_rank1_linear_writer")
        if value.output_scale_mode == "learned":
            payload.pop("output_scale_mode")
        if value.generation_process_every is None:
            payload.pop("generation_process_every")
        if (
            value.head_selection_method != "learned_contextual"
            and value.internal_iteration_score != "learned_contextual"
        ):
            payload.pop("head_selector_hidden_size")
            payload.pop("head_selector_use_identity_bias")
            if value.head_selection_method == "magnitude":
                payload.pop("head_selection_method")
        return payload
    if isinstance(value, DiffusionProcessorConfig):
        from gcw.config.serialization import to_dict

        return to_dict(value)
    if isinstance(value, PerformanceConfig):
        payload = {
            field.name: _config_payload(getattr(value, field.name))
            for field in fields(value)
        }
        if value.reference_loss.mode == "legacy":
            payload.pop("reference_loss")
        if not value.longshot_full_history.enabled:
            payload.pop("longshot_full_history")
        if not value.kv_score_sidecar.enabled:
            payload.pop("kv_score_sidecar")
        if not value.triton_cache_movement.enabled:
            payload.pop("triton_cache_movement")
        if not value.optimized_step.enabled:
            payload.pop("optimized_step")
        if value.noisy_diffusion.recurrence_backend == "eager":
            payload.pop("noisy_diffusion")
        return payload
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _config_payload(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, (list, tuple)):
        return [_config_payload(item) for item in value]
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        return {key: _config_payload(item) for key, item in mapping.items()}
    return value


def _has_fields(value: object, *names: str) -> bool:
    return all(hasattr(value, name) for name in names)


_MODEL_CONFIG_VALIDATION_CODES = {
    "diffusion_variant",
    "diffusion_recurrence",
    "fixed_output_scale_variant",
    "fixed_output_scale_readout_init",
    "scheduled_diffusion_readout_init",
    "flow_recurrence",
    "flow_variant",
    "linear_state_encoder",
    "linear_state_family",
    "linear_state_query",
    "linear_state_variant",
    "linear_state_write",
}
_TRAIN_CONFIG_VALIDATION_CODES = {
    "backbone_with_init",
    "canonical_resume_manifest",
    "init_with_resume",
    "pretrain_data",
    "processor_training_without_processor",
    "sft_data",
    "unknown_checkpoint_format",
    "unknown_data_kind",
}


@dataclass(frozen=True)
class CompatibilityIssue:
    code: str
    message: str
    path: str


@dataclass(frozen=True)
class CompatibilityReport:
    issues: tuple[CompatibilityIssue, ...] = ()

    @property
    def compatible(self) -> bool:
        return not self.issues

    def require(self) -> None:
        if self.issues:
            raise CompatibilityError(self)


class CompatibilityError(ValueError):
    def __init__(self, report: CompatibilityReport) -> None:
        self.report = report
        super().__init__("; ".join(issue.message for issue in report.issues))


def require_model_config(config: ModelConfig) -> None:
    from gcw.architecture.validation import _model_issues

    issues = tuple(
        issue
        for issue in _model_issues(config)
        if issue.code in _MODEL_CONFIG_VALIDATION_CODES
    )
    CompatibilityReport(issues).require()


def require_train_config(config: TrainConfig) -> None:
    from gcw.architecture.validation import _experiment_issues

    issues = tuple(
        issue
        for issue in _experiment_issues(config)
        if issue.code in _TRAIN_CONFIG_VALIDATION_CODES
    )
    CompatibilityReport(issues).require()
