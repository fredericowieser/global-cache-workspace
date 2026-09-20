"""Training configuration and the admission gate for execution features.

``TrainConfig`` binds a :class:`~gcw.config.ModelConfig`, data, optimizer, and
the opt-in ``performance`` execution policies. Its ``__post_init__`` keeps
field-level sanity (ranges, enum membership, coercion) and delegates the
cross-feature composition admission to :func:`gcw.architecture.execution.resolve_execution_plan`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal, get_args

from gcw.config.data import DataConfig
from gcw.config.model import ModelConfig
from gcw.config.types import (
    ActivationCheckpointingPolicy,
    BackboneChunkBackend,
    LossBackend,
    TrainableComponents,
    TrainingPrecision,
)
from gcw.performance.config import PerformanceConfig


@dataclass(frozen=True)
class ResumeConfig:
    checkpoint: str
    topology: Literal["same", "elastic_ddp"] = "same"
    expected_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("resume.checkpoint must not be empty")
        if self.topology not in {"same", "elastic_ddp"}:
            raise ValueError("resume.topology must be 'same' or 'elastic_ddp'")
        digest = self.expected_manifest_sha256
        if digest is not None and (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "resume.expected_manifest_sha256 must be a lowercase sha256 or null"
            )


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "global-cache-workspace"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    mode: str | None = None
    id: str | None = None
    resume: Literal["allow", "must", "never", "auto"] | None = None
    tags: list[str] = field(default_factory=list)
    log_every: int = 1

    def __post_init__(self) -> None:
        if self.resume not in {None, "allow", "must", "never", "auto"}:
            raise ValueError(
                "wandb.resume must be 'allow', 'must', 'never', 'auto', or null"
            )
        if self.resume == "must" and not self.id:
            raise ValueError("wandb.id is required when wandb.resume='must'")
        if not isinstance(self.tags, list) or any(
            not isinstance(tag, str) or not 1 <= len(tag) <= 64
            for tag in self.tags
        ):
            raise ValueError(
                "wandb.tags must contain only strings between 1 and 64 characters"
            )


@dataclass(frozen=True)
class LongshotSelectorTrainingConfig:
    warmup_end: float = 0.05
    soft_end: float = 0.15
    second_hardening_start: float = 0.60
    hard_start: float = 0.90
    initial_temperature: float = 2.0
    intermediate_temperature: float = 0.35
    final_temperature: float = 0.1
    initial_suppression: float = 0.0
    intermediate_suppression: float = 2.0
    final_suppression: float = 5.0

    def __post_init__(self) -> None:
        boundaries = (
            self.warmup_end,
            self.soft_end,
            self.second_hardening_start,
            self.hard_start,
        )
        if not all(math.isfinite(value) for value in boundaries) or not (
            0.0
            <= self.warmup_end
            <= self.soft_end
            < self.second_hardening_start
            < self.hard_start
            <= 1.0
        ):
            raise ValueError(
                "longshot_selector_training phase boundaries must satisfy "
                "0 <= warmup_end <= soft_end < second_hardening_start < hard_start <= 1"
            )
        temperatures = (
            self.initial_temperature,
            self.intermediate_temperature,
            self.final_temperature,
        )
        if not all(
            math.isfinite(value) and value > 0.0 for value in temperatures
        ) or not (
            self.initial_temperature
            >= self.intermediate_temperature
            >= self.final_temperature
        ):
            raise ValueError(
                "longshot_selector_training temperatures must be finite, positive, and non-increasing"
            )
        suppressions = (
            self.initial_suppression,
            self.intermediate_suppression,
            self.final_suppression,
        )
        if not all(
            math.isfinite(value) and value >= 0.0 for value in suppressions
        ) or not (
            self.initial_suppression
            <= self.intermediate_suppression
            <= self.final_suppression
        ):
            raise ValueError(
                "longshot_selector_training suppressions must be finite, non-negative, and non-decreasing"
            )


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mode: Literal["pretrain", "sft"] = "pretrain"
    distributed: Literal["none", "ddp", "fsdp"] = "none"
    trainable: TrainableComponents = "all"
    activation_checkpointing: ActivationCheckpointingPolicy = "none"
    backbone_chunk_backend: BackboneChunkBackend = "eager"
    loss_backend: LossBackend = "reference"
    longshot_gradient_checkpointing_seq_len_threshold: int | None = None
    backbone_checkpoint: str | None = None
    init_checkpoint: str | None = None
    resume: ResumeConfig | None = None
    checkpoint_format: str = "native_safetensors"
    output_dir: str = "runs/smoke"
    batch_size: int = 1
    sequence_len: int = 128
    max_steps: int = 2
    total_batch_size: int = 0
    global_batch_rows: int = 0
    logical_pair_grouping: bool = False
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    longshot_selector_training: LongshotSelectorTrainingConfig | None = None
    learning_rate: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    warmup_steps: int = 1
    warmdown_ratio: float = 0.2
    final_lr_frac: float = 0.0
    grad_clip: float = 1.0
    precision: TrainingPrecision | None = None
    bf16: bool = False
    loss: str | None = None
    # Save only cache-processor weights; the backbone is reconstructed from
    # ``backbone_checkpoint`` on load. ``None`` follows ``freeze_backbone``.
    save_cache_processor_only: bool | None = None
    freeze_backbone: bool = False
    activation_checkpoint_windows: int = 0
    activation_checkpoint_min_padded_sequence_length: int = 0
    # Number of cache-processing windows kept in one autograd horizon. Zero
    # preserves full BPTT; positive values detach recurrent cache state at each
    # horizon boundary without changing the forward recurrence.
    truncated_bptt_windows: int = 0
    seed: int = 24
    save_every: int = 0
    save_final: bool = True
    eval_every: int = 0
    eval_batches: int = 10
    eval_on_start: bool = False
    wandb: WandbConfig = field(default_factory=WandbConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.wandb, dict):
            self.wandb = WandbConfig(**self.wandb)
        if isinstance(self.resume, dict):
            self.resume = ResumeConfig(**self.resume)
        if isinstance(self.performance, dict):
            self.performance = PerformanceConfig(**self.performance)
        elif not isinstance(self.performance, PerformanceConfig):
            raise ValueError("performance must be a mapping")
        if isinstance(self.longshot_selector_training, dict):
            self.longshot_selector_training = LongshotSelectorTrainingConfig(
                **self.longshot_selector_training
            )
        elif self.longshot_selector_training is not None and not isinstance(
            self.longshot_selector_training, LongshotSelectorTrainingConfig
        ):
            raise ValueError("longshot_selector_training must be a mapping")
        selector_training = self.longshot_selector_training
        if selector_training is not None:
            processor = self.model.cache_processor
            if self.model.variant != "longshot" or processor is None:
                raise ValueError(
                    "longshot_selector_training requires a Longshot cache processor"
                )
            learned_initial_selection = (
                processor.head_selection_method == "learned_contextual"
            )
            learned_iterative_selection = (
                processor.iteration_mode == "internal"
                and processor.iterations > 1
                and processor.internal_iteration_score == "learned_contextual"
            )
            if (
                not learned_initial_selection
                and not learned_iterative_selection
            ):
                raise ValueError(
                    "longshot_selector_training requires learned_contextual initial "
                    "selection or learned contextual reselection across multiple "
                    "internal iterations"
                )
            if self.model.flow_processor is not None:
                raise ValueError(
                    "longshot_selector_training cannot be combined with flow"
                )
            if (
                processor.iteration_mode == "internal"
                and processor.iterations > 1
                and processor.internal_iteration_score != "learned_contextual"
            ):
                raise ValueError(
                    "soft Longshot selector training with multiple internal iterations requires "
                    "internal_iteration_score='learned_contextual'"
                )
        if self.freeze_backbone and self.trainable == "all":
            self.trainable = "cache_processor"
        if self.save_cache_processor_only is None:
            self.save_cache_processor_only = self.freeze_backbone
        if self.precision is None:
            self.precision = "bf16_mixed" if self.bf16 else "fp32"
        elif self.precision not in get_args(TrainingPrecision):
            raise ValueError(
                f"unsupported training precision {self.precision!r}; "
                f"use one of {', '.join(get_args(TrainingPrecision))}"
            )
        elif self.bf16 and self.precision == "fp32":
            raise ValueError("bf16=true conflicts with precision='fp32'")
        self.bf16 = self.precision != "fp32"
        if self.loss_backend not in {"reference", "target_only", "liger"}:
            raise ValueError(
                "loss_backend must be 'reference', 'target_only', or 'liger'"
            )
        if self.trainable not in get_args(TrainableComponents):
            raise ValueError(
                "trainable must be 'all', 'backbone', 'lora', "
                "'cache_processor', or 'lora_and_cache_processor'"
            )
        if self.activation_checkpointing not in {
            "none",
            "standard_processor_layers",
        }:
            raise ValueError(
                "activation_checkpointing must be 'none' or 'standard_processor_layers'"
            )
        if (
            self.activation_checkpointing == "standard_processor_layers"
            and self.model.variant not in {"bt", "bgt"}
        ):
            raise ValueError(
                "activation_checkpointing='standard_processor_layers' requires a BT or BGT cache processor"
            )
        # Backbone chunk compilation: an eager-vs-compiled windowed-Qwen3 backend
        # switch. Non-eager requires a windowed processor trained cache-processor-only.
        if self.backbone_chunk_backend not in {
            "eager",
            "compile",
            "compile_full_windows",
            "cuda_graph",
        }:
            raise ValueError(
                "backbone_chunk_backend must be 'eager', 'compile', 'compile_full_windows', or 'cuda_graph'"
            )
        # Optional Longshot segmented activation-checkpointing threshold.
        threshold = self.longshot_gradient_checkpointing_seq_len_threshold
        if threshold is not None:
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, int)
                or threshold <= 0
            ):
                raise ValueError(
                    "longshot_gradient_checkpointing_seq_len_threshold must be a positive integer or null"
                )
            processor = self.model.cache_processor
            if self.model.variant != "longshot" or processor is None:
                raise ValueError(
                    "longshot_gradient_checkpointing_seq_len_threshold requires a Longshot cache processor"
                )
            if not processor.gradient_checkpointing:
                raise ValueError(
                    "longshot_gradient_checkpointing_seq_len_threshold requires "
                    "model.cache_processor.gradient_checkpointing=true"
                )
        if self.trainable in {"lora", "lora_and_cache_processor"}:
            if self.model.lora is None:
                raise ValueError(
                    f"trainable={self.trainable!r} requires model.lora"
                )
        if self.trainable == "lora_and_cache_processor":
            if self.model.cache_processor is None or self.model.variant == "tf":
                raise ValueError(
                    "trainable='lora_and_cache_processor' requires a cache processor"
                )
            if self.freeze_backbone or self.save_cache_processor_only:
                raise ValueError(
                    "trainable='lora_and_cache_processor' requires "
                    "freeze_backbone=false and save_cache_processor_only=false "
                    "so checkpoints retain both adapters and processor weights"
                )
        if self.total_batch_size and self.global_batch_rows:
            raise ValueError(
                "set only one of total_batch_size and global_batch_rows"
            )
        if self.global_batch_rows < 0:
            raise ValueError("global_batch_rows must be non-negative")
        if isinstance(
            self.activation_checkpoint_windows, bool
        ) or not isinstance(self.activation_checkpoint_windows, int):
            raise ValueError("activation_checkpoint_windows must be an integer")
        if self.activation_checkpoint_windows < 0:
            raise ValueError(
                "activation_checkpoint_windows must be non-negative"
            )
        if isinstance(
            self.activation_checkpoint_min_padded_sequence_length, bool
        ) or not isinstance(
            self.activation_checkpoint_min_padded_sequence_length, int
        ):
            raise ValueError(
                "activation_checkpoint_min_padded_sequence_length must be an integer"
            )
        if self.activation_checkpoint_min_padded_sequence_length < 0:
            raise ValueError(
                "activation_checkpoint_min_padded_sequence_length must be non-negative"
            )
        if isinstance(self.truncated_bptt_windows, bool) or not isinstance(
            self.truncated_bptt_windows, int
        ):
            raise ValueError("truncated_bptt_windows must be an integer")
        if self.truncated_bptt_windows < 0:
            raise ValueError("truncated_bptt_windows must be non-negative")
        if self.truncated_bptt_windows and self.model.variant == "tf":
            raise ValueError(
                "truncated_bptt_windows requires a cache-processor variant"
            )
        if self.activation_checkpoint_windows and self.model.variant == "tf":
            raise ValueError(
                "activation_checkpoint_windows requires a recurrent cache-processor variant"
            )
        if self.activation_checkpoint_windows and self.model.family != "qwen3":
            raise ValueError(
                "activation_checkpoint_windows currently supports only the qwen3 backbone family"
            )
        if self.activation_checkpoint_windows and self.distributed == "fsdp":
            raise ValueError(
                "activation_checkpoint_windows cannot be combined with FSDP layer checkpointing"
            )
        if not isinstance(self.logical_pair_grouping, bool):
            raise ValueError("logical_pair_grouping must be a boolean")
        from gcw.architecture.execution import resolve_execution_plan

        resolve_execution_plan(self)
        if not math.isfinite(self.grad_clip) or self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be finite and positive")
        from gcw.architecture import require_train_config

        require_train_config(self)


@dataclass
class EvalConfig:
    batch_size: int | Literal["auto"] = 8
    max_batch_size: int = 64
    limit: int | None = None
    apply_chat_template: bool = False
    enable_thinking: bool = False
    cache_update_metrics_path: str | None = None
    log_samples: bool = False

    def __post_init__(self) -> None:
        if self.batch_size != "auto" and (
            type(self.batch_size) is not int or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be 'auto' or a positive integer")
        if type(self.max_batch_size) is not int or self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if self.limit is not None and (
            type(self.limit) is not int or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer")
        if type(self.apply_chat_template) is not bool:
            raise ValueError("apply_chat_template must be boolean")
        if type(self.log_samples) is not bool:
            raise ValueError("log_samples must be boolean")
        if type(self.enable_thinking) is not bool:
            raise ValueError("enable_thinking must be boolean")
        if self.cache_update_metrics_path is not None and (
            not isinstance(self.cache_update_metrics_path, str)
            or not self.cache_update_metrics_path.strip()
        ):
            raise ValueError(
                "cache_update_metrics_path must be a nonempty string"
            )
        if self.enable_thinking and not self.apply_chat_template:
            raise ValueError(
                "enable_thinking requires apply_chat_template=True"
            )

    @property
    def fixed_batch_size(self) -> int:
        if self.batch_size == "auto":
            raise ValueError(
                "batch_size='auto' is supported only by lm-eval suites"
            )
        return self.batch_size

    @property
    def default_max_new_tokens(self) -> int:
        return 4096 if self.enable_thinking else 2048
