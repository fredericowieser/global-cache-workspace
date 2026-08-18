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
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("resume.expected_manifest_sha256 must be a lowercase sha256 or null")


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
            raise ValueError("wandb.resume must be 'allow', 'must', 'never', 'auto', or null")
        if self.resume == "must" and not self.id:
            raise ValueError("wandb.id is required when wandb.resume='must'")
        if not isinstance(self.tags, list) or any(
            not isinstance(tag, str) or not 1 <= len(tag) <= 64 for tag in self.tags
        ):
            raise ValueError("wandb.tags must contain only strings between 1 and 64 characters")


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
        if self.freeze_backbone and self.trainable == "all":
            self.trainable = "cache_processor"
        if self.save_cache_processor_only is None:
            self.save_cache_processor_only = self.freeze_backbone
        if self.precision is None:
            self.precision = "bf16_mixed" if self.bf16 else "fp32"
        elif self.precision not in {"fp32", "bf16_mixed", "bf16"}:
            raise ValueError(f"unsupported training precision {self.precision!r}")
        elif self.bf16 and self.precision == "fp32":
            raise ValueError("bf16=true conflicts with precision='fp32'")
        self.bf16 = self.precision != "fp32"
        if self.loss_backend not in {"reference", "target_only", "liger"}:
            raise ValueError("loss_backend must be 'reference', 'target_only', or 'liger'")
        if self.trainable not in get_args(TrainableComponents):
            raise ValueError("trainable must be 'all', 'backbone', 'lora', or 'cache_processor'")
        if self.activation_checkpointing not in {"none", "standard_processor_layers"}:
            raise ValueError("activation_checkpointing must be 'none' or 'standard_processor_layers'")
        if self.activation_checkpointing == "standard_processor_layers" and self.model.variant not in {"bt", "bgt"}:
            raise ValueError(
                "activation_checkpointing='standard_processor_layers' requires a BT or BGT cache processor"
            )
        # Backbone chunk compilation: an eager-vs-compiled windowed-Qwen3 backend
        # switch. Non-eager requires a windowed processor trained cache-processor-only.
        if self.backbone_chunk_backend not in {"eager", "compile", "compile_full_windows", "cuda_graph"}:
            raise ValueError(
                "backbone_chunk_backend must be 'eager', 'compile', 'compile_full_windows', or 'cuda_graph'"
            )
        # Optional Longshot segmented activation-checkpointing threshold.
        threshold = self.longshot_gradient_checkpointing_seq_len_threshold
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0:
                raise ValueError("longshot_gradient_checkpointing_seq_len_threshold must be a positive integer or null")
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
        if self.trainable == "lora" and self.model.lora is None:
            raise ValueError("trainable='lora' requires model.lora")
        if self.total_batch_size and self.global_batch_rows:
            raise ValueError("set only one of total_batch_size and global_batch_rows")
        if self.global_batch_rows < 0:
            raise ValueError("global_batch_rows must be non-negative")
        if isinstance(self.activation_checkpoint_windows, bool) or not isinstance(
            self.activation_checkpoint_windows, int
        ):
            raise ValueError("activation_checkpoint_windows must be an integer")
        if self.activation_checkpoint_windows < 0:
            raise ValueError("activation_checkpoint_windows must be non-negative")
        if isinstance(self.activation_checkpoint_min_padded_sequence_length, bool) or not isinstance(
            self.activation_checkpoint_min_padded_sequence_length, int
        ):
            raise ValueError("activation_checkpoint_min_padded_sequence_length must be an integer")
        if self.activation_checkpoint_min_padded_sequence_length < 0:
            raise ValueError("activation_checkpoint_min_padded_sequence_length must be non-negative")
        if self.activation_checkpoint_windows and self.model.variant == "tf":
            raise ValueError("activation_checkpoint_windows requires a recurrent cache-processor variant")
        if self.activation_checkpoint_windows and self.model.family != "qwen3":
            raise ValueError("activation_checkpoint_windows currently supports only the qwen3 backbone family")
        if self.activation_checkpoint_windows and self.distributed == "fsdp":
            raise ValueError("activation_checkpoint_windows cannot be combined with FSDP layer checkpointing")
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
    tasks: list[str] = field(default_factory=list)
    batch_size: int = 8
    limit: int | None = None
