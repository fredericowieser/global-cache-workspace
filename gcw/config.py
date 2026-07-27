from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import yaml


Variant = Literal["tf", "bt", "bgt", "icarus", "longshot"]
Family = Literal["smollm", "llama", "qwen3", "qwen3_5_moe"]
RotaryStyle = Literal["llama", "ppt"]
MoeExpertsImplementation = Literal["grouped_mm"]
CacheUpdateStream = Literal["kv", "v"]
CacheOutProjInit = Literal["normal", "zero"]
CacheGateMode = Literal["sigmoid", "direct"]
CacheProcessorMode = Literal["standard", "workspace_broadcast", "head_mask_broadcast"]
CacheEncoderMethod = Literal["linear", "query_attention"]
CacheEncoderQuerySource = Literal["learned", "final_token"]
CacheProcessorIterationMode = Literal["internal", "external"]
CacheProcessorInternalIterationScore = Literal["attn_l2", "cache_head_attention", "kv_norm"]

_FLOAT32_MAX = 3.4028234663852886e38


def _validated_float32(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or abs(result) > _FLOAT32_MAX:
        raise ValueError(f"{name} must be finite and representable as float32")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass
class CacheProcessorConfig:
    process_every: int = 64
    topk: int = 32
    topk_heads: int = 32
    score: str = "attn_l2"
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_layers: int = 1
    num_heads: int = 8
    num_kv_heads: int = 8
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    gate_init: float = -5.0
    attention_dropout: float = 0.0
    per_cache_head_projections: bool = False
    gradient_checkpointing: bool = False
    update_stream: CacheUpdateStream = "kv"
    out_proj_init: CacheOutProjInit = "normal"
    gate_mode: CacheGateMode = "sigmoid"
    processor_mode: CacheProcessorMode = "standard"
    writer_head_fraction: float = 1.0
    salience_temperature: float = 1.0
    kv_score_init_k: float = 0.25
    kv_score_init_v: float = 0.75
    encoder_method: CacheEncoderMethod = "linear"
    encoder_seq_len: int | None = None
    encoder_query_source: CacheEncoderQuerySource = "learned"
    unrope_keys: bool = False
    include_self_head_kv: bool = False
    iterations: int = 1
    iteration_mode: CacheProcessorIterationMode = "internal"
    internal_iteration_score: CacheProcessorInternalIterationScore = "attn_l2"

    def __post_init__(self) -> None:
        if self.score not in {"attn_l2", "kv_norm"}:
            raise ValueError("cache_processor.score must be 'attn_l2' or 'kv_norm'")
        if self.topk_heads <= 0:
            raise ValueError("cache_processor.topk_heads must be positive")
        if self.update_stream not in {"kv", "v"}:
            raise ValueError("cache_processor.update_stream must be 'kv' or 'v'")
        if self.out_proj_init not in {"normal", "zero"}:
            raise ValueError("cache_processor.out_proj_init must be 'normal' or 'zero'")
        if self.gate_mode not in {"sigmoid", "direct"}:
            raise ValueError("cache_processor.gate_mode must be 'sigmoid' or 'direct'")
        if self.processor_mode not in {"standard", "workspace_broadcast", "head_mask_broadcast"}:
            raise ValueError("cache_processor.processor_mode must be 'standard', 'workspace_broadcast', or 'head_mask_broadcast'")
        if not 0.0 < self.writer_head_fraction <= 1.0:
            raise ValueError("cache_processor.writer_head_fraction must be in (0, 1]")
        if self.salience_temperature <= 0.0:
            raise ValueError("cache_processor.salience_temperature must be positive")
        if self.kv_score_init_k <= 0.0 or self.kv_score_init_v <= 0.0:
            raise ValueError("cache_processor.kv_score_init_k and kv_score_init_v must be positive")
        if self.encoder_method not in {"linear", "query_attention"}:
            raise ValueError("cache_processor.encoder_method must be 'linear' or 'query_attention'")
        if self.encoder_seq_len is not None and self.encoder_seq_len <= 0:
            raise ValueError("cache_processor.encoder_seq_len must be positive when set")
        if self.encoder_query_source not in {"learned", "final_token"}:
            raise ValueError("cache_processor.encoder_query_source must be 'learned' or 'final_token'")
        if self.iterations < 1:
            raise ValueError("cache_processor.iterations must be positive")
        if self.iteration_mode not in {"internal", "external"}:
            raise ValueError("cache_processor.iteration_mode must be 'internal' or 'external'")
        if self.internal_iteration_score not in {"attn_l2", "cache_head_attention", "kv_norm"}:
            raise ValueError(
                "cache_processor.internal_iteration_score must be 'attn_l2', 'cache_head_attention', or 'kv_norm'"
            )

    @property
    def stream_factor(self) -> int:
        if self.update_stream == "kv":
            return 2
        if self.update_stream == "v":
            return 1
        raise ValueError(f"unsupported cache_processor.update_stream: {self.update_stream}")


@dataclass
class FlowProcessorConfig:
    num_steps: int = 1
    time_conditioning: bool = True
    eta_max: float = 1.0
    flow_gate_init: float = -4.0

    def __post_init__(self) -> None:
        if isinstance(self.num_steps, bool) or not isinstance(self.num_steps, int) or self.num_steps <= 0:
            raise ValueError("flow_processor.num_steps must be a positive integer")
        if not isinstance(self.time_conditioning, bool):
            raise ValueError("flow_processor.time_conditioning must be a boolean")
        self.eta_max = _validated_float32("flow_processor.eta_max", self.eta_max, positive=True)
        self.flow_gate_init = _validated_float32("flow_processor.flow_gate_init", self.flow_gate_init)


@dataclass
class ModelConfig:
    variant: Variant = "tf"
    family: Family = "smollm"
    sequence_len: int = 2048
    vocab_size: int = 49152
    hidden_size: int = 960
    intermediate_size: int = 2560
    num_layers: int = 32
    num_heads: int = 15
    num_kv_heads: int = 5
    head_dim: int | None = 64
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    qk_norm: bool = False
    rotary_style: RotaryStyle = "ppt"
    cache_processor: CacheProcessorConfig | None = None
    flow_processor: FlowProcessorConfig | None = None
    legacy_bottleneck_generate: bool = False
    layer_types: list[str] | None = None
    processor_layer_indices: list[int] | None = None
    partial_rotary_factor: float = 1.0
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    moe_intermediate_size: int | None = None
    num_experts: int = 0
    num_experts_per_tok: int = 0
    shared_expert_intermediate_size: int | None = None
    hidden_act: str = "silu"
    moe_experts_implementation: MoeExpertsImplementation = "grouped_mm"
    router_aux_loss_coef: float = 0.0
    output_router_logits: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.cache_processor, dict):
            self.cache_processor = CacheProcessorConfig(**self.cache_processor)
        if isinstance(self.flow_processor, dict):
            self.flow_processor = FlowProcessorConfig(**self.flow_processor)
        if self.variant in {"bt", "bgt", "icarus", "longshot"} and self.cache_processor is None:
            self.cache_processor = CacheProcessorConfig()
        if self.flow_processor is not None:
            if self.variant not in {"bgt", "icarus", "longshot"}:
                raise ValueError("flow_processor requires variant 'bgt', 'icarus', or 'longshot'")
            if self.cache_processor is None:
                raise ValueError("flow_processor requires cache_processor configuration")
            if self.variant == "longshot" and self.cache_processor.iteration_mode == "internal" and self.cache_processor.iterations > 1:
                raise ValueError("Flow-Longshot does not support internal cache_processor iterations greater than one")
        if self.layer_types is not None:
            self.layer_types = list(self.layer_types)
            if len(self.layer_types) != self.num_layers:
                raise ValueError("layer_types must have length num_layers")
            unknown = sorted(set(self.layer_types) - {"full_attention", "linear_attention"})
            if unknown:
                raise ValueError(f"unsupported layer_types: {unknown}")
        if self.processor_layer_indices is not None:
            self.processor_layer_indices = [int(i) for i in self.processor_layer_indices]
            if any(i < 0 or i >= self.num_layers for i in self.processor_layer_indices):
                raise ValueError("processor_layer_indices entries must be valid layer indices")
            if self.processor_layer_indices != sorted(set(self.processor_layer_indices)):
                raise ValueError("processor_layer_indices must be sorted and unique")
            if self.layer_types is not None:
                non_cache = [i for i in self.processor_layer_indices if self.layer_types[i] != "full_attention"]
                if non_cache:
                    raise ValueError("processor_layer_indices must reference full_attention layers")
        if self.family == "qwen3_5_moe":
            if self.moe_experts_implementation != "grouped_mm":
                raise ValueError("qwen3_5_moe supports only moe_experts_implementation='grouped_mm'")
            if self.layer_types is None:
                raise ValueError("qwen3_5_moe requires layer_types")
            if self.moe_intermediate_size is None:
                self.moe_intermediate_size = self.intermediate_size
            if self.shared_expert_intermediate_size is None:
                self.shared_expert_intermediate_size = self.moe_intermediate_size
            if self.linear_key_head_dim is None:
                self.linear_key_head_dim = int(self.head_dim or self.hidden_size // self.num_heads)
            if self.linear_value_head_dim is None:
                self.linear_value_head_dim = self.linear_key_head_dim
            if self.linear_num_key_heads is None:
                self.linear_num_key_heads = self.num_kv_heads
            if self.linear_num_value_heads is None:
                self.linear_num_value_heads = self.num_heads
            if self.num_experts <= 0:
                raise ValueError("qwen3_5_moe requires num_experts")
            if self.num_experts_per_tok <= 0:
                raise ValueError("qwen3_5_moe requires num_experts_per_tok")
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if not 0.0 < self.partial_rotary_factor <= 1.0:
            raise ValueError("partial_rotary_factor must be in (0, 1]")
    @property
    def architecture_name(self) -> str:
        if self.flow_processor is not None:
            return f"flow_{self.variant}"
        return self.variant

    @property
    def cache_layer_indices(self) -> list[int]:
        if self.processor_layer_indices is not None:
            return self.processor_layer_indices
        if self.layer_types is not None:
            return [i for i, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"]
        return list(range(self.num_layers))

    @property
    def cache_num_layers(self) -> int:
        return len(self.cache_layer_indices)

    @property
    def cache_num_heads(self) -> int:
        return self.cache_num_layers * self.num_kv_heads

    @property
    def _experts_implementation(self) -> str:
        return self.moe_experts_implementation


@dataclass
class DataConfig:
    kind: str = "fineweb"
    data_dir: str | None = None
    dataset: str | None = None
    tokenizer: str = "cosmo2"
    tokenizer_source: str | None = None
    text_field: str = "text"
    max_length: int = 2048
    sft_skip_rows: int = 0
    sft_take_rows: int | None = None
    response_prefix: str = ""


@dataclass(frozen=True)
class ResumeConfig:
    checkpoint: str
    topology: Literal["same", "elastic_ddp"] = "same"

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("resume.checkpoint must not be empty")
        if self.topology not in {"same", "elastic_ddp"}:
            raise ValueError("resume.topology must be 'same' or 'elastic_ddp'")


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


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mode: Literal["pretrain", "sft"] = "pretrain"
    distributed: Literal["none", "ddp", "fsdp"] = "none"
    freeze_backbone: bool = False
    backbone_checkpoint: str | None = None
    resume: ResumeConfig | None = None
    loss: str = "next_token_ce"
    checkpoint_format: str = "native_safetensors"
    output_dir: str = "runs/smoke"
    batch_size: int = 1
    sequence_len: int = 128
    max_steps: int = 2
    total_batch_size: int = 0
    learning_rate: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    warmup_steps: int = 1
    warmdown_ratio: float = 0.2
    final_lr_frac: float = 0.0
    grad_clip: float = 1.0
    bf16: bool = False
    seed: int = 24
    save_every: int = 0
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
        if self.resume is not None and self.backbone_checkpoint is not None:
            raise ValueError("resume and backbone_checkpoint cannot be configured together")


@dataclass
class EvalConfig:
    tasks: list[str] = field(default_factory=list)
    batch_size: int = 8
    limit: int | None = None


SMOLLM_CONFIGS: dict[str, ModelConfig] = {
    "smollm-135m": ModelConfig(num_layers=30, hidden_size=576, num_heads=9, num_kv_heads=3, head_dim=64, intermediate_size=1536),
    "smollm-360m": ModelConfig(num_layers=32, hidden_size=960, num_heads=15, num_kv_heads=5, head_dim=64, intermediate_size=2560),
    "smollm-1.7b": ModelConfig(num_layers=24, hidden_size=2048, num_heads=32, num_kv_heads=32, head_dim=64, intermediate_size=8192),
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a mapping")
    return data


def load_train_config(path: str | Path) -> TrainConfig:
    return TrainConfig(**load_yaml(path))


def to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: to_dict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, list):
        return [to_dict(x) for x in value]
    if isinstance(value, dict):
        return {k: to_dict(v) for k, v in value.items()}
    return value


def dump_yaml(path: str | Path, value: Any) -> None:
    data = to_dict(value)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
