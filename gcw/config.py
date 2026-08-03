from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import yaml


Variant = Literal["tf", "bt", "bgt", "icarus", "longshot"]
Family = Literal["smollm", "llama", "qwen3", "qwen3_5", "qwen3_5_moe", "gemma4"]
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
_FLOAT32_MIN_SUBNORMAL = 2.0**-149


def _validated_float32(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be finite and representable as float32") from None
    if not math.isfinite(result) or abs(result) > _FLOAT32_MAX:
        raise ValueError(f"{name} must be finite and representable as float32")
    if positive and result < _FLOAT32_MIN_SUBNORMAL:
        raise ValueError(f"{name} must remain positive when represented as float32")
    return result


def gemma4_default_layer_types(num_layers: int) -> list[str]:
    layer_types = ["sliding_attention" if bool((i + 1) % 6) else "full_attention" for i in range(num_layers)]
    if layer_types:
        layer_types[-1] = "full_attention"
    return layer_types


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
    sliding_window: int | None = None
    num_global_key_value_heads: int | None = None
    global_head_dim: int | None = None
    attention_k_eq_v: bool = False
    num_kv_shared_layers: int = 0
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int | None = None
    final_logit_softcapping: float | None = None
    rope_parameters: dict[str, Any] | None = None
    use_double_wide_mlp: bool = False

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
        if self.family == "gemma4" and self.layer_types is None:
            self.layer_types = gemma4_default_layer_types(self.num_layers)
        if self.layer_types is not None:
            self.layer_types = list(self.layer_types)
            if len(self.layer_types) != self.num_layers:
                raise ValueError("layer_types must have length num_layers")
            allowed = {"full_attention", "linear_attention", "sliding_attention"} if self.family == "gemma4" else {
                "full_attention",
                "linear_attention",
            }
            unknown = sorted(set(self.layer_types) - allowed)
            if unknown:
                raise ValueError(f"unsupported layer_types: {unknown}")
            if self.family == "gemma4" and self.layer_types[-1] != "full_attention":
                self.layer_types[-1] = "full_attention"
        if self.processor_layer_indices is not None:
            self.processor_layer_indices = [int(i) for i in self.processor_layer_indices]
            if any(i < 0 or i >= self.num_layers for i in self.processor_layer_indices):
                raise ValueError("processor_layer_indices entries must be valid layer indices")
            if self.processor_layer_indices != sorted(set(self.processor_layer_indices)):
                raise ValueError("processor_layer_indices must be sorted and unique")
            if self.layer_types is not None:
                cache_layers = set(self._default_cache_layer_indices())
                non_cache = [i for i in self.processor_layer_indices if i not in cache_layers]
                if non_cache:
                    raise ValueError("processor_layer_indices must reference cache-bearing layers")
        if self.family in {"qwen3_5", "qwen3_5_moe"}:
            if self.layer_types is None:
                raise ValueError(f"{self.family} requires layer_types")
            if self.linear_key_head_dim is None:
                self.linear_key_head_dim = int(self.head_dim or self.hidden_size // self.num_heads)
            if self.linear_value_head_dim is None:
                self.linear_value_head_dim = self.linear_key_head_dim
            if self.linear_num_key_heads is None:
                self.linear_num_key_heads = self.num_kv_heads
            if self.linear_num_value_heads is None:
                self.linear_num_value_heads = self.num_heads
        if self.family == "qwen3_5_moe":
            if self.moe_experts_implementation != "grouped_mm":
                raise ValueError("qwen3_5_moe supports only moe_experts_implementation='grouped_mm'")
            if self.moe_intermediate_size is None:
                self.moe_intermediate_size = self.intermediate_size
            if self.shared_expert_intermediate_size is None:
                self.shared_expert_intermediate_size = self.moe_intermediate_size
            if self.num_experts <= 0:
                raise ValueError("qwen3_5_moe requires num_experts")
            if self.num_experts_per_tok <= 0:
                raise ValueError("qwen3_5_moe requires num_experts_per_tok")
        if self.family == "gemma4":
            if self.layer_types is None:
                raise ValueError("gemma4 requires layer_types")
            if any(layer_type == "linear_attention" for layer_type in self.layer_types):
                raise ValueError("gemma4 does not support linear_attention layers")
            if self.sliding_window is None:
                self.sliding_window = 512
            if self.global_head_dim is None:
                self.global_head_dim = self.head_dim
            if self.vocab_size_per_layer_input is None:
                self.vocab_size_per_layer_input = self.vocab_size
            if self.num_kv_shared_layers < 0 or self.num_kv_shared_layers > self.num_layers:
                raise ValueError("num_kv_shared_layers must be in [0, num_layers]")
            if self.rope_parameters is None:
                self.rope_parameters = {
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                    "full_attention": {
                        "rope_type": "proportional",
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                    },
                }
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
        return self._default_cache_layer_indices()

    def _default_cache_layer_indices(self) -> list[int]:
        if self.layer_types is not None:
            if self.family == "gemma4":
                first_shared = self.num_layers - self.num_kv_shared_layers
                return [
                    i
                    for i, layer_type in enumerate(self.layer_types)
                    if layer_type in {"full_attention", "sliding_attention"} and i < first_shared
                ]
            return [i for i, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"]
        return list(range(self.num_layers))

    @property
    def cache_num_layers(self) -> int:
        return len(self.cache_layer_indices)

    @property
    def cache_num_heads(self) -> int:
        if self.family == "gemma4":
            return sum(self.kv_heads_for_layer(i) for i in self.cache_layer_indices)
        return self.cache_num_layers * self.num_kv_heads

    @property
    def stored_cache_shapes(self) -> list[tuple[int, int]]:
        return [(self.kv_heads_for_layer(i), self.head_dim_for_layer(i)) for i in self.cache_layer_indices]

    @property
    def stored_cache_shape(self) -> tuple[int, int]:
        shapes = self.stored_cache_shapes
        if not shapes:
            if self.head_dim is None:
                raise ValueError("stored cache shape requires a resolved head_dim")
            return self.num_kv_heads, self.head_dim
        unique = set(shapes)
        if len(unique) > 1:
            raise ValueError("stored cache layers do not have a uniform KV shape")
        return shapes[0]

    def ensure_uniform_stored_cache_shape(self, owner: str) -> None:
        if self.family != "gemma4":
            return
        if len(set(self.stored_cache_shapes)) > 1:
            raise ValueError(f"{owner} requires uniform stored KV shapes for Gemma4 cache processing")

    def kv_heads_for_layer(self, layer_idx: int) -> int:
        if (
            self.family == "gemma4"
            and self.layer_types is not None
            and self.layer_types[layer_idx] == "full_attention"
            and self.attention_k_eq_v
        ):
            return int(self.num_global_key_value_heads or self.num_kv_heads)
        return self.num_kv_heads

    def head_dim_for_layer(self, layer_idx: int) -> int:
        if self.family == "gemma4" and self.layer_types is not None and self.layer_types[layer_idx] == "full_attention":
            head_dim = self.global_head_dim or self.head_dim
        else:
            head_dim = self.head_dim
        if head_dim is None:
            raise ValueError("cache head dimension has not been resolved")
        return head_dim

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


@dataclass(frozen=True)
class DistillationConfig:
    epochs: int = 1
    overfit_diagnostic: bool = False
    expected_locked_examples: int | None = None
    teacher_thinking: bool = True
    student_thinking: bool = False
    max_teacher_reasoning_tokens: int = 1024
    max_response_tokens: int = 1024
    temperature: float = 1.1
    top_p: float = 1.0
    top_k: int = 0
    token_clip: float | None = None
    kl_sequence_chunk_size: int = 32
    response_process_every: int = 64
    boundary_selection: Literal["configured", "all"] = "configured"
    gradient_accumulation_steps: int = 1
    reasoning_open_text: str = "<think>\n"
    reasoning_close_text: str = "</think>\n\n"
    close_reasoning_at_eos: bool = True
    skip_invalid_traces: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("distillation epochs must be positive")
        if self.epochs != 1 and not self.overfit_diagnostic:
            raise ValueError("repeated distillation epochs require an explicit overfit diagnostic")
        if self.overfit_diagnostic and (
            self.expected_locked_examples is None or self.expected_locked_examples <= 0
        ):
            raise ValueError("overfit diagnostic requires positive expected_locked_examples")
        if not self.overfit_diagnostic and self.expected_locked_examples is not None:
            raise ValueError("expected_locked_examples is only valid for an overfit diagnostic")
        if not self.teacher_thinking or self.student_thinking:
            raise ValueError("cot_distillation requires a thinking teacher and thinking-off student")
        if self.max_teacher_reasoning_tokens <= 0 or self.max_response_tokens <= 0:
            raise ValueError("distillation token limits must be positive")
        if self.temperature <= 0:
            raise ValueError("distillation temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("distillation top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("distillation top_k must be non-negative")
        if self.token_clip is not None:
            raise ValueError(
                "signed per-vocabulary KL clipping is invalid; distillation token_clip must be null"
            )
        if self.kl_sequence_chunk_size <= 0:
            raise ValueError("distillation kl_sequence_chunk_size must be positive")
        if self.response_process_every <= 0:
            raise ValueError("distillation response_process_every must be positive")
        if self.boundary_selection not in {"configured", "all"}:
            raise ValueError("distillation boundary_selection must be 'configured' or 'all'")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("distillation gradient_accumulation_steps must be positive")
        if not self.reasoning_close_text:
            raise ValueError("distillation reasoning_close_text must not be empty")


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mode: Literal["pretrain", "sft", "cot_distillation"] = "pretrain"
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
    distillation: DistillationConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.wandb, dict):
            self.wandb = WandbConfig(**self.wandb)
        if isinstance(self.distillation, dict):
            self.distillation = DistillationConfig(**self.distillation)
        if isinstance(self.resume, dict):
            self.resume = ResumeConfig(**self.resume)
        if self.resume is not None and self.backbone_checkpoint is not None:
            raise ValueError("resume and backbone_checkpoint cannot be configured together")
        if self.mode == "cot_distillation":
            if self.distillation is None:
                raise ValueError("cot_distillation requires distillation configuration")
            if self.model.variant != "bgt":
                raise ValueError("cot_distillation V0 requires a BGT model")
            processor = self.model.cache_processor
            if processor is None:
                raise ValueError("cot_distillation V0 requires a cache processor")
            if processor.attention_dropout != 0.0:
                raise ValueError("cot_distillation V0 requires zero processor attention dropout")
            if processor.processor_mode != "standard":
                raise ValueError("cot_distillation V0 requires standard cache processor mode")
            if self.model.flow_processor is not None:
                raise ValueError("cot_distillation V0 does not support Flow processors")
            if self.model.family != "qwen3":
                raise ValueError("cot_distillation V0 supports only the qwen3 family")
            if not self.freeze_backbone:
                raise ValueError("cot_distillation requires a frozen backbone")
            if self.backbone_checkpoint is None:
                raise ValueError("cot_distillation requires backbone_checkpoint")
            if self.batch_size != 1:
                raise ValueError("cot_distillation V0 requires batch_size=1")
            if self.data.kind != "openmath_prompts":
                raise ValueError("cot_distillation requires openmath_prompts data")
            if not self.data.tokenizer.startswith("hf:"):
                raise ValueError("cot_distillation requires one Hugging Face chat tokenizer")
            if self.distributed != "none":
                raise ValueError("cot_distillation V0 supports only single-process training")
            if self.resume is not None:
                raise ValueError("cot_distillation V0 does not yet support exact optimiser resume")
            if self.max_steps != 1:
                raise ValueError("cot_distillation uses one finite epoch and requires max_steps=1")


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
