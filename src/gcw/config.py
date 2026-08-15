from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import yaml

from gcw.performance.config import PerformanceConfig


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
CacheLinearStateQueryMode = Literal["per_linear_state", "shared", "global"]
WorkspaceFlowSolver = Literal["euler", "midpoint"]
TrainableComponents = Literal["all", "backbone", "lora", "cache_processor"]
ActivationCheckpointingPolicy = Literal["none", "standard_processor_layers"]
BackboneChunkBackend = Literal["eager", "compile", "compile_full_windows"]
LossBackend = Literal["reference", "target_only", "liger"]
SFTLossScope = Literal["answer_only", "full_sequence"]
ProcessorGenerationMode = Literal["windowed", "post_prefill_relative"]
DiffusionInferenceMode = Literal["deterministic", "stochastic"]

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
    topk_linear_heads: int | None = None
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
    linear_state_cache: bool = False
    linear_state_query_mode: CacheLinearStateQueryMode = "global"
    gate_linear_state_delta: bool = True

    def __post_init__(self) -> None:
        if self.score not in {"attn_l2", "kv_norm"}:
            raise ValueError("cache_processor.score must be 'attn_l2' or 'kv_norm'")
        if self.topk_heads <= 0:
            raise ValueError("cache_processor.topk_heads must be positive")
        if self.topk_linear_heads is not None and self.topk_linear_heads < 0:
            raise ValueError("cache_processor.topk_linear_heads must be non-negative when set")
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
        if self.linear_state_query_mode not in {"per_linear_state", "shared", "global"}:
            raise ValueError(
                "cache_processor.linear_state_query_mode must be 'per_linear_state', 'shared', or 'global'"
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
class DiffusionProcessorConfig:
    num_steps: int = 4
    time_conditioning: bool = True
    eta_max: float = 1.0
    refinement_gate_init: float = -4.0
    noise_scale: float = 0.15
    conditioning_scale: float = 1.0
    inference_mode: DiffusionInferenceMode = "deterministic"

    def __post_init__(self) -> None:
        if isinstance(self.num_steps, bool) or not isinstance(self.num_steps, int) or self.num_steps <= 0:
            raise ValueError("diffusion_processor.num_steps must be a positive integer")
        if not isinstance(self.time_conditioning, bool):
            raise ValueError("diffusion_processor.time_conditioning must be a boolean")
        if self.inference_mode not in {"deterministic", "stochastic"}:
            raise ValueError("diffusion_processor.inference_mode must be 'deterministic' or 'stochastic'")
        self.eta_max = _validated_float32("diffusion_processor.eta_max", self.eta_max, positive=True)
        self.refinement_gate_init = _validated_float32(
            "diffusion_processor.refinement_gate_init", self.refinement_gate_init
        )
        self.noise_scale = _validated_float32("diffusion_processor.noise_scale", self.noise_scale)
        self.conditioning_scale = _validated_float32(
            "diffusion_processor.conditioning_scale", self.conditioning_scale
        )
        if self.noise_scale < 0.0:
            raise ValueError("diffusion_processor.noise_scale must be non-negative")
        if self.conditioning_scale < 0.0:
            raise ValueError("diffusion_processor.conditioning_scale must be non-negative")


@dataclass
class WorkspaceFlowRefinementConfig:
    solver: WorkspaceFlowSolver = "euler"
    solver_steps: int = 1
    max_velocity: float = 0.75

    def __post_init__(self) -> None:
        if self.solver not in {"euler", "midpoint"}:
            raise ValueError("workspace_flow_refinement.solver must be 'euler' or 'midpoint'")
        if isinstance(self.solver_steps, bool) or not isinstance(self.solver_steps, int) or self.solver_steps <= 0:
            raise ValueError("workspace_flow_refinement.solver_steps must be a positive integer")
        self.max_velocity = _validated_float32(
            "workspace_flow_refinement.max_velocity",
            self.max_velocity,
            positive=True,
        )


@dataclass
class LoRAConfig:
    rank: int = 32
    alpha: float = 32.0
    dropout: float = 0.0
    targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or self.rank <= 0:
            raise ValueError("lora.rank must be a positive integer")
        self.alpha = _validated_float32("lora.alpha", self.alpha, positive=True)
        self.dropout = _validated_float32("lora.dropout", self.dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("lora.dropout must be in [0, 1)")
        self.targets = tuple(str(target) for target in self.targets)
        if not self.targets or any(not target for target in self.targets):
            raise ValueError("lora.targets must contain module names")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("lora.targets must be unique")


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
    lora: LoRAConfig | None = None
    flow_processor: FlowProcessorConfig | None = None
    diffusion_processor: DiffusionProcessorConfig | None = None
    processor_generation_mode: ProcessorGenerationMode = "windowed"
    workspace_flow_refinement: WorkspaceFlowRefinementConfig | None = None
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
        if isinstance(self.lora, dict):
            self.lora = LoRAConfig(**self.lora)
        if isinstance(self.flow_processor, dict):
            self.flow_processor = FlowProcessorConfig(**self.flow_processor)
        if isinstance(self.diffusion_processor, dict):
            self.diffusion_processor = DiffusionProcessorConfig(**self.diffusion_processor)
        if self.flow_processor is not None and self.diffusion_processor is not None:
            raise ValueError("flow_processor and diffusion_processor are mutually exclusive")
        if self.processor_generation_mode not in {"windowed", "post_prefill_relative"}:
            raise ValueError("processor_generation_mode must be 'windowed' or 'post_prefill_relative'")
        if self.variant == "tf" and self.processor_generation_mode != "windowed":
            raise ValueError("processor_generation_mode requires a cache processor")
        if isinstance(self.workspace_flow_refinement, dict):
            self.workspace_flow_refinement = WorkspaceFlowRefinementConfig(**self.workspace_flow_refinement)
        if self.variant in {"bt", "bgt", "icarus", "longshot"} and self.cache_processor is None:
            self.cache_processor = CacheProcessorConfig()
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
        if self.cache_processor is not None and self.cache_processor.unrope_keys:
            rotary_widths = {
                int(self.head_dim_for_layer(layer_idx) * self.partial_rotary_factor)
                for layer_idx in self.cache_layer_indices
            }
            if any(width <= 0 or width % 2 for width in rotary_widths):
                raise ValueError(
                    "cache_processor.unrope_keys requires a positive even rotary width "
                    "for every stored cache layer"
                )
        from gcw.architecture import require_model_config

        require_model_config(self)

    @property
    def architecture_name(self) -> str:
        if self.diffusion_processor is not None:
            return f"diffusion_{self.variant}"
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


def linear_attention_following_cache_indices(config: ModelConfig, *, require_layer_types: bool = False) -> list[tuple[int, int]]:
    if config.layer_types is None:
        if require_layer_types:
            raise ValueError("Longshot linear_state_cache requires layer_types")
        return []
    pairs: list[tuple[int, int]] = []
    cache_layer_indices = config.cache_layer_indices
    for layer_idx, layer_type in enumerate(config.layer_types):
        if layer_type != "linear_attention":
            continue
        following = next((cache_idx for cache_idx, full_idx in enumerate(cache_layer_indices) if full_idx > layer_idx), None)
        if following is None:
            raise ValueError(f"linear_attention layer {layer_idx} has no following full-attention cache layer")
        pairs.append((layer_idx, following))
    return pairs


@dataclass
class DataConfig:
    kind: Literal["random", "fineweb", "sft"] = "fineweb"
    data_dir: str | None = None
    dataset: str | None = None
    validation_dataset: str | None = None
    tokenizer: str = "cosmo2"
    tokenizer_source: str | None = None
    text_field: str = "text"
    max_length: int = 2048
    sft_skip_rows: int = 0
    sft_take_rows: int | None = None
    response_prefix: str = ""
    prompt_suffix: str = ""
    sft_chat_template: bool = False
    sft_enable_thinking: bool = False
    sft_loss_scope: SFTLossScope = "answer_only"
    canonical_execution: bool = False
    expected_artifact_sha256: str | None = None
    expected_source_ids_sha256: str | None = None
    experiment_manifest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sft_chat_template, bool) or not isinstance(self.sft_enable_thinking, bool):
            raise ValueError("SFT chat-template controls must be booleans")
        if self.sft_chat_template and self.response_prefix:
            raise ValueError("data.response_prefix must be empty with sft_chat_template")
        if self.sft_loss_scope not in {"answer_only", "full_sequence"}:
            raise ValueError("data.sft_loss_scope must be 'answer_only' or 'full_sequence'")
        hashes = (self.expected_artifact_sha256, self.expected_source_ids_sha256)
        authenticated = all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            and len(set(value)) > 1
            for value in hashes
        )
        if self.canonical_execution and not authenticated:
            raise ValueError(
                "canonical data requires authenticated artifact and source-ID sha256 hashes"
            )


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
    bf16: bool = False
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
        if self.loss_backend not in {"reference", "target_only", "liger"}:
            raise ValueError("loss_backend must be 'reference', 'target_only', or 'liger'")
        if self.trainable not in {"all", "backbone", "lora", "cache_processor"}:
            raise ValueError("trainable must be 'all', 'backbone', 'lora', or 'cache_processor'")
        if self.activation_checkpointing not in {"none", "standard_processor_layers"}:
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
        if self.backbone_chunk_backend not in {"eager", "compile", "compile_full_windows"}:
            raise ValueError(
                "backbone_chunk_backend must be 'eager', 'compile', or 'compile_full_windows'"
            )
        if self.backbone_chunk_backend != "eager":
            selected_backend = self.backbone_chunk_backend
            if self.model.family != "qwen3":
                raise ValueError(f"backbone_chunk_backend={selected_backend} requires model.family='qwen3'")
            if self.model.variant == "tf" or self.model.cache_processor is None:
                raise ValueError(
                    f"backbone_chunk_backend={selected_backend} requires a windowed cache processor"
                )
            if self.trainable != "cache_processor":
                raise ValueError(f"backbone_chunk_backend={selected_backend} requires trainable='cache_processor'")
            if self.model.cache_processor.process_every != 64:
                raise ValueError(
                    f"backbone_chunk_backend={selected_backend} requires cache_processor.process_every=64"
                )
            if self.sequence_len < 128:
                raise ValueError(f"backbone_chunk_backend={selected_backend} requires sequence_len >= 128")
            if self.distributed == "fsdp":
                raise ValueError(f"backbone_chunk_backend={selected_backend} is not supported with FSDP")
        threshold = self.longshot_gradient_checkpointing_seq_len_threshold
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0:
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
        if self.trainable == "lora" and self.model.lora is None:
            raise ValueError("trainable='lora' requires model.lora")
        if self.total_batch_size and self.global_batch_rows:
            raise ValueError("set only one of total_batch_size and global_batch_rows")
        if self.global_batch_rows < 0:
            raise ValueError("global_batch_rows must be non-negative")
        if isinstance(self.activation_checkpoint_windows, bool) or not isinstance(self.activation_checkpoint_windows, int):
            raise ValueError("activation_checkpoint_windows must be an integer")
        if self.activation_checkpoint_windows < 0:
            raise ValueError("activation_checkpoint_windows must be non-negative")
        if (
            isinstance(self.activation_checkpoint_min_padded_sequence_length, bool)
            or not isinstance(self.activation_checkpoint_min_padded_sequence_length, int)
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
        if self.logical_pair_grouping:
            if self.mode != "sft" or self.data.kind != "sft":
                raise ValueError("logical_pair_grouping requires SFT data")
            if self.data.sft_loss_scope != "answer_only":
                raise ValueError("logical_pair_grouping requires answer-only SFT loss")
            if (
                isinstance(self.batch_size, bool)
                or not isinstance(self.batch_size, int)
                or self.batch_size < 2
                or self.batch_size % 2
            ):
                raise ValueError("logical_pair_grouping requires an even physical batch_size of at least 2")
            if (
                isinstance(self.global_batch_rows, bool)
                or not isinstance(self.global_batch_rows, int)
                or self.global_batch_rows <= 0
                or self.global_batch_rows % 2
            ):
                raise ValueError("logical_pair_grouping requires a positive even global_batch_rows")
            if self.global_batch_rows % self.batch_size:
                raise ValueError("logical_pair_grouping requires global_batch_rows divisible by physical batch_size")
            if self.model.router_aux_loss_coef:
                raise ValueError("logical_pair_grouping does not support router auxiliary loss")
            diffusion = self.model.diffusion_processor
            if diffusion is not None and diffusion.noise_scale > 0.0:
                raise ValueError("logical_pair_grouping rejects noisy Diffusion because RNG assignment changes with batching")
        longshot_history = self.performance.longshot_full_history
        if longshot_history.enabled:
            processor = self.model.cache_processor
            if self.model.variant != "longshot" or processor is None:
                raise ValueError(
                    "performance.longshot_full_history requires a Longshot cache processor"
                )
            if self.trainable != "cache_processor":
                raise ValueError(
                    "performance.longshot_full_history requires trainable='cache_processor'"
                )
            if self.distributed != "none":
                raise ValueError(
                    "performance.longshot_full_history supports only single-process distributed='none'"
                )
            if self.backbone_chunk_backend != "eager":
                raise ValueError(
                    "performance.longshot_full_history requires backbone_chunk_backend='eager'"
                )
            if self.performance.adaptive_batching.enabled:
                raise ValueError(
                    "performance.longshot_full_history does not compose with adaptive batching"
                )
            if processor.score != "kv_norm":
                raise ValueError(
                    "performance.longshot_full_history requires cache_processor.score='kv_norm'"
                )
            if processor.attention_dropout != 0.0:
                raise ValueError(
                    "performance.longshot_full_history requires cache_processor.attention_dropout=0"
                )
            if processor.linear_state_cache:
                raise ValueError(
                    "performance.longshot_full_history does not support linear recurrent cache state"
                )
            if self.model.flow_processor is not None or self.model.workspace_flow_refinement is not None:
                raise ValueError(
                    "performance.longshot_full_history does not support Flow or workspace refinement"
                )
            if self.activation_checkpointing != "none":
                raise ValueError(
                    "performance.longshot_full_history supports Longshot and segmented checkpointing only"
                )
            if (
                processor.iterations > 1
                and processor.iteration_mode == "internal"
                and processor.internal_iteration_score != "kv_norm"
            ):
                raise ValueError(
                    "performance.longshot_full_history internal recurrence requires "
                    "cache_processor.internal_iteration_score='kv_norm'"
                )
            if self.model.family not in {"smollm", "llama", "qwen3"}:
                raise ValueError(
                    "performance.longshot_full_history requires a dense WindowedCache backbone family"
                )
            head_dim = self.model.stored_cache_shape[1]
            if processor.unrope_keys:
                rotary_dim = int(head_dim * self.model.partial_rotary_factor)
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
        sidecar = self.performance.kv_score_sidecar
        if sidecar.enabled:
            processor = self.model.cache_processor
            if self.model.variant != "longshot" or processor is None:
                raise ValueError("performance.kv_score_sidecar requires a Longshot cache processor")
            if self.performance.longshot_full_history.enabled:
                raise ValueError(
                    "performance.kv_score_sidecar does not compose with performance.longshot_full_history"
                )
            if self.batch_size != 2:
                raise ValueError("performance.kv_score_sidecar requires fixed eager batch_size=2")
            if self.distributed != "none":
                raise ValueError(
                    "performance.kv_score_sidecar supports only single-process distributed='none'"
                )
            if self.backbone_chunk_backend != "eager":
                raise ValueError("performance.kv_score_sidecar requires backbone_chunk_backend='eager'")
            if self.performance.adaptive_batching.enabled:
                raise ValueError("performance.kv_score_sidecar does not compose with adaptive batching")
            if processor.score != "kv_norm":
                raise ValueError(
                    "performance.kv_score_sidecar requires cache_processor.score='kv_norm'"
                )
            if processor.attention_dropout != 0.0:
                raise ValueError(
                    "performance.kv_score_sidecar requires cache_processor.attention_dropout=0"
                )
            if processor.linear_state_cache:
                raise ValueError("performance.kv_score_sidecar rejects linear recurrent cache state")
            if (
                self.model.flow_processor is not None
                or self.model.diffusion_processor is not None
                or self.model.workspace_flow_refinement is not None
            ):
                raise ValueError(
                    "performance.kv_score_sidecar does not compose with Flow, Diffusion, or workspace refinement"
                )
            if self.activation_checkpointing != "none":
                raise ValueError(
                    "performance.kv_score_sidecar supports Longshot native and segmented checkpointing only"
                )
            if self.model.family not in {"smollm", "llama", "qwen3"}:
                raise ValueError(
                    "performance.kv_score_sidecar requires a dense WindowedCache-compatible family"
                )
        noisy_diffusion_backend = self.performance.noisy_diffusion.recurrence_backend
        if noisy_diffusion_backend == "compile":
            diffusion = self.model.diffusion_processor
            if self.model.variant != "bgt" or diffusion is None or diffusion.noise_scale <= 0.0:
                raise ValueError(
                    "compiled noisy-Diffusion recurrence requires BGT with "
                    "diffusion_processor.noise_scale>0"
                )
            processor = self.model.cache_processor
            if processor is not None and processor.attention_dropout != 0.0:
                raise ValueError(
                    "compiled noisy-Diffusion recurrence requires "
                    "cache_processor.attention_dropout=0"
                )
            if self.distributed != "none":
                raise ValueError(
                    "compiled noisy-Diffusion recurrence currently supports distributed='none'"
                )
            if self.performance.adaptive_batching.enabled:
                raise ValueError(
                    "compiled noisy-Diffusion recurrence does not enable adaptive batching"
                )
        reference_loss = self.performance.reference_loss
        if reference_loss.mode != "legacy" and self.loss_backend != "reference":
            raise ValueError("bounded reference-loss execution requires loss_backend='reference'")
        adaptive_batching = self.performance.adaptive_batching
        if adaptive_batching.enabled:
            if not self.logical_pair_grouping:
                raise ValueError("adaptive batching requires logical_pair_grouping=true")
            if self.batch_size != 2:
                raise ValueError("adaptive batching requires batch_size=2 as the adjacent-pair source stream")
            if self.global_batch_rows != 64:
                raise ValueError("adaptive batching v1 requires global_batch_rows=64")
            if self.distributed != "none":
                raise ValueError("adaptive batching v1 supports only single-process distributed='none' execution")
            if self.loss_backend != "reference":
                raise ValueError("adaptive batching requires loss_backend='reference'")
            processor = self.model.cache_processor
            if processor is not None and processor.attention_dropout != 0.0:
                raise ValueError("adaptive batching requires cache_processor.attention_dropout=0")
            if self.model.lora is not None and self.model.lora.dropout != 0.0:
                raise ValueError("adaptive batching requires model.lora.dropout=0")
            if self.sequence_len > 4096:
                raise ValueError("adaptive batching v1 supports sequence_len up to 4096")
            physical_rows = tuple(band.physical_rows for band in adaptive_batching.bands)
            if physical_rows != (16, 8, 4, 2):
                raise ValueError("adaptive batching v1 physical_rows must be exactly 16, 8, 4, 2")
            if adaptive_batching.bands[-1].max_padded_tokens < self.sequence_len:
                raise ValueError("adaptive batching fallback band must cover sequence_len")
        if reference_loss.mode == "bounded":
            if self.distributed != "none":
                raise ValueError("bounded reference-loss execution currently supports only distributed='none'")
            maximum_physical_rows = (
                max(band.physical_rows for band in adaptive_batching.bands)
                if adaptive_batching.enabled
                else self.batch_size
            )
            if reference_loss.max_projected_positions < maximum_physical_rows:
                raise ValueError(
                    "performance.reference_loss.max_projected_positions must cover the maximum physical rows"
                )
        if not math.isfinite(self.grad_clip) or self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be finite and positive")
        from gcw.architecture import require_train_config

        require_train_config(self)


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


def load_train_config_bytes(payload: bytes) -> TrainConfig:
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("training config must be UTF-8 YAML") from exc
    data = yaml.safe_load(decoded)
    if not isinstance(data, dict):
        raise ValueError("training config must decode to a mapping")
    return TrainConfig(**data)


def load_train_config(path: str | Path) -> TrainConfig:
    return load_train_config_bytes(Path(path).read_bytes())


def to_dict(value: Any) -> Any:
    if isinstance(value, PerformanceConfig):
        payload = {field.name: to_dict(getattr(value, field.name)) for field in fields(value)}
        if not value.longshot_full_history.enabled:
            payload.pop("longshot_full_history")
        if not value.kv_score_sidecar.enabled:
            payload.pop("kv_score_sidecar")
        return payload
    if is_dataclass(value):
        return {f.name: to_dict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [to_dict(x) for x in value]
    if isinstance(value, dict):
        return {k: to_dict(v) for k, v in value.items()}
    return value



def dump_yaml(path: str | Path, value: Any) -> None:
    data = to_dict(value)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
