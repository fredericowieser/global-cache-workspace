"""Model configuration: backbone, cache-processor, Flow/Diffusion, and LoRA specs.

``ModelConfig`` is the single source of truth for architecture shape and the
``backbone x cache_processor`` composition. Its ``__post_init__`` resolves
cross-field defaults (per-family dimensions, layer types) and fails closed on
incompatible variant/family/dimension combinations before any model is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gcw.config.types import (
    CacheEncoderMethod,
    CacheEncoderQuerySource,
    CacheGateMode,
    CacheLinearStateQueryMode,
    CacheLinearStateUpdateMode,
    CacheOutProjInit,
    CacheProcessorIterationMode,
    CacheProcessorInternalIterationScore,
    CacheProcessorMode,
    CacheUpdateStream,
    DiffusionInferenceMode,
    Family,
    MoeExpertsImplementation,
    ProcessorGenerationMode,
    RotaryStyle,
    Variant,
    WorkspaceFlowSolver,
    _validated_float32,
)


def gemma4_default_layer_types(num_layers: int) -> list[str]:
    layer_types = [
        "sliding_attention" if bool((i + 1) % 6) else "full_attention"
        for i in range(num_layers)
    ]
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
    linear_state_update_mode: CacheLinearStateUpdateMode = "outer_product"
    gate_linear_state_delta: bool = True

    def __post_init__(self) -> None:
        if self.score not in {"attn_l2", "kv_norm"}:
            raise ValueError(
                "cache_processor.score must be 'attn_l2' or 'kv_norm'"
            )
        if self.topk_heads <= 0:
            raise ValueError("cache_processor.topk_heads must be positive")
        if self.topk_linear_heads is not None and self.topk_linear_heads < 0:
            raise ValueError(
                "cache_processor.topk_linear_heads must be non-negative when set"
            )
        if self.update_stream not in {"kv", "v"}:
            raise ValueError(
                "cache_processor.update_stream must be 'kv' or 'v'"
            )
        if self.out_proj_init not in {"normal", "zero"}:
            raise ValueError(
                "cache_processor.out_proj_init must be 'normal' or 'zero'"
            )
        if self.gate_mode not in {"sigmoid", "direct"}:
            raise ValueError(
                "cache_processor.gate_mode must be 'sigmoid' or 'direct'"
            )
        if self.processor_mode not in {
            "standard",
            "workspace_broadcast",
            "head_mask_broadcast",
        }:
            raise ValueError(
                "cache_processor.processor_mode must be 'standard', 'workspace_broadcast', or 'head_mask_broadcast'"
            )
        if not 0.0 < self.writer_head_fraction <= 1.0:
            raise ValueError(
                "cache_processor.writer_head_fraction must be in (0, 1]"
            )
        if self.salience_temperature <= 0.0:
            raise ValueError(
                "cache_processor.salience_temperature must be positive"
            )
        if self.kv_score_init_k <= 0.0 or self.kv_score_init_v <= 0.0:
            raise ValueError(
                "cache_processor.kv_score_init_k and kv_score_init_v must be positive"
            )
        if self.encoder_method not in {"linear", "query_attention"}:
            raise ValueError(
                "cache_processor.encoder_method must be 'linear' or 'query_attention'"
            )
        if self.encoder_seq_len is not None and self.encoder_seq_len <= 0:
            raise ValueError(
                "cache_processor.encoder_seq_len must be positive when set"
            )
        if self.encoder_query_source not in {"learned", "final_token"}:
            raise ValueError(
                "cache_processor.encoder_query_source must be 'learned' or 'final_token'"
            )
        if self.iterations < 1:
            raise ValueError("cache_processor.iterations must be positive")
        if self.iteration_mode not in {"internal", "external"}:
            raise ValueError(
                "cache_processor.iteration_mode must be 'internal' or 'external'"
            )
        if self.internal_iteration_score not in {
            "attn_l2",
            "cache_head_attention",
            "kv_norm",
        }:
            raise ValueError(
                "cache_processor.internal_iteration_score must be 'attn_l2', 'cache_head_attention', or 'kv_norm'"
            )
        if self.linear_state_query_mode not in {
            "per_linear_state",
            "shared",
            "global",
        }:
            raise ValueError(
                "cache_processor.linear_state_query_mode must be 'per_linear_state', 'shared', or 'global'"
            )
        if self.linear_state_update_mode not in {
            "outer_product",
            "gated_delta",
        }:
            raise ValueError(
                "cache_processor.linear_state_update_mode must be 'outer_product' or 'gated_delta'"
            )
        if self.linear_state_update_mode == "gated_delta":
            self.gate_linear_state_delta = False

    @property
    def stream_factor(self) -> int:
        if self.update_stream == "kv":
            return 2
        if self.update_stream == "v":
            return 1
        raise ValueError(
            f"unsupported cache_processor.update_stream: {self.update_stream}"
        )


@dataclass
class FlowProcessorConfig:
    num_steps: int = 1
    time_conditioning: bool = True
    eta_max: float = 1.0
    flow_gate_init: float = -4.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_steps, bool)
            or not isinstance(self.num_steps, int)
            or self.num_steps <= 0
        ):
            raise ValueError(
                "flow_processor.num_steps must be a positive integer"
            )
        if not isinstance(self.time_conditioning, bool):
            raise ValueError(
                "flow_processor.time_conditioning must be a boolean"
            )
        self.eta_max = _validated_float32(
            "flow_processor.eta_max", self.eta_max, positive=True
        )
        self.flow_gate_init = _validated_float32(
            "flow_processor.flow_gate_init", self.flow_gate_init
        )


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
        if (
            isinstance(self.num_steps, bool)
            or not isinstance(self.num_steps, int)
            or self.num_steps <= 0
        ):
            raise ValueError(
                "diffusion_processor.num_steps must be a positive integer"
            )
        if not isinstance(self.time_conditioning, bool):
            raise ValueError(
                "diffusion_processor.time_conditioning must be a boolean"
            )
        if self.inference_mode not in {"deterministic", "stochastic"}:
            raise ValueError(
                "diffusion_processor.inference_mode must be 'deterministic' or 'stochastic'"
            )
        self.eta_max = _validated_float32(
            "diffusion_processor.eta_max", self.eta_max, positive=True
        )
        self.refinement_gate_init = _validated_float32(
            "diffusion_processor.refinement_gate_init",
            self.refinement_gate_init,
        )
        self.noise_scale = _validated_float32(
            "diffusion_processor.noise_scale", self.noise_scale
        )
        self.conditioning_scale = _validated_float32(
            "diffusion_processor.conditioning_scale", self.conditioning_scale
        )
        if self.noise_scale < 0.0:
            raise ValueError(
                "diffusion_processor.noise_scale must be non-negative"
            )
        if self.conditioning_scale < 0.0:
            raise ValueError(
                "diffusion_processor.conditioning_scale must be non-negative"
            )


@dataclass
class WorkspaceFlowRefinementConfig:
    solver: WorkspaceFlowSolver = "euler"
    solver_steps: int = 1
    max_velocity: float = 0.75

    def __post_init__(self) -> None:
        if self.solver not in {"euler", "midpoint"}:
            raise ValueError(
                "workspace_flow_refinement.solver must be 'euler' or 'midpoint'"
            )
        if (
            isinstance(self.solver_steps, bool)
            or not isinstance(self.solver_steps, int)
            or self.solver_steps <= 0
        ):
            raise ValueError(
                "workspace_flow_refinement.solver_steps must be a positive integer"
            )
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
    legacy_bottleneck_generate: bool | None = None

    def __post_init__(self) -> None:
        if self.legacy_bottleneck_generate is not None:
            if not isinstance(self.legacy_bottleneck_generate, bool):
                raise ValueError("legacy_bottleneck_generate must be boolean")
            self.processor_generation_mode = (
                "post_prefill_relative"
                if self.legacy_bottleneck_generate
                else "windowed"
            )
        if isinstance(self.cache_processor, dict):
            self.cache_processor = CacheProcessorConfig(**self.cache_processor)
        if isinstance(self.lora, dict):
            self.lora = LoRAConfig(**self.lora)
        if isinstance(self.flow_processor, dict):
            self.flow_processor = FlowProcessorConfig(**self.flow_processor)
        if isinstance(self.diffusion_processor, dict):
            self.diffusion_processor = DiffusionProcessorConfig(
                **self.diffusion_processor
            )
        if (
            self.flow_processor is not None
            and self.diffusion_processor is not None
        ):
            raise ValueError(
                "flow_processor and diffusion_processor are mutually exclusive"
            )
        if self.processor_generation_mode not in {
            "windowed",
            "post_prefill_relative",
        }:
            raise ValueError(
                "processor_generation_mode must be 'windowed' or 'post_prefill_relative'"
            )
        if (
            self.variant == "tf"
            and self.processor_generation_mode != "windowed"
        ):
            raise ValueError(
                "processor_generation_mode requires a cache processor"
            )
        if isinstance(self.workspace_flow_refinement, dict):
            self.workspace_flow_refinement = WorkspaceFlowRefinementConfig(
                **self.workspace_flow_refinement
            )
        if (
            self.variant in {"bt", "bgt", "icarus", "longshot"}
            and self.cache_processor is None
        ):
            self.cache_processor = CacheProcessorConfig()
        if self.family == "gemma4" and self.layer_types is None:
            self.layer_types = gemma4_default_layer_types(self.num_layers)
        if self.layer_types is not None:
            self.layer_types = list(self.layer_types)
            if len(self.layer_types) != self.num_layers:
                raise ValueError("layer_types must have length num_layers")
            allowed = (
                {"full_attention", "linear_attention", "sliding_attention"}
                if self.family == "gemma4"
                else {
                    "full_attention",
                    "linear_attention",
                }
            )
            unknown = sorted(set(self.layer_types) - allowed)
            if unknown:
                raise ValueError(f"unsupported layer_types: {unknown}")
            if (
                self.family == "gemma4"
                and self.layer_types[-1] != "full_attention"
            ):
                self.layer_types[-1] = "full_attention"
        if self.processor_layer_indices is not None:
            self.processor_layer_indices = [
                int(i) for i in self.processor_layer_indices
            ]
            if any(
                i < 0 or i >= self.num_layers
                for i in self.processor_layer_indices
            ):
                raise ValueError(
                    "processor_layer_indices entries must be valid layer indices"
                )
            if self.processor_layer_indices != sorted(
                set(self.processor_layer_indices)
            ):
                raise ValueError(
                    "processor_layer_indices must be sorted and unique"
                )
            if self.layer_types is not None:
                cache_layers = set(self._default_cache_layer_indices())
                non_cache = [
                    i
                    for i in self.processor_layer_indices
                    if i not in cache_layers
                ]
                if non_cache:
                    raise ValueError(
                        "processor_layer_indices must reference cache-bearing layers"
                    )
        if self.family in {"qwen3_5", "qwen3_5_moe"}:
            if self.layer_types is None:
                raise ValueError(f"{self.family} requires layer_types")
            if self.linear_key_head_dim is None:
                self.linear_key_head_dim = int(
                    self.head_dim or self.hidden_size // self.num_heads
                )
            if self.linear_value_head_dim is None:
                self.linear_value_head_dim = self.linear_key_head_dim
            if self.linear_num_key_heads is None:
                self.linear_num_key_heads = self.num_kv_heads
            if self.linear_num_value_heads is None:
                self.linear_num_value_heads = self.num_heads
        if self.family == "qwen3_5_moe":
            if self.moe_experts_implementation != "grouped_mm":
                raise ValueError(
                    "qwen3_5_moe supports only moe_experts_implementation='grouped_mm'"
                )
            if self.moe_intermediate_size is None:
                self.moe_intermediate_size = self.intermediate_size
            if self.shared_expert_intermediate_size is None:
                self.shared_expert_intermediate_size = (
                    self.moe_intermediate_size
                )
            if self.num_experts <= 0:
                raise ValueError("qwen3_5_moe requires num_experts")
            if self.num_experts_per_tok <= 0:
                raise ValueError("qwen3_5_moe requires num_experts_per_tok")
        if self.family == "gemma4":
            if self.layer_types is None:
                raise ValueError("gemma4 requires layer_types")
            if any(
                layer_type == "linear_attention"
                for layer_type in self.layer_types
            ):
                raise ValueError(
                    "gemma4 does not support linear_attention layers"
                )
            if self.sliding_window is None:
                self.sliding_window = 512
            if self.global_head_dim is None:
                self.global_head_dim = self.head_dim
            if self.vocab_size_per_layer_input is None:
                self.vocab_size_per_layer_input = self.vocab_size
            if (
                self.num_kv_shared_layers < 0
                or self.num_kv_shared_layers > self.num_layers
            ):
                raise ValueError(
                    "num_kv_shared_layers must be in [0, num_layers]"
                )
            if self.rope_parameters is None:
                self.rope_parameters = {
                    "sliding_attention": {
                        "rope_type": "default",
                        "rope_theta": 10000.0,
                    },
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
        if (
            self.cache_processor is not None
            and self.cache_processor.unrope_keys
        ):
            rotary_widths = {
                int(
                    self.head_dim_for_layer(layer_idx)
                    * self.partial_rotary_factor
                )
                for layer_idx in self.cache_layer_indices
            }
            if any(width <= 0 or width % 2 for width in rotary_widths):
                raise ValueError(
                    "cache_processor.unrope_keys requires a positive even rotary width for every stored cache layer"
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
                    if layer_type in {"full_attention", "sliding_attention"}
                    and i < first_shared
                ]
            return [
                i
                for i, layer_type in enumerate(self.layer_types)
                if layer_type == "full_attention"
            ]
        return list(range(self.num_layers))

    @property
    def cache_num_layers(self) -> int:
        return len(self.cache_layer_indices)

    @property
    def cache_num_heads(self) -> int:
        if self.family == "gemma4":
            return sum(
                self.kv_heads_for_layer(i) for i in self.cache_layer_indices
            )
        return self.cache_num_layers * self.num_kv_heads

    @property
    def stored_cache_shapes(self) -> list[tuple[int, int]]:
        return [
            (self.kv_heads_for_layer(i), self.head_dim_for_layer(i))
            for i in self.cache_layer_indices
        ]

    @property
    def stored_cache_shape(self) -> tuple[int, int]:
        shapes = self.stored_cache_shapes
        if not shapes:
            if self.head_dim is None:
                raise ValueError(
                    "stored cache shape requires a resolved head_dim"
                )
            return self.num_kv_heads, self.head_dim
        unique = set(shapes)
        if len(unique) > 1:
            raise ValueError(
                "stored cache layers do not have a uniform KV shape"
            )
        return shapes[0]

    def ensure_uniform_stored_cache_shape(self, owner: str) -> None:
        if self.family != "gemma4":
            return
        if len(set(self.stored_cache_shapes)) > 1:
            raise ValueError(
                f"{owner} requires uniform stored KV shapes for Gemma4 cache processing"
            )

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
        if (
            self.family == "gemma4"
            and self.layer_types is not None
            and self.layer_types[layer_idx] == "full_attention"
        ):
            head_dim = self.global_head_dim or self.head_dim
        else:
            head_dim = self.head_dim
        if head_dim is None:
            raise ValueError("cache head dimension has not been resolved")
        return head_dim

    @property
    def _experts_implementation(self) -> str:
        return self.moe_experts_implementation


def linear_attention_following_cache_indices(
    config: ModelConfig, *, require_layer_types: bool = False
) -> list[tuple[int, int]]:
    if config.layer_types is None:
        if require_layer_types:
            raise ValueError("Longshot linear_state_cache requires layer_types")
        return []
    pairs: list[tuple[int, int]] = []
    cache_layer_indices = config.cache_layer_indices
    for layer_idx, layer_type in enumerate(config.layer_types):
        if layer_type != "linear_attention":
            continue
        following = next(
            (
                cache_idx
                for cache_idx, full_idx in enumerate(cache_layer_indices)
                if full_idx > layer_idx
            ),
            None,
        )
        if following is None:
            raise ValueError(
                f"linear_attention layer {layer_idx} has no following full-attention cache layer"
            )
        pairs.append((layer_idx, following))
    return pairs


SMOLLM_CONFIGS: dict[str, ModelConfig] = {
    "smollm-135m": ModelConfig(
        num_layers=30,
        hidden_size=576,
        num_heads=9,
        num_kv_heads=3,
        head_dim=64,
        intermediate_size=1536,
    ),
    "smollm-360m": ModelConfig(
        num_layers=32,
        hidden_size=960,
        num_heads=15,
        num_kv_heads=5,
        head_dim=64,
        intermediate_size=2560,
    ),
    "smollm-1.7b": ModelConfig(
        num_layers=24,
        hidden_size=2048,
        num_heads=32,
        num_kv_heads=32,
        head_dim=64,
        intermediate_size=8192,
    ),
}
