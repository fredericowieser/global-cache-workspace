from __future__ import annotations

import math
from numbers import Real
from typing import Literal

Variant = Literal["tf", "bt", "bgt", "icarus", "longshot"]
Family = Literal["smollm", "llama", "qwen3", "qwen3_5", "qwen3_5_moe", "gemma4"]
RotaryStyle = Literal["llama", "ppt"]
MoeExpertsImplementation = Literal["grouped_mm"]
CacheUpdateStream = Literal["kv", "v"]
CacheOutProjInit = Literal["normal", "zero"]
CacheGateMode = Literal["sigmoid", "direct"]
CacheProcessorMode = Literal[
    "standard", "workspace_broadcast", "head_mask_broadcast"
]
CacheEncoderMethod = Literal["linear", "query_attention"]
CacheEncoderQuerySource = Literal["learned", "final_token"]
CacheHeadSelectionMethod = Literal["magnitude", "learned_contextual"]
CacheProcessorIterationMode = Literal["internal", "external"]
CacheProcessorInternalIterationScore = Literal[
    "attn_l2", "cache_head_attention", "kv_norm", "learned_contextual"
]
CacheLinearStateQueryMode = Literal["per_linear_state", "shared", "global"]
CacheLinearStateUpdateMode = Literal["outer_product", "gated_delta"]
TrainableComponents = Literal["all", "backbone", "lora", "cache_processor"]
ActivationCheckpointingPolicy = Literal["none", "standard_processor_layers"]
BackboneChunkBackend = Literal[
    "eager", "compile", "compile_full_windows", "cuda_graph"
]
LossBackend = Literal["reference", "target_only", "liger"]
TrainingPrecision = Literal["fp32", "bf16_mixed", "bf16"]
SFTLossScope = Literal["answer_only", "full_sequence"]
ProcessorGenerationMode = Literal["windowed", "post_prefill_relative"]
DiffusionInferenceMode = Literal["deterministic", "stochastic"]

_FLOAT32_MAX = 3.4028234663852886e38
_FLOAT32_MIN_SUBNORMAL = 2.0**-149


def _validated_float32(
    name: str, value: object, *, positive: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(
            f"{name} must be finite and representable as float32"
        ) from None
    if not math.isfinite(result) or abs(result) > _FLOAT32_MAX:
        raise ValueError(f"{name} must be finite and representable as float32")
    if positive and result < _FLOAT32_MIN_SUBNORMAL:
        raise ValueError(
            f"{name} must remain positive when represented as float32"
        )
    return result
