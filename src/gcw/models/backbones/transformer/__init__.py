from __future__ import annotations

from gcw.models.backbones.transformer.attention import CausalSelfAttention
from gcw.models.backbones.transformer.block import Block
from gcw.models.backbones.transformer.cache import (
    CacheLike,
    LinearStateLayout,
    ProcessorExecutionFactory,
    ProcessorFn,
    WindowedCacheLayout,
    _flatten_windowed_cache,
    _unflatten_windowed_cache,
    _window_checkpoint_contexts,
)
from gcw.models.backbones.transformer.generation import (
    estimate_non_embedding_params,
    _matches_stop_sequence,
    _should_process_decoding_cache,
)
from gcw.models.backbones.transformer.layers import (
    Linear,
    MLP,
    RMSNorm,
    RotaryCache,
    SDPBackend,
    apply_rotary,
    causal_lower_right,
    flash_sdpa,
    sdpa_kernel,
    _autocast_dtype,
    _causal_chunk_bias,
    _causal_chunk_mask,
)
from gcw.models.backbones.transformer.tf import TF
from gcw.performance.losses import (
    bounded_reference_loss,
    liger_fused_linear_cross_entropy,
    reference_loss_execution,
    require_loss_backend,
)

__all__ = [
    "Block",
    "CacheLike",
    "CausalSelfAttention",
    "Linear",
    "LinearStateLayout",
    "MLP",
    "ProcessorExecutionFactory",
    "ProcessorFn",
    "RMSNorm",
    "RotaryCache",
    "SDPBackend",
    "TF",
    "WindowedCacheLayout",
    "apply_rotary",
    "bounded_reference_loss",
    "causal_lower_right",
    "estimate_non_embedding_params",
    "flash_sdpa",
    "liger_fused_linear_cross_entropy",
    "reference_loss_execution",
    "require_loss_backend",
    "sdpa_kernel",
    "_autocast_dtype",
    "_causal_chunk_bias",
    "_causal_chunk_mask",
    "_flatten_windowed_cache",
    "_matches_stop_sequence",
    "_should_process_decoding_cache",
    "_unflatten_windowed_cache",
    "_window_checkpoint_contexts",
]
