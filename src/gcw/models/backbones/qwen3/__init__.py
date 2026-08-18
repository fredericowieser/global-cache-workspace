from __future__ import annotations

from gcw.models.backbones.qwen3.attention import (
    Qwen35Attention,
    _apply_partial_rotary,
)
from gcw.models.backbones.qwen3.blocks import (
    Qwen35Block,
    Qwen35MLP,
)
from gcw.models.backbones.qwen3.gated_delta import (
    Kernel,
    Qwen35GatedDeltaDynamics,
    _HF_ACT2FN,
    _causal_conv1d_fn,
    _fla_chunk_gated_delta_rule,
    _fla_recurrent_gated_delta_rule,
    _has_fla_gated_delta_rule,
    _has_hf_torch_gated_delta_rule,
    _hf_qwen35_moe_torch_chunk_gated_delta_rule,
    _hf_qwen35_moe_torch_recurrent_gated_delta_rule,
    _hf_qwen35_torch_chunk_gated_delta_rule,
    _hf_qwen35_torch_recurrent_gated_delta_rule,
    _hf_torch_chunk_gated_delta_rule,
    _hf_torch_recurrent_gated_delta_rule,
    _hf_use_experts_implementation,
    _qwen35_linear_state_dynamics,
    qwen35_gated_delta_rule,
)
from gcw.models.backbones.qwen3.linear_attention import (
    Qwen35GatedDeltaNet,
    Qwen35MoeGatedDeltaNet,
)
from gcw.models.backbones.qwen3.models import (
    Qwen35MoeTF,
    Qwen35TF,
    Qwen3TF,
)
from gcw.models.backbones.qwen3.moe import (
    Qwen35MoeBlock,
    Qwen35MoeExperts,
    Qwen35MoeSparseMoeBlock,
    _Qwen35MoeExperts,
    qwen35_moe_load_balancing_loss,
)
from gcw.models.backbones.qwen3.norms import (
    Qwen35MoeRMSNorm,
    Qwen35RMSNorm,
    Qwen35RMSNormGated,
)

__all__ = [
    "Kernel",
    "Qwen35Attention",
    "Qwen35Block",
    "Qwen35GatedDeltaDynamics",
    "Qwen35GatedDeltaNet",
    "Qwen35MLP",
    "Qwen35MoeBlock",
    "Qwen35MoeExperts",
    "Qwen35MoeGatedDeltaNet",
    "Qwen35MoeRMSNorm",
    "Qwen35MoeSparseMoeBlock",
    "Qwen35MoeTF",
    "Qwen35RMSNorm",
    "Qwen35RMSNormGated",
    "Qwen35TF",
    "Qwen3TF",
    "_HF_ACT2FN",
    "_Qwen35MoeExperts",
    "_apply_partial_rotary",
    "_causal_conv1d_fn",
    "_fla_chunk_gated_delta_rule",
    "_fla_recurrent_gated_delta_rule",
    "_has_fla_gated_delta_rule",
    "_has_hf_torch_gated_delta_rule",
    "_hf_qwen35_moe_torch_chunk_gated_delta_rule",
    "_hf_qwen35_moe_torch_recurrent_gated_delta_rule",
    "_hf_qwen35_torch_chunk_gated_delta_rule",
    "_hf_qwen35_torch_recurrent_gated_delta_rule",
    "_hf_torch_chunk_gated_delta_rule",
    "_hf_torch_recurrent_gated_delta_rule",
    "_hf_use_experts_implementation",
    "_qwen35_linear_state_dynamics",
    "qwen35_gated_delta_rule",
    "qwen35_moe_load_balancing_loss",
]
