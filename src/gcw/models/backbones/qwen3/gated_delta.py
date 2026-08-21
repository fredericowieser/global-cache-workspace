from __future__ import annotations

from typing import Any, Callable
import torch
import torch.nn as nn

Kernel = Callable[..., Any]
_fla_chunk_gated_delta_rule: Kernel | None = None
_fla_recurrent_gated_delta_rule: Kernel | None = None
_causal_conv1d_fn: Kernel | None = None
_hf_qwen35_torch_chunk_gated_delta_rule: Kernel | None = None
_hf_qwen35_torch_recurrent_gated_delta_rule: Kernel | None = None
_hf_qwen35_moe_torch_chunk_gated_delta_rule: Kernel | None = None
_hf_qwen35_moe_torch_recurrent_gated_delta_rule: Kernel | None = None
_HF_ACT2FN: Any = None
_hf_use_experts_implementation: Kernel | None = None

try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _imported_fla_chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule as _imported_fla_recurrent_gated_delta_rule,
    )
except Exception:
    pass
else:
    _fla_chunk_gated_delta_rule = _imported_fla_chunk_gated_delta_rule
    _fla_recurrent_gated_delta_rule = _imported_fla_recurrent_gated_delta_rule

try:
    from causal_conv1d import causal_conv1d_fn as _imported_causal_conv1d_fn
except Exception:
    pass
else:
    _causal_conv1d_fn = _imported_causal_conv1d_fn

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule as _imported_hf_qwen35_torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule as _imported_hf_qwen35_torch_recurrent_gated_delta_rule,
    )
except Exception:
    pass
else:
    _hf_qwen35_torch_chunk_gated_delta_rule = (
        _imported_hf_qwen35_torch_chunk_gated_delta_rule
    )
    _hf_qwen35_torch_recurrent_gated_delta_rule = (
        _imported_hf_qwen35_torch_recurrent_gated_delta_rule
    )

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        torch_chunk_gated_delta_rule as _imported_hf_qwen35_moe_torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule as _imported_hf_qwen35_moe_torch_recurrent_gated_delta_rule,
    )
except Exception:
    pass
else:
    _hf_qwen35_moe_torch_chunk_gated_delta_rule = (
        _imported_hf_qwen35_moe_torch_chunk_gated_delta_rule
    )
    _hf_qwen35_moe_torch_recurrent_gated_delta_rule = (
        _imported_hf_qwen35_moe_torch_recurrent_gated_delta_rule
    )

_hf_torch_chunk_gated_delta_rule = (
    _hf_qwen35_torch_chunk_gated_delta_rule
    or _hf_qwen35_moe_torch_chunk_gated_delta_rule
)
_hf_torch_recurrent_gated_delta_rule = (
    _hf_qwen35_torch_recurrent_gated_delta_rule
    or _hf_qwen35_moe_torch_recurrent_gated_delta_rule
)

try:
    from transformers.activations import ACT2FN as _imported_hf_act2fn
    from transformers.integrations import (
        use_experts_implementation as _imported_hf_use_experts_implementation,
    )
except Exception:
    pass
else:
    _HF_ACT2FN = _imported_hf_act2fn
    _hf_use_experts_implementation = _imported_hf_use_experts_implementation


def _has_fla_gated_delta_rule() -> bool:
    return (
        _fla_chunk_gated_delta_rule is not None
        and _fla_recurrent_gated_delta_rule is not None
    )


def _has_hf_torch_gated_delta_rule() -> bool:
    return (
        _hf_torch_chunk_gated_delta_rule is not None
        and _hf_torch_recurrent_gated_delta_rule is not None
    )


class Qwen35GatedDeltaDynamics(nn.Module):
    def __init__(self, num_v_heads: int):
        super().__init__()
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))
        self.A_log = nn.Parameter(torch.empty(num_v_heads))

    @torch.no_grad()
    def init_weights(self) -> None:
        nn.init.ones_(self.dt_bias)
        self.A_log.copy_(torch.empty_like(self.A_log).uniform_(1e-4, 16).log_())


def qwen35_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    use_fla = query.device.type == "cuda" and _has_fla_gated_delta_rule()
    if not use_fla and not _has_hf_torch_gated_delta_rule():
        raise RuntimeError(
            "CPU Qwen3.5 gated-delta execution requires Transformers' torch fallback."
        )
    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (query, key, value, g, beta, initial_state)
    )
    if query.shape[1] == 1 and not needs_grad:
        recurrent_impl = (
            _fla_recurrent_gated_delta_rule
            if use_fla
            else _hf_torch_recurrent_gated_delta_rule
        )
        if recurrent_impl is None:
            raise RuntimeError(
                "Qwen3.5 recurrent gated-delta kernel is unavailable"
            )
        return recurrent_impl(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    chunk_impl = (
        _fla_chunk_gated_delta_rule
        if use_fla
        else _hf_torch_chunk_gated_delta_rule
    )
    if chunk_impl is None:
        raise RuntimeError("Qwen3.5 chunked gated-delta kernel is unavailable")
    return chunk_impl(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )


def _qwen35_linear_state_dynamics(
    backbone: nn.Module,
) -> dict[int, Qwen35GatedDeltaDynamics]:
    from gcw.models.backbones.qwen3.linear_attention import Qwen35GatedDeltaNet

    dynamics_by_layer: dict[int, Qwen35GatedDeltaDynamics] = {}
    for layer in getattr(backbone, "layers", []):
        linear_attn = getattr(layer, "linear_attn", None)
        if isinstance(linear_attn, Qwen35GatedDeltaNet):
            dynamics_by_layer[int(layer.layer_idx)] = linear_attn.dynamics
    return dynamics_by_layer
