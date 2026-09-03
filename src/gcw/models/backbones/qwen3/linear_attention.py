from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcw.config import ModelConfig
from gcw.models.backbones.transformer import Linear
from gcw.models.backbones.qwen3.gated_delta import (
    _HF_ACT2FN,
    _causal_conv1d_fn,
    _has_fla_gated_delta_rule,
    _has_hf_torch_gated_delta_rule,
    qwen35_gated_delta_rule,
    Qwen35GatedDeltaDynamics,
)
from gcw.models.backbones.qwen3.norms import Qwen35RMSNormGated


class Qwen35GatedDeltaNet(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        if (
            not _has_fla_gated_delta_rule()
            and not _has_hf_torch_gated_delta_rule()
        ):
            raise ImportError(
                "Qwen3.5 linear attention requires FLA kernels or Transformers' Qwen3.5 torch fallback functions."
            )
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.hidden_size = config.hidden_size
        self.num_v_heads = int(
            config.linear_num_value_heads or config.num_heads
        )
        self.num_k_heads = int(
            config.linear_num_key_heads or config.num_kv_heads
        )
        default_head_dim = config.head_dim_for_layer(layer_idx)
        self.head_k_dim = int(config.linear_key_head_dim or default_head_dim)
        self.head_v_dim = int(config.linear_value_head_dim or default_head_dim)
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = int(config.linear_conv_kernel_dim)
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.dynamics = Qwen35GatedDeltaDynamics(self.num_v_heads)
        self.norm = Qwen35RMSNormGated(self.head_v_dim, config.norm_eps)
        self.out_proj = Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = Linear(self.hidden_size, self.num_v_heads, bias=False)

    @property
    def dt_bias(self) -> nn.Parameter:
        return self.dynamics.dt_bias

    @property
    def A_log(self) -> nn.Parameter:
        return self.dynamics.A_log

    def effective_backends(self, device: torch.device) -> dict[str, str]:
        use_fast_gated_delta = (
            device.type == "cuda" and _has_fla_gated_delta_rule()
        )
        return {
            "causal_conv": (
                "causal-conv1d"
                if device.type == "cuda" and _causal_conv1d_fn is not None
                else "torch-conv1d"
            ),
            "gated_delta": (
                "fla" if use_fast_gated_delta else "transformers-torch"
            ),
        }

    @torch.no_grad()
    def init_state_weights(self) -> None:
        self.dynamics.init_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_state: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).view(
            batch, seq_len, -1, self.head_v_dim
        )
        beta = torch.sigmoid(self.in_proj_b(hidden_states))
        a = self.in_proj_a(hidden_states)
        recurrent_state = past_state[0] if past_state is not None else None
        conv_state = past_state[1] if past_state is not None else None

        conv_out, new_conv_state = self._causal_conv(mixed_qkv, conv_state)
        query, key, value = torch.split(
            conv_out.transpose(1, 2),
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
        g = self.dynamics(a)
        if self.num_v_heads // self.num_k_heads > 1:
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)
        core_attn_out, new_recurrent_state = self._gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
        )
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z).reshape(batch, seq_len, -1)
        return self.out_proj(core_attn_out), (
            new_recurrent_state,
            new_conv_state,
        )

    def _causal_conv(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_len = self.conv_kernel_size - 1
        if conv_state is not None:
            mixed = torch.cat(
                [conv_state.to(mixed_qkv.dtype), mixed_qkv], dim=-1
            )
        else:
            mixed = mixed_qkv
        if mixed_qkv.device.type == "cuda" and _causal_conv1d_fn is not None:
            out = _causal_conv1d_fn(
                x=mixed,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        else:
            out = self.conv1d(mixed)[:, :, : mixed.shape[-1]]
            out = (
                _HF_ACT2FN[self.activation](out)
                if _HF_ACT2FN is not None
                else F.silu(out)
            )
        out = out[:, :, -mixed_qkv.shape[-1] :]
        if state_len > 0:
            new_state = (
                mixed[:, :, -state_len:].detach()
                if not mixed.requires_grad
                else mixed[:, :, -state_len:]
            )
        else:
            new_state = mixed[:, :, :0]
        return out.to(mixed_qkv.dtype), new_state

    def _gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return qwen35_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
        )


Qwen35MoeGatedDeltaNet = Qwen35GatedDeltaNet
