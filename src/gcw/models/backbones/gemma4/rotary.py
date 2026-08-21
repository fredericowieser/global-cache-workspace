from __future__ import annotations

import torch
import torch.nn as nn

from gcw.config import ModelConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return (x * cos[None, :, None, :]) + (
        _rotate_half(x) * sin[None, :, None, :]
    )


class Gemma4RotaryCache(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.rope_parameters = config.rope_parameters or {}
        for layer_type in set(config.layer_types or ["full_attention"]):
            head_dim = self._head_dim(layer_type)
            inv_freq = self._inv_freq(layer_type, head_dim, torch.device("cpu"))
            self.register_buffer(
                f"{layer_type}_inv_freq", inv_freq, persistent=False
            )

    def get(
        self,
        positions: torch.Tensor,
        layer_type: str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq").to(positions.device)
        freqs = positions.float().unsqueeze(-1) * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _head_dim(self, layer_type: str) -> int:
        if layer_type == "full_attention":
            head_dim = self.config.global_head_dim or self.config.head_dim
        else:
            head_dim = self.config.head_dim
        if head_dim is None:
            raise ValueError("Gemma4 rotary cache requires a resolved head_dim")
        return head_dim

    def _inv_freq(
        self, layer_type: str, head_dim: int, device: torch.device
    ) -> torch.Tensor:
        params = self.rope_parameters.get(layer_type, {})
        base = float(params.get("rope_theta", 10000.0))
        rope_type = params.get("rope_type", "default")
        if rope_type == "proportional":
            proportion = float(params.get("partial_rotary_factor", 1.0))
            rope_angles = int(proportion * head_dim // 2)
            if rope_angles > 0:
                inv_rotated = 1.0 / (
                    base
                    ** (
                        torch.arange(
                            0,
                            2 * rope_angles,
                            2,
                            dtype=torch.float32,
                            device=device,
                        )
                        / float(head_dim)
                    )
                )
            else:
                inv_rotated = torch.empty(0, dtype=torch.float32, device=device)
            nope_angles = head_dim // 2 - rope_angles
            if nope_angles > 0:
                inv_freq = torch.cat(
                    [
                        inv_rotated,
                        torch.zeros(
                            nope_angles, dtype=torch.float32, device=device
                        ),
                    ]
                )
            else:
                inv_freq = inv_rotated
        elif rope_type == "default":
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(
                        0, head_dim, 2, dtype=torch.float32, device=device
                    )
                    / float(head_dim)
                )
            )
        else:
            raise NotImplementedError(
                f"Gemma4 rope_type={rope_type!r} is not implemented"
            )
        return inv_freq / float(params.get("factor", 1.0))
