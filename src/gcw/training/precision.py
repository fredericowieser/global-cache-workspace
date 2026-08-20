from __future__ import annotations

from typing import Any
import torch

from gcw.config import TrainConfig


def _uses_bf16_autocast(config: TrainConfig, device: torch.device) -> bool:
    return config.precision == "bf16_mixed" and device.type == "cuda"


def _validate_precision(config: TrainConfig, device: torch.device) -> None:
    if (
        config.precision in {"bf16_mixed", "bf16"}
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            f"precision={config.precision!r} requires a CUDA device with BF16 support"
        )


def _autocast_kwargs(config: TrainConfig, device: torch.device) -> dict[str, Any]:
    if config.precision == "bf16_mixed":
        return {"device_type": device.type, "dtype": torch.bfloat16, "enabled": device.type == "cuda"}
    if config.precision == "bf16":
        return {"device_type": device.type, "dtype": torch.bfloat16, "enabled": False}
    return {"device_type": device.type, "dtype": torch.float32, "enabled": False}


def _precision_summary(config: TrainConfig, device: torch.device) -> str:
    """Describe the precision actually in effect, including autocast that a device disables."""
    parameters = "bfloat16" if config.precision == "bf16" else "float32"
    autocast = _autocast_kwargs(config, device)
    if autocast["enabled"]:
        compute = f"autocast {str(autocast['dtype']).removeprefix('torch.')}"
    elif config.precision == "bf16_mixed":
        compute = f"autocast disabled on {device.type}, computing in float32"
    else:
        compute = f"no autocast, computing in {parameters}"
    return f"precision: {config.precision} (parameters {parameters}, {compute})"
