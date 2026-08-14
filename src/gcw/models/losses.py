from __future__ import annotations

from importlib import metadata
from typing import Any, Mapping, Protocol, cast

import torch

from gcw.config import LossBackend

LIGER_KERNEL_VERSION = "0.8.1"


class FusedLinearCrossEntropy(Protocol):
    def __call__(
        self,
        lin_weight: torch.Tensor,
        input: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class FusedLinearCrossEntropyFactory(Protocol):
    def __call__(self, **kwargs: Any) -> FusedLinearCrossEntropy: ...


def _load_liger_factory() -> FusedLinearCrossEntropyFactory:
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    except (ImportError, OSError) as exc:
        raise RuntimeError("loss_backend='liger' requires the CUDA GPU extra with liger-kernel==0.8.1") from exc
    return LigerFusedLinearCrossEntropyLoss


def require_loss_backend(backend: LossBackend, device: torch.device) -> None:
    """Fail before model allocation when an explicitly requested backend cannot run."""
    if backend not in {"reference", "target_only", "liger"}:
        raise ValueError(f"unknown loss backend: {backend!r}")
    if backend != "liger":
        return
    if device.type != "cuda":
        raise RuntimeError("loss_backend='liger' requires a CUDA device")
    try:
        version = metadata.version("liger-kernel")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("loss_backend='liger' requires the CUDA GPU extra with liger-kernel==0.8.1") from exc
    if version != LIGER_KERNEL_VERSION:
        raise RuntimeError(f"loss_backend='liger' requires liger-kernel=={LIGER_KERNEL_VERSION}; found {version}")
    _load_liger_factory()


def loss_backend_metadata(backend: LossBackend) -> dict[str, Any]:
    if backend not in {"reference", "target_only", "liger"}:
        raise ValueError(f"unknown loss backend: {backend!r}")
    result: dict[str, Any] = {
        "name": backend,
        "implementation": {
            "reference": "torch_full_projection_chunked_ce",
            "target_only": "torch_target_only_chunked_ce",
            "liger": "liger_fused_linear_cross_entropy",
        }[backend],
        "accumulation_dtype": "float32",
    }
    if backend == "liger":
        result.update(
            {
                "package": "liger-kernel",
                "package_version": metadata.version("liger-kernel"),
                "reduction_none": {
                    "effective_backend": "target_only",
                    "reason": "liger-kernel 0.8.1 fused linear CE has no per-token reduction",
                },
            }
        )
    return result


def checkpoint_loss_backend(state: Mapping[str, Any], device: torch.device) -> LossBackend:
    """Recover and validate the backend recorded by a final or training checkpoint."""
    recorded = state.get("loss_backend", "reference")
    name = recorded.get("name") if isinstance(recorded, dict) else recorded
    if name not in {"reference", "target_only", "liger"}:
        raise ValueError("checkpoint loss_backend provenance is invalid")
    backend = cast(LossBackend, name)
    if backend == "liger" and isinstance(recorded, dict):
        version = recorded.get("package_version")
        if version != LIGER_KERNEL_VERSION:
            raise RuntimeError(f"checkpoint requires liger-kernel=={LIGER_KERNEL_VERSION}; recorded {version!r}")
    require_loss_backend(backend, device)
    return backend


def liger_fused_linear_cross_entropy(
    weight: torch.Tensor,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str,
    softcap: float | None = None,
) -> torch.Tensor:
    """Apply Liger FLCE without selecting target rows or constructing full logits."""
    if reduction not in {"mean", "sum"}:
        raise ValueError("Liger fused linear cross entropy supports only mean and sum in this wrapper")
    if hidden.shape[:-1] != targets.shape:
        raise ValueError("hidden leading dimensions must match targets")
    if weight.ndim != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError("projection weight shape must be (vocab_size, hidden_size)")
    require_loss_backend("liger", hidden.device)
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    normalized_targets = torch.where(flat_targets >= 0, flat_targets, flat_targets.new_full((), -100))

    # Liger 0.8.1 does not define this edge safely for mean. This synchronization is preferable
    # to a NaN and does not gather target indices or materialize vocabulary logits.
    if not bool(torch.any(normalized_targets != -100).item()):
        return (flat_hidden.reshape(-1)[:1].float().sum() + weight.reshape(-1)[:1].float().sum()) * 0.0

    criterion = _load_liger_factory()(
        ignore_index=-100,
        reduction=reduction,
        accum_dtype=torch.float32,
        softcap=softcap,
    )
    return criterion(weight, flat_hidden, normalized_targets, None).float()
