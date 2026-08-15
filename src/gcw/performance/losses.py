from __future__ import annotations

from importlib import metadata
from typing import Any, Callable, Mapping, Protocol, cast

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from gcw.config import LossBackend
from gcw.performance.config import ReferenceLossConfig
from gcw.provenance import canonical_json_sha256

LIGER_KERNEL_VERSION = "0.8.1"
_REFERENCE_LOSS_EXECUTION_ATTRIBUTE = "_gcw_reference_loss_execution"


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
                "reduction_row_sum": {
                    "effective_backend": "liger_per_row_sum_v1",
                    "physical_calls_per_row": 1,
                    "reason": "one fused sum reduction per physical row preserves row structure without logits",
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


def configure_reference_loss_execution(model: torch.nn.Module, config: ReferenceLossConfig) -> None:
    """Attach state-neutral reference-loss execution policy to the owning backbone."""
    backbone = getattr(model, "backbone", model)
    object.__setattr__(backbone, _REFERENCE_LOSS_EXECUTION_ATTRIBUTE, config)


def reference_loss_execution(model: torch.nn.Module) -> ReferenceLossConfig:
    """Return the configured policy, preserving legacy execution by default."""
    value = getattr(model, _REFERENCE_LOSS_EXECUTION_ATTRIBUTE, None)
    if value is None:
        return ReferenceLossConfig()
    if not isinstance(value, ReferenceLossConfig):
        raise TypeError("configured reference-loss execution policy has an invalid type")
    return value


def reference_loss_execution_metadata(config: ReferenceLossConfig) -> dict[str, Any]:
    """Return authenticated provenance for non-scientific reference-loss execution."""
    execution = {
        "schema": "gcw.reference_loss_execution.v1",
        "mode": config.mode,
        "max_projected_positions": config.max_projected_positions,
        "checkpoint_chunks": config.checkpoint_chunks,
        "logical_sequence_chunk_cuda": 256,
        "logical_sequence_chunk_cpu": 64,
        "physical_chunk_rule": "min(logical_chunk,max(1,max_projected_positions//batch_rows))",
        "projects_ignored_positions": True,
        "vocabulary_projection": "full",
        "cross_entropy": "torch.nn.functional.cross_entropy_reduction_none",
        "accumulation_dtype": "float32",
        "torch_version": str(torch.__version__),
    }
    return {
        **execution,
        "requested_via": "TrainConfig.performance.reference_loss",
        "implementation_module": "gcw.performance.losses",
        "implementation_symbol": "gcw.performance.losses.bounded_reference_loss",
        "fingerprint_sha256": canonical_json_sha256(execution),
    }


def bounded_reference_loss(
    logits_from_hidden: Callable[[torch.Tensor], torch.Tensor],
    hidden: torch.Tensor,
    targets: torch.Tensor,
    reduction: str,
    config: ReferenceLossConfig,
) -> torch.Tensor:
    """Run full-vocabulary reference CE with a bounded physical projection chunk."""
    if config.mode != "bounded":
        raise ValueError("bounded_reference_loss requires mode='bounded'")
    if reduction not in {"mean", "sum", "none", "row_sum"}:
        raise ValueError("reference loss reduction must be 'mean', 'sum', 'none', or 'row_sum'")
    if hidden.ndim != 3 or targets.shape != hidden.shape[:2]:
        raise ValueError("reference loss requires hidden [rows, tokens, width] and matching targets")
    if hidden.shape[0] == 0:
        raise ValueError("reference loss requires at least one physical row")
    if config.max_projected_positions < hidden.shape[0]:
        raise ValueError("reference-loss max_projected_positions must be at least the physical batch row count")

    logical_chunk_size = 256 if hidden.is_cuda else 64
    physical_chunk_size = min(
        logical_chunk_size,
        max(1, config.max_projected_positions // hidden.shape[0]),
    )
    loss_sum = hidden.new_zeros((), dtype=torch.float32)
    row_loss_sums = hidden.new_zeros((hidden.shape[0],), dtype=torch.float32)
    token_count = hidden.new_zeros((), dtype=torch.float32)
    per_token: list[torch.Tensor] = []
    checkpoint_active = config.checkpoint_chunks and torch.is_grad_enabled()

    for logical_start in range(0, hidden.shape[1], logical_chunk_size):
        logical_end = min(logical_start + logical_chunk_size, hidden.shape[1])
        logical_losses: list[torch.Tensor] = []
        logical_valid: list[torch.Tensor] = []
        for start in range(logical_start, logical_end, physical_chunk_size):
            end = min(start + physical_chunk_size, logical_end)
            hidden_chunk = hidden[:, start:end]
            target_chunk = targets[:, start:end]
            if checkpoint_active:
                chunk_loss = activation_checkpoint(
                    lambda values, labels: _full_projection_loss_chunk(logits_from_hidden, values, labels),
                    hidden_chunk,
                    target_chunk,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                chunk_loss = _full_projection_loss_chunk(logits_from_hidden, hidden_chunk, target_chunk)
            logical_losses.append(chunk_loss)
            logical_valid.append(target_chunk >= 0)

        block_loss = torch.cat(logical_losses, dim=1)
        block_valid = torch.cat(logical_valid, dim=1)
        loss_sum = loss_sum + block_loss.sum()
        token_count = token_count + block_valid.sum().to(torch.float32)
        if reduction == "row_sum":
            row_loss_sums = row_loss_sums + block_loss.sum(dim=1)
        if reduction == "none":
            per_token.append(block_loss)

    if reduction == "none":
        return torch.cat(per_token, dim=1)
    if reduction == "row_sum":
        return row_loss_sums
    if reduction == "sum":
        return loss_sum
    return loss_sum / token_count.clamp_min(1.0)


def _full_projection_loss_chunk(
    logits_from_hidden: Callable[[torch.Tensor], torch.Tensor],
    hidden: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    logits = logits_from_hidden(hidden)
    normalized_targets = targets.reshape(-1).clone()
    normalized_targets[normalized_targets < 0] = -100
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        normalized_targets,
        ignore_index=-100,
        reduction="none",
    ).reshape(targets.shape)


def liger_fused_linear_cross_entropy_row_sum(
    weight: torch.Tensor,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    *,
    softcap: float | None = None,
) -> torch.Tensor:
    """Return structured row sums through one fused Liger call per row."""
    if hidden.ndim != 3 or hidden.shape[:-1] != targets.shape:
        raise ValueError("Liger row-sum loss requires hidden [rows, tokens, width] and matching targets")
    if weight.ndim != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError("projection weight shape must be (vocab_size, hidden_size)")
    require_loss_backend("liger", hidden.device)
    criterion = _load_liger_factory()(
        ignore_index=-100,
        reduction="sum",
        accum_dtype=torch.float32,
        softcap=softcap,
    )
    row_sums: list[torch.Tensor] = []
    for row in range(hidden.shape[0]):
        row_hidden = hidden[row].reshape(-1, hidden.shape[-1])
        row_targets = targets[row].reshape(-1)
        normalized = torch.where(row_targets >= 0, row_targets, row_targets.new_full((), -100))
        # Sum reduction is well-defined for an all-ignored row and preserves
        # differentiable zero gradients without a host synchronization.
        row_sums.append(criterion(weight, row_hidden, normalized, None).float())
    return torch.stack(row_sums)


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
