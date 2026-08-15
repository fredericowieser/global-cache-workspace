"""Typed configuration for explicit performance execution policies.

This module deliberately has no imports from :mod:`gcw.config`, model code, or
training code. Keeping these dataclasses dependency-free prevents the
performance package from creating configuration import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class PhysicalBatchBandConfig:
    """One maximum padded width and the physical row count allowed below it."""

    max_padded_tokens: int
    physical_rows: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_padded_tokens, bool)
            or not isinstance(self.max_padded_tokens, int)
            or self.max_padded_tokens <= 0
        ):
            raise ValueError("adaptive batching max_padded_tokens must be a positive integer")
        if (
            isinstance(self.physical_rows, bool)
            or not isinstance(self.physical_rows, int)
            or self.physical_rows < 2
            or self.physical_rows % 2
        ):
            raise ValueError("adaptive batching physical_rows must be an even integer of at least 2")


_DEFAULT_ADAPTIVE_BANDS = (
    PhysicalBatchBandConfig(max_padded_tokens=1024, physical_rows=16),
    PhysicalBatchBandConfig(max_padded_tokens=2048, physical_rows=8),
    PhysicalBatchBandConfig(max_padded_tokens=3072, physical_rows=4),
    PhysicalBatchBandConfig(max_padded_tokens=4096, physical_rows=2),
)


@dataclass(frozen=True)
class AdaptiveBatchingConfig:
    """Default-off policy for contiguous adjacent-pair physical grouping."""

    enabled: bool = False
    bands: tuple[PhysicalBatchBandConfig, ...] = field(default_factory=lambda: _DEFAULT_ADAPTIVE_BANDS)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("performance.adaptive_batching.enabled must be a boolean")
        raw_bands: Any = self.bands
        if not isinstance(raw_bands, (list, tuple)):
            raise ValueError("performance.adaptive_batching.bands must be a list or tuple")
        converted: list[PhysicalBatchBandConfig] = []
        for band in raw_bands:
            if isinstance(band, dict):
                converted.append(PhysicalBatchBandConfig(**band))
            elif isinstance(band, PhysicalBatchBandConfig):
                converted.append(band)
            else:
                raise ValueError("performance.adaptive_batching.bands must contain band mappings")
        bands = tuple(converted)
        if not bands:
            raise ValueError("performance.adaptive_batching.bands must not be empty")
        physical_rows = tuple(band.physical_rows for band in bands)
        thresholds = tuple(band.max_padded_tokens for band in bands)
        if any(left <= right for left, right in zip(physical_rows, physical_rows[1:])):
            raise ValueError("adaptive batching physical_rows must be strictly decreasing")
        if any(left >= right for left, right in zip(thresholds, thresholds[1:])):
            raise ValueError("adaptive batching max_padded_tokens must be strictly increasing")
        if bands[-1].physical_rows != 2:
            raise ValueError("adaptive batching must end with a two-row fallback band")
        object.__setattr__(self, "bands", bands)


@dataclass(frozen=True)
class ReferenceLossConfig:
    """Default-compatible execution policy for the full-vocabulary reference loss."""

    mode: Literal["legacy", "bounded"] = "legacy"
    max_projected_positions: int = 1024
    checkpoint_chunks: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"legacy", "bounded"}:
            raise ValueError("performance.reference_loss.mode must be 'legacy' or 'bounded'")
        if (
            isinstance(self.max_projected_positions, bool)
            or not isinstance(self.max_projected_positions, int)
            or self.max_projected_positions <= 0
        ):
            raise ValueError("performance.reference_loss.max_projected_positions must be a positive integer")
        if not isinstance(self.checkpoint_chunks, bool):
            raise ValueError("performance.reference_loss.checkpoint_chunks must be a boolean")
        if self.mode == "legacy" and self.checkpoint_chunks:
            raise ValueError("reference-loss chunk checkpointing requires mode='bounded'")


@dataclass(frozen=True)
class LongshotFullHistoryConfig:
    """Default-off selected-history execution for Longshot KV-norm selection."""

    enabled: bool = False
    max_temp_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("performance.longshot_full_history.enabled must be a boolean")
        if (
            isinstance(self.max_temp_bytes, bool)
            or not isinstance(self.max_temp_bytes, int)
            or self.max_temp_bytes <= 0
        ):
            raise ValueError("performance.longshot_full_history.max_temp_bytes must be a positive integer")


@dataclass(frozen=True)
class KVScoreSidecarConfig:
    """Default-off exact FP32 Longshot selector-score cache."""

    enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("performance.kv_score_sidecar.enabled must be a boolean")


@dataclass(frozen=True)
class TritonCacheMovementConfig:
    """Default-off raw-bit cache gather/layout backend."""

    enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("performance.triton_cache_movement.enabled must be a boolean")


@dataclass(frozen=True)
class NoisyDiffusionBackendConfig:
    """Execution policy for noisy Diffusion-BGT's deterministic recurrence."""

    recurrence_backend: Literal["eager", "compile"] = "eager"

    def __post_init__(self) -> None:
        if self.recurrence_backend not in {"eager", "compile"}:
            raise ValueError("performance.noisy_diffusion.recurrence_backend must be 'eager' or 'compile'")


@dataclass(frozen=True)
class PerformanceConfig:
    """Training performance policies whose defaults preserve reference execution."""

    adaptive_batching: AdaptiveBatchingConfig = field(default_factory=AdaptiveBatchingConfig)
    reference_loss: ReferenceLossConfig = field(default_factory=ReferenceLossConfig)
    longshot_full_history: LongshotFullHistoryConfig = field(default_factory=LongshotFullHistoryConfig)
    kv_score_sidecar: KVScoreSidecarConfig = field(default_factory=KVScoreSidecarConfig)
    triton_cache_movement: TritonCacheMovementConfig = field(default_factory=TritonCacheMovementConfig)
    noisy_diffusion: NoisyDiffusionBackendConfig = field(default_factory=NoisyDiffusionBackendConfig)

    def __post_init__(self) -> None:
        value: Any = self.adaptive_batching
        if isinstance(value, dict):
            object.__setattr__(self, "adaptive_batching", AdaptiveBatchingConfig(**value))
        elif not isinstance(value, AdaptiveBatchingConfig):
            raise ValueError("performance.adaptive_batching must be a mapping")
        reference_loss: Any = self.reference_loss
        if isinstance(reference_loss, dict):
            object.__setattr__(self, "reference_loss", ReferenceLossConfig(**reference_loss))
        elif not isinstance(reference_loss, ReferenceLossConfig):
            raise ValueError("performance.reference_loss must be a mapping")
        longshot_full_history: Any = self.longshot_full_history
        if isinstance(longshot_full_history, dict):
            object.__setattr__(
                self,
                "longshot_full_history",
                LongshotFullHistoryConfig(**longshot_full_history),
            )
        elif not isinstance(longshot_full_history, LongshotFullHistoryConfig):
            raise ValueError("performance.longshot_full_history must be a mapping")
        kv_score_sidecar: Any = self.kv_score_sidecar
        if isinstance(kv_score_sidecar, dict):
            object.__setattr__(
                self,
                "kv_score_sidecar",
                KVScoreSidecarConfig(**kv_score_sidecar),
            )
        elif not isinstance(kv_score_sidecar, KVScoreSidecarConfig):
            raise ValueError("performance.kv_score_sidecar must be a mapping")
        triton_cache_movement: Any = self.triton_cache_movement
        if isinstance(triton_cache_movement, dict):
            object.__setattr__(
                self,
                "triton_cache_movement",
                TritonCacheMovementConfig(**triton_cache_movement),
            )
        elif not isinstance(triton_cache_movement, TritonCacheMovementConfig):
            raise ValueError("performance.triton_cache_movement must be a mapping")
        noisy_diffusion: Any = self.noisy_diffusion
        if isinstance(noisy_diffusion, dict):
            object.__setattr__(
                self,
                "noisy_diffusion",
                NoisyDiffusionBackendConfig(**noisy_diffusion),
            )
        elif not isinstance(noisy_diffusion, NoisyDiffusionBackendConfig):
            raise ValueError("performance.noisy_diffusion must be a mapping")
