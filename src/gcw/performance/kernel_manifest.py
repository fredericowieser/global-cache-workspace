"""Closed launch manifest for production-candidate handwritten kernels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Literal

from gcw.provenance import canonical_json_bytes

CACHE_FEATURE_BLOCK = 128
CACHE_COPY_BLOCK = 256
CACHE_NUM_WARPS = 4
KERNEL_MANIFEST_SCHEMA = "gcw.kernel_manifest.v1"


@dataclass(frozen=True)
class KernelLaunchSpec:
    name: str
    operation: Literal["copy", "gather", "scatter", "pointwise"]
    block: int
    num_warps: int
    input_dtypes: tuple[str, ...]
    output_dtype: str
    layouts: tuple[str, ...]
    autotuning: Literal[False] = False

    def __post_init__(self) -> None:
        if not self.name or self.block <= 0 or self.num_warps <= 0:
            raise ValueError("kernel launch manifest entries require a name and positive launch dimensions")
        if not self.input_dtypes or not self.layouts:
            raise ValueError("kernel launch manifest entries require closed dtypes and layouts")
        if self.autotuning is not False:
            raise ValueError("production kernel manifest forbids runtime autotuning")


KERNEL_LAUNCHES = (
    KernelLaunchSpec(
        "copy_kv_bits",
        "copy",
        CACHE_COPY_BLOCK,
        CACHE_NUM_WARPS,
        ("bf16_bits",),
        "bf16_bits",
        ("layer_major",),
    ),
    KernelLaunchSpec(
        "gather_per_cache_head_bits",
        "gather",
        CACHE_FEATURE_BLOCK,
        CACHE_NUM_WARPS,
        ("bf16_bits", "int64_indices"),
        "bf16_bits",
        ("per_cache_head",),
    ),
    KernelLaunchSpec(
        "gather_shared_kv_bits",
        "gather",
        CACHE_FEATURE_BLOCK,
        CACHE_NUM_WARPS,
        ("bf16_bits", "int64_indices"),
        "bf16_bits",
        ("layer", "global"),
    ),
    KernelLaunchSpec(
        "scatter_shared_update_bits",
        "scatter",
        CACHE_FEATURE_BLOCK,
        CACHE_NUM_WARPS,
        ("bf16_bits", "int64_indices"),
        "bf16_bits",
        ("layer", "global"),
    ),
    KernelLaunchSpec(
        "scatter_shared_update_fp32_to_bf16",
        "scatter",
        CACHE_FEATURE_BLOCK,
        CACHE_NUM_WARPS,
        ("float32", "int64_indices"),
        "bf16",
        ("layer", "global"),
    ),
    KernelLaunchSpec(
        "scatter_shared_kv_grad_bits",
        "scatter",
        CACHE_FEATURE_BLOCK,
        CACHE_NUM_WARPS,
        ("bf16_bits", "int64_indices"),
        "bf16_bits",
        ("layer", "global"),
    ),
    KernelLaunchSpec(
        "scatter_per_cache_head_grad_bits",
        "scatter",
        CACHE_FEATURE_BLOCK,
        CACHE_NUM_WARPS,
        ("bf16_bits", "int64_indices"),
        "bf16_bits",
        ("per_cache_head",),
    ),
)


def kernel_manifest_payload() -> dict[str, object]:
    """Return the canonical launch/runtime domain bound by qualification."""
    return {
        "schema": KERNEL_MANIFEST_SCHEMA,
        "runtime": {
            "compute_capability": [9, 0],
            "triton_version": "3.6.0",
            "physical_batch_rows": 2,
            "runtime_autotuning": False,
        },
        "launches": [asdict(spec) for spec in KERNEL_LAUNCHES],
    }


def kernel_manifest_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(kernel_manifest_payload())).hexdigest()
