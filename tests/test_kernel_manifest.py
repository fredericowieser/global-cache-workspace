from __future__ import annotations

from gcw.performance.kernel_manifest import (
    CACHE_COPY_BLOCK,
    CACHE_FEATURE_BLOCK,
    KERNEL_LAUNCHES,
    CACHE_NUM_WARPS,
    kernel_manifest_payload,
    kernel_manifest_sha256,
)


def test_kernel_manifest_is_closed_deterministic_and_autotune_free() -> None:
    payload = kernel_manifest_payload()
    assert payload["runtime"] == {
        "compute_capability": [9, 0],
        "triton_version": "3.6.0",
        "physical_batch_rows": 2,
        "runtime_autotuning": False,
    }
    names = tuple(spec.name for spec in KERNEL_LAUNCHES)
    assert names == (
        "copy_kv_bits",
        "gather_per_cache_head_bits",
        "gather_shared_kv_bits",
        "scatter_shared_update_bits",
        "scatter_shared_update_fp32_to_bf16",
        "scatter_shared_kv_grad_bits",
        "scatter_per_cache_head_grad_bits",
        "processor_silu_mul_forward",
        "processor_silu_mul_backward",
    )
    assert all(spec.autotuning is False for spec in KERNEL_LAUNCHES)
    assert {spec.block for spec in KERNEL_LAUNCHES} == {
        CACHE_FEATURE_BLOCK,
        CACHE_COPY_BLOCK,
    }
    assert {spec.num_warps for spec in KERNEL_LAUNCHES} == {CACHE_NUM_WARPS}
    assert kernel_manifest_sha256() == "528c52aac8d5b2ee057bf56ecbc2adb545c5f16e1bba5ea1a833e59d64ec2466"
