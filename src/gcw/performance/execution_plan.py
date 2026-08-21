"""Closed receipt-bound plan for the composed optimized-step candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Literal

from gcw.performance.config import OptimizedStepConfig
from gcw.performance.kernel_manifest import kernel_manifest_sha256
from gcw.provenance import canonical_json_bytes
from gcw.qualification.receipts import capture_hmac_key, load_and_verify_receipt

OPTIMIZED_STEP_RECEIPT_SCHEMA = "gcw.optimized_step_receipt.v1"
COMBINED_V1_CAMPAIGN_ADMITTED = False


@dataclass(frozen=True)
class OptimizedComponent:
    name: str
    version: str
    status: Literal["qualified", "candidate", "infrastructure"]


COMBINED_V1_COMPONENTS = (
    OptimizedComponent("static_workspaces", "v1", "infrastructure"),
    OptimizedComponent("ragged_pair_plans", "v1", "infrastructure"),
    OptimizedComponent("activation_lifetime", "v1", "infrastructure"),
)


@dataclass(frozen=True)
class OptimizedStepAuthority:
    """Externally observed source/config/runtime identity for receipt verification."""

    source_tree_sha256: str
    config_sha256: str
    accelerator: Literal["h100", "h200"]
    gpu_uuid: str
    torch_version: str
    triton_version: str

    def __post_init__(self) -> None:
        for name in ("source_tree_sha256", "config_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    f"optimized-step authority {name} must be lowercase SHA-256"
                )
        if (
            not self.gpu_uuid
            or not self.torch_version
            or not self.triton_version
        ):
            raise ValueError(
                "optimized-step authority runtime identity must be complete"
            )


@dataclass(frozen=True)
class VerifiedOptimizedStepPlan:
    profile: Literal["combined_v1"]
    plan_sha256: str
    evidence_sha256: tuple[str, ...]
    authority: OptimizedStepAuthority


def combined_v1_plan_payload() -> dict[str, object]:
    """Return the deterministic physical-execution composition under test."""
    return {
        "schema": "gcw.optimized_step_plan.v1",
        "profile": "combined_v1",
        "single_device_only": True,
        "physical_batch_rows": 2,
        "pairs_per_step": 32,
        "pair_weight": "1/32",
        "eager_fallback": False,
        "campaign_admitted": COMBINED_V1_CAMPAIGN_ADMITTED,
        "kernel_manifest_sha256": kernel_manifest_sha256(),
        "components": [
            asdict(component) for component in COMBINED_V1_COMPONENTS
        ],
        "excluded": [
            "adaptive_pair_batching",
            "regional_inductor",
            "triton_cache_training",
            "regional_cuda_graphs",
            "pair_waves",
            "triton_processor_pointwise",
            "fused_liger_row_sum",
            "evaluation_or_generation_dispatch",
        ],
    }


def combined_v1_plan_sha256() -> str:
    return hashlib.sha256(
        canonical_json_bytes(combined_v1_plan_payload())
    ).hexdigest()


def optimized_step_receipt_context(
    config: OptimizedStepConfig,
    authority: OptimizedStepAuthority,
) -> dict[str, object]:
    if not config.enabled or config.qualification_id is None:
        raise ValueError(
            "optimized-step receipt context requires an enabled bound configuration"
        )
    return {
        "audience": "gcw.training.optimized_step",
        "qualification_id": config.qualification_id,
        "profile": config.profile,
        "plan_sha256": combined_v1_plan_sha256(),
        "kernel_manifest_sha256": kernel_manifest_sha256(),
        **asdict(authority),
    }


def require_optimized_step_admission(
    config: OptimizedStepConfig,
    *,
    authority: OptimizedStepAuthority | None = None,
) -> VerifiedOptimizedStepPlan | None:
    """Fail closed unless this build and one keyed receipt admit the exact plan."""
    if not config.enabled:
        return None
    if not COMBINED_V1_CAMPAIGN_ADMITTED:
        raise RuntimeError(
            "optimized-step combined_v1 is not campaign-admitted by this source build"
        )
    if authority is None:
        raise RuntimeError(
            "enabled optimized-step execution requires complete runtime authority"
        )
    assert config.receipt_path is not None
    assert config.key_path is not None
    assert config.key_id is not None
    key = capture_hmac_key(Path(config.key_path), expected_key_id=config.key_id)
    payload = load_and_verify_receipt(
        Path(config.receipt_path),
        expected_schema=OPTIMIZED_STEP_RECEIPT_SCHEMA,
        expected_kind=config.profile,
        expected_context=optimized_step_receipt_context(config, authority),
        key=key,
    )
    expected_keys = {
        "admitted",
        "profile",
        "plan_sha256",
        "evidence_sha256",
        "qualified_accelerators",
    }
    if set(payload) != expected_keys:
        raise RuntimeError("optimized-step receipt payload schema differs")
    evidence = payload["evidence_sha256"]
    if (
        payload["admitted"] is not True
        or payload["profile"] != config.profile
        or payload["plan_sha256"] != combined_v1_plan_sha256()
        or payload["qualified_accelerators"] != ["h100", "h200"]
        or type(evidence) is not list
        or not evidence
        or any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in evidence
        )
    ):
        raise RuntimeError(
            "optimized-step receipt does not admit the closed H100/H200 plan"
        )
    return VerifiedOptimizedStepPlan(
        profile="combined_v1",
        plan_sha256=combined_v1_plan_sha256(),
        evidence_sha256=tuple(evidence),
        authority=authority,
    )
