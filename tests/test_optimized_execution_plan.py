from __future__ import annotations

import os
from pathlib import Path

import pytest

from gcw.performance import OptimizedStepConfig
from gcw.performance import execution_plan as plan_module
from gcw.performance.execution_plan import (
    OPTIMIZED_STEP_RECEIPT_SCHEMA,
    OptimizedStepAuthority,
    combined_v1_plan_payload,
    combined_v1_plan_sha256,
    optimized_step_receipt_context,
    require_optimized_step_admission,
)
from gcw.qualification.receipts import QualificationHMACKey, issue_receipt, write_receipt


def _authority() -> OptimizedStepAuthority:
    return OptimizedStepAuthority(
        source_tree_sha256="a" * 64,
        config_sha256="b" * 64,
        accelerator="h200",
        gpu_uuid="GPU-authority",
        torch_version="2.10.0+cu128",
        triton_version="3.6.0",
    )


def test_combined_plan_is_closed_default_off_and_records_rejections() -> None:
    payload = combined_v1_plan_payload()
    assert payload["campaign_admitted"] is False
    assert payload["physical_batch_rows"] == 2
    assert payload["pairs_per_step"] == 32
    assert payload["pair_weight"] == "1/32"
    assert payload["eager_fallback"] is False
    assert "fused_liger_row_sum" in payload["excluded"]
    assert "regional_cuda_graphs" in payload["excluded"]
    assert "pair_waves" in payload["excluded"]
    assert "triton_processor_pointwise" in payload["excluded"]
    assert combined_v1_plan_sha256() == "19e9d6f7981e6236dc31a5f521a81658aa524286d77afc60cf31de7bda1093a0"
    assert require_optimized_step_admission(OptimizedStepConfig()) is None
    with pytest.raises(ValueError, match="requires receipt_path"):
        OptimizedStepConfig(enabled=True)


def test_unadmitted_combined_plan_fails_before_opening_receipt_paths() -> None:
    config = OptimizedStepConfig(
        enabled=True,
        receipt_path="/does/not/exist/receipt",
        key_path="/does/not/exist/key",
        key_id="key-v1",
        qualification_id="challenge-v1",
    )
    with pytest.raises(RuntimeError, match="not campaign-admitted"):
        require_optimized_step_admission(config, authority=_authority())


def test_admitted_build_still_requires_exact_keyed_h100_h200_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plan_module, "COMBINED_V1_CAMPAIGN_ADMITTED", True)
    secret = bytes(range(32))
    key_path = tmp_path / "key.bin"
    key_path.write_bytes(secret)
    key_path.chmod(0o600)
    receipt_path = tmp_path / "receipt.json"
    config = OptimizedStepConfig(
        enabled=True,
        receipt_path=str(receipt_path),
        key_path=str(key_path),
        key_id="key-v1",
        qualification_id="challenge-v1",
    )
    evidence = ["c" * 64, "d" * 64]
    receipt = issue_receipt(
        {
            "admitted": True,
            "profile": "combined_v1",
            "plan_sha256": combined_v1_plan_sha256(),
            "evidence_sha256": evidence,
            "qualified_accelerators": ["h100", "h200"],
        },
        schema=OPTIMIZED_STEP_RECEIPT_SCHEMA,
        kind="combined_v1",
        context=optimized_step_receipt_context(config, _authority()),
        key=QualificationHMACKey("key-v1", secret),
    )
    write_receipt(receipt_path, receipt)

    verified = require_optimized_step_admission(config, authority=_authority())
    assert verified is not None
    assert verified.plan_sha256 == combined_v1_plan_sha256()
    assert verified.evidence_sha256 == tuple(evidence)
    assert os.stat(key_path).st_mode & 0o077 == 0
