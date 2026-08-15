from __future__ import annotations

import pytest
import torch

from gcw.qualification import resources as resources_module
from gcw.qualification.resources import (
    QualificationResourceError,
    _NvidiaSMIRow,
    _classify_accelerator,
    _match_gpu_row,
    capture_resource_identity,
    observe_cuda_release,
)


def test_cpu_resource_identity_separates_semantic_and_authority_local() -> None:
    if torch.cuda.is_available():
        pytest.skip("CPU resource identity test requires a CPU-only process")
    identity = capture_resource_identity(expected_accelerator="cpu")
    assert identity.semantic.accelerator == "cpu"
    assert identity.semantic.compute_capability is None
    assert identity.authority_local.pid > 0
    assert identity.authority_local.parent_pid > 0
    assert identity.authority_local.gpu_uuid is None


def test_accelerator_classification_is_closed_to_hopper_authority() -> None:
    assert _classify_accelerator("NVIDIA H100 80GB HBM3") == "h100"
    assert _classify_accelerator("NVIDIA H200") == "h200"
    with pytest.raises(QualificationResourceError, match="unsupported"):
        _classify_accelerator("NVIDIA A100-SXM4-80GB")


def test_gpu_row_matching_requires_unambiguous_uuid_or_single_visibility() -> None:
    h100 = _NvidiaSMIRow("GPU-one", "00000000:01:00.0", "570.1")
    h200 = _NvidiaSMIRow("GPU-two", "00000000:02:00.0", "570.1")
    assert _match_gpu_row((h100, h200), "GPU-two", 0, 1) == h200
    assert _match_gpu_row((h100,), None, 0, 1) == h100
    assert _match_gpu_row((h100, h200), None, 0, 1) is None
    assert _match_gpu_row((h100, h100), "GPU-one", 0, 1) is None


def test_parent_observes_cuda_process_rows_reach_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [((101, 4096),), ((101, 1024),), ()]
    sleeps: list[float] = []

    def query(gpu_uuid: str) -> tuple[tuple[int, int], ...]:
        assert gpu_uuid == "GPU-authority"
        return rows.pop(0)

    monkeypatch.setattr(resources_module, "_query_compute_processes", query)
    monkeypatch.setattr(resources_module.time, "sleep", sleeps.append)
    observation = observe_cuda_release(
        "GPU-authority",
        timeout_seconds=10,
        poll_interval_seconds=0.25,
    )
    assert observation.polls == (((101, 4096),), ((101, 1024),), ())
    assert sleeps == [0.25, 0.25]
    assert observation.cleared_monotonic_ns >= observation.started_monotonic_ns


def test_expected_gpu_fails_when_cuda_is_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA-unavailable gate requires a CPU-only process")
    with pytest.raises(QualificationResourceError, match="CUDA is unavailable"):
        capture_resource_identity(expected_accelerator="h200")
