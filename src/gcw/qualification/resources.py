"""Semantic and authority-local CUDA resource identity."""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import subprocess
import time
from typing import Literal

import torch

AcceleratorClass = Literal["cpu", "h100", "h200"]


class QualificationResourceError(RuntimeError):
    """Raised when hardware authority cannot be established."""


@dataclass(frozen=True)
class SemanticDeviceIdentity:
    accelerator: AcceleratorClass
    device_name: str
    compute_capability: tuple[int, int] | None
    total_memory_bytes: int
    torch_version: str
    cuda_runtime: str | None
    cudnn_version: int | None


@dataclass(frozen=True)
class AuthorityLocalIdentity:
    hostname: str
    pid: int
    parent_pid: int
    cuda_visible_devices: str | None
    visible_cuda_devices: int
    cuda_ordinal: int | None
    gpu_uuid: str | None
    pci_bus_id: str | None
    driver_version: str | None
    slurm_job_id: str | None
    slurm_step_id: str | None
    slurm_node_id: str | None


@dataclass(frozen=True)
class QualificationResourceIdentity:
    semantic: SemanticDeviceIdentity
    authority_local: AuthorityLocalIdentity


@dataclass(frozen=True)
class CUDAReleaseObservation:
    """Parent-observed process rows ending in an empty authority-local GPU."""

    gpu_uuid: str
    started_monotonic_ns: int
    cleared_monotonic_ns: int
    polls: tuple[tuple[tuple[int, int], ...], ...]

    def __post_init__(self) -> None:
        if not self.gpu_uuid or not self.polls or self.polls[-1] != ():
            raise QualificationResourceError(
                "CUDA release observation must end with an empty GPU process set"
            )
        if self.cleared_monotonic_ns < self.started_monotonic_ns:
            raise QualificationResourceError(
                "CUDA release observation timestamps are reversed"
            )


def observe_cuda_release(
    gpu_uuid: str,
    *,
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.5,
) -> CUDAReleaseObservation:
    """Poll NVIDIA process authority until this GPU is observably unused."""
    if not gpu_uuid or timeout_seconds <= 0 or poll_interval_seconds <= 0:
        raise ValueError(
            "CUDA release observation requires a UUID and positive timing bounds"
        )
    started = time.monotonic_ns()
    deadline = time.monotonic() + timeout_seconds
    polls: list[tuple[tuple[int, int], ...]] = []
    while True:
        rows = _query_compute_processes(gpu_uuid)
        polls.append(rows)
        if not rows:
            return CUDAReleaseObservation(
                gpu_uuid, started, time.monotonic_ns(), tuple(polls)
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise QualificationResourceError(
                f"CUDA processes remained on {gpu_uuid} after worker exit: {rows}"
            )
        time.sleep(min(poll_interval_seconds, remaining))


def capture_resource_identity(
    *,
    expected_accelerator: AcceleratorClass | None = None,
    require_single_cuda: bool = False,
    require_gpu_authority: bool = False,
) -> QualificationResourceIdentity:
    """Capture device semantics separately from process/allocation authority."""
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if require_single_cuda and visible != 1:
        raise QualificationResourceError(
            f"qualification requires exactly one visible CUDA GPU; found {visible}"
        )
    if visible == 0:
        if expected_accelerator not in {None, "cpu"}:
            assert expected_accelerator is not None
            raise QualificationResourceError(
                f"expected {expected_accelerator.upper()} but CUDA is unavailable"
            )
        semantic = SemanticDeviceIdentity(
            accelerator="cpu",
            device_name=platform.processor() or "cpu",
            compute_capability=None,
            total_memory_bytes=0,
            torch_version=str(torch.__version__),
            cuda_runtime=None,
            cudnn_version=None,
        )
        authority = _authority_local(visible=0, ordinal=None, gpu=None)
        return QualificationResourceIdentity(semantic, authority)

    ordinal = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(ordinal)
    accelerator = _classify_accelerator(properties.name)
    if expected_accelerator is not None and accelerator != expected_accelerator:
        raise QualificationResourceError(
            f"expected {expected_accelerator.upper()} but found {properties.name}"
        )
    gpu_rows = _query_nvidia_smi()
    property_uuid = _normalized_optional(getattr(properties, "uuid", None))
    if property_uuid is not None and not property_uuid.startswith("GPU-"):
        property_uuid = f"GPU-{property_uuid}"
    gpu = _match_gpu_row(gpu_rows, property_uuid, ordinal, visible)
    if require_gpu_authority and (
        gpu is None
        or gpu.uuid is None
        or gpu.pci_bus_id is None
        or gpu.driver_version is None
    ):
        raise QualificationResourceError(
            "GPU UUID, PCI bus ID, and driver authority are required"
        )
    semantic = SemanticDeviceIdentity(
        accelerator=accelerator,
        device_name=properties.name,
        compute_capability=torch.cuda.get_device_capability(ordinal),
        total_memory_bytes=int(properties.total_memory),
        torch_version=str(torch.__version__),
        cuda_runtime=str(torch.version.cuda),
        cudnn_version=torch.backends.cudnn.version(),
    )
    authority = _authority_local(visible=visible, ordinal=ordinal, gpu=gpu)
    return QualificationResourceIdentity(semantic, authority)


@dataclass(frozen=True)
class _NvidiaSMIRow:
    uuid: str | None
    pci_bus_id: str | None
    driver_version: str | None


def _classify_accelerator(name: str) -> Literal["h100", "h200"]:
    normalized = name.upper()
    if "H100" in normalized:
        return "h100"
    if "H200" in normalized:
        return "h200"
    raise QualificationResourceError(f"unsupported CUDA accelerator: {name}")


def _query_nvidia_smi() -> tuple[_NvidiaSMIRow, ...]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,pci.bus_id,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if result.returncode != 0:
        return ()
    rows: list[_NvidiaSMIRow] = []
    for raw in result.stdout.splitlines():
        fields = tuple(field.strip() for field in raw.split(","))
        if len(fields) != 3:
            return ()
        rows.append(
            _NvidiaSMIRow(*(_normalized_optional(field) for field in fields))
        )
    return tuple(rows)


def _query_compute_processes(gpu_uuid: str) -> tuple[tuple[int, int], ...]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QualificationResourceError(
            "cannot observe CUDA process release"
        ) from exc
    if result.returncode != 0:
        raise QualificationResourceError(
            "nvidia-smi rejected CUDA process release query"
        )
    rows: list[tuple[int, int]] = []
    for raw in result.stdout.splitlines():
        fields = tuple(field.strip() for field in raw.split(","))
        if len(fields) != 3:
            raise QualificationResourceError(
                "CUDA process release query returned malformed rows"
            )
        row_uuid, pid, memory = fields
        if row_uuid != gpu_uuid:
            continue
        try:
            parsed = (int(pid), int(memory))
        except ValueError as exc:
            raise QualificationResourceError(
                "CUDA process release row is not numeric"
            ) from exc
        if parsed[0] <= 0 or parsed[1] < 0:
            raise QualificationResourceError(
                "CUDA process release row has invalid values"
            )
        rows.append(parsed)
    if len({pid for pid, _memory in rows}) != len(rows):
        raise QualificationResourceError(
            "CUDA process release query returned duplicate PIDs"
        )
    return tuple(sorted(rows))


def _match_gpu_row(
    rows: tuple[_NvidiaSMIRow, ...],
    property_uuid: str | None,
    ordinal: int,
    visible: int,
) -> _NvidiaSMIRow | None:
    if property_uuid is not None:
        matches = tuple(row for row in rows if row.uuid == property_uuid)
        if len(matches) == 1:
            return matches[0]
        return None
    if len(rows) == visible == 1 and ordinal == 0:
        return rows[0]
    return None


def _normalized_optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "[NOT SUPPORTED]"}:
        return None
    return text


def _authority_local(
    *,
    visible: int,
    ordinal: int | None,
    gpu: _NvidiaSMIRow | None,
) -> AuthorityLocalIdentity:
    return AuthorityLocalIdentity(
        hostname=platform.node(),
        pid=os.getpid(),
        parent_pid=os.getppid(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        visible_cuda_devices=visible,
        cuda_ordinal=ordinal,
        gpu_uuid=None if gpu is None else gpu.uuid,
        pci_bus_id=None if gpu is None else gpu.pci_bus_id,
        driver_version=None if gpu is None else gpu.driver_version,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_step_id=os.environ.get("SLURM_STEP_ID"),
        slurm_node_id=os.environ.get("SLURMD_NODENAME"),
    )
