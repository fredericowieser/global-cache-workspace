from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from gcw.qualification import process as process_module
from gcw.qualification.process import (
    QualificationProcessError,
    WorkerSpec,
    run_producer_resumer,
    run_worker,
)
from gcw.qualification.resources import CUDAReleaseObservation


def _environment(root: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
    }


def test_producer_and_resumer_are_distinct_sequential_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    worker = root / "tests/qualification_process_worker.py"
    producer_output = tmp_path / "producer.json"
    resumer_output = tmp_path / "resumer.json"
    producer = WorkerSpec.create(
        role="producer",
        argv=(sys.executable, str(worker), "--role", "producer", "--output", str(producer_output)),
        cwd=tmp_path,
        environment=_environment(root),
        output=producer_output,
    )
    resumer = WorkerSpec.create(
        role="resumer",
        argv=(
            sys.executable,
            str(worker),
            "--role",
            "resumer",
            "--output",
            str(resumer_output),
        ),
        cwd=tmp_path,
        environment=_environment(root),
        output=resumer_output,
    )
    release_calls: list[str] = []

    def observe(gpu_uuid: str, *, timeout_seconds: float) -> CUDAReleaseObservation:
        assert timeout_seconds == 7
        release_calls.append(gpu_uuid)
        return CUDAReleaseObservation(gpu_uuid, len(release_calls), len(release_calls), ((),))

    monkeypatch.setattr(process_module, "observe_cuda_release", observe)
    result = run_producer_resumer(
        producer,
        resumer,
        timeout_seconds=30,
        gpu_uuid="GPU-authority",
        release_timeout_seconds=7,
    )
    assert result.producer.pid != result.resumer.pid
    assert result.producer.finished_monotonic_ns <= result.resumer.started_monotonic_ns
    resumer_payload = json.loads(resumer_output.read_text())
    assert resumer_payload["producer_pid"] == result.producer.pid
    assert resumer_payload["pid"] == result.resumer.pid
    assert len(result.producer.output.sha256) == len(result.resumer.output.sha256) == 64
    assert result.producer.input is None
    assert result.resumer.input is not None
    assert result.resumer.input.sha256 == result.producer.output.sha256
    assert result.resumer.input.size_bytes == result.producer.output.size_bytes
    assert release_calls == ["GPU-authority", "GPU-authority"]
    assert result.producer_release is not None and result.producer_release.polls == ((),)
    assert result.resumer_release is not None and result.resumer_release.polls == ((),)


def test_worker_rejects_existing_output_before_launch(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "occupied.json"
    output.write_text("owned")
    spec = WorkerSpec.create(
        role="producer",
        argv=(sys.executable, "-c", "raise SystemExit(99)"),
        cwd=tmp_path,
        environment=_environment(root),
        output=output,
    )
    with pytest.raises(QualificationProcessError, match="already exists"):
        run_worker(spec, timeout_seconds=5)
