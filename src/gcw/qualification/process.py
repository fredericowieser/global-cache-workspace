"""Fresh-process qualification orchestration without shell execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time
from typing import Mapping, Sequence

from gcw.qualification.inputs import BlobIdentity, capture_file


class QualificationProcessError(RuntimeError):
    """Raised when a qualification worker violates its declared protocol."""


@dataclass(frozen=True)
class WorkerSpec:
    """Closed invocation and exclusive output expected from one worker."""

    role: str
    argv: tuple[str, ...]
    cwd: Path
    environment: tuple[tuple[str, str], ...]
    output: Path

    @classmethod
    def create(
        cls,
        *,
        role: str,
        argv: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
        output: Path,
    ) -> WorkerSpec:
        return cls(role, tuple(argv), cwd, tuple(sorted(environment.items())), output)

    def __post_init__(self) -> None:
        if not self.role or not self.argv or any(not value for value in self.argv):
            raise ValueError("worker role and argv must be nonempty")
        if not self.cwd.is_absolute() or not self.output.is_absolute():
            raise ValueError("worker cwd and output must be absolute")
        names = tuple(name for name, _value in self.environment)
        if names != tuple(sorted(set(names))) or any(not name for name in names):
            raise ValueError("worker environment names must be unique, sorted, and nonempty")


@dataclass(frozen=True)
class WorkerResult:
    role: str
    pid: int
    parent_pid: int
    started_monotonic_ns: int
    finished_monotonic_ns: int
    returncode: int
    stdout: str
    stderr: str
    output: BlobIdentity


@dataclass(frozen=True)
class ProducerResumerResult:
    producer: WorkerResult
    resumer: WorkerResult


def run_worker(spec: WorkerSpec, *, timeout_seconds: float) -> WorkerResult:
    """Run one worker to completion and capture its exclusive output once."""
    if timeout_seconds <= 0:
        raise ValueError("worker timeout must be positive")
    if not spec.cwd.is_dir() or spec.cwd.is_symlink():
        raise QualificationProcessError(f"worker cwd is unavailable: {spec.cwd}")
    if spec.output.exists() or spec.output.is_symlink():
        raise QualificationProcessError(f"worker output already exists: {spec.output}")
    environment = {name: value for name, value in spec.environment}
    started = time.monotonic_ns()
    try:
        process = subprocess.Popen(
            spec.argv,
            cwd=spec.cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise QualificationProcessError(f"failed to start {spec.role} worker") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        stdout, stderr = process.communicate()
        raise QualificationProcessError(
            f"{spec.role} worker timed out; stdout={stdout!r} stderr={stderr!r}"
        ) from exc
    finished = time.monotonic_ns()
    if process.returncode != 0:
        raise QualificationProcessError(
            f"{spec.role} worker exited {process.returncode}; stdout={stdout!r} stderr={stderr!r}"
        )
    try:
        captured = capture_file(spec.output, role=f"worker_output:{spec.role}")
    except ValueError as exc:
        raise QualificationProcessError(f"{spec.role} worker produced no valid output") from exc
    return WorkerResult(
        role=spec.role,
        pid=process.pid,
        parent_pid=os.getpid(),
        started_monotonic_ns=started,
        finished_monotonic_ns=finished,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        output=captured.identity,
    )


def run_producer_resumer(
    producer: WorkerSpec,
    resumer: WorkerSpec,
    *,
    timeout_seconds: float,
) -> ProducerResumerResult:
    """Require complete producer exit before starting a distinct resumer."""
    if producer.role != "producer" or resumer.role != "resumer":
        raise ValueError("fresh-process protocol requires producer and resumer roles")
    producer_result = run_worker(producer, timeout_seconds=timeout_seconds)
    resumer_result = run_worker(resumer, timeout_seconds=timeout_seconds)
    if producer_result.pid == resumer_result.pid:
        raise QualificationProcessError("producer and resumer unexpectedly reused one PID")
    if resumer_result.started_monotonic_ns < producer_result.finished_monotonic_ns:
        raise QualificationProcessError("resumer started before producer completion")
    return ProducerResumerResult(producer_result, resumer_result)
