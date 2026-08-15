"""Fresh-process qualification orchestration without shell execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import time
from typing import Mapping, Sequence

from gcw.qualification.inputs import BlobIdentity, CapturedBlob, capture_file


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
    input: BlobIdentity | None


@dataclass(frozen=True)
class ProducerResumerResult:
    producer: WorkerResult
    resumer: WorkerResult


def run_worker(spec: WorkerSpec, *, timeout_seconds: float) -> WorkerResult:
    """Run one worker to completion and capture its exclusive output once."""
    result, _captured = _run_worker_capture(spec, timeout_seconds=timeout_seconds, stdin_payload=None)
    return result


def _run_worker_capture(
    spec: WorkerSpec,
    *,
    timeout_seconds: float,
    stdin_payload: bytes | None,
) -> tuple[WorkerResult, CapturedBlob]:
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
            stdin=subprocess.DEVNULL if stdin_payload is None else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise QualificationProcessError(f"failed to start {spec.role} worker") from exc
    try:
        stdout_bytes, stderr_bytes = process.communicate(input=stdin_payload, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        stdout_bytes, stderr_bytes = process.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        raise QualificationProcessError(f"{spec.role} worker timed out; stdout={stdout!r} stderr={stderr!r}") from exc
    finished = time.monotonic_ns()
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    if process.returncode != 0:
        raise QualificationProcessError(
            f"{spec.role} worker exited {process.returncode}; stdout={stdout!r} stderr={stderr!r}"
        )
    try:
        captured = capture_file(spec.output, role=f"worker_output:{spec.role}")
    except ValueError as exc:
        raise QualificationProcessError(f"{spec.role} worker produced no valid output") from exc
    input_identity = (
        None
        if stdin_payload is None
        else BlobIdentity(
            role=f"worker_input:{spec.role}",
            size_bytes=len(stdin_payload),
            sha256=hashlib.sha256(stdin_payload).hexdigest(),
        )
    )
    return (
        WorkerResult(
            role=spec.role,
            pid=process.pid,
            parent_pid=os.getpid(),
            started_monotonic_ns=started,
            finished_monotonic_ns=finished,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            output=captured.identity,
            input=input_identity,
        ),
        captured,
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
    producer_result, producer_output = _run_worker_capture(
        producer,
        timeout_seconds=timeout_seconds,
        stdin_payload=None,
    )
    resumer_result, _resumer_output = _run_worker_capture(
        resumer,
        timeout_seconds=timeout_seconds,
        stdin_payload=producer_output.content,
    )
    if producer_result.pid == resumer_result.pid:
        raise QualificationProcessError("producer and resumer unexpectedly reused one PID")
    if resumer_result.started_monotonic_ns < producer_result.finished_monotonic_ns:
        raise QualificationProcessError("resumer started before producer completion")
    if resumer_result.input is None or resumer_result.input.sha256 != producer_result.output.sha256:
        raise QualificationProcessError("resumer input differs from parent-captured producer output")
    return ProducerResumerResult(producer_result, resumer_result)
