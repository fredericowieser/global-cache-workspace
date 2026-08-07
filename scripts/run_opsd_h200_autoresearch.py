#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable

import yaml

from gcw.distill.data import iter_openmath_prompts
from gcw.autoresearch import (
    choose_capacity_cells,
    decision_dose_steps,
    feasible_dose_steps,
    materialize_train_config,
    matrix_generation_count,
    merge_opsd_task_shards,
    project_end_to_end_training,
    require_complete_capacity_promotion,
    require_heldout_improvement,
    require_realized_teacher_batch,
    select_disjoint_heldout_indices,
    select_evaluation_limit,
    validate_autoresearch_protocol,
    validate_opsd_task_shard,
)
from gcw.opsd_eval import aggregate_opsd_screen


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _runtime_matrix_reserve(
    *,
    deadline_seconds: float,
    configured_reserve_seconds: float,
    closure_reserve_seconds: float,
) -> float:
    values = (
        deadline_seconds,
        configured_reserve_seconds,
        closure_reserve_seconds,
    )
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("campaign deadline and reserves must be finite and positive")
    if deadline_seconds <= 2.0 * closure_reserve_seconds:
        raise ValueError("campaign deadline must leave work time before closure reserve")
    return min(configured_reserve_seconds, deadline_seconds / 2.0)


def _validate_spend_approval(
    path: Path,
    *,
    source_revision: str,
    protocol_sha256: str,
    requested_hours: float,
    provider_project_id: str = "project-e01d0ymrpr004k0s993tba",
    provider_region: str = "eu-west1",
) -> dict[str, Any]:
    approval = json.loads(path.resolve(strict=True).read_text())
    expected = {
        "kind": "gcw_h200_spend_approval_v1",
        "source_revision": source_revision,
        "protocol_sha256": protocol_sha256,
        "provider": "nebius",
        "project_id": provider_project_id,
        "region": provider_region,
        "gpu_model": "H200",
        "gpu_count": 1,
        "regular_capacity": True,
    }
    for key, value in expected.items():
        if approval.get(key) != value:
            raise RuntimeError(f"spend approval mismatch for {key}")
    approved_hours = float(approval.get("approved_hours", 0.0))
    hourly_quote = float(approval.get("quoted_usd_per_gpu_hour", 0.0))
    disk_allowance = float(approval.get("maximum_disk_spend_usd", 0.0))
    maximum_spend = float(approval.get("maximum_spend_usd", 0.0))
    approval_phrase = approval.get("approval_phrase")
    envelopes = {
        "APPROVE 24H $110": (24.0, 110.0),
        "APPROVE 12H $55": (12.0, 55.0),
    }
    if approval_phrase not in envelopes:
        raise RuntimeError("spend approval lacks the canonical user approval phrase")
    phrase_hours, phrase_maximum = envelopes[approval_phrase]
    monetary_values = (hourly_quote, disk_allowance, maximum_spend)
    if not all(math.isfinite(value) for value in monetary_values):
        raise RuntimeError("spend approval monetary fields must be finite")
    if not math.isfinite(approved_hours) or not math.isfinite(requested_hours):
        raise RuntimeError("spend approval runtime fields must be finite")
    if approved_hours != phrase_hours or maximum_spend != phrase_maximum:
        raise RuntimeError("approval phrase differs from exact approved envelope")
    if requested_hours <= 0 or requested_hours > approved_hours:
        raise RuntimeError("requested runtime exceeds exact approved hours")
    if hourly_quote != 4.5 or disk_allowance < 0 or maximum_spend <= 0:
        raise RuntimeError("spend approval has invalid quoted price fields")
    if hourly_quote * approved_hours + disk_allowance > maximum_spend:
        raise RuntimeError("approved maximum spend does not cover the requested envelope")
    return approval


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int
    error_kind: str | None = None
    error: str | None = None


class CampaignCommandTimeout(TimeoutError):
    """A bounded campaign phase reached its reviewed time allocation."""


_ACTIVE_PROCESSES: dict[int, subprocess.Popen[str]] = {}
_PROCESS_CONDITION = threading.Condition()
_SPAWNING_PROCESSES = 0
_SHUTDOWN_REQUESTED = False
_SHUTDOWN_PIPE_WRITE: int | None = None


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_registered_process_groups(
    processes: tuple[subprocess.Popen[str], ...], *, grace_seconds: float = 10.0
) -> None:
    process_groups = {process.pid for process in processes}
    for process_group in process_groups:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        for process in processes:
            process.poll()
        survivors = {
            process_group
            for process_group in process_groups
            if _process_group_exists(process_group)
        }
        if not survivors:
            break
        time.sleep(0.05)
    else:
        survivors = {
            process_group
            for process_group in process_groups
            if _process_group_exists(process_group)
        }
    for process_group in survivors:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    _terminate_registered_process_groups((process,))


def _shutdown_manager(read_fd: int) -> None:
    while os.read(read_fd, 1):
        with _PROCESS_CONDITION:
            while _SPAWNING_PROCESSES:
                _PROCESS_CONDITION.wait(timeout=0.05)
            processes = tuple(_ACTIVE_PROCESSES.values())
        _terminate_registered_process_groups(processes)


def _request_campaign_shutdown(_signum: int, _frame: Any) -> None:
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    if _SHUTDOWN_PIPE_WRITE is not None:
        try:
            os.write(_SHUTDOWN_PIPE_WRITE, b"!")
        except BlockingIOError:
            pass


def _install_shutdown_manager() -> None:
    global _SHUTDOWN_PIPE_WRITE
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    _SHUTDOWN_PIPE_WRITE = write_fd
    threading.Thread(
        target=_shutdown_manager,
        args=(read_fd,),
        name="campaign-shutdown-manager",
        daemon=True,
    ).start()
    signal.signal(signal.SIGTERM, _request_campaign_shutdown)
    signal.signal(signal.SIGINT, _request_campaign_shutdown)


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    log: Any,
    timeout: float,
) -> ProcessOutcome:
    global _SPAWNING_PROCESSES
    with _PROCESS_CONDITION:
        _SPAWNING_PROCESSES += 1
        if _SHUTDOWN_REQUESTED:
            _SPAWNING_PROCESSES -= 1
            _PROCESS_CONDITION.notify_all()
            return ProcessOutcome(130, "interrupted", "campaign shutdown was requested")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        with _PROCESS_CONDITION:
            _SPAWNING_PROCESSES -= 1
            _PROCESS_CONDITION.notify_all()
        return ProcessOutcome(127, "launch_error", f"{type(exc).__name__}: {exc}")
    except BaseException:
        with _PROCESS_CONDITION:
            _SPAWNING_PROCESSES -= 1
            _PROCESS_CONDITION.notify_all()
        raise
    with _PROCESS_CONDITION:
        _ACTIVE_PROCESSES[process.pid] = process
        _SPAWNING_PROCESSES -= 1
        _PROCESS_CONDITION.notify_all()
    try:
        try:
            return ProcessOutcome(process.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            return ProcessOutcome(
                124,
                "timeout",
                f"process group exceeded bounded timeout of {timeout:.3f} seconds",
            )
        except BaseException:
            _terminate_process_group(process)
            raise
    finally:
        with _PROCESS_CONDITION:
            _ACTIVE_PROCESSES.pop(process.pid, None)
            _PROCESS_CONDITION.notify_all()


class Campaign:
    def __init__(
        self,
        *,
        repo: Path,
        protocol_path: Path,
        source_revision: str,
        deadline_seconds: int,
        verify_source: bool = True,
    ) -> None:
        self.repo = repo.resolve(strict=True)
        self.protocol_path = protocol_path.resolve(strict=True)
        self.protocol = json.loads(self.protocol_path.read_text())
        self.source_revision = source_revision
        self.verify_source = verify_source
        self.root = Path(self.protocol["output_root"])
        if self.root.exists():
            raise FileExistsError(f"refusing to overwrite campaign root: {self.root}")
        self.root.mkdir(parents=True)
        self.logs = self.root / "logs"
        self.configs = self.root / "configs"
        self.logs.mkdir()
        self.configs.mkdir()
        self.started = time.monotonic()
        self.deadline_seconds = deadline_seconds
        self.state_path = self.root / "campaign-state.json"
        self.state: dict[str, Any] = {
            "schema_version": 1,
            "kind": "opsd_h200_autoresearch_state",
            "status": "running",
            "source_revision": source_revision,
            "protocol_path": str(self.protocol_path),
            "protocol_sha256": _sha256(self.protocol_path),
            "campaign_root": str(self.root),
            "deadline_seconds": deadline_seconds,
            "phases": [],
        }
        self._publish_state()

    def assert_source_identity(self) -> None:
        if not self.verify_source:
            return
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=self.repo, text=True
        )
        symbolic = subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"],
            cwd=self.repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if actual != self.source_revision or dirty or symbolic.returncode == 0:
            raise RuntimeError(
                "campaign source drift: expected clean detached revision "
                f"{self.source_revision}, got revision={actual}, "
                f"dirty={bool(dirty)}, detached={symbolic.returncode != 0}"
            )

    def remaining(self) -> float:
        return self.deadline_seconds - (time.monotonic() - self.started)

    def _publish_state(self) -> None:
        self.state["elapsed_seconds"] = time.monotonic() - self.started
        self.state["remaining_seconds"] = max(0.0, self.remaining())
        _atomic_json(self.state_path, self.state)

    def phase(self, name: str, status: str, **details: Any) -> None:
        self.state["phases"].append(
            {
                "name": name,
                "status": status,
                "elapsed_seconds": time.monotonic() - self.started,
                **details,
            }
        )
        self._publish_state()

    def run(self, name: str, command: list[str], *, timeout: float | None = None) -> float:
        allowed = max(1.0, self.remaining() - 300.0)
        if timeout is not None:
            allowed = min(allowed, timeout)
        if allowed <= 1.0:
            raise TimeoutError("campaign deadline leaves no command budget")
        self.assert_source_identity()
        log_path = self.logs / f"{name}.log"
        started = time.monotonic()
        with log_path.open("x", encoding="utf-8") as log:
            outcome = _run_process(
                command,
                cwd=self.repo,
                log=log,
                timeout=allowed,
            )
        elapsed = time.monotonic() - started
        self.assert_source_identity()
        status = "completed"
        if outcome.error_kind == "timeout":
            status = "timed_out"
        elif outcome.returncode != 0:
            status = "failed"
        self.phase(
            name,
            status,
            returncode=outcome.returncode,
            error_kind=outcome.error_kind,
            error=outcome.error,
            command=command,
            log=str(log_path),
            command_elapsed_seconds=elapsed,
        )
        if outcome.error_kind == "timeout":
            raise CampaignCommandTimeout(
                f"campaign command reached its allocation: {name}; see {log_path}"
            )
        if outcome.returncode != 0:
            raise RuntimeError(f"campaign command failed: {name}; see {log_path}")
        return elapsed

    def run_parallel(
        self,
        commands: dict[str, list[str]],
        *,
        max_workers: int,
        max_attempts: int = 3,
        reserve_seconds: float = 300.0,
        maximum_seconds: float | None = None,
        success_validators: dict[str, Callable[[], bool]] | None = None,
    ) -> float:
        if (
            max_workers <= 0
            or max_attempts <= 0
            or reserve_seconds < 0
            or (maximum_seconds is not None and maximum_seconds <= 0)
        ):
            raise ValueError("parallel campaign workers, attempts, and budgets must be positive")
        if not commands:
            return 0.0
        parallel_started = time.monotonic()
        pending = dict(commands)
        workers = max_workers

        def one(
            item: tuple[str, list[str]], attempt: int, timeout: float
        ) -> tuple[str, ProcessOutcome, float, Path, bool]:
            name, command = item
            log_path = self.logs / f"{name}.attempt-{attempt}.log"
            started = time.monotonic()
            try:
                with log_path.open("x", encoding="utf-8") as log:
                    outcome = _run_process(
                        command,
                        cwd=self.repo,
                        log=log,
                        timeout=timeout,
                    )
            except Exception as exc:
                outcome = ProcessOutcome(
                    125,
                    "worker_error",
                    f"{type(exc).__name__}: {exc}",
                )
            oom = False
            if outcome.returncode != 0 and log_path.exists():
                text = log_path.read_text(errors="replace").lower()
                oom = "out of memory" in text or "cuda error: out of memory" in text
            return name, outcome, time.monotonic() - started, log_path, oom

        def validate_published_output(name: str) -> bool | None:
            validator = (
                success_validators.get(name)
                if success_validators is not None
                else None
            )
            if validator is None:
                return None
            try:
                return bool(validator())
            except Exception as exc:
                self.phase(
                    name,
                    "output_validation_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                return False

        for attempt in range(1, max_attempts + 1):
            if _SHUTDOWN_REQUESTED:
                raise RuntimeError("campaign shutdown requested during parallel execution")
            if success_validators:
                for name in tuple(pending):
                    if validate_published_output(name) is True:
                        pending.pop(name)
                        self.phase(
                            name,
                            "recovered_published_output",
                            attempt=attempt,
                        )
                if not pending:
                    return time.monotonic() - parallel_started
            failures: dict[str, list[str]] = {}
            saw_oom = False
            items = list(pending.items())
            for offset in range(0, len(items), workers):
                campaign_available = self.remaining() - reserve_seconds
                phase_available = (
                    math.inf
                    if maximum_seconds is None
                    else maximum_seconds - (time.monotonic() - parallel_started)
                )
                timeout = min(campaign_available, phase_available)
                if timeout <= 0:
                    unstarted = dict(items[offset:])
                    failures.update(unstarted)
                    self.phase(
                        "evaluation-deadline-reserve",
                        "stopped",
                        attempt=attempt,
                        unstarted_shards=sorted(unstarted),
                        campaign_available_seconds=campaign_available,
                        phase_available_seconds=phase_available,
                    )
                    raise RuntimeError(
                        "campaign deadline reserve reached before queued shards started"
                    )
                batch = items[offset : offset + workers]
                self.assert_source_identity()
                with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                    futures = {
                        pool.submit(one, item, attempt, timeout): item[0]
                        for item in batch
                    }
                    for future in as_completed(futures):
                        name, outcome, elapsed, log_path, oom = future.result()
                        validation = validate_published_output(name)
                        recovered = outcome.returncode != 0 and validation is True
                        invalid_success = outcome.returncode == 0 and validation is False
                        status = "completed"
                        if recovered:
                            status = "recovered_published_output"
                        elif invalid_success:
                            status = "invalid_output"
                        elif outcome.error_kind == "timeout":
                            status = "timed_out"
                        elif outcome.returncode != 0:
                            status = "failed"
                        self.phase(
                            name,
                            status,
                            attempt=attempt,
                            returncode=outcome.returncode,
                            error_kind=outcome.error_kind,
                            error=outcome.error,
                            log=str(log_path),
                            command_elapsed_seconds=elapsed,
                            oom=oom,
                        )
                        if recovered:
                            continue
                        if outcome.returncode != 0 or invalid_success:
                            failures[name] = pending[name]
                            saw_oom = saw_oom or oom
                self.assert_source_identity()
            if not failures:
                return time.monotonic() - parallel_started
            pending = failures
            if saw_oom and workers > 1:
                workers = max(1, workers // 2)
                self.phase(
                    "evaluation-concurrency-backoff",
                    "completed",
                    attempt=attempt,
                    next_max_workers=workers,
                    failed_shards=sorted(pending),
                )
            if self.remaining() <= reserve_seconds:
                break
        raise RuntimeError(
            "parallel campaign commands exhausted retries: "
            f"{sorted(pending)}"
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


def _write_config(campaign: Campaign, experiment_id: str, config: dict[str, Any]) -> Path:
    path = campaign.configs / f"{experiment_id}.yaml"
    _atomic_yaml(path, config)
    return path


def _copy_trace(trace: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(trace, output_dir / "teacher-traces.jsonl")


def _require_contiguous_prompt_prefix(source: Path, *, rows: int) -> None:
    if rows <= 0:
        raise ValueError("teacher prefix row count must be positive")
    indices = [
        record.source_index
        for record in iter_openmath_prompts(source, skip_rows=0, take_rows=rows)
    ]
    if indices != list(range(rows)):
        raise RuntimeError(
            "registered teacher prompt prefix is not contiguous; range sharding is unsafe"
        )


def _teacher_shard_config(
    base: dict[str, Any],
    *,
    output_dir: Path,
    start_row: int,
    stop_row: int,
) -> dict[str, Any]:
    if start_row < 0 or stop_row <= start_row:
        raise ValueError("teacher shard config range is invalid")
    config = copy.deepcopy(base)
    config["output_dir"] = str(output_dir)
    config["data"]["sft_skip_rows"] = start_row
    config["data"]["sft_take_rows"] = stop_row - start_row
    config["wandb"]["enabled"] = False
    return config


def _teacher_shard_ranges(
    start: int,
    stop: int,
    *,
    workers: int,
    batch_size: int,
) -> list[tuple[int, int]]:
    if start < 0 or stop <= start or workers <= 0 or batch_size <= 0:
        raise ValueError("teacher shard range contract is invalid")
    total = stop - start
    batch_groups = math.ceil(total / batch_size)
    active_workers = min(workers, batch_groups)
    base_groups, extra_groups = divmod(batch_groups, active_workers)
    ranges: list[tuple[int, int]] = []
    cursor = start
    for worker_index in range(active_workers):
        groups = base_groups + int(worker_index < extra_groups)
        shard_stop = min(stop, cursor + groups * batch_size)
        ranges.append((cursor, shard_stop))
        cursor = shard_stop
    if cursor != stop:
        raise RuntimeError("teacher shard planning did not close the requested row range")
    return ranges


def _merge_teacher_trace_shards(
    shard_dirs: list[Path],
    target_dir: Path,
    *,
    expected_source_indices: range,
    source_revision: str,
    batch_size: int,
    elapsed_seconds: float,
    historical_shard_count: int = 0,
) -> dict[str, Any]:
    if (
        not shard_dirs
        or batch_size <= 0
        or elapsed_seconds <= 0
        or historical_shard_count < 0
        or historical_shard_count >= len(shard_dirs)
    ):
        raise ValueError("teacher shard merge contract is invalid")
    expected = list(expected_source_indices)
    if not expected or expected != sorted(set(expected)):
        raise ValueError("expected teacher source indices must be non-empty and unique")

    rows: dict[int, str] = {}
    manifests: list[dict[str, Any]] = []
    fingerprint: str | None = None
    for shard_dir in shard_dirs:
        manifest_path = shard_dir / "teacher-precompute.json"
        trace_path = shard_dir / "teacher-traces.jsonl"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("source_revision") != source_revision
            or int(manifest.get("batch_size", 0)) != batch_size
            or "H200" not in str(manifest.get("gpu_name"))
            or manifest.get("trace_sha256") != _sha256(trace_path)
        ):
            raise RuntimeError(f"teacher shard provenance mismatch: {shard_dir}")
        shard_fingerprint = str(manifest.get("teacher_fingerprint", ""))
        if not shard_fingerprint:
            raise RuntimeError(f"teacher shard lacks a fingerprint: {shard_dir}")
        if fingerprint is None:
            fingerprint = shard_fingerprint
        elif shard_fingerprint != fingerprint:
            raise RuntimeError("teacher shards have conflicting fingerprints")

        shard_records = 0
        with trace_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                source_index = int(payload["source_index"])
                if payload.get("fingerprint") != shard_fingerprint:
                    raise RuntimeError(
                        f"teacher trace fingerprint mismatch at source index {source_index}"
                    )
                if "failure_reason" not in payload:
                    reasoning_ids = payload.get("reasoning_ids")
                    visible_ids = payload.get("visible_response_ids")
                    full_ids = payload.get("full_completion_ids")
                    if (
                        not isinstance(reasoning_ids, list)
                        or not isinstance(visible_ids, list)
                        or not isinstance(full_ids, list)
                        or reasoning_ids + visible_ids != full_ids
                    ):
                        raise RuntimeError(
                            f"malformed teacher native trajectory at source index {source_index}"
                        )
                if source_index in rows:
                    raise RuntimeError(
                        f"duplicate teacher trace source index: {source_index}"
                    )
                rows[source_index] = json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ) + "\n"
                shard_records += 1
        if shard_records != int(manifest.get("records", -1)):
            raise RuntimeError(f"teacher shard record count mismatch: {shard_dir}")
        manifests.append(manifest)

    actual = sorted(rows)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            f"teacher shard coverage mismatch: missing={missing[:20]} extra={extra[:20]}"
        )

    if target_dir.exists() and not target_dir.is_symlink():
        raise RuntimeError("canonical teacher target must be absent or a version symlink")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    version_dir = Path(
        tempfile.mkdtemp(
            dir=target_dir.parent,
            prefix=f".{target_dir.name}-version-",
        )
    )
    trace_path = version_dir / "teacher-traces.jsonl"
    with trace_path.open("x", encoding="utf-8") as handle:
        for source_index in expected:
            handle.write(rows[source_index])
        handle.flush()
        os.fsync(handle.fileno())

    valid = sum(
        1 for line in rows.values() if "failure_reason" not in json.loads(line)
    )
    historical_manifests = manifests[:historical_shard_count]
    current_manifests = manifests[historical_shard_count:]
    historical_peak = max(
        (int(row.get("peak_cuda_memory_bytes", 0)) for row in historical_manifests),
        default=0,
    )
    current_peak = sum(
        int(row.get("peak_cuda_memory_bytes", 0)) for row in current_manifests
    )
    current_completion_tokens = sum(
        int(row.get("completion_tokens", 0)) for row in current_manifests
    )
    total_completion_tokens = sum(
        int(row.get("completion_tokens", 0)) for row in manifests
    )
    manifest = {
        "schema_version": 2,
        "kind": "opsd_native_teacher_trace_precompute_sharded",
        "source_revision": source_revision,
        "teacher_fingerprint": fingerprint,
        "trace_path": str((target_dir / "teacher-traces.jsonl").absolute()),
        "trace_sha256": _sha256(trace_path),
        "batch_size": batch_size,
        "worker_count": len(current_manifests),
        "historical_shard_count": historical_shard_count,
        "elapsed_seconds": elapsed_seconds,
        "completion_tokens_per_second": current_completion_tokens / elapsed_seconds,
        "device": "cuda",
        "gpu_name": "NVIDIA H200",
        "peak_cuda_memory_bytes": max(historical_peak, current_peak),
        "historical_peak_cuda_memory_bytes": historical_peak,
        "current_peak_cuda_memory_bytes": current_peak,
        "records": len(rows),
        "cached": sum(int(row.get("cached", 0)) for row in manifests),
        "generated": sum(int(row.get("generated", 0)) for row in manifests),
        "invocation_cached": sum(
            int(row.get("cached", 0)) for row in current_manifests
        ),
        "invocation_generated": sum(
            int(row.get("generated", 0)) for row in current_manifests
        ),
        "valid": valid,
        "invalid": len(rows) - valid,
        "completion_tokens": total_completion_tokens,
        "invocation_completion_tokens": current_completion_tokens,
        "generation_batches": sum(
            int(row.get("generation_batches", 0)) for row in manifests
        ),
        "singleton_batches": sum(
            int(row.get("singleton_batches", 0)) for row in manifests
        ),
        "max_observed_batch_size": max(
            int(row.get("max_observed_batch_size", 0)) for row in manifests
        ),
        "shard_manifests": [
            str((shard_dir / "teacher-precompute.json").resolve())
            for shard_dir in shard_dirs
        ],
    }
    _atomic_json(version_dir / "teacher-precompute.json", manifest)
    directory_fd = os.open(version_dir, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    temporary_link = target_dir.parent / (
        f".{target_dir.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    temporary_link.symlink_to(version_dir.name, target_is_directory=True)
    os.replace(temporary_link, target_dir)
    parent_fd = os.open(target_dir.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return manifest


def _run_sharded_teacher_precompute(
    campaign: Campaign,
    *,
    uv: str,
    base_config: dict[str, Any],
    target_dir: Path,
    start_row: int,
    stop_row: int,
    workers: int,
    batch_size: int,
    source_revision: str,
    phase_name: str,
    reserve_seconds: float,
    maximum_seconds: float,
) -> tuple[dict[str, Any], float]:
    ranges = _teacher_shard_ranges(
        start_row,
        stop_row,
        workers=workers,
        batch_size=batch_size,
    )
    shard_root = campaign.root / "teacher-shards" / phase_name
    shard_root.mkdir(parents=True, exist_ok=False)
    shard_dirs: list[Path] = []
    if start_row > 0:
        existing = shard_root / "existing"
        existing.mkdir()
        shutil.copy2(target_dir / "teacher-traces.jsonl", existing / "teacher-traces.jsonl")
        shutil.copy2(
            target_dir / "teacher-precompute.json",
            existing / "teacher-precompute.json",
        )
        shard_dirs.append(existing)

    commands: dict[str, list[str]] = {}
    validators: dict[str, Callable[[], bool]] = {}
    for worker_index, (shard_start, shard_stop) in enumerate(ranges):
        worker_name = f"{phase_name}-worker-{worker_index}"
        worker_output = shard_root / f"worker-{worker_index}"
        worker_config = _teacher_shard_config(
            base_config,
            output_dir=worker_output,
            start_row=shard_start,
            stop_row=shard_stop,
        )
        config_path = _write_config(campaign, worker_name, worker_config)
        commands[worker_name] = [
            uv,
            "run",
            "--locked",
            "--extra",
            "gpu",
            "--extra",
            "convert",
            "python",
            "scripts/precompute_opsd_teacher_traces.py",
            "--config",
            str(config_path),
            "--batch-size",
            str(batch_size),
            "--source-revision",
            source_revision,
        ]

        def validate(
            output: Path = worker_output,
            expected: list[int] = list(range(shard_start, shard_stop)),
        ) -> bool:
            manifest_path = output / "teacher-precompute.json"
            trace_path = output / "teacher-traces.jsonl"
            if not manifest_path.is_file() or not trace_path.is_file():
                return False
            manifest = json.loads(manifest_path.read_text())
            return (
                manifest.get("source_revision") == source_revision
                and int(manifest.get("batch_size", 0)) == batch_size
                and manifest.get("trace_sha256") == _sha256(trace_path)
                and _all_trace_indices(trace_path) == expected
            )

        validators[worker_name] = validate
        shard_dirs.append(worker_output)

    elapsed = campaign.run_parallel(
        commands,
        max_workers=min(workers, len(commands)),
        max_attempts=1,
        reserve_seconds=reserve_seconds,
        maximum_seconds=maximum_seconds,
        success_validators=validators,
    )
    manifest = _merge_teacher_trace_shards(
        shard_dirs,
        target_dir,
        expected_source_indices=range(stop_row),
        source_revision=source_revision,
        batch_size=batch_size,
        elapsed_seconds=elapsed,
        historical_shard_count=int(start_row > 0),
    )
    campaign.phase(
        phase_name,
        "completed",
        worker_count=len(commands),
        start_row=start_row,
        stop_row=stop_row,
        records=int(manifest["records"]),
        valid=int(manifest["valid"]),
        elapsed_seconds=elapsed,
    )
    return manifest, elapsed


def _valid_trace_indices(trace: Path) -> list[int]:
    indices = []
    with trace.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if "failure_reason" not in payload:
                indices.append(int(payload["source_index"]))
    indices.sort()
    if len(indices) != len(set(indices)):
        raise RuntimeError("teacher trace cache contains duplicate valid source indices")
    return indices


def _all_trace_indices(trace_path: Path) -> list[int]:
    indices = []
    with trace_path.open() as handle:
        for line in handle:
            if line.strip():
                indices.append(int(json.loads(line)["source_index"]))
    return sorted(indices)


def _benchmark_choices(document: dict[str, Any]) -> tuple[int, int, float, float, int]:
    rows = list(document["rows"])
    teacher_rows = [row for row in rows if row.get("teacher_status") == "passed"]
    student_rows = [row for row in rows if row.get("student_status") == "passed"]
    try:
        teacher_baseline = next(
            row for row in teacher_rows if int(row["batch_size"]) == 1
        )
        student_baseline = next(
            row for row in student_rows if int(row["batch_size"]) == 1
        )
    except StopIteration as exc:
        raise RuntimeError("batch benchmark lacks independent batch-1 baselines") from exc
    baseline_teacher = float(teacher_baseline["teacher_tokens_per_second"])
    baseline_student = float(student_baseline["student_tokens_per_second"])
    teacher_eligible = [
        row
        for row in teacher_rows
        if row.get("teacher_batch_matches_serial") is True
        and int(row["teacher_peak_cuda_memory_bytes"]) <= 120_000_000_000
    ]
    student_eligible = [
        row
        for row in student_rows
        if row.get("student_batch_matches_serial") is True
        and row.get("student_event_schedules_match_serial") is True
        and int(row["student_peak_cuda_memory_bytes"]) <= 120_000_000_000
    ]
    if not teacher_eligible or not student_eligible:
        raise RuntimeError("no semantically valid H200 teacher/student batch candidate")
    teacher_best = max(
        teacher_eligible,
        key=lambda row: (float(row["teacher_tokens_per_second"]), -int(row["batch_size"])),
    )
    student_best = max(
        student_eligible,
        key=lambda row: (float(row["student_tokens_per_second"]), -int(row["batch_size"])),
    )
    teacher_speedup = float(teacher_best["teacher_tokens_per_second"]) / baseline_teacher
    student_speedup = float(student_best["student_tokens_per_second"]) / baseline_student
    selected_peak = max(
        int(teacher_best["teacher_peak_cuda_memory_bytes"]),
        int(student_best["student_peak_cuda_memory_bytes"]),
    )
    return (
        int(teacher_best["batch_size"]),
        int(student_best["batch_size"]),
        teacher_speedup,
        student_speedup,
        selected_peak,
    )


def _capacity_summary(output_dir: Path) -> dict[str, float]:
    summary = json.loads((output_dir / "summary.json").read_text())
    diagnostic = summary["overfit_diagnostic"]
    return {
        "initial_loss": float(diagnostic["initial"]["objective_mean_loss"]),
        "final_loss": float(diagnostic["pre_save"]["objective_mean_loss"]),
    }


def _capacity_config(
    base: dict[str, Any],
    *,
    cell: dict[str, Any],
    output_dir: Path,
    take_rows: int,
    expected_locked_examples: int,
    epochs: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["output_dir"] = str(output_dir)
    config["learning_rate"] = float(cell["learning_rate"])
    config["model"]["cache_processor"]["gate_init"] = float(cell["gate_init"])
    config["data"]["sft_take_rows"] = take_rows
    distill = config["distillation"]
    distill.update(
        {
            "epochs": epochs,
            "overfit_diagnostic": True,
            "expected_locked_examples": expected_locked_examples,
            "gradient_accumulation_steps": 1,
            "execution_microbatch_size": 1,
            "require_precomputed_teacher_traces": True,
            "max_optimizer_steps": expected_locked_examples * epochs,
            "record_cache_displacements": True,
            "cache_displacement_audit_every_optimizer_steps": 1,
            "visible_trajectory_mode": cell.get("visible_trajectory_mode", "both"),
        }
    )
    config["wandb"]["name"] = f"autoresearch-capacity-{cell['id']}"
    config["wandb"]["group"] = "opsd-h200-autoresearch-v1-capacity"
    config["metadata"]["campaign"] = "qwen3_06b_opsd_h200_autoresearch_v1"
    config["metadata"]["experiment_id"] = cell["id"]
    config["metadata"]["scientific_role"] = "locked_example_capacity_screen"
    return config


def _publish_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise ValueError(f"existing campaign JSON conflicts with reconstructed payload: {path}")
        return
    _atomic_json(path, payload)


def _run_evaluation_scope(
    campaign: Campaign,
    *,
    uv: str,
    checkpoint: Path,
    source_revision: str,
    tokenizer_source: str,
    datasets_cache: str,
    evaluation: dict[str, Any],
    batch_size: int,
    limit: int | None,
    max_workers: int,
    scope_name: str,
    full_task_counts: dict[str, int],
) -> tuple[Path, float, int]:
    tasks = tuple(str(task) for task in evaluation["tasks"])
    conditions = tuple(evaluation["conditions"])
    scope_dir = campaign.root / scope_name
    shard_root = scope_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    commands: dict[str, list[str]] = {}
    success_validators: dict[str, Callable[[], bool]] = {}
    shard_paths: dict[tuple[str, str], Path] = {}

    def shard_validator(path: Path, condition_name: str, task: str) -> Callable[[], bool]:
        def validate() -> bool:
            if not path.exists():
                return False
            validate_opsd_task_shard(
                json.loads(path.read_text()),
                condition_name=condition_name,
                task=task,
                limit=limit,
                full_task_counts=full_task_counts,
            )
            return True

        return validate

    for condition in conditions:
        condition_name = str(condition["name"])
        condition_dir = shard_root / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            output = condition_dir / f"{task}.json"
            shard_paths[(condition_name, task)] = output
            if output.exists():
                validate_opsd_task_shard(
                    json.loads(output.read_text()),
                    condition_name=condition_name,
                    task=task,
                    limit=limit,
                    full_task_counts=full_task_counts,
                )
                continue
            command = [
                uv,
                "run",
                "--locked",
                "--extra",
                "gpu",
                "--extra",
                "eval",
                "gcw",
                "eval",
                "opsd-screen",
                "--checkpoint",
                str(checkpoint),
                "--source-revision",
                source_revision,
                "--tokenizer-source",
                tokenizer_source,
                "--datasets-cache",
                datasets_cache,
                "--thinking",
                str(condition["thinking"]),
                "--processor",
                str(condition["processor"]),
                "--tasks",
                task,
                "--batch-size",
                str(batch_size),
                "--out",
                str(output),
            ]
            if limit is not None:
                command.extend(("--limit", str(limit)))
            command_name = f"{scope_name}-{condition_name}-{task}"
            commands[command_name] = command
            success_validators[command_name] = shard_validator(
                output, condition_name, task
            )
    elapsed = campaign.run_parallel(
        commands,
        max_workers=max_workers,
        reserve_seconds=float(evaluation["closure_reserve_seconds"]),
        success_validators=success_validators,
    )

    merged_paths = []
    for condition in conditions:
        condition_name = str(condition["name"])
        shards = []
        for task in tasks:
            path = shard_paths[(condition_name, task)]
            if not path.is_file():
                raise FileNotFoundError(f"evaluation shard is missing after execution: {path}")
            document = json.loads(path.read_text())
            validate_opsd_task_shard(
                document,
                condition_name=condition_name,
                task=task,
                limit=limit,
                full_task_counts=full_task_counts,
            )
            shards.append(document)
        merged = merge_opsd_task_shards(
            shards,
            tasks=tasks,
            full_task_counts=full_task_counts,
        )
        merged_path = scope_dir / f"{condition_name}.json"
        _publish_json_once(merged_path, merged)
        merged_paths.append(merged_path)
    aggregate_path = scope_dir / "aggregate.json"
    condition_documents = {
        path.stem: {
            "path": str(path),
            "sha256": _sha256(path),
        }
        for path in merged_paths
    }
    aggregate = aggregate_opsd_screen(
        [json.loads(path.read_text()) for path in merged_paths]
    )
    aggregate["condition_documents"] = condition_documents
    aggregate["condition_documents_sha256"] = _canonical_sha256(condition_documents)
    _publish_json_once(aggregate_path, aggregate)
    generations = matrix_generation_count(
        tasks,
        conditions=len(conditions),
        limit=limit,
        full_task_counts=full_task_counts,
    )
    campaign.phase(
        f"{scope_name}-closure",
        "completed",
        limit_per_task=limit,
        generations=generations,
        pending_shards_executed=len(commands),
        aggregate=str(aggregate_path),
        aggregate_sha256=_sha256(aggregate_path),
        condition_documents_sha256=aggregate["condition_documents_sha256"],
    )
    return aggregate_path, elapsed, generations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--spend-approval", type=Path, required=True)
    parser.add_argument("--provider-project-id", required=True)
    parser.add_argument("--provider-region", required=True)
    parser.add_argument("--hours", type=float, default=24.0)
    args = parser.parse_args()
    _install_shutdown_manager()
    if args.hours <= 0 or args.hours > 24:
        raise ValueError("hours must lie in (0, 24]")
    repo = args.repo.resolve(strict=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != args.source_revision:
        raise RuntimeError(f"source revision mismatch: {actual} != {args.source_revision}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True):
        raise RuntimeError("campaign requires a clean checkout")
    if subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0:
        raise RuntimeError("campaign requires an immutable detached checkout")
    protocol_path = args.protocol.resolve(strict=True)
    canonical_protocol = (
        repo / "configs/protocols/qwen3_06b_opsd_h200_autoresearch_v1.json"
    ).resolve(strict=True)
    if protocol_path != canonical_protocol:
        raise RuntimeError("campaign protocol must be the canonical reviewed in-repository file")
    protocol_relative = protocol_path.relative_to(repo).as_posix()
    reviewed_protocol = subprocess.check_output(
        ["git", "show", f"{args.source_revision}:{protocol_relative}"], cwd=repo
    )
    if protocol_path.read_bytes() != reviewed_protocol:
        raise RuntimeError("campaign protocol bytes differ from the reviewed source blob")
    spend_approval = _validate_spend_approval(
        args.spend_approval,
        source_revision=args.source_revision,
        protocol_sha256=hashlib.sha256(reviewed_protocol).hexdigest(),
        requested_hours=args.hours,
        provider_project_id=args.provider_project_id,
        provider_region=args.provider_region,
    )
    campaign = Campaign(
        repo=repo,
        protocol_path=protocol_path,
        source_revision=args.source_revision,
        deadline_seconds=int(args.hours * 3600),
    )
    protocol = campaign.protocol
    matrix_reserve = _runtime_matrix_reserve(
        deadline_seconds=float(campaign.deadline_seconds),
        configured_reserve_seconds=float(protocol["matrix_reserve_seconds"]),
        closure_reserve_seconds=float(protocol["evaluation"]["closure_reserve_seconds"]),
    )
    campaign.state["spend_approval"] = spend_approval
    campaign.state["spend_approval_sha256"] = _sha256(args.spend_approval)
    campaign.state["configured_matrix_reserve_seconds"] = float(
        protocol["matrix_reserve_seconds"]
    )
    campaign.state["effective_matrix_reserve_seconds"] = matrix_reserve
    campaign._publish_state()
    base = _load_yaml(repo / protocol["base_config"])
    validate_autoresearch_protocol(protocol, base)
    teacher_rows = int(protocol["teacher_precompute_rows"])
    _require_contiguous_prompt_prefix(
        Path(base["data"]["data_dir"]),
        rows=teacher_rows,
    )
    campaign.phase(
        "teacher-shard-input-gate",
        "passed",
        contiguous_source_rows=teacher_rows,
    )
    uv = "uv"

    try:
        runtime_waiver = campaign.root / "runtime-waiver.json"
        campaign.run(
            "runtime-waiver",
            [
                uv,
                "run",
                "--locked",
                "--extra",
                "gpu",
                "--extra",
                "convert",
                "python",
                "scripts/validate_opsd_pragmatic_waiver.py",
                "--repo",
                str(repo),
                "--config",
                str(repo / protocol["base_config"]),
                "--waiver",
                str(
                    repo
                    / "configs/protocols/qwen3_06b_opsd_pragmatic_parity_waiver_v2.json"
                ),
                "--source-revision",
                args.source_revision,
                "--out",
                str(runtime_waiver),
            ],
            timeout=600.0,
        )
        runtime_evidence = json.loads(runtime_waiver.read_text())
        if (
            runtime_evidence.get("gpu_family") != "H200"
            or runtime_evidence.get("parity_gate_remains_failed") is not True
            or runtime_evidence.get("backbone_label")
            != "converted_GCW_Qwen3_0.6B_base_not_numerically_identical_to_HF"
        ):
            raise RuntimeError("runtime waiver evidence does not authorize this H200 campaign")

        benchmark_out = campaign.root / "batch-benchmark.json"
        campaign.run(
            "batch-benchmark",
            [
                uv,
                "run",
                "--locked",
                "--extra",
                "gpu",
                "--extra",
                "convert",
                "python",
                "scripts/benchmark_opsd_h200_batching.py",
                "--config",
                str(repo / protocol["base_config"]),
                "--source-revision",
                args.source_revision,
                "--batch-sizes",
                ",".join(
                    str(value)
                    for value in sorted(
                        set(protocol["teacher_batch_candidates"])
                        | set(protocol["execution_microbatch_candidates"])
                    )
                ),
                "--tokens",
                str(protocol["batch_benchmark_tokens"]),
                "--out",
                str(benchmark_out),
            ],
            timeout=3600,
        )
        benchmark_document = json.loads(benchmark_out.read_text())
        if (
            benchmark_document.get("source_revision") != args.source_revision
            or benchmark_document.get("config")
            != str((repo / protocol["base_config"]).resolve())
            or "H200" not in str(benchmark_document.get("gpu_name"))
            or int(benchmark_document.get("tokens_per_sequence", -1))
            != int(protocol["batch_benchmark_tokens"])
        ):
            raise RuntimeError("batch benchmark provenance does not match the reviewed H200 request")
        (
            selected_teacher_batch,
            selected_batch,
            teacher_batch_speedup,
            student_batch_speedup,
            selected_peak,
        ) = _benchmark_choices(benchmark_document)
        campaign.state["selected_teacher_batch"] = selected_teacher_batch
        campaign.state["selected_execution_microbatch"] = selected_batch
        campaign.state["measured_teacher_exact_shape_speedup"] = teacher_batch_speedup
        campaign.state["measured_student_exact_shape_speedup"] = student_batch_speedup
        campaign.state["selected_batch_peak_cuda_memory_bytes"] = selected_peak
        campaign._publish_state()

        teacher_output = campaign.root / "teacher-cache"
        teacher_id = "teacher-cache"
        teacher_config = materialize_train_config(
            base,
            experiment_id=teacher_id,
            output_dir=str(teacher_output),
            learning_rate=0.0001,
            gate_init=-5.0,
            effective_batch=16,
            execution_microbatch=min(selected_batch, 16),
            optimizer_steps=1,
            take_rows=int(protocol["teacher_precompute_initial_rows"]),
            visible_trajectory_mode="both",
            displacement_audit_every=16,
        )
        _write_config(campaign, teacher_id, teacher_config)
        teacher_manifest, teacher_pilot_seconds = _run_sharded_teacher_precompute(
            campaign,
            uv=uv,
            base_config=teacher_config,
            target_dir=teacher_output,
            start_row=0,
            stop_row=int(protocol["teacher_precompute_initial_rows"]),
            workers=int(protocol["teacher_precompute_workers"]),
            batch_size=selected_teacher_batch,
            source_revision=args.source_revision,
            phase_name="teacher-precompute-pilot",
            reserve_seconds=matrix_reserve,
            maximum_seconds=float(protocol["teacher_precompute_pilot_max_seconds"]),
        )
        trace = teacher_output / "teacher-traces.jsonl"
        if (
            teacher_manifest.get("source_revision") != args.source_revision
            or "H200" not in str(teacher_manifest.get("gpu_name"))
            or teacher_manifest.get("trace_sha256") != _sha256(trace)
        ):
            raise RuntimeError("teacher pilot provenance does not match the reviewed H200 request")
        teacher_pilot_rows = int(teacher_manifest["records"])
        teacher_pilot_valid = int(teacher_manifest["valid"])
        teacher_pilot_invalid = int(teacher_manifest["invalid"])
        expected_pilot_rows = int(protocol["teacher_precompute_initial_rows"])
        if (
            teacher_pilot_rows != expected_pilot_rows
            or teacher_pilot_valid + teacher_pilot_invalid != expected_pilot_rows
        ):
            raise RuntimeError("teacher trace pilot did not close exactly 64 source rows")
        teacher_valid_yield = teacher_pilot_valid / teacher_pilot_rows
        if teacher_valid_yield < float(protocol["minimum_teacher_valid_yield_fraction"]):
            raise RuntimeError(
                "teacher trace pilot valid yield is below the reviewed threshold: "
                f"{teacher_valid_yield:.6f}"
            )
        if int(teacher_manifest["batch_size"]) != selected_teacher_batch:
            raise RuntimeError("teacher precompute manifest changed the selected batch size")
        if int(teacher_manifest["peak_cuda_memory_bytes"]) > 120_000_000_000:
            raise RuntimeError("representative teacher trace pilot exceeded 120 GB")
        if _all_trace_indices(trace) != list(range(expected_pilot_rows)):
            raise RuntimeError("teacher trace pilot source-index coverage is not exactly 0..63")
        generated_rows = int(teacher_manifest["generated"])
        singleton_batches = int(teacher_manifest["singleton_batches"])
        nonsingleton_fraction = (
            (generated_rows - singleton_batches) / generated_rows
            if generated_rows
            else 0.0
        )
        if nonsingleton_fraction < float(
            protocol["minimum_teacher_nonsingleton_fraction"]
        ):
            raise RuntimeError("teacher trace pilot real batching occupancy is below threshold")
        realized_teacher_batch = int(teacher_manifest["max_observed_batch_size"])
        minimum_realized_batch = int(protocol["minimum_teacher_realized_batch_size"])
        require_realized_teacher_batch(
            selected_batch=selected_teacher_batch,
            realized_batch=realized_teacher_batch,
            minimum_batch=minimum_realized_batch,
        )
        campaign.phase(
            "teacher-pilot-gate",
            "passed",
            valid_yield_fraction=teacher_valid_yield,
            nonsingleton_fraction=nonsingleton_fraction,
            records=teacher_pilot_rows,
            valid=teacher_pilot_valid,
            generation_batches=int(teacher_manifest["generation_batches"]),
            singleton_batches=int(teacher_manifest["singleton_batches"]),
            max_observed_batch_size=int(teacher_manifest["max_observed_batch_size"]),
            completion_tokens_per_second=float(
                teacher_manifest["completion_tokens_per_second"]
            ),
        )
        valid_indices = _valid_trace_indices(trace)
        dose = protocol["decision_dose"]
        heldout_candidate_count = int(dose["heldout_candidate_examples"])
        heldout_minimum = int(dose["minimum_heldout_examples"])
        required_locked = int(protocol["capacity"]["locked_examples"])
        integrated_probe_examples = int(protocol["integrated_probe_examples"])
        heldout_candidates = select_disjoint_heldout_indices(
            valid_indices,
            locked_examples=required_locked,
            candidate_examples=heldout_candidate_count,
            minimum_examples=heldout_minimum,
            reserved_non_heldout_examples=integrated_probe_examples,
        )
        if len(valid_indices) < required_locked:
            raise RuntimeError("teacher precompute produced too few valid traces for capacity screen")
        if set(valid_indices[:required_locked]) & set(heldout_candidates):
            raise RuntimeError("capacity/model-selection examples overlap the fixed held-out pool")
        capacity_take_rows = valid_indices[required_locked - 1] + 1
        capacity_summaries: dict[str, dict[str, float]] = {}
        capacity_outputs: dict[str, Path] = {}
        cells = list(protocol["capacity_cells"])
        capacity_reserve = matrix_reserve
        minimum_cell_budget = float(
            protocol["capacity"].get("minimum_cell_budget_seconds", 1800)
        )
        if campaign.remaining() <= capacity_reserve + minimum_cell_budget:
            raise RuntimeError("insufficient budget for the complete capacity grid")
        capacity_commands: dict[str, list[str]] = {}
        capacity_validators: dict[str, Callable[[], bool]] = {}
        for cell in cells:
            output = campaign.root / "capacity" / cell["id"]
            _copy_trace(trace, output)
            config = _capacity_config(
                base,
                cell=cell,
                output_dir=output,
                take_rows=capacity_take_rows,
                expected_locked_examples=required_locked,
                epochs=int(protocol["capacity"]["epochs"]),
            )
            config_path = _write_config(campaign, f"capacity-{cell['id']}", config)
            command_name = f"capacity-{cell['id']}"
            capacity_commands[command_name] = [
                uv,
                "run",
                "--locked",
                "--extra",
                "gpu",
                "--extra",
                "convert",
                "gcw",
                "train",
                "--config",
                str(config_path),
            ]

            def validate(output_dir: Path = output) -> bool:
                try:
                    _capacity_summary(output_dir)
                except (FileNotFoundError, KeyError, TypeError, ValueError):
                    return False
                return True

            capacity_validators[command_name] = validate
        campaign.run_parallel(
            capacity_commands,
            max_workers=int(protocol["capacity_workers"]),
            max_attempts=1,
            reserve_seconds=capacity_reserve,
            success_validators=capacity_validators,
        )
        for cell in cells:
            output = campaign.root / "capacity" / cell["id"]
            capacity_summaries[cell["id"]] = _capacity_summary(output)
            capacity_outputs[cell["id"]] = output
        promoted = choose_capacity_cells(
            capacity_summaries,
            minimum_reduction_fraction=float(
                protocol["capacity"]["minimum_loss_reduction_fraction"]
            ),
            maximum_cells=int(protocol["capacity"]["maximum_cells_promoted"]),
        )
        try:
            best_capacity = require_complete_capacity_promotion(
                [str(cell["id"]) for cell in cells],
                capacity_summaries,
                promoted,
            )
        except RuntimeError as exc:
            campaign.phase(
                "capacity-gate",
                "failed",
                summaries=capacity_summaries,
                error=str(exc),
            )
            raise
        campaign.phase(
            "capacity-gate", "passed", summaries=capacity_summaries, promoted=promoted
        )

        selected_cell = next(cell for cell in cells if cell["id"] == best_capacity)
        integrated_speedup: float | None = None
        end_to_end_speedup: float | None = None
        integrated_step_seconds: float | None = None
        training_valid_fraction: float | None = None
        if (
            len(valid_indices) - len(heldout_candidates) >= integrated_probe_examples
            and campaign.remaining() > matrix_reserve + 1800.0
        ):
            probe_id = f"integrated-probe-{best_capacity}-bs16"
            probe_output = campaign.root / "integrated-probe" / probe_id
            _copy_trace(trace, probe_output)
            probe_config = materialize_train_config(
                base,
                experiment_id=probe_id,
                output_dir=str(probe_output),
                learning_rate=float(selected_cell["learning_rate"]),
                gate_init=float(selected_cell["gate_init"]),
                effective_batch=16,
                execution_microbatch=min(selected_batch, 16),
                optimizer_steps=1,
                take_rows=int(protocol["teacher_precompute_rows"]),
                visible_trajectory_mode=selected_cell.get(
                    "visible_trajectory_mode", "both"
                ),
                displacement_audit_every=1,
            )
            probe_config["distillation"]["heldout_source_indices"] = heldout_candidates
            probe_config["distillation"]["minimum_heldout_examples"] = heldout_minimum
            probe_path = _write_config(campaign, probe_id, probe_config)
            campaign.run(
                probe_id,
                [
                    uv,
                    "run",
                    "--locked",
                    "--extra",
                    "gpu",
                    "--extra",
                    "convert",
                    "gcw",
                    "train",
                    "--config",
                    str(probe_path),
                ],
                timeout=campaign.remaining() - matrix_reserve,
            )
            probe_summary = json.loads((probe_output / "summary.json").read_text())
            if (
                int(probe_summary["optimizer_steps"]) != 1
                or int(probe_summary["effective_batch_examples"]) != 16
            ):
                raise RuntimeError("integrated probe did not execute one effective-batch-16 step")
            probe_seen = int(probe_summary["seen_prompts"])
            probe_valid = int(probe_summary["valid_prompts"])
            if probe_seen <= 0 or probe_valid != 16:
                raise RuntimeError("integrated probe reported invalid prompt-yield evidence")
            training_valid_fraction = probe_valid / probe_seen
            integrated_peak_memory = int(probe_summary["peak_cuda_memory_bytes"])
            if integrated_peak_memory < 0 or integrated_peak_memory > 120_000_000_000:
                raise RuntimeError(
                    "integrated real-data student probe exceeded the 120 GB memory gate"
                )
            integrated_step_seconds = float(probe_summary["elapsed_seconds"])
            if integrated_step_seconds <= 0:
                raise RuntimeError("integrated probe reported non-positive elapsed time")
            integrated_speedup = (
                float(protocol["baseline_effective_batch16_seconds"])
                / integrated_step_seconds
            )
            projection = project_end_to_end_training(
                baseline_effective_step_seconds=float(
                    protocol["baseline_effective_batch16_seconds"]
                ),
                integrated_step_seconds=integrated_step_seconds,
                teacher_pilot_seconds=teacher_pilot_seconds,
                teacher_pilot_rows=teacher_pilot_rows,
                valid_teacher_traces=max(
                    1,
                    math.floor(teacher_pilot_rows * training_valid_fraction),
                ),
                optimizer_steps=1000,
                effective_batch=16,
            )
            end_to_end_speedup = float(projection["end_to_end_speedup"])
            campaign.state["integrated_effective_batch16_seconds"] = (
                integrated_step_seconds
            )
            campaign.state["cached_training_speedup"] = integrated_speedup
            campaign.state["training_valid_fraction"] = training_valid_fraction
            campaign.state["integrated_peak_cuda_memory_bytes"] = integrated_peak_memory
            campaign.state["end_to_end_training_projection"] = projection
            campaign.state["end_to_end_training_speedup"] = end_to_end_speedup
            campaign.state["projected_1000_step_days"] = (
                float(projection["projected_total_seconds"]) / 86400.0
            )
            campaign._publish_state()

        requested_optimizer_steps = (
            decision_dose_steps(
                end_to_end_speedup,
                below_4x=int(dose["maximum_optimizer_steps_below_4x"]),
                middle=int(dose["maximum_optimizer_steps_from_4x_to_7_5x"]),
                high=int(dose["maximum_optimizer_steps_at_or_above_7_5x"]),
            )
            if end_to_end_speedup is not None
            else 0
        )
        required_valid_traces = requested_optimizer_steps * 16 + heldout_candidate_count
        teacher_total_seconds = teacher_pilot_seconds
        extension_safety = float(protocol["teacher_extension_eta_safety_factor"])
        if requested_optimizer_steps > 0 and training_valid_fraction is None:
            raise RuntimeError("decision dose was requested without integrated valid-yield evidence")
        estimated_successful_examples = (
            math.floor(teacher_pilot_rows * training_valid_fraction)
            if training_valid_fraction is not None
            else 0
        )
        if requested_optimizer_steps > 0 and estimated_successful_examples < required_valid_traces:
            assert training_valid_fraction is not None
            raw_target = math.ceil(
                required_valid_traces / training_valid_fraction * extension_safety
            )
            maximum_rows = int(protocol["teacher_precompute_rows"])
            available_extension_seconds = campaign.remaining() - matrix_reserve - 1800.0
            pilot_rows_per_second = teacher_pilot_rows / teacher_pilot_seconds
            affordable_rows = teacher_pilot_rows + max(
                0,
                math.floor(
                    available_extension_seconds
                    * pilot_rows_per_second
                    / extension_safety
                ),
            )
            extension_rows = min(maximum_rows, raw_target, affordable_rows)
            if extension_rows > teacher_pilot_rows:
                extension_id = f"teacher-cache-extend-{extension_rows}"
                extension_config = materialize_train_config(
                    base,
                    experiment_id=extension_id,
                    output_dir=str(teacher_output),
                    learning_rate=0.0001,
                    gate_init=-5.0,
                    effective_batch=16,
                    execution_microbatch=min(selected_batch, 16),
                    optimizer_steps=1,
                    take_rows=extension_rows,
                    visible_trajectory_mode="both",
                    displacement_audit_every=16,
                )
                _write_config(campaign, extension_id, extension_config)
                teacher_manifest, extension_seconds = _run_sharded_teacher_precompute(
                    campaign,
                    uv=uv,
                    base_config=extension_config,
                    target_dir=teacher_output,
                    start_row=teacher_pilot_rows,
                    stop_row=extension_rows,
                    workers=int(protocol["teacher_precompute_workers"]),
                    batch_size=selected_teacher_batch,
                    source_revision=args.source_revision,
                    phase_name=extension_id,
                    reserve_seconds=matrix_reserve + 1800.0,
                    maximum_seconds=available_extension_seconds,
                )
                teacher_total_seconds += extension_seconds
                valid_indices = _valid_trace_indices(trace)
                extension_records = int(teacher_manifest["records"])
                extension_valid = int(teacher_manifest["valid"])
                extension_invalid = int(teacher_manifest["invalid"])
                if (
                    teacher_manifest.get("source_revision") != args.source_revision
                    or "H200" not in str(teacher_manifest.get("gpu_name"))
                    or teacher_manifest.get("trace_sha256") != _sha256(trace)
                    or int(teacher_manifest["batch_size"]) != selected_teacher_batch
                    or extension_records != extension_rows
                    or extension_valid + extension_invalid != extension_rows
                    or int(teacher_manifest["peak_cuda_memory_bytes"]) > 120_000_000_000
                    or _all_trace_indices(trace) != list(range(extension_rows))
                ):
                    raise RuntimeError("teacher extension failed provenance or exact row closure")
                extension_generated = int(teacher_manifest["generated"])
                extension_singletons = int(teacher_manifest["singleton_batches"])
                extension_nonsingleton_fraction = (
                    (extension_generated - extension_singletons) / extension_generated
                    if extension_generated
                    else 0.0
                )
                if extension_nonsingleton_fraction < float(
                    protocol["minimum_teacher_nonsingleton_fraction"]
                ):
                    raise RuntimeError("teacher extension real batching occupancy is below threshold")
                campaign.phase(
                    "teacher-extension-gate",
                    "completed",
                    target_rows=extension_rows,
                    valid_traces=len(valid_indices),
                    required_valid_traces=required_valid_traces,
                    elapsed_seconds=extension_seconds,
                    generation_batches=int(teacher_manifest["generation_batches"]),
                    singleton_batches=int(teacher_manifest["singleton_batches"]),
                )
        final_teacher_rows = int(teacher_manifest["records"])
        conservative_training_examples = (
            math.floor(
                final_teacher_rows * training_valid_fraction / extension_safety
            )
            if training_valid_fraction is not None
            else 0
        )
        available_training_examples = max(
            0,
            len(valid_indices) - len(heldout_candidates),
        )
        post_extension_requested_steps = 0
        if integrated_step_seconds is not None:
            final_projection = project_end_to_end_training(
                baseline_effective_step_seconds=float(
                    protocol["baseline_effective_batch16_seconds"]
                ),
                integrated_step_seconds=integrated_step_seconds,
                teacher_pilot_seconds=teacher_total_seconds,
                teacher_pilot_rows=final_teacher_rows,
                valid_teacher_traces=max(1, conservative_training_examples),
                optimizer_steps=1000,
                effective_batch=16,
            )
            post_extension_requested_steps = decision_dose_steps(
                float(final_projection["end_to_end_speedup"]),
                below_4x=int(dose["maximum_optimizer_steps_below_4x"]),
                middle=int(dose["maximum_optimizer_steps_from_4x_to_7_5x"]),
                high=int(dose["maximum_optimizer_steps_at_or_above_7_5x"]),
            )
            campaign.state["end_to_end_training_projection_after_extension"] = (
                final_projection
            )
        final_policy_steps = min(
            requested_optimizer_steps,
            post_extension_requested_steps,
        )
        optimizer_steps = feasible_dose_steps(
            final_policy_steps,
            valid_traces=available_training_examples,
            effective_batch=16,
        )
        campaign.state["requested_decision_optimizer_steps"] = requested_optimizer_steps
        campaign.state["post_extension_requested_optimizer_steps"] = (
            post_extension_requested_steps
        )
        campaign.state["trace_feasible_decision_optimizer_steps"] = optimizer_steps
        campaign.state["valid_teacher_traces"] = len(valid_indices)
        campaign.state["teacher_trace_total_seconds"] = teacher_total_seconds
        campaign._publish_state()
        if optimizer_steps <= 0:
            raise RuntimeError(
                "integrated throughput and authenticated trace capacity did not authorize a decision dose"
            )
        assert integrated_step_seconds is not None
        projected_decision_seconds = (
            optimizer_steps
            * integrated_step_seconds
            * float(dose["training_eta_safety_factor"])
        )
        campaign.state["projected_decision_training_seconds"] = projected_decision_seconds
        campaign._publish_state()
        if campaign.remaining() <= matrix_reserve + 1800.0 + projected_decision_seconds:
            raise RuntimeError("insufficient pre-evaluation budget for the authorized decision dose")
        if len(heldout_candidates) < heldout_minimum:
            raise RuntimeError(
                "teacher cache lacks the minimum disjoint held-out candidates after the training dose"
            )
        decision_take_rows = final_teacher_rows
        experiment_id = f"decision-{best_capacity}-bs16-steps{optimizer_steps}"
        decision_output = campaign.root / "decision" / experiment_id
        _copy_trace(trace, decision_output)
        decision_config = materialize_train_config(
            base,
            experiment_id=experiment_id,
            output_dir=str(decision_output),
            learning_rate=float(selected_cell["learning_rate"]),
            gate_init=float(selected_cell["gate_init"]),
            effective_batch=16,
            execution_microbatch=min(selected_batch, 16),
            optimizer_steps=optimizer_steps,
            take_rows=decision_take_rows,
            visible_trajectory_mode=selected_cell.get(
                "visible_trajectory_mode", "both"
            ),
            displacement_audit_every=int(
                dose["cache_displacement_audit_every_optimizer_steps"]
            ),
        )
        decision_config["distillation"]["heldout_source_indices"] = heldout_candidates
        decision_config["distillation"]["minimum_heldout_examples"] = heldout_minimum
        decision_path = _write_config(campaign, experiment_id, decision_config)
        try:
            campaign.run(
                experiment_id,
                [uv, "run", "--locked", "--extra", "gpu", "--extra", "convert", "gcw", "train", "--config", str(decision_path)],
                timeout=campaign.remaining() - matrix_reserve,
            )
        except CampaignCommandTimeout as exc:
            campaign.phase(
                "decision-gate",
                "failed",
                reason="decision dose reached evaluation reserve",
            )
            raise RuntimeError("decision dose did not complete before the evaluation reserve") from exc
        decision_summary = json.loads((decision_output / "summary.json").read_text())
        trained_source_indices = {
            int(index) for index in decision_summary["trained_source_indices"]
        }
        overlap = trained_source_indices & set(heldout_candidates)
        if overlap:
            raise RuntimeError(
                f"final trained source indices overlap fixed held-out indices: {sorted(overlap)}"
            )
        heldout_evidence = require_heldout_improvement(
            decision_summary,
            minimum_reduction_fraction=float(
                dose["minimum_heldout_kl_reduction_fraction"]
            ),
        )
        campaign.phase(
            "heldout-kl-gate",
            "passed",
            experiment_id=experiment_id,
            evidence=heldout_evidence,
        )
        checkpoint = decision_output / "final"

        evaluation = protocol["evaluation"]
        eval_batch = min(selected_batch, max(int(value) for value in evaluation["batch_candidates"]))
        concurrency = int(evaluation["max_workers"])
        full_task_counts = {
            str(task): int(count)
            for task, count in evaluation["full_task_sample_counts"].items()
        }
        tokenizer_source = str(Path(base["data"]["tokenizer"].removeprefix("hf:")))
        datasets_cache = "/mnt/jfs-01/shared/gcw/hf_datasets_eval_v3"
        sentinel_limit = int(evaluation["sentinel_limit_per_task"])
        sentinel_aggregate, sentinel_elapsed, sentinel_generations = _run_evaluation_scope(
            campaign,
            uv=uv,
            checkpoint=checkpoint,
            source_revision=args.source_revision,
            tokenizer_source=tokenizer_source,
            datasets_cache=datasets_cache,
            evaluation=evaluation,
            batch_size=eval_batch,
            limit=sentinel_limit,
            max_workers=concurrency,
            scope_name="matrix-sentinel",
            full_task_counts=full_task_counts,
        )
        if sentinel_elapsed <= 0:
            raise RuntimeError("fresh campaign sentinel produced no measurable execution time")
        selected_limit = select_evaluation_limit(
            sentinel_elapsed_seconds=sentinel_elapsed,
            sentinel_generations=sentinel_generations,
            remaining_seconds=campaign.remaining(),
            closure_reserve_seconds=float(evaluation["closure_reserve_seconds"]),
            tasks=tuple(str(task) for task in evaluation["tasks"]),
            conditions=len(evaluation["conditions"]),
            candidate_limits=tuple(int(value) for value in evaluation["partial_limit_candidates"]),
            full_task_counts=full_task_counts,
            safety_factor=float(evaluation["eta_safety_factor"]),
        )
        campaign.state["sentinel_elapsed_seconds"] = sentinel_elapsed
        campaign.state["sentinel_generations"] = sentinel_generations
        campaign.state["selected_evaluation_limit_per_task"] = selected_limit
        campaign._publish_state()
        if selected_limit in {0, sentinel_limit}:
            aggregate = sentinel_aggregate
            evaluation_status = (
                "sentinel_only_due_to_deadline" if selected_limit == 0 else "partial"
            )
            selected_limit = sentinel_limit
        else:
            scope_name = "matrix-full" if selected_limit is None else f"matrix-limit-{selected_limit}"
            aggregate, _, _ = _run_evaluation_scope(
                campaign,
                uv=uv,
                checkpoint=checkpoint,
                source_revision=args.source_revision,
                tokenizer_source=tokenizer_source,
                datasets_cache=datasets_cache,
                evaluation=evaluation,
                batch_size=eval_batch,
                limit=selected_limit,
                max_workers=concurrency,
                scope_name=scope_name,
                full_task_counts=full_task_counts,
            )
            evaluation_status = "full" if selected_limit is None else "partial"
        campaign.state["status"] = "completed"
        campaign.state["selected_checkpoint"] = str(checkpoint)
        campaign.state["evaluation_status"] = evaluation_status
        campaign.state["selected_evaluation_limit_per_task"] = selected_limit
        campaign.state["matrix_aggregate"] = str(aggregate)
        campaign._publish_state()
    except BaseException as exc:
        campaign.state["status"] = "failed"
        campaign.state["error"] = f"{type(exc).__name__}: {exc}"
        campaign._publish_state()
        raise


if __name__ == "__main__":
    main()
