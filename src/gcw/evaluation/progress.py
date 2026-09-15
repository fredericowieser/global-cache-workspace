from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _json_safe(payload: dict) -> dict:
    safe = json.loads(json.dumps(payload, sort_keys=True, default=str))
    if not isinstance(safe, dict):
        raise ValueError("authenticated JSON payload must be an object")
    return safe


def _payload_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _atomic_write_authenticated_json(
    path: Path, *, kind: str, payload: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(payload)
    document = {
        "schema_version": 1,
        "kind": kind,
        "payload": safe,
        "payload_sha256": _payload_sha256(safe),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _load_authenticated_json(path: Path, *, kind: str) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"authenticated {kind} JSON is unreadable") from exc
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "kind",
        "payload",
        "payload_sha256",
    }:
        raise ValueError(f"authenticated {kind} JSON schema differs")
    payload = document.get("payload")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != kind
        or not isinstance(payload, dict)
        or document.get("payload_sha256") != _payload_sha256(payload)
    ):
        raise ValueError(f"authenticated {kind} JSON integrity check failed")
    return payload


def _write_partial_results(results: dict) -> None:
    if not is_main_process():
        return
    partial_dir = os.environ.get("GCW_EVAL_PARTIAL_DIR")
    if not partial_dir:
        return
    _atomic_write_authenticated_json(
        Path(partial_dir) / "partial.json",
        kind="flow_math_eval_partial_results",
        payload=results,
    )


def _empty_eval_results() -> dict:
    return {
        "results": {},
        "groups": {},
        "n-samples": {},
        "n-shot": {},
        "versions": {},
        "higher_is_better": {},
        "gcw_runtime": {},
    }


def _validate_partial_results(
    payload: dict, runs: list[tuple[list[str], int]]
) -> dict:
    expected_keys = set(_empty_eval_results())
    if set(payload) != expected_keys or any(
        not isinstance(payload[key], dict) for key in expected_keys
    ):
        raise ValueError("authenticated partial results schema differs")
    requested = {task for tasks, _fewshot in runs for task in tasks}
    for section in ("results", "groups"):
        for task in payload[section]:
            if not any(
                task == requested_task or task.startswith(f"{requested_task}_")
                for requested_task in requested
            ):
                raise ValueError(
                    f"authenticated partial results contain unexpected task: {task}"
                )
    runtime = payload["gcw_runtime"]
    required_runtime_fields = {
        "requested_batch_size",
        "configured_auto_max_batch_size",
        "successful_batch_sizes",
    }
    for tasks, _fewshot in runs:
        if not all(
            task in payload["results"] or task in payload["groups"]
            for task in tasks
        ):
            continue
        task_runtime = runtime.get(",".join(tasks))
        if not isinstance(
            task_runtime, dict
        ) or not required_runtime_fields <= set(task_runtime):
            raise ValueError(
                "partial results lack current evaluation runtime measurements; "
                "a fresh evaluation is required"
            )
    return payload


def _load_partial_results(runs: list[tuple[list[str], int]]) -> dict | None:
    partial_dir = os.environ.get("GCW_EVAL_PARTIAL_DIR")
    if not partial_dir:
        return None
    path = Path(partial_dir) / "partial.json"
    if not path.exists():
        return None
    payload = _load_authenticated_json(
        path, kind="flow_math_eval_partial_results"
    )
    return _validate_partial_results(payload, runs)


def _progress_path() -> Path | None:
    if os.environ.get("GCW_EVAL_PROGRESS_PATH"):
        return Path(os.environ["GCW_EVAL_PROGRESS_PATH"])
    if os.environ.get("GCW_EVAL_PARTIAL_DIR"):
        return Path(os.environ["GCW_EVAL_PARTIAL_DIR"]) / "progress.json"
    return None


def _validate_progress(
    payload: dict, runs: list[tuple[list[str], int]]
) -> dict:
    if set(payload) != {
        "started_at",
        "started_time",
        "updated_at",
        "total_runs",
        "completed_runs",
        "completed_tasks",
        "elapsed_sec",
        "eta_sec",
        "runs",
    }:
        raise ValueError("authenticated progress schema differs")
    entries = payload.get("runs")
    if not isinstance(entries, list) or len(entries) != len(runs):
        raise ValueError("authenticated progress run list differs")
    allowed_entry_keys = {
        "tasks",
        "fewshot",
        "status",
        "start_time",
        "end_time",
        "started_at",
        "ended_at",
        "duration_sec",
        "skipped",
        "error",
    }
    expected_runs = [
        {"tasks": tasks, "fewshot": fewshot} for tasks, fewshot in runs
    ]
    actual_runs = []
    for entry in entries:
        if not isinstance(entry, dict) or not set(entry).issubset(
            allowed_entry_keys
        ):
            raise ValueError("authenticated progress entry schema differs")
        if not {
            "tasks",
            "fewshot",
            "status",
            "start_time",
            "end_time",
        }.issubset(entry):
            raise ValueError("authenticated progress entry is incomplete")
        actual_runs.append(
            {"tasks": entry["tasks"], "fewshot": entry["fewshot"]}
        )
    if payload.get("total_runs") != len(runs) or actual_runs != expected_runs:
        raise ValueError(
            "authenticated progress does not match requested evaluation runs"
        )
    return payload


def _format_time(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _estimate_eta(progress: dict) -> float | None:
    runs = progress.get("runs", [])
    completed = [
        float(item["duration_sec"])
        for item in runs
        if item.get("status") == "completed"
        and not item.get("skipped")
        and item.get("duration_sec")
    ]
    remaining = sum(
        1 for item in runs if item.get("status") not in {"completed", "failed"}
    )
    if not completed or remaining <= 0:
        return None
    return sum(completed) / len(completed) * remaining


def _new_progress(runs: list[tuple[list[str], int]]) -> dict:
    path = _progress_path()
    if path is not None and path.exists():
        payload = _load_authenticated_json(path, kind="flow_math_eval_progress")
        return _validate_progress(payload, runs)
    now = time.time()
    return {
        "started_at": _format_time(now),
        "started_time": now,
        "updated_at": _format_time(now),
        "total_runs": len(runs),
        "completed_runs": 0,
        "completed_tasks": [],
        "elapsed_sec": 0.0,
        "eta_sec": None,
        "runs": [
            {
                "tasks": tasks,
                "fewshot": fewshot,
                "status": "pending",
                "start_time": None,
                "end_time": None,
            }
            for tasks, fewshot in runs
        ],
    }


def _update_progress(
    run_idx: int,
    status: str,
    runs: list[tuple[list[str], int]],
    merged: dict,
    *,
    progress: dict | None = None,
    skipped: bool = False,
    error: str | None = None,
) -> dict:
    progress = progress or _new_progress(runs)
    now = time.time()
    entry = progress["runs"][run_idx]
    if entry.get("start_time") is None:
        entry["start_time"] = now
        entry["started_at"] = _format_time(now)
    entry["status"] = status
    entry["skipped"] = bool(skipped)
    if error:
        entry["error"] = error
    if status in {"completed", "failed"}:
        entry["end_time"] = now
        entry["ended_at"] = _format_time(now)
        entry["duration_sec"] = max(0.0, now - float(entry["start_time"]))
    progress["updated_at"] = _format_time(now)
    progress["completed_runs"] = sum(
        1 for item in progress["runs"] if item.get("status") == "completed"
    )
    progress["completed_tasks"] = sorted(merged.get("results", {}))
    progress["elapsed_sec"] = max(
        0.0, now - float(progress.get("started_time", now))
    )
    progress["eta_sec"] = _estimate_eta(progress)
    return progress


def _write_progress(progress: dict, merged: dict) -> None:
    if not is_main_process():
        return
    path = _progress_path()
    if path is None:
        return
    progress["completed_tasks"] = sorted(merged.get("results", {}))
    _atomic_write_authenticated_json(
        path, kind="flow_math_eval_progress", payload=progress
    )
