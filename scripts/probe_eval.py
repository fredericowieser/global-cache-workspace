from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from extract_paper_table import TASK_COLUMNS, _extract_row, _markdown

PAPER7_TASKS = tuple(TASK_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe GCW eval progress, results, and rough ETA."
    )
    parser.add_argument(
        "--json",
        default="results/llama32_1b_bgt_paper7_fast.json",
        help="Final eval JSON path",
    )
    parser.add_argument(
        "--partial",
        default="results/paper7_fast_partial/partial.json",
        help="Partial eval JSON path",
    )
    parser.add_argument(
        "--progress",
        default="results/paper7_fast_partial/progress.json",
        help="Progress sidecar JSON path",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Optional eval log path for timestamp fallback",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        default=None,
        help="Optional Slurm allocation id for GPU status",
    )
    parser.add_argument(
        "--skip-tasks",
        default="",
        help="Comma-separated tasks to omit from expected progress",
    )
    parser.add_argument("--tasks", choices=["paper7"], default="paper7")
    args = parser.parse_args()

    tasks = tuple(
        task
        for task in PAPER7_TASKS
        if task not in _skip_tasks(args.skip_tasks)
    )
    final_path = Path(args.json)
    partial_path = Path(args.partial)
    progress_path = Path(args.progress)
    result_path = final_path if final_path.exists() else partial_path
    payload = _read_json(result_path) or {}
    progress = _read_json(progress_path) or {}

    status = (
        "complete"
        if final_path.exists()
        else "running" if payload else "not-started"
    )
    completed = sorted(payload.get("results", {}))
    total = len(tasks)
    eta = _eta(progress, completed, total, args.log)

    print(f"Status: {status}")
    print(f"Result: {result_path if payload else 'none'}")
    print(
        f"Completed: {len(completed)}/{total} ({', '.join(completed) if completed else 'none'})"
    )
    if progress:
        print(f"Updated: {progress.get('updated_at', 'unknown')}")
    print(f"Elapsed: {_fmt_seconds(_elapsed(progress, args.log))}")
    print(f"ETA: {_fmt_seconds(eta) if eta is not None else 'unknown'}")
    if args.job_id is not None:
        gpu = _gpu_status(args.job_id)
        if gpu:
            print("GPU:")
            print(gpu)
    if final_path.exists():
        print()
        print(_markdown(_extract_row(payload)))


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _skip_tasks(value: str) -> set[str]:
    return {task.strip() for task in value.split(",") if task.strip()}


def _elapsed(progress: dict, log: str | None) -> float | None:
    if progress.get("elapsed_sec") is not None:
        return float(progress["elapsed_sec"])
    if progress.get("started_time") is not None:
        return max(0.0, time.time() - float(progress["started_time"]))
    if log and Path(log).exists():
        return max(0.0, time.time() - Path(log).stat().st_ctime)
    return None


def _eta(
    progress: dict, completed: list[str], total: int, log: str | None
) -> float | None:
    if progress.get("eta_sec") is not None:
        return float(progress["eta_sec"])
    elapsed = _elapsed(progress, log)
    if elapsed is None or not completed:
        return None
    remaining = max(0, total - len(completed))
    return elapsed / len(completed) * remaining


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _gpu_status(job_id: int) -> str:
    try:
        out = subprocess.check_output(
            [
                "srun",
                f"--jobid={job_id}",
                "--overlap",
                "bash",
                "-lc",
                "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader",
            ],
            text=True,
            timeout=15,
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return ""
    return out.strip()


if __name__ == "__main__":
    main()
