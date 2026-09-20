#!/usr/bin/env python3
"""Read-only recovery of GCW KOR-Bench outputs from CPython 3.12 on Linux."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time


class RecoveryError(RuntimeError):
    pass


class PythonMemory:
    """Bounded decoding of the installed 64-bit CPython 3.12 object layouts."""

    def __init__(self, pid: int, deadline: float):
        self.fd = os.open(f"/proc/{pid}/mem", os.O_RDONLY)
        self.deadline = deadline
        self.bytes_read = 0
        self.type_names = {}

    def close(self):
        os.close(self.fd)

    def read(self, address: int, size: int) -> bytes:
        if time.monotonic() >= self.deadline:
            raise RecoveryError("Snapshot time limit exceeded")
        if address < 4096 or not 0 <= size <= 32 * 1024 * 1024:
            raise RecoveryError("Invalid memory read address or size")
        self.bytes_read += size
        if self.bytes_read > 256 * 1024 * 1024:
            raise RecoveryError("Snapshot read budget exceeded")
        value = os.pread(self.fd, size, address)
        if len(value) != size:
            raise RecoveryError("Incomplete process memory read")
        return value

    def unpack(self, fmt, address):
        return struct.unpack(fmt, self.read(address, struct.calcsize(fmt)))[0]

    def pointer(self, address):
        return self.unpack("<Q", address)

    def kind(self, address):
        type_address = self.pointer(address + 8)
        if type_address not in self.type_names:
            name_address = self.pointer(type_address + 24)
            name = self.read(name_address, 128).split(b"\0", 1)[0]
            self.type_names[type_address] = name.decode("ascii")
        return self.type_names[type_address]

    def uncell(self, address):
        if self.kind(address) == "cell":
            return self.pointer(address + 16)
        return address

    def sequence(self, address):
        address = self.uncell(address)
        kind = self.kind(address)
        if kind not in {"list", "tuple"}:
            raise RecoveryError(f"Expected list/tuple, got {kind}")
        length = self.unpack("<q", address + 16)
        if not 0 <= length <= 100_000:
            raise RecoveryError("Invalid sequence length")
        if not length:
            return []
        data = self.pointer(address + 24) if kind == "list" else address + 24
        return list(struct.unpack(f"<{length}Q", self.read(data, 8 * length)))

    def dict_entries(self, keys, values=0):
        header = self.read(keys, 32)
        index_bytes_log2, kind = header[9], header[10]
        count = struct.unpack_from("<q", header, 24)[0]
        if (
            kind not in (0, 1, 2)
            or index_bytes_log2 > 24
            or not 0 <= count <= 100_000
        ):
            raise RecoveryError("Invalid dictionary layout")
        stride = 24 if kind == 0 else 16
        data = self.read(keys + 32 + (1 << index_bytes_log2), count * stride)
        split = self.read(values, 8 * count) if values and count else None
        entries = []
        for index in range(count):
            offset = index * stride + (8 if kind == 0 else 0)
            key, value = struct.unpack_from("<QQ", data, offset)
            if split is not None:
                value = struct.unpack_from("<Q", split, index * 8)[0]
            if key and value:
                entries.append((key, value))
        return entries

    def dictionary(self, address):
        if self.kind(address) != "dict":
            raise RecoveryError("Expected dictionary")
        return self.dict_entries(
            self.pointer(address + 32), self.pointer(address + 40)
        )

    def attributes(self, address):
        address = self.uncell(address)
        type_address = self.pointer(address + 8)
        flags = self.pointer(type_address + 168)
        if flags & (1 << 4):
            storage = self.pointer(address - 24)
            if storage & 1:
                entries = self.dict_entries(
                    self.pointer(type_address + 880), storage + 1
                )
            elif storage:
                entries = self.dictionary(storage)
            else:
                entries = []
        else:
            offset = self.unpack("<q", type_address + 288)
            if offset <= 0:
                raise RecoveryError(
                    f"Unsupported instance dictionary: {self.kind(address)}"
                )
            storage = self.pointer(address + offset)
            entries = self.dictionary(storage) if storage else []
        return {self.decode(key): value for key, value in entries}

    def decode(self, address, depth=0):
        if depth > 20:
            raise RecoveryError("Object nesting limit exceeded")
        address = self.uncell(address)
        kind = self.kind(address)
        if kind == "NoneType":
            return None
        if kind in {"int", "bool"}:
            tag = self.pointer(address + 16)
            length, sign = tag >> 3, tag & 3
            if length > 32 or sign > 2:
                raise RecoveryError("Invalid integer layout")
            digits = self.read(address + 24, length * 4)
            value = sum(
                struct.unpack_from("<I", digits, i * 4)[0] << (30 * i)
                for i in range(length)
            )
            value = -value if sign == 2 else value
            return bool(value) if kind == "bool" else value
        if kind == "float":
            return self.unpack("<d", address + 16)
        if kind == "str":
            length = self.unpack("<q", address + 16)
            state = self.unpack("<I", address + 32)
            width, compact, ascii_only = (
                (state >> 2) & 7,
                state & 32,
                state & 64,
            )
            if width not in (1, 2, 4) or not 0 <= length <= 8_000_000:
                raise RecoveryError("Invalid Unicode layout")
            data = (
                address + (40 if ascii_only else 56)
                if compact
                else self.pointer(address + 56)
            )
            return self.read(data, length * width).decode(
                {1: "latin1", 2: "utf-16-le", 4: "utf-32-le"}[width]
            )
        if kind in {"list", "tuple"}:
            items = [
                self.decode(item, depth + 1) for item in self.sequence(address)
            ]
            return tuple(items) if kind == "tuple" else items
        if kind == "dict":
            return {
                self.decode(key, depth + 1): self.decode(value, depth + 1)
                for key, value in self.dictionary(address)
            }
        raise RecoveryError(f"Unsupported Python object type: {kind}")


def process_identity(pid):
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return stat[19], stat[0]


@contextmanager
def stopped_process(pid, seconds=30):
    identity, state = process_identity(pid)
    if state in {"T", "t", "Z"}:
        raise RecoveryError("Target is already stopped, traced, or exited")
    read_fd, write_fd = os.pipe()
    watchdog = os.fork()
    if watchdog == 0:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.close(write_fd)
        try:
            ready, _, _ = select.select([read_fd], [], [], seconds)
            disarmed = bool(ready) and os.read(read_fd, 1) == b"x"
            if not disarmed and process_identity(pid)[0] == identity:
                os.kill(pid, signal.SIGCONT)
        except (OSError, ProcessLookupError):
            pass
        finally:
            os._exit(0)
    os.close(read_fd)
    try:
        os.kill(pid, signal.SIGSTOP)
        deadline = time.monotonic() + seconds - 5
        while True:
            states = [
                process_identity(int(path.name))[1]
                for path in Path(f"/proc/{pid}/task").iterdir()
            ]
            if states and all(state in {"T", "t"} for state in states):
                break
            if time.monotonic() >= deadline:
                raise RecoveryError(
                    "Timed out waiting for target threads to stop"
                )
            time.sleep(0.01)
        yield deadline
    finally:
        try:
            if process_identity(pid)[0] == identity:
                os.kill(pid, signal.SIGCONT)
        finally:
            try:
                os.write(write_fd, b"x")
            except BrokenPipeError:
                pass
            os.close(write_fd)
            os.waitpid(watchdog, 0)


def extract_snapshot(memory, traces):
    candidates = [
        (trace, frame)
        for trace in traces
        for frame in trace["frames"]
        if frame["filename"].endswith("gcw/evaluation/lm_eval_adapter.py")
        and frame["name"].split(".")[-1] == "generate_until"
    ]
    if len(candidates) != 1:
        raise RecoveryError(
            "Expected one active GCW generate_until frame; retry later"
        )
    trace, frame = candidates[0]
    local = {item["name"]: item["addr"] for item in frame["locals"]}
    requests = memory.sequence(local["requests"])
    outs = memory.decode(local["outs"])
    if len(outs) != len(requests):
        raise RecoveryError("Request/output populations differ")
    request_attributes = [memory.attributes(address) for address in requests]
    keys = []
    occurrences = {}
    for index, attrs in enumerate(request_attributes):
        task = memory.decode(attrs["task_name"]) or "requests"
        doc_id = memory.decode(attrs["doc_id"])
        example = str(index if doc_id is None else doc_id)
        idx = str(memory.decode(attrs["idx"])) if "idx" in attrs else "0"
        base = (task, example, idx)
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        repeats = (
            memory.decode(attrs["repeats"]) if "repeats" in attrs else None
        )
        if isinstance(repeats, int) and repeats > 0:
            occurrence %= repeats
        keys.append((task, example, f"{idx}:{occurrence}"))
    if "metric_keys" in local and memory.decode(local["metric_keys"]) != keys:
        raise RecoveryError("Live metric keys differ from request identities")
    completed = {
        index: value for index, value in enumerate(outs) if value is not None
    }
    sources = {index: "outs" for index in completed}
    for current in trace["frames"]:
        if not current["filename"].endswith(
            "gcw/evaluation/lm_eval_adapter.py"
        ):
            continue
        if current["name"].split(".")[-1] != "_run_batches":
            continue
        values = {item["name"]: item["addr"] for item in current["locals"]}
        if "results" not in values or "rows" not in values:
            continue
        results = memory.decode(values["results"])
        rows = memory.sequence(values["rows"])
        if len(results) > len(rows):
            raise RecoveryError(
                "Batch output population exceeds input population"
            )
        for row, value in zip(rows, results):
            index = memory.decode(memory.sequence(row)[0])
            if index in completed and completed[index] != value:
                raise RecoveryError("Conflicting completed responses")
            completed[index] = value
            sources[index] = "current_group_completed_batches"
    diagnostics = {}
    diagnostic_error = None
    try:
        evaluator = memory.attributes(local["self"])
        collector = memory.attributes(evaluator["update_metrics"])
        diagnostics = memory.decode(collector["requests"])
    except (KeyError, OSError, RecoveryError, UnicodeError) as exc:
        diagnostic_error = str(exc)
    records = []
    for index, response in sorted(completed.items()):
        if not isinstance(response, str) or not 0 <= index < len(requests):
            raise RecoveryError("Invalid completed response")
        attrs = request_attributes[index]
        task = memory.decode(attrs["task_name"])
        doc_id = memory.decode(attrs["doc_id"])
        key = keys[index]
        if tuple(key[:2]) != (task, str(doc_id)):
            raise RecoveryError("Request/metric identity mismatch")
        record = {
            "request_index": index,
            "task": task,
            "doc_id": doc_id,
            "example_id": key[1],
            "request_id": key[2],
            "doc": memory.decode(attrs["doc"]),
            "arguments": memory.decode(attrs["arguments"]),
            "response": response,
            "recovered_from": sources[index],
        }
        if key in diagnostics:
            record["cache_update_metrics"] = diagnostics[key]
        records.append(record)
    return {
        "schema": "gcw.live_korbench_recovery.v1",
        "partial": True,
        "total_requests": len(requests),
        "completed_requests": len(records),
        "metrics_recovered_requests": sum(
            "cache_update_metrics" in row for row in records
        ),
        "metrics_error": diagnostic_error,
        "samples": records,
    }


def capture(pid, py_spy, seconds=30):
    executable = str(Path(f"/proc/{pid}/exe").resolve())
    version = subprocess.check_output(
        [
            executable,
            "-I",
            "-S",
            "-c",
            "import sys; print(sys.version_info[:2])",
        ],
        text=True,
        timeout=5,
    ).strip()
    if (
        version != "(3, 12)"
        or struct.calcsize("P") != 8
        or sys.byteorder != "little"
    ):
        raise RecoveryError(
            "Recovery supports only little-endian 64-bit CPython 3.12"
        )
    started = time.monotonic()
    with stopped_process(pid, seconds) as deadline:
        dump = subprocess.run(
            [
                str(py_spy),
                "dump",
                "--pid",
                str(pid),
                "--json",
                "--locals",
                "--nonblocking",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=min(10, seconds - 5),
        )
        traces = json.loads(dump.stdout)
        memory = PythonMemory(pid, deadline)
        try:
            snapshot = extract_snapshot(memory, traces)
        finally:
            memory.close()
    snapshot.update(
        pid=pid,
        hostname=socket.gethostname(),
        captured_at=datetime.now(timezone.utc).isoformat(),
        pause_seconds=time.monotonic() - started,
    )
    return snapshot


def atomic_json(path, payload):
    original_uid, original_gid = os.geteuid(), os.getegid()
    as_user = original_uid == 0 and "SUDO_UID" in os.environ
    name = None
    try:
        if as_user:
            os.setegid(int(os.environ["SUDO_GID"]))
            os.seteuid(int(os.environ["SUDO_UID"]))
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, 0o644)
        os.link(name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            if name is not None:
                Path(name).unlink(missing_ok=True)
        finally:
            if as_user:
                os.seteuid(original_uid)
                os.setegid(original_gid)


def commandline(pid):
    return (
        Path(f"/proc/{pid}/cmdline")
        .read_bytes()
        .rstrip(b"\0")
        .decode()
        .split("\0")
    )


def matching_processes(manifest, selected_pid=None):
    expected = {
        entry["out"]: (model, entry)
        for model in manifest["models"]
        for entry in model["evaluations"]
    }
    matches = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or (
            selected_pid and int(directory.name) != selected_pid
        ):
            continue
        pid = int(directory.name)
        try:
            args = commandline(pid)
            if (
                len(args) < 4
                or Path(args[1]).name != "gcw"
                or args[2:4] != ["eval", "korbench"]
            ):
                continue
            output = args[args.index("--out") + 1]
            if output not in expected:
                continue
            model, entry = expected[output]
            if args[args.index("--checkpoint") + 1] != model["checkpoint"]:
                raise RecoveryError(
                    "Live checkpoint differs from campaign manifest"
                )
            if (
                args[args.index("--enable-thinking") + 1]
                != str(entry["thinking"]).lower()
            ):
                raise RecoveryError(
                    "Live thinking setting differs from campaign manifest"
                )
            matches.append((pid, args, model, entry))
        except (
            FileNotFoundError,
            ProcessLookupError,
            PermissionError,
            ValueError,
        ):
            continue
    return sorted(matches)


def score(path):
    from gcw.evaluation.korbench import process_results
    from gcw.evaluation import korbench_scoring

    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    payload = json.loads(source)
    if payload.get("schema") != "gcw.live_korbench_recovery.v1":
        raise RecoveryError("Unrecognized recovery snapshot")
    scorer_digest = hashlib.sha256(
        Path(korbench_scoring.__file__).read_bytes()
    ).hexdigest()
    expected = payload["campaign_manifest"]["protocol"]["scorer_sha256"]
    if scorer_digest != expected:
        raise RecoveryError("Scorer differs from the campaign's pinned scorer")
    destination = path.with_name(path.stem + ".scored.json")
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous.get("source_sha256") != digest:
            raise RecoveryError(
                "Existing scored snapshot refers to different input"
            )
        print(f"Already scored: {destination}", flush=True)
        return
    groups = {}
    seen = set()
    for sample in payload["samples"]:
        key = (sample["task"], sample["doc_id"])
        if key in seen:
            raise RecoveryError("Duplicate recovered example")
        seen.add(key)
        metrics = process_results(sample["doc"], [sample["response"]])
        sample["scores"] = metrics
        groups.setdefault(sample["task"], []).append(metrics)
    payload["partial_scores"] = {
        task: {
            "count": len(rows),
            **{
                name: sum(row[name] for row in rows) / len(rows)
                for name in rows[0]
            },
        }
        for task, rows in sorted(groups.items())
    }
    payload["source_sha256"] = digest
    payload["scorer_sha256"] = scorer_digest
    atomic_json(destination, payload)
    print(
        json.dumps(
            {
                "path": str(destination),
                "completed": payload["completed_requests"],
                "total": payload["total_requests"],
                "scores": payload["partial_scores"],
            }
        ),
        flush=True,
    )


def archive_completed(manifest_path, destination):
    manifest = json.loads(manifest_path.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    for model in manifest["models"]:
        for entry in model["evaluations"]:
            source = Path(entry["out"])
            target = destination / (source.stem + ".completed.json")
            if target.exists() or not source.exists():
                continue
            try:
                result = json.loads(source.read_text())
            except json.JSONDecodeError:
                continue
            if not result.get("results"):
                continue
            record = {
                "source": str(source),
                "results": result,
                "campaign_manifest": manifest,
            }
            metrics = entry["cache_update_metrics"]
            if metrics:
                metrics_path = Path(metrics)
                if not metrics_path.exists():
                    print(
                        f"Waiting for completed metrics: {metrics_path}",
                        file=sys.stderr,
                    )
                    continue
                record["cache_update_metrics"] = [
                    json.loads(line)
                    for line in metrics_path.read_text().splitlines()
                ]
            try:
                atomic_json(target, record)
            except FileExistsError:
                continue
            print(f"Archived completed evaluation: {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("capture")
    inspect.add_argument("--manifest", required=True, type=Path)
    inspect.add_argument("--out-dir", required=True, type=Path)
    inspect.add_argument("--py-spy", required=True, type=Path)
    inspect.add_argument("--pid", type=int)
    inspect.add_argument("--list", action="store_true")
    scoring = commands.add_parser("score")
    scoring.add_argument("paths", nargs="+", type=Path)
    archive = commands.add_parser("archive-completed")
    archive.add_argument("--manifest", required=True, type=Path)
    archive.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "archive-completed":
        archive_completed(args.manifest, args.out_dir)
        return
    if args.command == "score":
        for path in args.paths:
            score(path)
        return
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    matches = matching_processes(manifest, args.pid)
    if not matches:
        raise SystemExit(
            "No matching live KOR-Bench evaluation processes on this node"
        )
    if args.list:
        for pid, _, model, entry in matches:
            print(f"{pid}: {model['method']} thinking={entry['thinking']}")
        return
    if not args.out_dir.is_dir():
        raise SystemExit(
            "Create --out-dir as your normal user before invoking sudo"
        )
    failures = 0
    for pid, cmd, model, entry in matches:
        label = Path(entry["out"]).stem
        print(f"Capturing {label}, PID {pid}...", flush=True)
        try:
            snapshot = capture(pid, args.py_spy)
            expected = sum(manifest["population"].values())
            if snapshot["total_requests"] != expected:
                raise RecoveryError(
                    "Live request count differs from campaign manifest"
                )
            snapshot.update(
                command=cmd,
                campaign_manifest=manifest,
                manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                method=model["method"],
                thinking=entry["thinking"],
                metrics_expected=entry["cache_update_metrics"] is not None,
            )
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            path = (
                args.out_dir
                / f"{label}.{socket.gethostname()}.{pid}.{stamp}.partial.json"
            )
            atomic_json(path, snapshot)
            print(
                f"Saved {snapshot['completed_requests']}/{expected} responses; {snapshot['metrics_recovered_requests']} metric records; pause {snapshot['pause_seconds']:.2f}s: {path}",
                flush=True,
            )
            if (
                snapshot["metrics_expected"]
                and snapshot["metrics_recovered_requests"]
                != snapshot["completed_requests"]
            ):
                print(
                    "Warning: some recovered responses lack diffusion metrics",
                    file=sys.stderr,
                )
            if snapshot["metrics_error"]:
                print(
                    f"Metric recovery warning: {snapshot['metrics_error']}",
                    file=sys.stderr,
                )
        except Exception as exc:
            failures += 1
            print(
                f"PID {pid}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
