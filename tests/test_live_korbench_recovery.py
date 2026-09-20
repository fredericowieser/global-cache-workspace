import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analysis/recover_live_korbench.py"
)
spec = importlib.util.spec_from_file_location("live_recovery", SCRIPT)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)

pytestmark = pytest.mark.skipif(
    sys.version_info[:2] != (3, 12) or sys.platform != "linux",
    reason="CPython 3.12 Linux recovery",
)


@pytest.fixture
def memory():
    reader = recovery.PythonMemory(os.getpid(), time.monotonic() + 10)
    yield reader
    reader.close()


def test_python_memory_decodes_builtin_values_and_instance_storage(memory):
    values = [
        None,
        True,
        False,
        0,
        1,
        -123456789123456789,
        1 << 100,
        3.14,
        "",
        "ascii",
        "caf\u00e9",
        "\u6d4b\u8bd5",
        "\U0001f680",
        [1, None, "x"],
        ("a", 4),
        {("a", "0", "0:0"): {"values": [0.1, 3.0], "applications": 2}},
    ]
    for value in values:
        assert memory.decode(id(value)) == value

    class Managed:
        pass

    for instance in (Managed(), types.SimpleNamespace()):
        instance.answer = "\u6d4b\u8bd5"
        attrs = memory.attributes(id(instance))
        assert memory.decode(attrs["answer"]) == "\u6d4b\u8bd5"
        instance.__dict__  # materialize managed dictionary
        assert (
            memory.decode(memory.attributes(id(instance))["answer"])
            == "\u6d4b\u8bd5"
        )

    def closure():
        return values

    assert memory.decode(id(closure.__closure__[0])) == values


@pytest.mark.parametrize("expose_metric_keys", [True, False])
def test_extracts_outs_pending_group_and_metric_linkage(
    memory, expose_metric_keys
):
    requests = [
        types.SimpleNamespace(
            task_name="korbench_logic",
            doc_id=i,
            doc={"answer": "yes"},
            arguments=(f"prompt{i}", {}),
        )
        for i in range(3)
    ]
    outs = ["first", None, None]
    keys = [("korbench_logic", str(i), "0:0") for i in range(3)]
    metrics = {
        keys[0]: {
            "1:0:value": {
                "sum": [1.0, 2.0, 1.5, 0.5],
                "applications": 1,
                "heads": 2,
            }
        }
    }
    evaluator = types.SimpleNamespace(
        update_metrics=types.SimpleNamespace(requests=metrics)
    )
    results = ["second"]
    rows = [(1, [1]), (2, [2])]
    filename = "/test/gcw/evaluation/lm_eval_adapter.py"
    traces = [
        {
            "frames": [
                {
                    "filename": filename,
                    "name": "GCWEvalLM._run_batches",
                    "locals": [
                        {"name": "results", "addr": id(results)},
                        {"name": "rows", "addr": id(rows)},
                    ],
                },
                {
                    "filename": filename,
                    "name": "GCWEvalLM.generate_until",
                    "locals": [
                        {"name": name, "addr": id(value)}
                        for name, value in [
                            ("requests", requests),
                            ("outs", outs),
                            ("metric_keys", keys),
                            ("self", evaluator),
                        ]
                    ],
                },
            ]
        }
    ]
    if not expose_metric_keys:
        traces[0]["frames"][1]["locals"] = [
            item
            for item in traces[0]["frames"][1]["locals"]
            if item["name"] != "metric_keys"
        ]
    result = recovery.extract_snapshot(memory, traces)
    assert result["completed_requests"] == 2
    assert result["total_requests"] == 3
    assert result["metrics_recovered_requests"] == 1
    assert result["metrics_error"] is None
    assert [row["response"] for row in result["samples"]] == ["first", "second"]
    assert (
        result["samples"][1]["recovered_from"]
        == "current_group_completed_batches"
    )


@pytest.fixture
def child(tmp_path):
    source = """
import ctypes, os, time
from types import SimpleNamespace
ctypes.CDLL(None).prctl(0x59616d61, ctypes.c_ulong(-1).value, 0, 0, 0)

def _run_batches(rows, run_batch):
    results = ["second \\u6d4b\\u8bd5"]
    run_batch(rows)
    return results

def generate_until(requests, self):
    outs = ["first", None, None]
    metric_keys = [("korbench_logic", str(i), "0:0") for i in range(3)]
    def run_batch(rows):
        captured = metric_keys, self
        print("ready", flush=True)
        time.sleep(120)
        return captured
    _run_batches([(1, [1]), (2, [2])], run_batch)
    return outs

requests = [SimpleNamespace(task_name="korbench_logic", doc_id=i, doc={"answer": "yes"}, arguments=(f"prompt{i}", {})) for i in range(3)]
metrics = {("korbench_logic", "0", "0:0"): {"1:0:value": {"sum": [1., 2., 1.5, .5], "applications": 1, "heads": 2}}}
generate_until(requests, SimpleNamespace(update_metrics=SimpleNamespace(requests=metrics)))
"""
    launcher = (
        "exec(compile("
        + repr(source)
        + ", '/test/gcw/evaluation/lm_eval_adapter.py', 'exec'))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", launcher], stdout=subprocess.PIPE, text=True
    )
    assert process.stdout.readline().strip() == "ready"
    yield process
    process.send_signal(signal.SIGCONT)
    process.terminate()
    process.wait(timeout=5)


def test_py_spy_capture_child_without_sudo(child):
    py_spy = Path.home() / ".local/bin/py-spy"
    if not py_spy.exists():
        pytest.skip("py-spy not installed")
    traces = json.loads(
        subprocess.check_output(
            [
                str(py_spy),
                "dump",
                "--pid",
                str(child.pid),
                "--json",
                "--locals",
                "--nonblocking",
            ],
            text=True,
        )
    )
    frame = next(
        frame
        for trace in traces
        for frame in trace["frames"]
        if frame["name"].split(".")[-1] == "generate_until"
    )
    assert "metric_keys" not in {item["name"] for item in frame["locals"]}
    snapshot = recovery.capture(child.pid, py_spy)
    assert snapshot["completed_requests"] == 2
    assert snapshot["metrics_recovered_requests"] == 1
    assert snapshot["samples"][1]["response"] == "second \u6d4b\u8bd5"
    assert recovery.process_identity(child.pid)[1] not in {"T", "t"}


def test_resume_on_extraction_failure(child):
    with pytest.raises(ValueError):
        with recovery.stopped_process(child.pid):
            assert recovery.process_identity(child.pid)[1] == "T"
            raise ValueError("failure")
    assert recovery.process_identity(child.pid)[1] not in {"T", "t"}


def test_atomic_json_exclusive(tmp_path):
    path = tmp_path / "snapshot.json"
    recovery.atomic_json(path, {"value": "\u6d4b\u8bd5"})
    assert json.loads(path.read_text()) == {"value": "\u6d4b\u8bd5"}
    with pytest.raises(FileExistsError):
        recovery.atomic_json(path, {"value": "overwrite"})
    assert json.loads(path.read_text()) == {"value": "\u6d4b\u8bd5"}


def test_watchdog_resumes_after_recovery_process_is_killed(child):
    launcher = f"""
import importlib.util, os, signal
spec = importlib.util.spec_from_file_location("recovery", {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with module.stopped_process({child.pid}, seconds=10):
    os.kill(os.getpid(), signal.SIGKILL)
"""
    owner = subprocess.Popen([sys.executable, "-c", launcher])
    assert owner.wait(timeout=5) == -signal.SIGKILL
    deadline = time.monotonic() + 3
    while recovery.process_identity(child.pid)[1] in {"T", "t"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_watchdog_resumes_after_timeout(child):
    with recovery.stopped_process(child.pid, seconds=1):
        deadline = time.monotonic() + 3
        while recovery.process_identity(child.pid)[1] in {"T", "t"}:
            assert time.monotonic() < deadline
            time.sleep(0.01)


def test_large_population_decode_with_diffusion_metrics(memory):
    from dataclasses import dataclass

    @dataclass
    class Request:
        task_name: str
        doc_id: int
        doc: dict
        arguments: tuple

    requests = [
        Request(
            "korbench_logic",
            i,
            {"answer": "yes", "prompt": "rule \u6d4b\u8bd5 " * 300},
            ("prompt " * 500, {}),
        )
        for i in range(1250)
    ]
    outs = [
        "response \u6d4b\u8bd5 " * 1000 if i < 1200 else None
        for i in range(1250)
    ]
    keys = [("korbench_logic", str(i), "0:0") for i in range(1250)]
    metrics = {
        keys[i]: {
            f"1:{step}:value": {
                "sum": [1.0, 2.0, 1.5, 0.5],
                "applications": 2000,
                "heads": 16,
            }
            for step in range(8)
        }
        for i in range(1200)
    }

    class Evaluator:
        pass

    evaluator = Evaluator()
    evaluator.update_metrics = Evaluator()
    evaluator.update_metrics.requests = metrics
    traces = [
        {
            "frames": [
                {
                    "filename": "/test/gcw/evaluation/lm_eval_adapter.py",
                    "name": "generate_until",
                    "locals": [
                        {"name": name, "addr": id(value)}
                        for name, value in [
                            ("requests", requests),
                            ("outs", outs),
                            ("metric_keys", keys),
                            ("self", evaluator),
                        ]
                    ],
                }
            ]
        }
    ]
    result = recovery.extract_snapshot(memory, traces)
    assert (
        result["completed_requests"]
        == result["metrics_recovered_requests"]
        == 1200
    )
    assert result["samples"][1199]["response"] == outs[1199]
    assert result["samples"][1199]["doc"] == requests[1199].doc


def test_capture_resumes_if_py_spy_fails(child):
    with pytest.raises(FileNotFoundError):
        recovery.capture(child.pid, Path("/nonexistent/py-spy"))
    assert recovery.process_identity(child.pid)[1] not in {"T", "t"}


def test_score_recovered_korbench_and_preserve_raw_snapshot(tmp_path):
    from gcw.evaluation import korbench_scoring
    import hashlib

    root = SCRIPT.parents[2]
    with (root / "data/eval-data/korbench/operation.jsonl").open() as stream:
        doc = json.loads(stream.readline())
    payload = {
        "schema": "gcw.live_korbench_recovery.v1",
        "partial": True,
        "completed_requests": 1,
        "total_requests": 1250,
        "campaign_manifest": {
            "protocol": {
                "scorer_sha256": hashlib.sha256(
                    Path(korbench_scoring.__file__).read_bytes()
                ).hexdigest()
            }
        },
        "samples": [
            {
                "task": "korbench_operation",
                "doc_id": 0,
                "doc": doc,
                "response": doc["answer"],
            }
        ],
    }
    path = tmp_path / "run.partial.json"
    recovery.atomic_json(path, payload)
    original = path.read_bytes()
    recovery.score(path)
    scored = json.loads((tmp_path / "run.partial.scored.json").read_text())
    assert scored["partial_scores"]["korbench_operation"] == {
        "count": 1,
        "accuracy": 1.0,
        "parse_failure": 0.0,
    }
    recovery.score(path)
    assert path.read_bytes() == original
    payload["campaign_manifest"]["protocol"]["scorer_sha256"] = "invalid"
    path.write_text(json.dumps(payload))
    with pytest.raises(recovery.RecoveryError, match="pinned scorer"):
        recovery.score(path)


def test_archive_completed_results_and_metrics(tmp_path):
    source = tmp_path / "result.json"
    metrics = tmp_path / "metrics.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "evaluations": [
                            {
                                "out": str(source),
                                "cache_update_metrics": str(metrics),
                            }
                        ]
                    }
                ]
            }
        )
    )
    destination = tmp_path / "archive"
    recovery.archive_completed(manifest_path, destination)
    assert list(destination.iterdir()) == []
    source.write_text('{"results":')
    recovery.archive_completed(manifest_path, destination)
    assert list(destination.iterdir()) == []
    source.write_text(
        json.dumps({"results": {"korbench_logic": {"accuracy,none": 1.0}}})
    )
    metrics.write_text('{"example_id":"0"}\n')
    recovery.archive_completed(manifest_path, destination)
    target = destination / "result.completed.json"
    record = json.loads(target.read_text())
    assert record["results"] == json.loads(source.read_text())
    assert record["cache_update_metrics"] == [{"example_id": "0"}]
    recovery.archive_completed(manifest_path, destination)


def test_shared_write_restores_identity_on_failure(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("SUDO_UID", "1234")
    monkeypatch.setenv("SUDO_GID", "5678")
    monkeypatch.setattr(recovery.os, "geteuid", lambda: 0)
    monkeypatch.setattr(recovery.os, "getegid", lambda: 0)
    monkeypatch.setattr(
        recovery.os, "seteuid", lambda uid: calls.append(("uid", uid))
    )
    monkeypatch.setattr(
        recovery.os, "setegid", lambda gid: calls.append(("gid", gid))
    )
    with pytest.raises(ValueError):
        recovery.atomic_json(tmp_path / "bad.json", {"value": float("nan")})
    assert calls == [("gid", 5678), ("uid", 1234), ("uid", 0), ("gid", 0)]
    assert list(tmp_path.iterdir()) == []
