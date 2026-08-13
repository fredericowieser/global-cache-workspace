import json
import subprocess
import sys


def test_probe_eval_reports_partial_progress(tmp_path):
    partial = tmp_path / "partial.json"
    progress = tmp_path / "progress.json"
    partial.write_text(
        json.dumps(
            {"results": {"gsm8k_openmath": {"exact_match,flexible": 0.25}}}
        )
    )
    progress.write_text(
        json.dumps(
            {
                "updated_at": "2026-06-22T00:00:00Z",
                "elapsed_sec": 60,
                "eta_sec": 300,
                "runs": [],
            }
        )
    )

    out = subprocess.check_output(
        [
            sys.executable,
            "scripts/probe_eval.py",
            "--json",
            str(tmp_path / "missing.json"),
            "--partial",
            str(partial),
            "--progress",
            str(progress),
            "--skip-tasks",
            "theoremqa",
        ],
        text=True,
    )

    assert "Status: running" in out
    assert "Completed: 1/5 (gsm8k_openmath)" in out
    assert "ETA: 5m 0s" in out
