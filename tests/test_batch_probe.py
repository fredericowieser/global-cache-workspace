import io
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from scripts.cluster import probe_batch as probe


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "__file__", str(tmp_path / "scripts/cluster/probe.py"))
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(probe.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path / "triton"))
    config = tmp_path / "train.yaml"
    base = {
        "model": {"variant": "tf", "cache_processor": None},
        "data": {"kind": "random"},
        "distributed": "ddp", "batch_size": 16, "sequence_len": 1024,
        "total_batch_size": 131072, "max_steps": 2000,
        "learning_rate": 0.0001, "output_dir": "runs/original",
    }
    config.write_text(yaml.safe_dump(base))
    return config, tmp_path / "probe", base


def mock_training(monkeypatch, outcomes):
    calls = []

    def launch(command, **kwargs):
        config = yaml.safe_load(Path(command[-1]).read_text())
        calls.append((command, config, kwargs["env"].copy()))
        status = next(outcomes)
        if status == "oom":
            output, code = "torch.OutOfMemoryError: CUDA out of memory\n", 1
        elif status == "error":
            output, code = "RuntimeError: invalid architecture\n", 1
        else:
            loss = "nan" if status == "nan" else "1.0"
            output = "".join(f"step {i}/3 loss={loss}\n" for i in range(1, 4))
            code = 0
        return type("Process", (), {"stdout": io.StringIO(output), "wait": lambda self: code})()

    monkeypatch.setattr(probe.subprocess, "Popen", launch)
    return calls


@pytest.mark.parametrize("gpus,expected", [(1, [128, 64]), (2, [64, 32])])
def test_accumulation_search_and_isolation(campaign, monkeypatch, gpus, expected):
    source, output, original = campaign
    calls = mock_training(monkeypatch, iter(["oom", "success"]))
    assert probe.main([str(source), "--gpus", str(gpus), "--result-dir", str(output)]) == 0
    assert [c[1]["batch_size"] for c in calls] == expected
    report = json.loads((output / "report.json").read_text())[0]
    assert report["optimal_batch_size"] == expected[-1]
    assert report["optimal_grad_accum_steps"] == 2
    assert yaml.safe_load(source.read_text()) == original
    for command, config, environment in calls:
        assert ("torchrun" in command) == (gpus == 2)
        assert config["total_batch_size"] == 131072
        assert config["learning_rate"] == original["learning_rate"]
        assert config["save_final"] is False and config["wandb"]["enabled"] is False
        assert config["output_dir"] != original["output_dir"]
        assert environment["TRITON_CACHE_DIR"] == str(source.parent / "triton")


def test_distributed_rank_cache_isolation(campaign, monkeypatch, tmp_path):
    source, output, _ = campaign
    monkeypatch.setenv("TRITON_CACHE_DIR", "/shared/inherited-cache")
    with monkeypatch.context() as mocked:
        calls = mock_training(mocked, iter(["success"]))
        assert probe.main([str(source), "--gpus", "2", "--result-dir", str(output)]) == 0
    command, _, environment = calls[0]
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$TRITON_CACHE_DIR" "$@"\n')
    python.chmod(0o755)
    launcher = command[command.index("bash"):]
    for rank in (0, 1):
        result = subprocess.run(
            launcher,
            env={**environment, "LOCAL_RANK": str(rank), "PATH": f"{tmp_path}:{os.environ['PATH']}"},
            capture_output=True, text=True, check=True,
        )
        lines = result.stdout.splitlines()
        assert lines[0] == f"{tmp_path}/triton-rank-{rank}"
        assert lines[1:] == ["-m", "gcw.cli", "train", "--config", command[-1]]


def test_non_power_of_two_divisors(campaign, monkeypatch):
    source, output, base = campaign
    base["global_batch_rows"] = 120
    base["total_batch_size"] = 0
    source.write_text(yaml.safe_dump(base))
    calls = mock_training(monkeypatch, iter(["oom", "oom", "success"]))
    assert probe.main([str(source), "--gpus", "2", "--result-dir", str(output)]) == 0
    assert [c[1]["batch_size"] for c in calls] == [60, 30, 20]
    report = json.loads((output / "report.json").read_text())[0]
    assert report["optimal_grad_accum_steps"] == 3
    assert all(c[1]["global_batch_rows"] == 120 for c in calls)


@pytest.mark.parametrize("failure", ["error", "nan"])
def test_non_oom_failures_do_not_retry(campaign, monkeypatch, failure):
    source, output, _ = campaign
    calls = mock_training(monkeypatch, iter([failure]))
    assert probe.main([str(source), "--result-dir", str(output)]) == 1
    assert len(calls) == 1
    report = json.loads((output / "report.json").read_text())[0]
    assert report["optimal_batch_size"] is None
    assert report["attempts"][0]["status"] == "error"


def test_missing_dataset_fails_before_launch(campaign, monkeypatch):
    source, output, base = campaign
    base["data"] = {"kind": "sft", "dataset": "missing.jsonl"}
    source.write_text(yaml.safe_dump(base))
    calls = mock_training(monkeypatch, iter([]))
    with pytest.raises(FileNotFoundError, match="dataset is missing"):
        probe.main([str(source), "--result-dir", str(output)])
    assert not calls and not output.exists()


def test_same_filename_configs_have_separate_outputs(campaign, monkeypatch, tmp_path):
    source, output, base = campaign
    second = tmp_path / "other/train.yaml"
    second.parent.mkdir()
    second.write_text(yaml.safe_dump(base))
    calls = mock_training(monkeypatch, iter(["success", "success"]))
    assert probe.main([str(source), str(second), "--result-dir", str(output)]) == 0
    assert calls[0][1]["output_dir"] != calls[1][1]["output_dir"]
