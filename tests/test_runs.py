from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gcw.config import ModelConfig, TrainConfig, WandbConfig
from gcw.runs import RunRecord
from gcw.tracking import RunTracker


@pytest.mark.parametrize("horizon", [0, 2])
@pytest.mark.parametrize("name", [None, "explicit-name"])
@pytest.mark.parametrize("tags", [[], ["user-tag"]])
def test_training_tracker_preserves_names_and_tags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    horizon: int,
    name: str | None,
    tags: list[str],
) -> None:
    monkeypatch.setenv("GCW_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    monkeypatch.setenv("GCW_WANDB_REQUIRED", "1")
    run = Mock(id="fake-id", url="https://wandb.invalid/fake-id", summary={})
    init = Mock(return_value=run)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    config = TrainConfig(
        model=ModelConfig(variant="bgt"),
        output_dir=str(tmp_path / "unchanged-name"),
        truncated_bptt_windows=horizon,
        wandb=WandbConfig(enabled=True, name=name, tags=list(tags)),
    )

    tracker = RunTracker.for_training(config, {})
    expected_name = name or "unchanged-name"
    assert tracker.record.paths.root.name.startswith(f"{expected_name}--")
    init.assert_called_once()
    kwargs = init.call_args.kwargs
    assert kwargs["name"] == expected_name
    assert kwargs["tags"] == config.wandb.tags == tags
    assert run.summary["truncated_bptt_windows"] == horizon
    assert run.summary["training_method"] == (
        "truncated_bptt" if horizon else "full_bptt"
    )
    assert kwargs["config"].get("truncated_bptt_windows", 0) == horizon
    tracker.finish()


def test_run_record_writes_logs_and_keeps_state_external(
    tmp_path: Path,
) -> None:
    state = tmp_path / "mutable" / "checkpoint"
    record = RunRecord.create(
        kind="training",
        experiment_name="tiny experiment",
        config={"model": "tf", "seed": 24},
        state_path=state,
        data_root=tmp_path / "data",
        execution_id="attempt-1",
        repository=tmp_path,
    )

    record.append_metrics({"train/loss": 1.25}, step=1)
    record.append_log("step 1")
    record.bind_wandb(
        run_id="abc", url="https://wandb.invalid/abc", mode="offline"
    )
    record.finish(summary={"train/loss": 1.25})

    assert record.paths.root.name == "tiny-experiment--attempt-1"
    yaml_text = record.paths.config.read_text()
    assert "seed: 24" in yaml_text
    assert json.loads(record.paths.metrics_stream.read_text())["step"] == 1
    assert json.loads(record.paths.status.read_text())["state"] == "succeeded"
    assert json.loads(record.paths.state.read_text())[
        "mutable_state_path"
    ] == str(state)
    assert not any(
        path.suffix in {".pt", ".bin", ".safetensors"}
        for path in record.paths.root.rglob("*")
    )
    with pytest.raises(RuntimeError, match="finalized"):
        record.append_log("late")


def test_run_record_rejects_collisions_and_normalizes_non_finite_metrics(
    tmp_path: Path,
) -> None:
    kwargs = {
        "kind": "evaluation",
        "experiment_name": "eval",
        "config": {},
        "state_path": None,
        "data_root": tmp_path,
        "execution_id": "same",
        "repository": tmp_path,
    }
    record = RunRecord.create(**kwargs)
    with pytest.raises(FileExistsError):
        RunRecord.create(**kwargs)
    record.append_metrics({"loss": float("nan")})
    assert json.loads(record.paths.metrics_stream.read_text())["loss"] is None
