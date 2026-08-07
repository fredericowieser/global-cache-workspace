from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml

import scripts.run_opsd_h200_autoresearch as autoresearch_runner
from gcw.autoresearch import (
    select_disjoint_heldout_indices,
    validate_autoresearch_protocol,
)
from scripts.run_opsd_h200_autoresearch import (
    Campaign,
    _merge_teacher_trace_shards,
    _require_contiguous_prompt_prefix,
    _run_sharded_teacher_precompute,
    _teacher_shard_config,
    _teacher_shard_ranges,
)


def test_teacher_shard_ranges_balance_exact_batch_groups() -> None:
    assert _teacher_shard_ranges(64, 2048, workers=4, batch_size=2) == [
        (64, 560),
        (560, 1056),
        (1056, 1552),
        (1552, 2048),
    ]
    assert _teacher_shard_ranges(64, 70, workers=4, batch_size=2) == [
        (64, 66),
        (66, 68),
        (68, 70),
    ]


@pytest.mark.parametrize(
    ("start", "stop", "workers", "batch_size"),
    [(1, 1, 1, 1), (2, 1, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0)],
)
def test_teacher_shard_ranges_reject_invalid_contract(
    start: int, stop: int, workers: int, batch_size: int
) -> None:
    with pytest.raises(ValueError):
        _teacher_shard_ranges(start, stop, workers=workers, batch_size=batch_size)


def test_contiguous_prompt_prefix_rejects_filtered_source_rows(tmp_path: Path) -> None:
    valid = tmp_path / "valid.jsonl"
    valid.write_text('{"question":"q0"}\n{"question":"q1"}\n')
    _require_contiguous_prompt_prefix(valid, rows=2)

    filtered = tmp_path / "filtered.jsonl"
    filtered.write_text('{"question":"q0"}\n{}\n{"question":"q2"}\n')
    with pytest.raises(RuntimeError, match="contiguous"):
        _require_contiguous_prompt_prefix(filtered, rows=2)


def test_parallel_rejects_zero_exit_with_invalid_output(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({"output_root": str(tmp_path / "campaign")}))
    campaign = Campaign(
        repo=repo,
        protocol_path=protocol_path,
        source_revision="a" * 40,
        deadline_seconds=60,
        verify_source=False,
    )

    with pytest.raises(RuntimeError, match="exhausted retries"):
        campaign.run_parallel(
            {"invalid-success": [sys.executable, "-c", "pass"]},
            max_workers=1,
            max_attempts=1,
            reserve_seconds=0.0,
            success_validators={"invalid-success": lambda: False},
        )


def _write_teacher_shard(
    root: Path,
    indices: list[int],
    *,
    source_revision: str = "a" * 40,
    fingerprint: str = "teacher-fingerprint",
    batch_size: int = 2,
) -> None:
    root.mkdir()
    trace = root / "teacher-traces.jsonl"
    trace.write_text(
        "".join(
            json.dumps(
                {
                    "source_index": index,
                    "prompt_sha256": f"prompt-{index}",
                    "fingerprint": fingerprint,
                    "reasoning_ids": [index + 1],
                    "visible_response_ids": [index + 2],
                    "full_completion_ids": [index + 1, index + 2],
                },
                sort_keys=True,
            )
            + "\n"
            for index in indices
        )
    )
    manifest = {
        "schema_version": 1,
        "kind": "opsd_native_teacher_trace_precompute",
        "source_revision": source_revision,
        "teacher_fingerprint": fingerprint,
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "batch_size": batch_size,
        "records": len(indices),
        "cached": 0,
        "generated": len(indices),
        "valid": len(indices),
        "invalid": 0,
        "completion_tokens": len(indices) * 2,
        "generation_batches": (len(indices) + batch_size - 1) // batch_size,
        "singleton_batches": len(indices) % batch_size,
        "max_observed_batch_size": min(batch_size, len(indices)),
        "elapsed_seconds": 3.0,
        "peak_cuda_memory_bytes": 2_000_000_000,
        "gpu_name": "NVIDIA H200",
    }
    (root / "teacher-precompute.json").write_text(json.dumps(manifest))


def test_merge_teacher_trace_shards_publishes_sorted_exact_closure(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    target = tmp_path / "target"
    _write_teacher_shard(first, [0, 2])
    _write_teacher_shard(second, [1, 3])

    manifest = _merge_teacher_trace_shards(
        [first, second],
        target,
        expected_source_indices=range(4),
        source_revision="a" * 40,
        batch_size=2,
        elapsed_seconds=4.0,
    )

    rows = [
        json.loads(line)
        for line in (target / "teacher-traces.jsonl").read_text().splitlines()
    ]
    assert [row["source_index"] for row in rows] == [0, 1, 2, 3]
    assert manifest["records"] == 4
    assert manifest["valid"] == 4
    assert manifest["worker_count"] == 2
    assert manifest["trace_sha256"] == hashlib.sha256(
        (target / "teacher-traces.jsonl").read_bytes()
    ).hexdigest()
    assert json.loads((target / "teacher-precompute.json").read_text()) == manifest


def test_merge_teacher_extension_separates_historical_accounting(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot"
    current_a = tmp_path / "current-a"
    current_b = tmp_path / "current-b"
    _write_teacher_shard(pilot, [0, 1])
    _write_teacher_shard(current_a, [2])
    _write_teacher_shard(current_b, [3])

    manifest = _merge_teacher_trace_shards(
        [pilot, current_a, current_b],
        tmp_path / "merged",
        expected_source_indices=range(4),
        source_revision="a" * 40,
        batch_size=2,
        elapsed_seconds=5.0,
        historical_shard_count=1,
    )

    assert manifest["worker_count"] == 2
    assert manifest["completion_tokens_per_second"] == 0.8
    assert manifest["peak_cuda_memory_bytes"] == 4_000_000_000
    assert manifest["historical_peak_cuda_memory_bytes"] == 2_000_000_000
    assert manifest["current_peak_cuda_memory_bytes"] == 4_000_000_000


def test_teacher_trace_manifest_pair_publishes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    target = tmp_path / "canonical"
    _write_teacher_shard(first, [0])
    _write_teacher_shard(second, [0])

    _merge_teacher_trace_shards(
        [first],
        target,
        expected_source_indices=range(1),
        source_revision="a" * 40,
        batch_size=2,
        elapsed_seconds=1.0,
    )
    assert target.is_symlink()
    published_version = target.resolve()
    trace_before = (target / "teacher-traces.jsonl").read_bytes()
    manifest_before = (target / "teacher-precompute.json").read_bytes()

    def fail_manifest(*args, **kwargs):
        raise OSError("injected manifest publication failure")

    monkeypatch.setattr(autoresearch_runner, "_atomic_json", fail_manifest)
    with pytest.raises(OSError, match="injected"):
        _merge_teacher_trace_shards(
            [second],
            target,
            expected_source_indices=range(1),
            source_revision="a" * 40,
            batch_size=2,
            elapsed_seconds=1.0,
        )

    assert target.resolve() == published_version
    assert (target / "teacher-traces.jsonl").read_bytes() == trace_before
    assert (target / "teacher-precompute.json").read_bytes() == manifest_before


def test_merge_teacher_trace_shards_rejects_duplicate_indices(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_teacher_shard(first, [0, 1])
    _write_teacher_shard(second, [1, 2])

    with pytest.raises(RuntimeError, match="duplicate teacher trace source index"):
        _merge_teacher_trace_shards(
            [first, second],
            tmp_path / "target",
            expected_source_indices=range(3),
            source_revision="a" * 40,
            batch_size=2,
            elapsed_seconds=4.0,
        )


def test_merge_teacher_trace_shards_rejects_malformed_trajectory(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shard"
    _write_teacher_shard(shard, [0])
    trace = shard / "teacher-traces.jsonl"
    payload = json.loads(trace.read_text())
    payload["full_completion_ids"] = [999]
    trace.write_text(json.dumps(payload, sort_keys=True) + "\n")
    manifest_path = shard / "teacher-precompute.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="trajectory"):
        _merge_teacher_trace_shards(
            [shard],
            tmp_path / "merged",
            expected_source_indices=range(1),
            source_revision="a" * 40,
            batch_size=2,
            elapsed_seconds=1.0,
        )


def test_teacher_shard_config_isolates_range_without_mutating_base(tmp_path: Path) -> None:
    base = {
        "output_dir": "/campaign/teacher-cache",
        "data": {"sft_skip_rows": 0, "sft_take_rows": 2048},
        "wandb": {"enabled": True},
    }
    before = json.loads(json.dumps(base))

    config = _teacher_shard_config(
        base,
        output_dir=tmp_path / "worker-0",
        start_row=64,
        stop_row=560,
    )

    assert base == before
    assert config["output_dir"] == str(tmp_path / "worker-0")
    assert config["data"]["sft_skip_rows"] == 64
    assert config["data"]["sft_take_rows"] == 496
    assert config["wandb"]["enabled"] is False


def test_run_sharded_teacher_precompute_launches_disjoint_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({"output_root": str(tmp_path / "campaign")}))
    campaign = Campaign(
        repo=repo,
        protocol_path=protocol_path,
        source_revision="a" * 40,
        deadline_seconds=3600,
        verify_source=False,
    )
    captured: dict[str, list[str]] = {}

    def fake_parallel(commands, **kwargs):
        assert kwargs["max_workers"] == 2
        assert kwargs["max_attempts"] == 1
        assert kwargs["maximum_seconds"] == 120.0
        captured.update(commands)
        for command in commands.values():
            config_path = Path(command[command.index("--config") + 1])
            config = yaml.safe_load(config_path.read_text())
            start = int(config["data"]["sft_skip_rows"])
            count = int(config["data"]["sft_take_rows"])
            _write_teacher_shard(
                Path(config["output_dir"]), list(range(start, start + count))
            )
        return 5.0

    monkeypatch.setattr(campaign, "run_parallel", fake_parallel)
    base = {
        "output_dir": "/unused",
        "data": {"sft_skip_rows": 0, "sft_take_rows": 8},
        "wandb": {"enabled": True},
    }

    manifest, elapsed = _run_sharded_teacher_precompute(
        campaign,
        uv="uv",
        base_config=base,
        target_dir=campaign.root / "teacher-cache",
        start_row=0,
        stop_row=8,
        workers=2,
        batch_size=2,
        source_revision="a" * 40,
        phase_name="teacher-pilot",
        reserve_seconds=300.0,
        maximum_seconds=120.0,
    )

    assert elapsed == 5.0
    assert sorted(captured) == ["teacher-pilot-worker-0", "teacher-pilot-worker-1"]
    assert manifest["records"] == 8
    assert manifest["worker_count"] == 2
    assert campaign.state["phases"][-1]["name"] == "teacher-pilot"
    assert campaign.state["phases"][-1]["status"] == "completed"
    assert campaign.state["phases"][-1]["worker_count"] == 2
    assert [
        json.loads(line)["source_index"]
        for line in (campaign.root / "teacher-cache/teacher-traces.jsonl")
        .read_text()
        .splitlines()
    ] == list(range(8))


def test_disjoint_heldout_selection_reserves_integrated_probe_examples() -> None:
    with pytest.raises(RuntimeError, match="minimum held-out"):
        select_disjoint_heldout_indices(
            list(range(22)),
            locked_examples=4,
            candidate_examples=32,
            minimum_examples=8,
            reserved_non_heldout_examples=16,
        )

    valid_indices = list(range(44))
    heldout = select_disjoint_heldout_indices(
        valid_indices,
        locked_examples=4,
        candidate_examples=32,
        minimum_examples=8,
        reserved_non_heldout_examples=16,
    )
    assert heldout == list(range(16, 44))
    assert not set(valid_indices[:16]) & set(heldout)

    with pytest.raises(RuntimeError, match="minimum held-out"):
        select_disjoint_heldout_indices(
            list(range(10)),
            locked_examples=4,
            candidate_examples=32,
            minimum_examples=8,
            reserved_non_heldout_examples=16,
        )


def test_protocol_rejects_unbounded_acceleration_workers() -> None:
    protocol = json.loads(
        Path("configs/protocols/qwen3_06b_opsd_h200_autoresearch_v1.json").read_text()
    )
    base = yaml.safe_load(
        Path(
            "configs/train/opsd/qwen3_06b_opsd_native_base_pragmatic_steps1000_bs16_v1.yaml"
        ).read_text()
    )
    validate_autoresearch_protocol(protocol, base)
    assert protocol["teacher_precompute_initial_rows"] == 128
    assert protocol["integrated_probe_examples"] == 16
    assert protocol["teacher_precompute_workers"] == 4
    assert protocol["capacity_workers"] == 2
    assert protocol["evaluation"]["max_workers"] == 2

    for path, value in (
        (("teacher_precompute_workers",), 0),
        (("teacher_precompute_workers",), 9),
        (("capacity_workers",), 0),
        (("capacity_workers",), 9),
        (("evaluation", "max_workers"), 0),
        (("evaluation", "max_workers"), 9),
    ):
        tampered = json.loads(json.dumps(protocol))
        target = tampered
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match="worker"):
            validate_autoresearch_protocol(tampered, base)
