import hashlib
import json

import pytest

from scripts.subset_reasoning_gym import prepare_subset


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    rows = [
        {
            "task": task,
            "split": "train",
            "source_id": f"{task}:{index}",
            "experiment_order": index,
            "prompt_tokens": 100,
            "target_tokens": 1900 if index % 2 else 100,
            "sequence_tokens": 2000 if index % 2 else 200,
        }
        for task, count in (("a", 12), ("b", 6), ("c", 6))
        for index in range(count)
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (root / "train.jsonl").write_bytes(payload)
    manifest = {
        "files": {
            "train": {
                "file": "train.jsonl",
                "rows": len(rows),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            "validation": {},
            "test": {},
        },
        "tokenizer": "fixture",
        "tokenizer_identity_sha256": "fixture",
        "format": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_subset_proportions_identity_and_reproducibility(source, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    report = prepare_subset(source, first, 12, 24)
    assert report == prepare_subset(source, second, 12, 24)
    assert report["task_counts"] == {"a": 6, "b": 3, "c": 3}
    payload = (first / "train.jsonl").read_bytes()
    assert payload == (second / "train.jsonl").read_bytes()
    selected = payload.splitlines()
    assert len(selected) == len(set(selected)) == 12
    assert set(selected) <= set(
        (source / "train.jsonl").read_bytes().splitlines()
    )
    assert hashlib.sha256(payload).hexdigest() == report["sha256"]
    truncated = sum(
        json.loads(row)["sequence_tokens"] > 1536 for row in selected
    )
    assert report["audit"]["truncated_rows"] == truncated
    assert report["audit"]["removed_tokens_mean_truncated"] == 464
    assert report["audit"]["prompt_only_rows"] == 0


def test_subset_rounding_and_seed(source, tmp_path):
    first = prepare_subset(source, tmp_path / "first", 7, 24)
    second = prepare_subset(source, tmp_path / "second", 7, 25)
    assert sum(first["task_counts"].values()) == 7
    assert first["task_counts"] == {"a": 3, "b": 2, "c": 2}
    assert first["sha256"] != second["sha256"]


def test_subset_audit_cap_does_not_change_selection(source, tmp_path):
    first = prepare_subset(source, tmp_path / "first", 12, 24)
    second = prepare_subset(
        source, tmp_path / "second", 12, 24, sequence_len=2048
    )
    assert first["sha256"] == second["sha256"]
    assert second["audit"]["sequence_len"] == 2048
    assert second["audit"]["truncated_rows"] == 0
    assert second["audit"]["removed_tokens_mean_truncated"] is None
    with pytest.raises(ValueError, match="sequence_len"):
        prepare_subset(source, tmp_path / "invalid", 12, 24, sequence_len=0)


def test_subset_rejects_changed_source(source, tmp_path):
    with (source / "train.jsonl").open("ab") as handle:
        handle.write(
            (source / "train.jsonl").read_bytes().splitlines()[0] + b"\n"
        )
    destination = tmp_path / "subset"
    with pytest.raises(ValueError, match="manifest"):
        prepare_subset(source, destination, 12, 24)
    assert not destination.exists()


def test_subset_refuses_overwrite_and_oversampling(source, tmp_path):
    with pytest.raises(FileExistsError):
        prepare_subset(source, source, 12, 24)
    with pytest.raises(ValueError, match="more rows"):
        prepare_subset(source, tmp_path / "oversized", 25, 24)
