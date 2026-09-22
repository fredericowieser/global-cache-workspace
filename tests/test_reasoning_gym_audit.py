import numpy as np
import pytest

from gcw.data._staging import dataset_staging, publish_dataset
from gcw.data import _staging
from scripts.audit_reasoning_gym import truncation_statistics


def test_truncation_audit_distinguishes_eos_loss_and_prompt_only():
    counts = np.asarray([(100, 413, 0), (600, 100, 1), (200, 312, 1)])
    report = truncation_statistics(counts, 512)
    assert report["truncated_rows"] == 2
    assert report["prompt_only_rows"] == 1
    assert report["eos_only_truncated_rows"] == 1
    assert report["removed_total_tokens_truncated_rows"]["mean"] == 94.5
    assert report["removed_total_tokens_truncated_rows"]["std"] == 93.5
    assert report["removed_answer_tokens_truncated_rows"]["total"] == 101
    assert report["lost_answer_token_percent"] == pytest.approx(100 * 101 / 825)
    complete = truncation_statistics(counts, 1024)
    assert complete["truncated_rows"] == 0
    assert complete["removed_total_tokens_truncated_rows"]["mean"] is None


def test_staging_cleans_failed_build_and_keeps_published_data(tmp_path):
    with pytest.raises(RuntimeError, match="failed build"):
        with dataset_staging(tmp_path / "published") as temporary:
            (temporary / "partial.jsonl").write_text("incomplete")
            raise RuntimeError("failed build")
    assert not temporary.exists()
    with dataset_staging(tmp_path / "published") as temporary:
        staging = temporary / "dataset"
        staging.mkdir()
        (staging / "train.jsonl").write_text("complete")
        staging.rename(tmp_path / "published")
    assert not temporary.exists()
    assert (tmp_path / "published" / "train.jsonl").read_text() == "complete"


def test_publication_replaces_only_prepared_datasets(tmp_path):
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "manifest.json").write_text(
        '{"schema":"gcw.reasoning_gym.sft.v1"}'
    )
    (destination / "train.jsonl").write_text("old rows")
    with dataset_staging(destination) as temporary:
        staging = temporary / "dataset"
        staging.mkdir()
        (staging / "train.jsonl").write_text("new rows")
        with pytest.raises(FileExistsError):
            publish_dataset(staging, destination)
        (destination / "unrelated.txt").write_text("keep")
        with pytest.raises(ValueError, match="unrelated"):
            publish_dataset(staging, destination, replace=True)
        assert (destination / "train.jsonl").read_text() == "old rows"
        (destination / "unrelated.txt").unlink()
        publish_dataset(staging, destination, replace=True)
    assert (destination / "train.jsonl").read_text() == "new rows"
    assert list(tmp_path.iterdir()) == [destination]


def test_failed_publication_restores_existing_dataset(tmp_path, monkeypatch):
    from pathlib import Path

    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "manifest.json").write_text(
        '{"schema":"gcw.reasoning_gym.sft.v1"}'
    )
    (destination / "train.jsonl").write_text("old rows")
    rename = Path.rename
    with dataset_staging(destination) as temporary:
        staging = temporary / "dataset"
        staging.mkdir()

        def fail_publish(path, target):
            if path == staging:
                raise OSError("publication failed")
            return rename(path, target)

        monkeypatch.setattr(Path, "rename", fail_publish)
        with pytest.raises(OSError, match="publication failed"):
            publish_dataset(staging, destination, replace=True)
    assert (destination / "train.jsonl").read_text() == "old rows"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("permanent", [False, True])
def test_backup_cleanup_retries_and_reports_published_state(
    tmp_path, monkeypatch, permanent
):
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "manifest.json").write_text(
        '{"schema":"gcw.reasoning_gym.sft.v1"}'
    )
    (destination / "train.jsonl").write_text("previous")
    remove = _staging.shutil.rmtree
    attempts = []

    def locked_backup(path):
        if (path / "train.jsonl").is_file():
            attempts.append(path)
            if permanent or len(attempts) < 3:
                raise PermissionError("locked by another process")
        remove(path)

    monkeypatch.setattr(_staging.shutil, "rmtree", locked_backup)
    monkeypatch.setattr(_staging.time, "sleep", lambda _: None)
    with dataset_staging(destination) as temporary:
        staging = temporary / "dataset"
        staging.mkdir()
        (staging / "train.jsonl").write_text("current")
        if permanent:
            with pytest.raises(
                _staging.PublishedDatasetCleanupError,
                match="regeneration is unnecessary",
            ) as caught:
                publish_dataset(staging, destination, replace=True)
            assert caught.value.output == destination
            assert caught.value.directory == attempts[-1]
            assert (
                caught.value.directory / "train.jsonl"
            ).read_text() == "previous"
        else:
            publish_dataset(staging, destination, replace=True)
            assert not attempts[-1].exists()
    assert len(attempts) == 3
    assert (destination / "train.jsonl").read_text() == "current"


def test_simultaneous_backup_and_staging_cleanup_failures(
    tmp_path, monkeypatch
):
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "manifest.json").write_text(
        '{"schema":"gcw.reasoning_gym.sft.v1"}'
    )
    (destination / "train.jsonl").write_text("previous")

    def locked_directory(path, parent):
        raise PermissionError("locked by another process")

    monkeypatch.setattr(_staging, "_remove_directory", locked_directory)
    with pytest.raises(
        _staging.PublishedDatasetCleanupError, match="superseded backup"
    ) as caught:
        with dataset_staging(destination) as temporary:
            staging = temporary / "dataset"
            staging.mkdir()
            (staging / "train.jsonl").write_text("current")
            publish_dataset(staging, destination, replace=True)
    assert caught.value.output == destination
    assert (caught.value.directory / "train.jsonl").read_text() == "previous"
    assert str(temporary) in str(caught.value.__cause__)
    assert (destination / "train.jsonl").read_text() == "current"
