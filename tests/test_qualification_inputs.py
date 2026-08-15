from __future__ import annotations

from pathlib import Path

import pytest

from gcw.qualification.inputs import (
    QualificationInputError,
    capture_file,
    capture_source,
    capture_train_config,
)


def test_config_snapshot_never_reopens_authored_path(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    authored = tmp_path / "train.yaml"
    original = (root / "configs/train/smoke_tf.yaml").read_bytes()
    authored.write_bytes(original)
    snapshot = capture_train_config(authored)
    authored.write_text("not: [valid")

    reconstructed = snapshot.load()
    assert snapshot.authored.content == original
    assert snapshot.authored.identity.size_bytes == len(original)
    assert reconstructed.model.variant == "tf"
    assert len(snapshot.resolved_sha256) == 64


def test_source_snapshot_is_sorted_content_addressed_and_materialized(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "b.py").write_text("B = 2\n")
    (source / "pkg").mkdir()
    (source / "pkg/a.py").write_text("A = 1\n")

    snapshot = capture_source(source, ("pkg/a.py", "b.py"))
    assert [member.relative_path for member in snapshot.members] == ["b.py", "pkg/a.py"]
    assert len(snapshot.tree_sha256) == 64

    (source / "b.py").write_text("B = 999\n")
    staged = tmp_path / "staged"
    snapshot.materialize(staged)
    assert (staged / "b.py").read_text() == "B = 2\n"
    assert (staged / "pkg/a.py").read_text() == "A = 1\n"
    with pytest.raises(QualificationInputError, match="already exists"):
        snapshot.materialize(staged)


def test_source_snapshot_rejects_ambiguous_paths_and_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "real.py").write_text("VALUE = 1\n")
    (source / "link.py").symlink_to(source / "real.py")

    with pytest.raises(QualificationInputError, match="unsafe"):
        capture_source(source, ("../real.py",))
    with pytest.raises(QualificationInputError, match="unique"):
        capture_source(source, ("real.py", "real.py"))
    with pytest.raises(QualificationInputError, match="symlink"):
        capture_source(source, ("link.py",))
    with pytest.raises(QualificationInputError, match="cannot open"):
        capture_file(source / "link.py", role="symlink")
