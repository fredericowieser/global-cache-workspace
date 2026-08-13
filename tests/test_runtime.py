from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gcw.provenance import (
    canonical_json_sha256,
    file_sha256,
    normalized_prompt_text_sha256,
    prompt_text_sha256,
    repository_state,
    tensor_state_sha256,
)
from gcw.runtime import atomic_write_bytes, atomic_write_json, atomic_write_text, fsync_directory


def test_atomic_write_json_preserves_nan_policy_and_custom_overwrite_error(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="Out of range float"):
        atomic_write_json(invalid, {"value": float("nan")}, allow_nan=False)
    assert not invalid.exists()

    existing = tmp_path / "existing.json"
    existing.write_text("original\n")
    with pytest.raises(FileExistsError, match="reviewed evidence"):
        atomic_write_json(
            existing,
            {"value": 2},
            exclusive=True,
            overwrite_error="refusing to overwrite reviewed evidence: {path}",
        )
    assert existing.read_text() == "original\n"


def test_atomic_writers_preserve_exact_bytes_exclusive_mode_and_permissions(tmp_path: Path) -> None:
    binary = tmp_path / "nested" / "payload.bin"
    atomic_write_bytes(binary, b"payload\x00bytes", exclusive=True, mode=0o444)
    assert binary.read_bytes() == b"payload\x00bytes"
    assert binary.stat().st_mode & 0o777 == 0o444

    text = tmp_path / "payload.txt"
    atomic_write_text(text, "exact text", exclusive=True)
    assert text.read_bytes() == b"exact text"

    with pytest.raises(FileExistsError, match="protected output"):
        atomic_write_text(
            text,
            "replacement",
            exclusive=True,
            overwrite_error="refusing to overwrite protected output: {path}",
        )
    assert text.read_bytes() == b"exact text"


def test_file_sha256_hashes_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload\x00bytes")

    assert file_sha256(path) == "9d0b49b81cde2c40b476ddd33e9e74a222720a9ae72a44c52b0f1bbaf69d7238"


def test_canonical_json_sha256_is_order_independent_and_policy_explicit() -> None:
    assert canonical_json_sha256({"b": [2, 1], "a": 3}) == canonical_json_sha256(
        {"a": 3, "b": [2, 1]}
    )
    assert canonical_json_sha256({"text": "£"}, ensure_ascii=False) == (
        "b2ab9db713c25dbf17bee8b24a8e30dd40a93b0b2985ae6cb97da1de0dbb4d37"
    )
    assert canonical_json_sha256({"not_finite": float("nan")}, allow_nan=True) == (
        "d79749486f8aadb22ef327e6da1dd50717b9408b733554f8be8be09baa93264b"
    )
    with pytest.raises(ValueError, match="JSON"):
        canonical_json_sha256({"not_finite": float("nan")})


def test_prompt_and_tensor_hashes_preserve_v1_encodings() -> None:
    state = {"weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}

    assert prompt_text_sha256("a  b\n") != normalized_prompt_text_sha256("a  b\n")
    assert normalized_prompt_text_sha256("a  b\n") == prompt_text_sha256("a b")
    assert tensor_state_sha256(state) == "da6d60be6a8dd647865cbd8f1ab93d7347f1bb4e9e4a6d9ed4f1686d4a88d943"


def test_repository_state_distinguishes_clean_dirty_and_unavailable(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()

    calls: list[tuple[str, ...]] = []

    def clean(command, **_kwargs):
        calls.append(tuple(command[1:]))
        return b"a" * 40 + b"\n" if command[1] == "rev-parse" else b""

    assert repository_state(repository, check_output=clean) == ("a" * 40, True)
    assert calls == [("rev-parse", "HEAD"), ("status", "--porcelain")]

    def dirty(command, **_kwargs):
        return b"a" * 40 + b"\n" if command[1] == "rev-parse" else b" M src/gcw/runtime.py\n"

    assert repository_state(repository, check_output=dirty) == ("a" * 40, False)

    def unavailable(_command, **_kwargs):
        raise OSError("git unavailable")

    assert repository_state(repository, check_output=unavailable) == (None, False)


def test_atomic_write_json_replaces_or_refuses_without_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "nested" / "payload.json"
    atomic_write_json(path, {"value": 1}, indent=2, sort_keys=True)
    assert path.read_text() == '{\n  "value": 1\n}\n'

    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"value": 2}, exclusive=True)
    assert path.read_text() == '{\n  "value": 1\n}\n'

    failed = tmp_path / "failed.json"

    def interrupt(_source: Path, _destination: Path) -> None:
        raise OSError("replace interrupted")

    monkeypatch.setattr("gcw.runtime.os.replace", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        atomic_write_json(failed, {"complete": True})
    assert not failed.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_fsync_directory_closes_descriptor_when_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[int] = []
    monkeypatch.setattr("gcw.runtime.os.open", lambda *_args: 37)
    monkeypatch.setattr("gcw.runtime.os.close", closed.append)

    def fail(_descriptor: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr("gcw.runtime.os.fsync", fail)
    with pytest.raises(OSError, match="fsync failed"):
        fsync_directory(tmp_path)
    assert closed == [37]
