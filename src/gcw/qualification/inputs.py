"""Single-read, content-addressed qualification inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Iterable

from gcw.config import TrainConfig, load_train_config_bytes, to_dict
from gcw.provenance import canonical_json_bytes
from gcw.runtime import atomic_write_bytes, fsync_directory


class QualificationInputError(ValueError):
    """Raised when a qualification input cannot be captured unambiguously."""


@dataclass(frozen=True)
class BlobIdentity:
    """Path-independent identity for one captured byte payload."""

    role: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class CapturedBlob:
    """One immutable in-memory payload read from its source exactly once."""

    identity: BlobIdentity
    content: bytes


@dataclass(frozen=True)
class ConfigSnapshot:
    """Authored configuration bytes and their resolved canonical form."""

    authored: CapturedBlob
    resolved_bytes: bytes
    resolved_sha256: str

    def load(self) -> TrainConfig:
        """Reconstruct a fresh config from captured bytes, never its source path."""
        return load_train_config_bytes(self.authored.content)


@dataclass(frozen=True)
class SourceMemberSnapshot:
    """One normalized member of a captured source tree."""

    relative_path: str
    mode: int
    blob: CapturedBlob


@dataclass(frozen=True)
class SourceSnapshot:
    """Closed source-member set and deterministic tree identity."""

    members: tuple[SourceMemberSnapshot, ...]
    tree_sha256: str

    def materialize(self, destination: Path) -> None:
        """Create a new source directory solely from captured member bytes."""
        if destination.exists() or destination.is_symlink():
            raise QualificationInputError("source materialization destination already exists")
        destination.mkdir(parents=True, mode=0o700)
        try:
            for member in self.members:
                target = destination.joinpath(*PurePosixPath(member.relative_path).parts)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                atomic_write_bytes(target, member.blob.content, mode=member.mode, exclusive=True)
            fsync_directory(destination)
        except BaseException:
            # Preserve a partially staged tree for forensic inspection. A later
            # run must use a new exclusive destination.
            raise


def capture_file(path: Path, *, role: str) -> CapturedBlob:
    """Read a regular non-symlink file once through one stable descriptor."""
    if not isinstance(role, str) or not role:
        raise QualificationInputError("captured blob role must be a nonempty string")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationInputError(f"cannot open qualification input {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise QualificationInputError(f"qualification input is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise QualificationInputError(f"qualification input changed while being read: {path}")
    content = b"".join(chunks)
    if len(content) != before.st_size:
        raise QualificationInputError(f"qualification input size changed while being read: {path}")
    return CapturedBlob(
        BlobIdentity(role, len(content), hashlib.sha256(content).hexdigest()),
        content,
    )


def capture_train_config(path: Path) -> ConfigSnapshot:
    """Capture authored YAML once and bind its fully resolved configuration."""
    authored = capture_file(path, role="train_config_authored")
    config = load_train_config_bytes(authored.content)
    resolved = canonical_json_bytes(to_dict(config))
    return ConfigSnapshot(authored, resolved, hashlib.sha256(resolved).hexdigest())


def capture_source(root: Path, relative_paths: Iterable[str]) -> SourceSnapshot:
    """Capture a sorted, unique, closed set of source members."""
    if root.is_symlink() or not root.is_dir():
        raise QualificationInputError("source root must be a real directory")
    normalized = tuple(sorted(_normalize_relative_path(value) for value in relative_paths))
    if not normalized or len(set(normalized)) != len(normalized):
        raise QualificationInputError("source member paths must be nonempty and unique")
    members: list[SourceMemberSnapshot] = []
    for relative in normalized:
        source = root.joinpath(*PurePosixPath(relative).parts)
        _reject_symlink_components(root, relative)
        blob = capture_file(source, role=f"source:{relative}")
        mode = stat.S_IMODE(source.stat(follow_symlinks=False).st_mode)
        members.append(SourceMemberSnapshot(relative, mode, blob))
    manifest = [
        {
            "path": member.relative_path,
            "mode": member.mode,
            "size_bytes": member.blob.identity.size_bytes,
            "sha256": member.blob.identity.sha256,
        }
        for member in members
    ]
    tree_sha256 = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return SourceSnapshot(tuple(members), tree_sha256)


def _normalize_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationInputError("source member path must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise QualificationInputError(f"unsafe source member path: {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise QualificationInputError(f"source member path is noncanonical: {value!r}")
    return normalized


def _reject_symlink_components(root: Path, relative: str) -> None:
    current = root
    for component in PurePosixPath(relative).parts:
        current = current / component
        try:
            metadata = current.stat(follow_symlinks=False)
        except OSError as exc:
            raise QualificationInputError(f"source member is missing: {relative}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise QualificationInputError(f"source member traverses a symlink: {relative}")
