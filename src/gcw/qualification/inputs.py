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
            raise QualificationInputError(
                "source materialization destination already exists"
            )
        destination.mkdir(parents=True, mode=0o700)
        try:
            for member in self.members:
                target = destination.joinpath(
                    *PurePosixPath(member.relative_path).parts
                )
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                atomic_write_bytes(
                    target,
                    member.blob.content,
                    mode=member.mode,
                    exclusive=True,
                )
            fsync_directory(destination)
        except BaseException:
            # Preserve a partially staged tree for forensic inspection. A later
            # run must use a new exclusive destination.
            raise


def capture_file(path: Path, *, role: str) -> CapturedBlob:
    """Read a regular non-symlink file once through one stable descriptor."""
    _validate_role(role)
    flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationInputError(
            f"cannot open qualification input {path}"
        ) from exc
    try:
        blob, _metadata = _capture_descriptor(
            descriptor, role=role, label=str(path)
        )
        return blob
    finally:
        os.close(descriptor)


def capture_train_config(path: Path) -> ConfigSnapshot:
    """Capture authored YAML once and bind its fully resolved configuration."""
    authored = capture_file(path, role="train_config_authored")
    config = load_train_config_bytes(authored.content)
    resolved = canonical_json_bytes(to_dict(config))
    return ConfigSnapshot(
        authored, resolved, hashlib.sha256(resolved).hexdigest()
    )


def capture_source(root: Path, relative_paths: Iterable[str]) -> SourceSnapshot:
    """Capture a sorted, unique, closed set through one descriptor-rooted tree."""
    normalized = tuple(
        sorted(_normalize_relative_path(value) for value in relative_paths)
    )
    if not normalized or len(set(normalized)) != len(normalized):
        raise QualificationInputError(
            "source member paths must be nonempty and unique"
        )
    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        root_descriptor = os.open(root, root_flags)
    except OSError as exc:
        raise QualificationInputError(
            "source root must be a real directory"
        ) from exc
    try:
        if not stat.S_ISDIR(os.fstat(root_descriptor).st_mode):
            raise QualificationInputError(
                "source root must be a real directory"
            )
        members = [
            _capture_source_member(root_descriptor, relative)
            for relative in normalized
        ]
    finally:
        os.close(root_descriptor)
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
        raise QualificationInputError(
            "source member path must be a nonempty string"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise QualificationInputError(f"unsafe source member path: {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise QualificationInputError(
            f"source member path is noncanonical: {value!r}"
        )
    return normalized


def _validate_role(role: str) -> None:
    if not isinstance(role, str) or not role:
        raise QualificationInputError(
            "captured blob role must be a nonempty string"
        )


def _capture_descriptor(
    descriptor: int,
    *,
    role: str,
    label: str,
) -> tuple[CapturedBlob, os.stat_result]:
    _validate_role(role)
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise QualificationInputError(
            f"qualification input is not a regular file: {label}"
        )
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 8 * 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            value.st_mode,
        )

    if identity(before) != identity(after):
        raise QualificationInputError(
            f"qualification input changed while being read: {label}"
        )
    content = b"".join(chunks)
    if len(content) != before.st_size:
        raise QualificationInputError(
            f"qualification input size changed while being read: {label}"
        )
    return (
        CapturedBlob(
            BlobIdentity(
                role, len(content), hashlib.sha256(content).hexdigest()
            ),
            content,
        ),
        before,
    )


def _capture_source_member(
    root_descriptor: int, relative: str
) -> SourceMemberSnapshot:
    parts = PurePosixPath(relative).parts
    directory = os.dup(root_descriptor)
    flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        for component in parts[:-1]:
            try:
                child = os.open(
                    component,
                    flags | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=directory,
                )
            except OSError as exc:
                raise QualificationInputError(
                    f"source member traverses an unsafe component: {relative}"
                ) from exc
            os.close(directory)
            directory = child
            if not stat.S_ISDIR(os.fstat(directory).st_mode):
                raise QualificationInputError(
                    f"source member traverses a non-directory: {relative}"
                )
        try:
            descriptor = os.open(parts[-1], flags, dir_fd=directory)
        except OSError as exc:
            raise QualificationInputError(
                f"source member is missing, unsafe, or symlinked: {relative}"
            ) from exc
        try:
            blob, metadata = _capture_descriptor(
                descriptor,
                role=f"source:{relative}",
                label=relative,
            )
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)
    return SourceMemberSnapshot(relative, stat.S_IMODE(metadata.st_mode), blob)
