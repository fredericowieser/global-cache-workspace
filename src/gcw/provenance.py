from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from typing import Any

import torch


SourceMember = tuple[str, bytes]
PackageManifest = dict[str, dict[str, int | str]]


def json_bytes(
    payload: Any,
    *,
    allow_nan: bool = False,
    default: Callable[[Any], Any] | None = None,
    ensure_ascii: bool = True,
    separators: tuple[str, str] | None = (",", ":"),
) -> bytes:
    return json.dumps(
        payload,
        allow_nan=allow_nan,
        default=default,
        ensure_ascii=ensure_ascii,
        sort_keys=True,
        separators=separators,
    ).encode("utf-8")


def json_sha256(payload: Any, **kwargs: Any) -> str:
    return hashlib.sha256(json_bytes(payload, **kwargs)).hexdigest()


def canonical_json_bytes(
    payload: Any,
    *,
    allow_nan: bool = False,
    ensure_ascii: bool = True,
) -> bytes:
    return json_bytes(payload, allow_nan=allow_nan, ensure_ascii=ensure_ascii)


def canonical_json_sha256(
    payload: Any,
    *,
    allow_nan: bool = False,
    ensure_ascii: bool = True,
) -> str:
    return json_sha256(payload, allow_nan=allow_nan, ensure_ascii=ensure_ascii)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes and exact bytes with the stable V1 encoding."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prompt_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_prompt_text_sha256(text: str) -> str:
    return prompt_text_sha256(" ".join(text.split()))


def repository_state(
    repository: Path,
    *,
    check_output: Callable[..., bytes] = subprocess.check_output,
) -> tuple[str | None, bool]:
    git = _git_executable()
    if git is None:
        return None, False
    try:
        revision = check_output(
            [git, "rev-parse", "HEAD"],
            cwd=repository,
            stderr=subprocess.DEVNULL,
        ).decode("ascii").strip()
        dirty = check_output(
            [git, "status", "--porcelain"],
            cwd=repository,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return None, False
    return revision, not bool(dirty)


def _logical_path(path: str) -> str:
    logical = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or logical.is_absolute()
        or any(part in ("", ".", "..") for part in logical.parts)
        or logical.as_posix() != path
    ):
        raise ValueError(f"source path must be a normalized relative POSIX path: {path!r}")
    return path


def source_tree_sha256(members: Iterable[SourceMember]) -> str:
    ordered: list[SourceMember] = []
    seen: set[str] = set()
    for path, payload in members:
        logical = _logical_path(path)
        if logical in seen:
            raise ValueError(f"duplicate logical source path: {logical}")
        if not isinstance(payload, bytes):
            raise TypeError(f"source payload must be bytes: {logical}")
        seen.add(logical)
        ordered.append((logical, payload))

    digest = hashlib.sha256()
    for logical, payload in sorted(ordered):
        encoded = logical.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def package_source_members(package_root: Path) -> tuple[SourceMember, ...]:
    package_root = package_root.resolve(strict=True)
    if not package_root.is_dir():
        raise ValueError(f"package root is not a directory: {package_root}")
    package_name = package_root.name
    members: list[SourceMember] = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(package_root).as_posix()
        members.append((f"{package_name}/{relative}", path.read_bytes()))
    return tuple(members)


def package_member_manifest(package_root: Path) -> PackageManifest:
    return {
        path: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for path, payload in package_source_members(package_root)
    }


def verify_package_member_manifest(package_root: Path, expected: PackageManifest) -> None:
    actual = package_member_manifest(package_root)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"package member set differs: missing={missing}, extra={extra}")
    for path in sorted(expected):
        if actual[path] != expected[path]:
            raise ValueError(f"package member differs: {path}")


def find_repository_root(start: Path) -> Path:
    current = start.resolve(strict=True)
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "uv.lock").is_file():
            return candidate
    raise ValueError(f"could not find GCW repository root from: {start}")


def repository_package_root(repository: Path) -> Path:
    roots = (repository / "src" / "gcw", repository / "gcw")
    existing = [path for path in roots if path.is_dir()]
    if len(existing) != 1:
        raise ValueError(f"repository must contain exactly one GCW package root: {existing}")
    return existing[0]


def _relative_file(repository: Path, path: Path) -> tuple[str, bytes]:
    resolved_repository = repository.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_repository).as_posix()
    except ValueError as error:
        raise ValueError(f"source file is outside repository: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"source path is not a file: {path}")
    return _logical_path(relative), resolved.read_bytes()


def _git_executable() -> str | None:
    git = shutil.which("git")
    if git is None and Path("/run/current-system/sw/bin/git").is_file():
        return "/run/current-system/sw/bin/git"
    return git


def _git_identity(
    repository: Path,
    *,
    status_paths: Sequence[str],
    require_git: bool,
    check_output: Callable[..., bytes] = subprocess.check_output,
) -> tuple[str | None, bytes | None]:
    git = _git_executable()
    if git is None:
        if require_git:
            raise RuntimeError("canonical execution requires Git provenance")
        return None, None
    try:
        commit = check_output(
            [git, "rev-parse", "HEAD"],
            cwd=repository,
            stderr=subprocess.DEVNULL,
        ).decode("ascii").strip()
        status = check_output(
            [git, "status", "--porcelain=v1", "--untracked-files=all", "--", *status_paths],
            cwd=repository,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as error:
        if require_git:
            raise RuntimeError("canonical execution requires Git provenance") from error
        return None, None
    return commit, status



def repository_source_identity(
    repository: Path,
    *,
    executed_files: Sequence[Path] = (),
    require_git: bool = False,
    check_output: Callable[..., bytes] = subprocess.check_output,
) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    package_root = repository_package_root(repository)
    package_members = package_source_members(package_root)
    required = (repository / "pyproject.toml", repository / "uv.lock")
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"repository identity source is missing: {missing[0]}")

    declared = tuple(_relative_file(repository, path) for path in executed_files)
    members = package_members + tuple(_relative_file(repository, path) for path in required) + declared
    package_status_path = package_root.relative_to(repository).as_posix()
    status_paths = (
        package_status_path,
        "pyproject.toml",
        "uv.lock",
        *(path for path, _ in declared),
    )
    commit, status = _git_identity(
        repository,
        status_paths=status_paths,
        require_git=require_git,
        check_output=check_output,
    )
    manifest = package_member_manifest(package_root)
    package_manifest_bytes = canonical_json_bytes(manifest)
    return {
        "schema": "gcw.source_code_identity.v2",
        "source_tree_sha256": source_tree_sha256(members),
        "package_tree_sha256": source_tree_sha256(package_members),
        "package_manifest_sha256": hashlib.sha256(package_manifest_bytes).hexdigest(),
        "package_manifest": manifest,
        "executed_files": [path for path, _ in declared],
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "git_status_sha256": None if status is None else hashlib.sha256(status).hexdigest(),
    }
