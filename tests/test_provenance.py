from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from gcw.provenance import (
    find_repository_root,
    package_member_manifest,
    package_source_members,
    repository_source_identity,
    source_tree_sha256,
    verify_package_member_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repository, text=True).strip()


def _initialise_repository(repository: Path) -> None:
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "gcw-tests@example.invalid")
    _git(repository, "config", "user.name", "GCW Tests")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")


def test_source_tree_digest_is_independent_of_physical_src_layout(tmp_path: Path) -> None:
    flat = tmp_path / "flat" / "gcw"
    src = tmp_path / "src-layout" / "src" / "gcw"
    for package in (flat, src):
        (package / "models").mkdir(parents=True)
        (package / "__init__.py").write_bytes(b"from gcw.models import Model\n")
        (package / "models" / "__init__.py").write_bytes(b"class Model: ...\n")
        (package / "py.typed").write_bytes(b"")

    flat_members = package_source_members(flat)
    src_members = package_source_members(src)

    assert flat_members == src_members
    assert source_tree_sha256(flat_members) == source_tree_sha256(src_members)
    assert [path for path, _ in flat_members] == [
        "gcw/__init__.py",
        "gcw/models/__init__.py",
        "gcw/py.typed",
    ]


@pytest.mark.parametrize("package_relative", [Path("gcw"), Path("src/gcw")])
def test_repository_root_discovery_is_layout_independent(
    tmp_path: Path,
    package_relative: Path,
) -> None:
    repository = tmp_path / "repository"
    module = repository / package_relative / "models" / "model.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(b"")
    (repository / "pyproject.toml").write_bytes(b"")
    (repository / "uv.lock").write_bytes(b"")

    assert find_repository_root(module) == repository.resolve()


def test_source_tree_digest_uses_unambiguous_framing_and_rejects_duplicates() -> None:
    first = source_tree_sha256((("gcw/a", b"bc"),))
    second = source_tree_sha256((("gcw/ab", b"c"),))

    assert first != second
    with pytest.raises(ValueError, match="duplicate logical source path"):
        source_tree_sha256((("gcw/a.py", b"one"), ("gcw/a.py", b"two")))
    with pytest.raises(ValueError, match="relative POSIX"):
        source_tree_sha256((("../outside.py", b"unsafe"),))


def test_package_member_manifest_verifies_exact_installed_subset(tmp_path: Path) -> None:
    package = tmp_path / "gcw"
    package.mkdir()
    (package / "__init__.py").write_bytes(b"VERSION = 1\n")
    manifest = package_member_manifest(package)

    verify_package_member_manifest(package, manifest)
    (package / "__init__.py").write_bytes(b"VERSION = 2\n")
    with pytest.raises(ValueError, match="package member differs"):
        verify_package_member_manifest(package, manifest)

    (package / "__init__.py").write_bytes(b"VERSION = 1\n")
    (package / "unexpected.py").write_bytes(b"pass\n")
    with pytest.raises(ValueError, match="package member set differs"):
        verify_package_member_manifest(package, manifest)


def test_repository_source_identity_v2_binds_declared_script_and_git_state(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    package = repository / "gcw"
    scripts = repository / "scripts"
    package.mkdir(parents=True)
    scripts.mkdir()
    (package / "__init__.py").write_bytes(b"VALUE = 1\n")
    (scripts / "run.py").write_bytes(b"print('one')\n")
    (scripts / "unrelated.py").write_bytes(b"print('ignored')\n")
    (repository / "pyproject.toml").write_bytes(b"[project]\nname='fixture'\n")
    (repository / "uv.lock").write_bytes(b"version = 1\n")
    _initialise_repository(repository)

    identity = repository_source_identity(
        repository,
        executed_files=(repository / "scripts" / "run.py",),
        require_git=True,
    )

    assert identity["schema"] == "gcw.source_code_identity.v2"
    assert identity["git_commit"] == _git(repository, "rev-parse", "HEAD")
    assert identity["git_dirty"] is False
    assert identity["executed_files"] == ["scripts/run.py"]
    assert identity["package_manifest"] == package_member_manifest(package)

    (scripts / "unrelated.py").write_bytes(b"print('changed but undeclared')\n")
    unrelated = repository_source_identity(
        repository,
        executed_files=(repository / "scripts" / "run.py",),
        require_git=True,
    )
    assert unrelated["source_tree_sha256"] == identity["source_tree_sha256"]
    assert unrelated["git_status_sha256"] == identity["git_status_sha256"]

    (scripts / "run.py").write_bytes(b"print('two')\n")
    changed = repository_source_identity(
        repository,
        executed_files=(repository / "scripts" / "run.py",),
        require_git=True,
    )
    assert changed["source_tree_sha256"] != identity["source_tree_sha256"]
    assert changed["git_dirty"] is True
    assert changed["git_status_sha256"] != identity["git_status_sha256"]


def test_repository_source_identity_requires_git_when_requested(tmp_path: Path) -> None:
    repository = tmp_path / "not-a-repository"
    (repository / "gcw").mkdir(parents=True)
    (repository / "gcw" / "__init__.py").write_bytes(b"")
    (repository / "pyproject.toml").write_bytes(b"")
    (repository / "uv.lock").write_bytes(b"")

    with pytest.raises(RuntimeError, match="requires Git provenance"):
        repository_source_identity(repository, require_git=True)



def test_package_manifest_has_canonical_json_stable_digest(tmp_path: Path) -> None:
    package = tmp_path / "gcw"
    package.mkdir()
    payload = b"x = 1\n"
    (package / "module.py").write_bytes(payload)

    manifest = package_member_manifest(package)

    assert manifest == {
        "gcw/module.py": {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == hashlib.sha256(encoded).hexdigest()
