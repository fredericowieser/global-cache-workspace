from __future__ import annotations

from configparser import ConfigParser
from importlib import import_module
from importlib.metadata import version
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest


@pytest.mark.parametrize("target", ["sdist", "wheel"])
def test_distribution_excludes_initialized_research_trees(
    tmp_path: Path, target: str
) -> None:
    pytest.importorskip(
        "hatchling", reason="requires the Hatchling build backend"
    )
    backend = import_module(f"hatchling.builders.{target}")
    builder_class = getattr(backend, f"{target.title()}Builder")
    root = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    shutil.copyfile(root / "pyproject.toml", project / "pyproject.toml")

    retained = {
        "src/gcw/__init__.py",
        "scripts/command.py",
        "examples/example.py",
        "configs/example.yaml",
        "docs/index.md",
        "README.md",
        "LICENSE",
        "uv.lock",
    }
    excluded = {
        "src/gcw/spikes/prototype.py",
        "src/future_research/nested/reference.py",
        "src/future_research/__init__.py",
        "src/unlisted.txt",
        "experiments/result.txt",
        "analysis/result.txt",
        "artifacts/result.txt",
        "data/result.txt",
        "tests/test_dummy.py",
    }
    submodules = ConfigParser()
    submodules.read(root / ".gitmodules")
    for section in submodules.sections():
        path = submodules[section]["path"]
        excluded.update({f"{path}/__init__.py", f"{path}/nested/reference.md"})
    for relative_path in retained | excluded:
        path = project / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Dummy packaging fixture\n")

    artifact = next(
        builder_class(str(project)).build(
            directory=str(tmp_path / "dist"), versions=["standard"]
        )
    )
    if target == "sdist":
        with tarfile.open(artifact) as archive:
            files = {
                PurePosixPath(*PurePosixPath(member.name).parts[1:]).as_posix()
                for member in archive.getmembers()
                if member.isfile()
            }
        assert retained | {"pyproject.toml", "PKG-INFO"} <= files
        assert not files & excluded
        assert all(
            name.startswith("src/gcw/")
            for name in files
            if name.startswith("src/")
        )
    else:
        with zipfile.ZipFile(artifact) as archive:
            files = set(archive.namelist())
        assert "gcw/__init__.py" in files
        assert "gcw/spikes/prototype.py" not in files
        assert all(
            name.startswith("gcw/")
            or PurePosixPath(name).parts[0].endswith(".dist-info")
            for name in files
        )


def test_package_version_and_curated_subpackage_exports() -> None:
    import gcw
    import gcw.performance as performance

    assert gcw.__version__ == version("gcw")
    assert gcw.__all__ == [
        "Backbone",
        "CacheProcessorBase",
        "CacheProcessorConfig",
        "EvalConfig",
        "LanguageModel",
        "ModelConfig",
        "TF",
        "TrainConfig",
        "build_model",
    ]
    assert performance.__all__ == (
        "AdaptiveBatchingConfig",
        "KVScoreSidecarConfig",
        "NoisyDiffusionBackendConfig",
        "OptimizedStepConfig",
        "PerformanceConfig",
        "PhysicalBatchBandConfig",
        "ReferenceLossConfig",
        "TritonCacheMovementConfig",
    )


def test_removed_compatibility_modules_are_not_packaged() -> None:
    legacy_modules = (
        "gcw.train",
        "gcw.eval",
        "gcw.training_checkpoint",
        "gcw.models.bgt",
        "gcw.models.bt",
        "gcw.models.gemma4",
        "gcw.models.llama",
        "gcw.models.qwen3",
        "gcw.models.backbones.backend",
        "gcw.models.cache_processors.backend",
        "gcw.models.cache_processors.longshot_scores",
        "gcw.models.losses",
    )
    for module in legacy_modules:
        try:
            import_module(module)
        except ModuleNotFoundError:
            continue
        raise AssertionError(
            f"legacy compatibility module still exists: {module}"
        )


def test_cpu_smoke_example_runs_from_the_source_layout() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(root / "scripts/quickstart/cpu_smoke.py")],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "TF loss=" in result.stdout
    assert "BGT loss=" in result.stdout
    assert "Longshot admitted" in result.stdout
    assert "Longshot rejected" in result.stdout


def test_flow_smoke_example_runs_from_the_source_layout() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/quickstart/flow_smoke.py"),
            "--device",
            "cpu",
            "--flow-steps",
            "2",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"architecture": "flow_bgt"' in result.stdout
    assert '"architecture": "flow_longshot"' in result.stdout


def test_flow_cpu_gate_uses_the_src_layout_mypy_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/cluster/slurm/flow_cpu_gate.sbatch").read_text()

    assert (
        '"$PYTHON" -m mypy \\\n  src/gcw \\\n  scripts/quickstart/cpu_smoke.py \\\n  scripts/quickstart/flow_smoke.py'
        in script
    )
    assert '"$PYTHON" -m mypy gcw' not in script
