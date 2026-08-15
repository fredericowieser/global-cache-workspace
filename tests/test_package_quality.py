from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys


def test_package_version_and_curated_subpackage_exports() -> None:
    import gcw
    import gcw.analysis as analysis
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
    assert analysis.__all__ == (
        "HeadMeanAblator",
        "UnigramBaseline",
        "deterministic_masks",
        "synergy_from_losses",
    )
    assert performance.__all__ == (
        "AdaptiveBatchingConfig",
        "PerformanceConfig",
        "PhysicalBatchBandConfig",
        "ReferenceLossConfig",
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
        raise AssertionError(f"legacy compatibility module still exists: {module}")


def test_cpu_smoke_example_runs_from_the_source_layout() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(root / "examples/cpu_smoke.py")],
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


def test_flow_cpu_gate_uses_the_src_layout_mypy_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/cluster/slurm/flow_cpu_gate.sbatch").read_text()

    assert '"$PYTHON" -m mypy \\\n  src/gcw/architecture.py' in script
    assert '"$PYTHON" -m mypy gcw' not in script
