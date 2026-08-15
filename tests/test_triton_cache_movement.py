from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from gcw.performance.triton import gather_per_cache_head_kv, require_triton_runtime


def test_triton_package_import_is_lazy() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, torch; import gcw.performance.triton; "
            "assert 'gcw.performance.triton._cache_kernels' not in sys.modules; print('lazy')",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lazy"


def test_triton_runtime_and_cache_gather_fail_closed_on_cpu() -> None:
    with pytest.raises(RuntimeError, match="require a CUDA tensor"):
        require_triton_runtime(torch.device("cpu"))

    keys = torch.zeros(2, 2, 1, 4, 8, dtype=torch.bfloat16)
    values = torch.zeros_like(keys)
    indices = torch.tensor([[0, 3], [1, 2]])
    with pytest.raises(ValueError, match="one CUDA device"):
        gather_per_cache_head_kv(keys, values, indices)
