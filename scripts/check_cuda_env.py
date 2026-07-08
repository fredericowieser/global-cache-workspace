#!/usr/bin/env python3
"""Fail fast if the PyTorch CUDA wheel environment is internally inconsistent."""

from __future__ import annotations

import ctypes
import importlib.metadata as metadata
import os
import subprocess
import sys
from pathlib import Path


BAD_PACKAGES = [
    "nvidia-cublas",
    "nvidia-cuda-cupti",
    "nvidia-cuda-nvrtc",
    "nvidia-cuda-runtime",
    "nvidia-cudnn-cu13",
    "nvidia-cufft",
    "nvidia-cufile",
    "nvidia-curand",
    "nvidia-cusolver",
    "nvidia-cusparse",
    "nvidia-cusparselt-cu13",
    "nvidia-nccl-cu13",
    "nvidia-nvjitlink",
    "nvidia-nvshmem-cu13",
    "nvidia-nvtx",
]


REQUIRED_LIBS = [
    "nvidia/cudnn/lib/libcudnn.so.9",
    "nvidia/cusparselt/lib/libcusparseLt.so.0",
    "nvidia/nccl/lib/libnccl.so.2",
    "nvidia/nvshmem/lib/libnvshmem_host.so.3",
]


def site_packages() -> Path:
    for path in sys.path:
        candidate = Path(path)
        if candidate.name == "site-packages" and candidate.exists():
            return candidate
    raise SystemExit("Could not locate site-packages on sys.path.")


def installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> int:
    bad_installed = {name: installed_version(name) for name in BAD_PACKAGES if installed_version(name)}
    if bad_installed:
        for name, version in sorted(bad_installed.items()):
            print(f"Unexpected CUDA 13 package installed: {name}=={version}", file=sys.stderr)
        return 1

    root = site_packages()
    missing = [rel for rel in REQUIRED_LIBS if not (root / rel).exists()]
    if missing:
        for rel in missing:
            print(f"Missing required PyTorch CUDA library: {root / rel}", file=sys.stderr)
        return 1

    try:
        import torch
    except Exception as exc:
        print(f"Failed to import torch: {exc}", file=sys.stderr)
        return 1

    if torch.version.cuda != "12.8":
        print(f"Expected torch CUDA runtime 12.8, got {torch.version.cuda}", file=sys.stderr)
        return 1

    nccl_lib = root / "nvidia/nccl/lib/libnccl.so.2"
    nccl = ctypes.CDLL(os.fspath(nccl_lib))
    version = ctypes.c_int()
    rc = nccl.ncclGetVersion(ctypes.byref(version))
    if rc != 0:
        print(f"ncclGetVersion failed with return code {rc}", file=sys.stderr)
        return 1
    if version.value != 22809:
        print(f"Expected NCCL version code 22809, got {version.value}", file=sys.stderr)
        return 1

    strings = subprocess.run(
        ["strings", os.fspath(nccl_lib)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout
    if "cuda13" in strings.lower():
        print(f"NCCL library appears to be a CUDA 13 build: {nccl_lib}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
