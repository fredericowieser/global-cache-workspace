from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist


LOG = logging.getLogger("gcw")


def setup_logging() -> None:
    logging.basicConfig(level=os.environ.get("GCW_LOGLEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")


def base_dir() -> Path:
    root = Path(os.environ.get("GCW_BASE_DIR", Path.home() / ".cache" / "gcw"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def print0(message: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(message, flush=True)


def dist_info() -> tuple[bool, int, int, int]:
    if dist.is_available() and dist.is_initialized():
        return True, dist.get_rank(), int(os.environ.get("LOCAL_RANK", "0")), dist.get_world_size()
    return False, 0, 0, 1


def init_distributed(kind: str) -> torch.device:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if kind in {"ddp", "fsdp"} and world_size > 1 and "RANK" in os.environ and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    _, _, local_rank, _ = dist_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def autocast_dtype(use_bf16: bool) -> torch.dtype:
    return torch.bfloat16 if use_bf16 and torch.cuda.is_available() else torch.float32
