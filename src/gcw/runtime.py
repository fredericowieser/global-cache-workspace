from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
import torch.distributed as dist


LOG = logging.getLogger("gcw")


def setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("GCW_LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )


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




def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _overwrite_message(path: Path, template: str | None) -> str:
    return template.format(path=path) if template is not None else str(path)


def atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    exclusive: bool = False,
    mode: int | None = None,
    overwrite_error: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and (path.exists() or path.is_symlink()):
        raise FileExistsError(_overwrite_message(path, overwrite_error))
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            temporary.chmod(mode)
        if exclusive:
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise FileExistsError(_overwrite_message(path, overwrite_error)) from None
            temporary.unlink()
        else:
            os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str, **kwargs: Any) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), **kwargs)


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    indent: int | None = None,
    sort_keys: bool = True,
    allow_nan: bool = True,
    exclusive: bool = False,
    mode: int | None = None,
    overwrite_error: str | None = None,
) -> None:
    text = json.dumps(payload, allow_nan=allow_nan, indent=indent, sort_keys=sort_keys) + "\n"
    atomic_write_text(
        path,
        text,
        exclusive=exclusive,
        mode=mode,
        overwrite_error=overwrite_error,
    )
