from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

from gcw.common import base_dir, dist_info

SHARD_DTYPE = np.uint16


def default_data_dir() -> Path:
    return base_dir() / "fineweb_edu_dedup_cosmo2"


def shard_path(data_dir: str | Path, split: str, index: int) -> Path:
    return Path(data_dir) / f"{split}_{index:06d}.bin"


def list_shards(data_dir: str | Path, split: str) -> list[str]:
    return sorted(glob.glob(str(Path(data_dir) / f"{split}_*.bin")))


def read_shard(path: str | Path) -> np.memmap:
    return np.memmap(path, dtype=SHARD_DTYPE, mode="r")


class ShardWriter:
    def __init__(self, data_dir: str | Path, split: str, shard_tokens: int):
        if shard_tokens <= 0:
            raise ValueError("shard_tokens must be positive")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_tokens = shard_tokens
        self.index = 0
        self.total_tokens = 0
        self._buf: list[np.ndarray] = []
        self._buf_len = 0

    def add_tokens(self, ids) -> None:
        arr = np.asarray(ids, dtype=SHARD_DTYPE)
        if arr.size == 0:
            return
        self._buf.append(arr)
        self._buf_len += arr.size
        self.total_tokens += arr.size
        while self._buf_len >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def close(self) -> int:
        if self._buf_len:
            self._flush(self._buf_len)
        return self.index

    def _flush(self, n: int) -> None:
        cat = np.concatenate(self._buf) if len(self._buf) > 1 else self._buf[0]
        out, rest = cat[:n], cat[n:]
        out.tofile(shard_path(self.data_dir, self.split, self.index))
        self.index += 1
        self._buf = [rest] if rest.size else []
        self._buf_len = rest.size


def data_loader(
    batch_size: int,
    sequence_len: int,
    split: str = "train",
    device: torch.device | str = "cpu",
    data_dir: str | Path | None = None,
    infinite: bool = True,
):
    data_dir = (
        Path(data_dir).expanduser()
        if data_dir is not None
        else default_data_dir()
    )
    shards = list_shards(data_dir, split)
    if not shards:
        raise FileNotFoundError(f"No {split} shards found in {data_dir}")
    _, rank, _, world = dist_info()
    step_tokens = batch_size * sequence_len * world
    rank_offset = rank * batch_size * sequence_len
    epoch = 0
    while True:
        for shard_file in shards:
            shard = read_shard(shard_file)
            steps = max(0, (len(shard) - 1) // step_tokens)
            for step in range(steps):
                base = step * step_tokens + rank_offset
                buf = torch.from_numpy(
                    shard[base : base + batch_size * sequence_len + 1].astype(
                        np.int64
                    )
                )
                x = buf[:-1].view(batch_size, sequence_len).to(device)
                y = buf[1:].view(batch_size, sequence_len).to(device)
                yield x, y, {
                    "epoch": epoch,
                    "shard": shard_file,
                    "step_in_shard": step + 1,
                }
        epoch += 1
        if not infinite:
            return
