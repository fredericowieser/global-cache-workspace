from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gcw.runtime import base_dir, dist_info

SHARD_DTYPE = np.uint16
SHARD_SCHEMA = "gcw.fineweb.shards.v1"


class ShardDataset:
    def __init__(self, data_dir: str | Path):
        self.path = Path(data_dir).expanduser()
        if (self.path / ".fineweb-incomplete").exists():
            raise ValueError(f"Incomplete FineWeb preparation: {self.path}")
        manifest_path = self.path / "manifest.json"
        self.manifest = None
        self.manifest_sha256 = None
        self.dtype = np.dtype(SHARD_DTYPE)
        self.inventory: dict[str, list[dict]] = {}
        if manifest_path.exists():
            raw = manifest_path.read_bytes()
            self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
            self.manifest = json.loads(raw)
            if (
                not isinstance(self.manifest, dict)
                or self.manifest.get("schema") != SHARD_SCHEMA
            ):
                raise ValueError("Unsupported FineWeb shard manifest schema")
            if self.manifest.get("dtype") not in {"<u2", "<u4"}:
                raise ValueError("FineWeb shard dtype must be <u2 or <u4")
            self.dtype = np.dtype(self.manifest["dtype"])
            identity = self.manifest.get("tokenizer", {})
            if (
                not isinstance(identity, dict)
                or not isinstance(identity.get("source"), str)
                or not identity["source"]
            ):
                raise ValueError("Invalid FineWeb tokenizer source")
            maximum = identity.get("max_token_id")
            eos = identity.get("eos_token_id")
            if (
                not isinstance(maximum, int)
                or not isinstance(eos, int)
                or not 0 <= eos <= maximum <= np.iinfo(self.dtype).max
                or not self._valid_digest(identity.get("sha256"))
            ):
                raise ValueError("Invalid FineWeb tokenizer metadata")
        for split in ("train", "val"):
            if self.manifest is None:
                entries = [
                    {
                        "name": Path(path).name,
                        "tokens": Path(path).stat().st_size
                        // self.dtype.itemsize,
                    }
                    for path in list_shards(self.path, split)
                ]
            else:
                inventory = self.manifest.get("shards")
                if not isinstance(inventory, dict):
                    raise ValueError("Missing FineWeb shard inventory")
                entries = inventory.get(split)
                if not isinstance(entries, list):
                    raise ValueError(f"Missing FineWeb {split} inventory")
            names = set()
            for entry in entries:
                if not isinstance(entry, dict):
                    raise TypeError(f"Invalid FineWeb {split} shard entry")
                name = entry.get("name", "")
                tokens = entry.get("tokens")
                if (
                    not isinstance(name, str)
                    or Path(name).name != name
                    or not name.startswith(f"{split}_")
                    or not name.endswith(".bin")
                    or name in names
                    or not isinstance(tokens, int)
                    or tokens <= 0
                ):
                    raise ValueError(f"Invalid FineWeb {split} shard entry")
                names.add(name)
                path = self.path / name
                if (
                    not path.is_file()
                    or path.stat().st_size != tokens * self.dtype.itemsize
                ):
                    raise ValueError(
                        f"Missing or truncated FineWeb shard: {path}"
                    )
                if self.manifest is not None and not self._valid_digest(
                    entry.get("sha256")
                ):
                    raise ValueError(f"Invalid FineWeb shard checksum: {name}")
            if names != {
                Path(path).name for path in list_shards(self.path, split)
            }:
                raise ValueError(
                    f"FineWeb {split} inventory differs from manifest"
                )
            self.inventory[split] = entries

    @staticmethod
    def _valid_digest(value) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    def validate_tokenizer(self, name: str, vocab_size: int) -> None:
        from gcw.data.tokenizers import (
            get_tokenizer,
            tokenizer_content_identity,
        )

        if self.manifest is None:
            if name != "cosmo2":
                raise ValueError(
                    "Legacy FineWeb shards require the Cosmo2 tokenizer"
                )
            return
        identity = tokenizer_content_identity(get_tokenizer(name))
        if identity != {
            key: self.manifest["tokenizer"][key] for key in identity
        }:
            raise ValueError(
                "FineWeb tokenizer does not match the shard manifest"
            )
        if identity["max_token_id"] >= vocab_size:
            raise ValueError(
                "Model vocabulary cannot represent FineWeb token IDs"
            )

    def shards(self, split: str, micro_tokens: int) -> list[Path]:
        entries = self.inventory[split]
        if not entries:
            raise FileNotFoundError(f"No {split} shards found in {self.path}")
        if micro_tokens <= 0 or not any(
            entry["tokens"] > micro_tokens for entry in entries
        ):
            raise ValueError(
                f"No {split} shard can supply a complete global microbatch"
            )
        return [self.path / entry["name"] for entry in entries]


def default_data_dir() -> Path:
    return base_dir() / "fineweb_edu_dedup_cosmo2"


def shard_path(data_dir: str | Path, split: str, index: int) -> Path:
    return Path(data_dir) / f"{split}_{index:06d}.bin"


def list_shards(data_dir: str | Path, split: str) -> list[str]:
    return sorted(glob.glob(str(Path(data_dir) / f"{split}_*.bin")))


def read_shard(path: str | Path, dtype=SHARD_DTYPE) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="r")


class ShardWriter:
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        shard_tokens: int,
        *,
        dtype=SHARD_DTYPE,
    ):
        if shard_tokens <= 0:
            raise ValueError("shard_tokens must be positive")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_tokens = shard_tokens
        self.index = 0
        self.total_tokens = 0
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("<u2"), np.dtype("<u4")):
            raise ValueError("Shard storage requires uint16 or uint32")
        self.inventory: list[dict] = []
        self._buf: list[np.ndarray] = []
        self._buf_len = 0

    def add_tokens(self, ids) -> None:
        arr = np.asarray(ids)
        if arr.size == 0:
            return
        if (
            arr.ndim != 1
            or arr.dtype.kind not in "iu"
            or arr.min() < 0
            or arr.max() > np.iinfo(self.dtype).max
        ):
            raise ValueError(
                "Token IDs must be nonnegative integers within the shard dtype range"
            )
        arr = arr.astype(self.dtype)
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
        path = shard_path(self.data_dir, self.split, self.index)
        with path.open("xb") as handle:
            out.tofile(handle)
        self.inventory.append(
            {
                "name": path.name,
                "tokens": n,
                "sha256": hashlib.sha256(out).hexdigest(),
            }
        )
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
    dataset = ShardDataset(data_dir)
    _, rank, _, world = dist_info()
    step_tokens = batch_size * sequence_len * world
    shards = dataset.shards(split, step_tokens)
    rank_offset = rank * batch_size * sequence_len
    epoch = 0
    while True:
        for shard_file in shards:
            shard = read_shard(shard_file, dataset.dtype)
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
