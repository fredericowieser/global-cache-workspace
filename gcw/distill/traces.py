from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TeacherTraceIndexEntry:
    prompt_sha256: str
    byte_offset: int


class TeacherTraceStore:
    """Append-only, fingerprinted teacher-reasoning cache with a compact offset index."""

    def __init__(self, path: str | Path, *, fingerprint: str) -> None:
        if not fingerprint:
            raise ValueError("teacher trace fingerprint must not be empty")
        self.path = Path(path)
        self.fingerprint = fingerprint
        self._index: dict[int, TeacherTraceIndexEntry] = {}
        self._unsynced_writes = 0
        if self.path.exists():
            self._load_index()

    def get(self, *, source_index: int, prompt: str) -> tuple[int, ...] | None:
        entry = self._index.get(source_index)
        if entry is None:
            return None
        prompt_hash = _prompt_hash(prompt)
        if entry.prompt_sha256 != prompt_hash:
            raise ValueError(f"teacher trace prompt hash mismatch for source_index={source_index}")
        with self.path.open("rb") as handle:
            handle.seek(entry.byte_offset)
            payload = _parse_payload(handle.readline(), source=f"source_index={source_index}")
        return _reasoning_ids(payload, source=f"source_index={source_index}")

    def put(self, *, source_index: int, prompt: str, reasoning_ids: tuple[int, ...]) -> None:
        if source_index < 0:
            raise ValueError("source_index must be non-negative")
        _validate_reasoning_ids(reasoning_ids, source=f"source_index={source_index}")
        prompt_hash = _prompt_hash(prompt)
        existing = self._index.get(source_index)
        if existing is not None:
            if existing.prompt_sha256 != prompt_hash or self.get(source_index=source_index, prompt=prompt) != reasoning_ids:
                raise ValueError(f"conflicting teacher trace for source_index={source_index}")
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": self.fingerprint,
            "source_index": source_index,
            "prompt_sha256": prompt_hash,
            "reasoning_ids": list(reasoning_ids),
        }
        encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        with self.path.open("ab") as handle:
            byte_offset = handle.tell()
            handle.write(encoded)
            handle.flush()
            self._unsynced_writes += 1
            if self._unsynced_writes >= 64:
                os.fsync(handle.fileno())
                self._unsynced_writes = 0
        self._index[source_index] = TeacherTraceIndexEntry(prompt_hash, byte_offset)

    def sync(self) -> None:
        if self._unsynced_writes == 0:
            return
        with self.path.open("ab") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        self._unsynced_writes = 0

    def _load_index(self) -> None:
        with self.path.open("rb") as handle:
            line_number = 0
            while True:
                byte_offset = handle.tell()
                line = handle.readline()
                if not line:
                    return
                line_number += 1
                if not line.strip():
                    continue
                source = f"row {line_number}"
                payload = _parse_payload(line, source=source)
                fingerprint = str(payload.get("fingerprint", ""))
                if fingerprint != self.fingerprint:
                    raise ValueError(
                        f"teacher trace fingerprint mismatch at {source}: "
                        f"expected {self.fingerprint}, got {fingerprint}"
                    )
                try:
                    source_index = int(payload["source_index"])
                    prompt_sha256 = str(payload["prompt_sha256"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid teacher trace {source}") from exc
                _reasoning_ids(payload, source=source)
                if source_index < 0 or len(prompt_sha256) != 64:
                    raise ValueError(f"invalid teacher trace {source}")
                entry = TeacherTraceIndexEntry(prompt_sha256, byte_offset)
                existing = self._index.get(source_index)
                if existing is not None:
                    raise ValueError(f"duplicate teacher trace source_index={source_index} at {source}")
                self._index[source_index] = entry


def _parse_payload(line: bytes, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid teacher trace {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid teacher trace {source}")
    return payload


def _reasoning_ids(payload: dict[str, Any], *, source: str) -> tuple[int, ...]:
    try:
        reasoning_ids = tuple(int(token_id) for token_id in payload["reasoning_ids"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid teacher trace {source}") from exc
    _validate_reasoning_ids(reasoning_ids, source=source)
    return reasoning_ids


def _validate_reasoning_ids(reasoning_ids: tuple[int, ...], *, source: str) -> None:
    if not reasoning_ids or any(token_id < 0 for token_id in reasoning_ids):
        raise ValueError(f"invalid teacher trace reasoning_ids for {source}")


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
