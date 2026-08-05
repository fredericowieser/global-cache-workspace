from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gcw.distill.teacher import NativeTeacherTrajectory


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

    def get_native_trajectory(
        self, *, source_index: int, prompt: str
    ) -> NativeTeacherTrajectory | None:
        entry = self._index.get(source_index)
        if entry is None:
            return None
        prompt_hash = _prompt_hash(prompt)
        if entry.prompt_sha256 != prompt_hash:
            raise ValueError(f"teacher trace prompt hash mismatch for source_index={source_index}")
        with self.path.open("rb") as handle:
            handle.seek(entry.byte_offset)
            payload = _parse_payload(handle.readline(), source=f"source_index={source_index}")
        reasoning_ids = _reasoning_ids(payload, source=f"source_index={source_index}")
        visible_ids = _token_tuple(
            payload, "visible_response_ids", source=f"source_index={source_index}"
        )
        full_ids = _token_tuple(
            payload, "full_completion_ids", source=f"source_index={source_index}"
        )
        if reasoning_ids + visible_ids != full_ids:
            raise ValueError(f"invalid native teacher trajectory for source_index={source_index}")
        return NativeTeacherTrajectory(reasoning_ids, visible_ids, full_ids)

    def put_native_trajectory(
        self,
        *,
        source_index: int,
        prompt: str,
        reasoning_ids: tuple[int, ...],
        visible_response_ids: tuple[int, ...],
        full_completion_ids: tuple[int, ...],
    ) -> None:
        if source_index < 0:
            raise ValueError("source_index must be non-negative")
        _validate_reasoning_ids(reasoning_ids, source=f"source_index={source_index}")
        _validate_token_ids(
            visible_response_ids,
            field="visible_response_ids",
            source=f"source_index={source_index}",
        )
        if reasoning_ids + visible_response_ids != full_completion_ids:
            raise ValueError(f"invalid native teacher trajectory for source_index={source_index}")
        prompt_hash = _prompt_hash(prompt)
        existing = self._index.get(source_index)
        if existing is not None:
            cached = self.get_native_trajectory(source_index=source_index, prompt=prompt)
            expected = NativeTeacherTrajectory(
                reasoning_ids, visible_response_ids, full_completion_ids
            )
            if existing.prompt_sha256 != prompt_hash or cached != expected:
                raise ValueError(f"conflicting teacher trace for source_index={source_index}")
            return
        self._append(
            source_index=source_index,
            prompt_hash=prompt_hash,
            payload={
                "fingerprint": self.fingerprint,
                "source_index": source_index,
                "prompt_sha256": prompt_hash,
                "reasoning_ids": list(reasoning_ids),
                "visible_response_ids": list(visible_response_ids),
                "full_completion_ids": list(full_completion_ids),
            },
        )

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

        self._append(
            source_index=source_index,
            prompt_hash=prompt_hash,
            payload={
                "fingerprint": self.fingerprint,
                "source_index": source_index,
                "prompt_sha256": prompt_hash,
                "reasoning_ids": list(reasoning_ids),
            },
        )

    def _append(
        self, *, source_index: int, prompt_hash: str, payload: dict[str, Any]
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
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


def _token_tuple(payload: dict[str, Any], field: str, *, source: str) -> tuple[int, ...]:
    try:
        token_ids = tuple(int(token_id) for token_id in payload[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid teacher trace {source}") from exc
    _validate_token_ids(token_ids, field=field, source=source)
    return token_ids


def _validate_token_ids(token_ids: tuple[int, ...], *, field: str, source: str) -> None:
    if not token_ids or any(token_id < 0 for token_id in token_ids):
        raise ValueError(f"invalid teacher trace {field} for {source}")


def _validate_reasoning_ids(reasoning_ids: tuple[int, ...], *, source: str) -> None:
    _validate_token_ids(reasoning_ids, field="reasoning_ids", source=source)


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
