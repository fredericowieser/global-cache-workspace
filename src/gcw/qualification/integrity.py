"""Strict integrity envelopes for qualification artifacts.

The SHA-256 digest detects accidental corruption and binds externally anchored
expected bytes.  It is deliberately not described as authentication: a party
that can replace the payload can also calculate a replacement digest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from gcw.provenance import canonical_json_bytes
from gcw.qualification.inputs import capture_file
from gcw.runtime import atomic_write_bytes


class QualificationIntegrityError(ValueError):
    """Raised when an integrity artifact is malformed or inconsistent."""


def integrity_document(
    payload: dict[str, Any],
    *,
    schema: str,
    kind: str,
) -> dict[str, Any]:
    """Return a strict canonical payload envelope and its unkeyed digest."""
    if not isinstance(schema, str) or not schema:
        raise QualificationIntegrityError(
            "integrity schema must be a nonempty string"
        )
    if not isinstance(kind, str) or not kind:
        raise QualificationIntegrityError(
            "integrity kind must be a nonempty string"
        )
    if type(payload) is not dict:
        raise QualificationIntegrityError(
            "integrity payload must be a dictionary"
        )
    payload_bytes = canonical_json_bytes(payload)
    safe_payload = json.loads(payload_bytes)
    return {
        "schema": schema,
        "kind": kind,
        "payload": safe_payload,
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
    }


def verify_integrity_document(
    document: object,
    *,
    expected_schema: str,
    expected_kind: str,
) -> dict[str, Any]:
    """Validate one envelope, including its unique canonical representation."""
    if type(document) is not dict:
        raise QualificationIntegrityError(
            "integrity document must be a dictionary"
        )
    if set(document) != {"schema", "kind", "payload", "payload_sha256"}:
        raise QualificationIntegrityError("integrity document schema differs")
    if (
        document["schema"] != expected_schema
        or document["kind"] != expected_kind
    ):
        raise QualificationIntegrityError("integrity document identity differs")
    payload = document["payload"]
    digest = document["payload_sha256"]
    if (
        type(payload) is not dict
        or type(digest) is not str
        or len(digest) != 64
    ):
        raise QualificationIntegrityError(
            "integrity payload or digest is malformed"
        )
    try:
        int(digest, 16)
    except ValueError as exc:
        raise QualificationIntegrityError(
            "integrity digest is not hexadecimal"
        ) from exc
    payload_bytes = canonical_json_bytes(payload)
    if hashlib.sha256(payload_bytes).hexdigest() != digest:
        raise QualificationIntegrityError("integrity payload digest differs")
    expected = integrity_document(
        payload, schema=expected_schema, kind=expected_kind
    )
    if document != expected:
        raise QualificationIntegrityError("integrity document is noncanonical")
    return payload


def serialize_integrity_document(document: object) -> bytes:
    """Serialize a previously validated generic integrity document."""
    if type(document) is not dict:
        raise QualificationIntegrityError(
            "integrity document must be a dictionary"
        )
    schema = document.get("schema")
    kind = document.get("kind")
    if type(schema) is not str or type(kind) is not str:
        raise QualificationIntegrityError(
            "integrity document identity is malformed"
        )
    verify_integrity_document(
        document, expected_schema=schema, expected_kind=kind
    )
    return canonical_json_bytes(document) + b"\n"


def deserialize_integrity_document(
    payload: bytes,
    *,
    expected_schema: str,
    expected_kind: str,
) -> dict[str, Any]:
    """Decode strict canonical JSON and validate its envelope."""
    if type(payload) is not bytes:
        raise QualificationIntegrityError(
            "serialized integrity document must be bytes"
        )
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationIntegrityError(
            "integrity document is not valid JSON"
        ) from exc
    result = verify_integrity_document(
        document,
        expected_schema=expected_schema,
        expected_kind=expected_kind,
    )
    if payload != canonical_json_bytes(document) + b"\n":
        raise QualificationIntegrityError(
            "serialized integrity document is noncanonical"
        )
    return result


def write_integrity_document(
    path: Path, document: object, *, exclusive: bool = True
) -> None:
    """Atomically write canonical envelope bytes and fsync their directory."""
    atomic_write_bytes(
        path, serialize_integrity_document(document), exclusive=exclusive
    )


def load_integrity_document(
    path: Path,
    *,
    expected_schema: str,
    expected_kind: str,
) -> dict[str, Any]:
    """Read an artifact once, then validate only the captured bytes."""
    try:
        captured = capture_file(path, role="qualification_integrity_artifact")
    except ValueError as exc:
        raise QualificationIntegrityError(
            "integrity artifact is unreadable"
        ) from exc
    return deserialize_integrity_document(
        captured.content,
        expected_schema=expected_schema,
        expected_kind=expected_kind,
    )
