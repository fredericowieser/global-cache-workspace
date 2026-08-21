"""Keyed, single-read authentication for qualification receipts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
from typing import Any

from gcw.provenance import canonical_json_bytes
from gcw.qualification.inputs import capture_file
from gcw.runtime import atomic_write_bytes

_RECEIPT_DOMAIN = b"gcw.qualification.receipt.v1\0"
_RECEIPT_KEYS = {
    "schema",
    "kind",
    "algorithm",
    "key_id",
    "context",
    "payload",
    "hmac_sha256",
}


class QualificationReceiptError(ValueError):
    """Raised when a keyed qualification receipt is unsafe or invalid."""


@dataclass(frozen=True, repr=False)
class QualificationHMACKey:
    key_id: str
    secret: bytes

    def __post_init__(self) -> None:
        if not self.key_id or len(self.secret) < 32:
            raise QualificationReceiptError(
                "qualification HMAC keys require an ID and at least 256 bits"
            )


def capture_hmac_key(
    path: Path, *, expected_key_id: str
) -> QualificationHMACKey:
    """Capture a private raw key through one stable descriptor with strict mode checks."""
    if not expected_key_id:
        raise QualificationReceiptError(
            "expected receipt key ID must be nonempty"
        )
    flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationReceiptError(
            "qualification HMAC key cannot be opened safely"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise QualificationReceiptError(
                "qualification HMAC key must be a regular file"
            )
        if before.st_uid != os.geteuid():
            raise QualificationReceiptError(
                "qualification HMAC key owner differs from verifier UID"
            )
        if stat.S_IMODE(before.st_mode) & 0o077:
            raise QualificationReceiptError(
                "qualification HMAC key must not grant group or other permissions"
            )
        if before.st_nlink != 1:
            raise QualificationReceiptError(
                "qualification HMAC key must have exactly one hard link"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            value.st_mode,
            value.st_uid,
            value.st_nlink,
        )

    if identity(before) != identity(after):
        raise QualificationReceiptError(
            "qualification HMAC key changed while being captured"
        )
    secret = b"".join(chunks)
    if len(secret) != before.st_size or len(secret) < 32:
        raise QualificationReceiptError(
            "qualification HMAC key must contain at least 32 raw bytes"
        )
    return QualificationHMACKey(expected_key_id, secret)


def issue_receipt(
    payload: dict[str, Any],
    *,
    schema: str,
    kind: str,
    context: dict[str, Any],
    key: QualificationHMACKey,
) -> bytes:
    """Create canonical authenticated bytes in a trusted supervisor."""
    _validate_identity(schema, kind)
    safe_context = _closed_json_mapping(context, field="context")
    safe_payload = _closed_json_mapping(payload, field="payload")
    unsigned = {
        "schema": schema,
        "kind": kind,
        "algorithm": "hmac-sha256",
        "key_id": key.key_id,
        "context": safe_context,
        "payload": safe_payload,
    }
    tag = hmac.new(
        key.secret,
        _RECEIPT_DOMAIN + canonical_json_bytes(unsigned),
        hashlib.sha256,
    ).hexdigest()
    return canonical_json_bytes({**unsigned, "hmac_sha256": tag}) + b"\n"


def verify_receipt_bytes(
    receipt: bytes,
    *,
    expected_schema: str,
    expected_kind: str,
    expected_context: dict[str, Any],
    key: QualificationHMACKey,
) -> dict[str, Any]:
    """Verify canonical captured bytes and return only the authenticated payload."""
    _validate_identity(expected_schema, expected_kind)
    expected_context_safe = _closed_json_mapping(
        expected_context, field="expected_context"
    )
    document = _strict_json_document(receipt)
    if set(document) != _RECEIPT_KEYS:
        raise QualificationReceiptError("qualification receipt schema differs")
    if (
        document["schema"] != expected_schema
        or document["kind"] != expected_kind
    ):
        raise QualificationReceiptError(
            "qualification receipt identity differs"
        )
    if (
        document["algorithm"] != "hmac-sha256"
        or document["key_id"] != key.key_id
    ):
        raise QualificationReceiptError(
            "qualification receipt algorithm or key ID differs"
        )
    context = _closed_json_mapping(document["context"], field="context")
    payload = _closed_json_mapping(document["payload"], field="payload")
    if context != expected_context_safe:
        raise QualificationReceiptError("qualification receipt context differs")
    tag = document["hmac_sha256"]
    if type(tag) is not str or len(tag) != 64 or tag != tag.lower():
        raise QualificationReceiptError(
            "qualification receipt tag is malformed"
        )
    try:
        bytes.fromhex(tag)
    except ValueError as exc:
        raise QualificationReceiptError(
            "qualification receipt tag is not hexadecimal"
        ) from exc
    unsigned = {
        name: document[name]
        for name in (
            "schema",
            "kind",
            "algorithm",
            "key_id",
            "context",
            "payload",
        )
    }
    expected_tag = hmac.new(
        key.secret,
        _RECEIPT_DOMAIN + canonical_json_bytes(unsigned),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(tag, expected_tag):
        raise QualificationReceiptError(
            "qualification receipt authentication failed"
        )
    if receipt != canonical_json_bytes(document) + b"\n":
        raise QualificationReceiptError(
            "qualification receipt bytes are noncanonical"
        )
    return payload


def load_and_verify_receipt(
    path: Path,
    *,
    expected_schema: str,
    expected_kind: str,
    expected_context: dict[str, Any],
    key: QualificationHMACKey,
) -> dict[str, Any]:
    """Read a receipt once and verify only that stable captured payload."""
    try:
        captured = capture_file(path, role="qualification_receipt")
    except ValueError as exc:
        raise QualificationReceiptError(
            "qualification receipt cannot be captured safely"
        ) from exc
    return verify_receipt_bytes(
        captured.content,
        expected_schema=expected_schema,
        expected_kind=expected_kind,
        expected_context=expected_context,
        key=key,
    )


def write_receipt(
    path: Path, receipt: bytes, *, exclusive: bool = True
) -> None:
    """Write already-issued canonical receipt bytes atomically."""
    document = _strict_json_document(receipt)
    if receipt != canonical_json_bytes(document) + b"\n":
        raise QualificationReceiptError(
            "qualification receipt bytes are noncanonical"
        )
    atomic_write_bytes(path, receipt, exclusive=exclusive)


def _validate_identity(schema: str, kind: str) -> None:
    if (
        not isinstance(schema, str)
        or not schema
        or not isinstance(kind, str)
        or not kind
    ):
        raise QualificationReceiptError(
            "qualification receipt schema and kind must be nonempty strings"
        )


def _closed_json_mapping(value: object, *, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise QualificationReceiptError(
            f"qualification receipt {field} must be a dictionary"
        )
    _validate_json_value(value, path=field)
    # Round-trip strips custom mapping/scalar subclasses after exact-type checks.
    return json.loads(
        canonical_json_bytes(value), parse_constant=_reject_constant
    )


def _validate_json_value(value: object, *, path: str) -> None:
    if value is None or type(value) in {str, bool}:
        return
    if type(value) is int:
        if not -(2**63) <= value < 2**63:
            raise QualificationReceiptError(
                f"qualification receipt integer is out of range at {path}"
            )
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for name, item in value.items():
            if type(name) is not str or not name:
                raise QualificationReceiptError(
                    f"qualification receipt key is invalid at {path}"
                )
            _validate_json_value(item, path=f"{path}.{name}")
        return
    raise QualificationReceiptError(
        f"qualification receipt value type is unsupported at {path}"
    )


def _strict_json_document(receipt: bytes) -> dict[str, Any]:
    if type(receipt) is not bytes:
        raise QualificationReceiptError("qualification receipt must be bytes")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise QualificationReceiptError(
                    f"qualification receipt has duplicate key {name!r}"
                )
            result[name] = value
        return result

    try:
        document = json.loads(
            receipt,
            object_pairs_hook=pairs_hook,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationReceiptError(
            "qualification receipt is not strict JSON"
        ) from exc
    if type(document) is not dict:
        raise QualificationReceiptError(
            "qualification receipt document must be a dictionary"
        )
    _validate_json_value(document, path="document")
    return document


def _reject_constant(value: str) -> None:
    raise QualificationReceiptError(
        f"qualification receipt forbids JSON constant {value}"
    )
