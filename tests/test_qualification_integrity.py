from __future__ import annotations

import json

import pytest

from gcw.qualification.integrity import (
    QualificationIntegrityError,
    deserialize_integrity_document,
    integrity_document,
    load_integrity_document,
    serialize_integrity_document,
    verify_integrity_document,
    write_integrity_document,
)

_SCHEMA = "gcw.test.integrity.v1"
_KIND = "gcw.test"


def test_integrity_document_round_trip_is_canonical_and_exclusive(tmp_path) -> None:
    document = integrity_document({"b": [2, 1], "a": 3}, schema=_SCHEMA, kind=_KIND)
    assert verify_integrity_document(
        document, expected_schema=_SCHEMA, expected_kind=_KIND
    ) == {"a": 3, "b": [2, 1]}
    payload = serialize_integrity_document(document)
    assert payload.endswith(b"\n")
    assert deserialize_integrity_document(
        payload, expected_schema=_SCHEMA, expected_kind=_KIND
    ) == {"a": 3, "b": [2, 1]}

    path = tmp_path / "receipt.json"
    write_integrity_document(path, document)
    assert load_integrity_document(
        path, expected_schema=_SCHEMA, expected_kind=_KIND
    ) == {"a": 3, "b": [2, 1]}
    with pytest.raises(FileExistsError):
        write_integrity_document(path, document)


def test_integrity_document_rejects_corruption_and_noncanonical_bytes() -> None:
    document = integrity_document({"value": 1}, schema=_SCHEMA, kind=_KIND)
    altered = dict(document)
    altered["payload"] = {"value": 2}
    with pytest.raises(QualificationIntegrityError, match="digest differs"):
        verify_integrity_document(altered, expected_schema=_SCHEMA, expected_kind=_KIND)

    pretty = json.dumps(document, indent=2, sort_keys=True).encode() + b"\n"
    with pytest.raises(QualificationIntegrityError, match="noncanonical"):
        deserialize_integrity_document(pretty, expected_schema=_SCHEMA, expected_kind=_KIND)

    with pytest.raises(ValueError, match="Out of range float"):
        integrity_document({"value": float("nan")}, schema=_SCHEMA, kind=_KIND)


def test_integrity_envelope_does_not_claim_authentication() -> None:
    original = integrity_document({"claim": "one"}, schema=_SCHEMA, kind=_KIND)
    replacement = integrity_document({"claim": "two"}, schema=_SCHEMA, kind=_KIND)
    assert original["payload_sha256"] != replacement["payload_sha256"]
    # A writer that replaces the payload can produce another internally valid
    # digest. Independent expected bytes or a trusted key remain necessary.
    assert verify_integrity_document(
        replacement, expected_schema=_SCHEMA, expected_kind=_KIND
    ) == {"claim": "two"}
