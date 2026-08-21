from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gcw.qualification.receipts import (
    QualificationHMACKey,
    QualificationReceiptError,
    capture_hmac_key,
    issue_receipt,
    load_and_verify_receipt,
    verify_receipt_bytes,
    write_receipt,
)

_SCHEMA = "gcw.optimized_step_receipt.v1"
_KIND = "combined_v1"
_CONTEXT = {
    "accelerator": "h200",
    "gpu_uuid": "GPU-authority",
    "kernel_manifest_sha256": "a" * 64,
    "qualification_id": "challenge-17",
    "source_tree_sha256": "b" * 64,
}
_PAYLOAD = {"admitted": True, "evidence_sha256": ["c" * 64], "steps": 3}


def _key() -> QualificationHMACKey:
    return QualificationHMACKey("qualification-key-v1", bytes(range(32)))


def _receipt() -> bytes:
    return issue_receipt(
        _PAYLOAD, schema=_SCHEMA, kind=_KIND, context=_CONTEXT, key=_key()
    )


def test_authenticated_receipt_is_canonical_context_bound_and_single_read(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    path = tmp_path / "receipt.json"
    write_receipt(path, receipt)
    assert receipt.endswith(b"\n")
    assert (
        load_and_verify_receipt(
            path,
            expected_schema=_SCHEMA,
            expected_kind=_KIND,
            expected_context=_CONTEXT,
            key=_key(),
        )
        == _PAYLOAD
    )

    forged = json.loads(receipt)
    forged["payload"]["steps"] = 4
    forged_bytes = (
        json.dumps(forged, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    path.write_bytes(forged_bytes)
    with pytest.raises(
        QualificationReceiptError, match="authentication failed"
    ):
        load_and_verify_receipt(
            path,
            expected_schema=_SCHEMA,
            expected_kind=_KIND,
            expected_context=_CONTEXT,
            key=_key(),
        )


def test_receipt_rejects_context_transplant_key_change_and_noncanonical_bytes() -> (
    None
):
    receipt = _receipt()
    with pytest.raises(QualificationReceiptError, match="context differs"):
        verify_receipt_bytes(
            receipt,
            expected_schema=_SCHEMA,
            expected_kind=_KIND,
            expected_context={**_CONTEXT, "gpu_uuid": "GPU-other"},
            key=_key(),
        )
    with pytest.raises(QualificationReceiptError, match="key ID differs"):
        verify_receipt_bytes(
            receipt,
            expected_schema=_SCHEMA,
            expected_kind=_KIND,
            expected_context=_CONTEXT,
            key=QualificationHMACKey("other-key", bytes(range(32))),
        )
    with pytest.raises(QualificationReceiptError, match="noncanonical"):
        verify_receipt_bytes(
            receipt.replace(b'"admitted":true', b'"admitted": true'),
            expected_schema=_SCHEMA,
            expected_kind=_KIND,
            expected_context=_CONTEXT,
            key=_key(),
        )


def test_receipt_rejects_duplicate_keys_floats_and_oversized_integers() -> None:
    duplicate = b'{"schema":"a","schema":"b"}\n'
    with pytest.raises(QualificationReceiptError, match="duplicate key"):
        verify_receipt_bytes(
            duplicate,
            expected_schema=_SCHEMA,
            expected_kind=_KIND,
            expected_context=_CONTEXT,
            key=_key(),
        )
    with pytest.raises(QualificationReceiptError, match="unsupported"):
        issue_receipt(
            {"metric": 1.5},
            schema=_SCHEMA,
            kind=_KIND,
            context=_CONTEXT,
            key=_key(),
        )
    with pytest.raises(QualificationReceiptError, match="out of range"):
        issue_receipt(
            {"counter": 2**63},
            schema=_SCHEMA,
            kind=_KIND,
            context=_CONTEXT,
            key=_key(),
        )


def test_private_key_capture_rejects_permissions_symlinks_and_hardlinks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "key.bin"
    path.write_bytes(os.urandom(32))
    path.chmod(0o600)
    key = capture_hmac_key(path, expected_key_id="key-v1")
    assert key.key_id == "key-v1"
    assert "secret" not in repr(key)

    path.chmod(0o640)
    with pytest.raises(QualificationReceiptError, match="group or other"):
        capture_hmac_key(path, expected_key_id="key-v1")
    path.chmod(0o600)
    link = tmp_path / "key-link.bin"
    os.link(path, link)
    with pytest.raises(QualificationReceiptError, match="hard link"):
        capture_hmac_key(path, expected_key_id="key-v1")
    link.unlink()
    symlink = tmp_path / "key-symlink.bin"
    symlink.symlink_to(path)
    with pytest.raises(QualificationReceiptError, match="opened safely"):
        capture_hmac_key(symlink, expected_key_id="key-v1")
