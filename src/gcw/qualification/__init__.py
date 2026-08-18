"""Private qualification primitives.

The supported package API remains :mod:`gcw`; this namespace is intentionally
private. Only the receipt-bound admission primitives
(:mod:`~gcw.qualification.receipts`) and the capture inputs they consume
(:mod:`~gcw.qualification.inputs`) are live. The training-state
capture/comparison substrate was removed as dead code.
"""

from gcw.qualification.inputs import (
    BlobIdentity,
    CapturedBlob,
    ConfigSnapshot,
    QualificationInputError,
    SourceMemberSnapshot,
    SourceSnapshot,
    capture_file,
    capture_source,
    capture_train_config,
)
from gcw.qualification.receipts import (
    QualificationHMACKey,
    QualificationReceiptError,
    capture_hmac_key,
    issue_receipt,
    load_and_verify_receipt,
    verify_receipt_bytes,
    write_receipt,
)

__all__ = (
    "BlobIdentity",
    "CapturedBlob",
    "ConfigSnapshot",
    "QualificationHMACKey",
    "QualificationInputError",
    "QualificationReceiptError",
    "SourceMemberSnapshot",
    "SourceSnapshot",
    "capture_file",
    "capture_hmac_key",
    "capture_source",
    "capture_train_config",
    "issue_receipt",
    "load_and_verify_receipt",
    "verify_receipt_bytes",
    "write_receipt",
)
