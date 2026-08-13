from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from gcw.config import TrainConfig


@dataclass(frozen=True)
class CampaignContract:
    schema: str
    dataset_sha256: str
    tokenizer_identity_sha256: str
    backbone_sha256: str
    sequence_length: int
    global_batch_rows: int
    max_steps: int
    dataset_path: str | None = None
    tokenizer_source: str | None = None
    backbone_checkpoint: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "CampaignContract":
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict):
            raise ValueError("campaign manifest must contain an object")
        return cls(**payload)

    def validate(self, config: TrainConfig) -> None:
        mismatches: list[str] = []
        checks: dict[str, tuple[Any, Any]] = {
            "sequence_length": (config.sequence_len, self.sequence_length),
            "global_batch_rows": (config.global_batch_rows, self.global_batch_rows),
            "max_steps": (config.max_steps, self.max_steps),
            "dataset": (str(Path(config.data.dataset or "").expanduser().resolve()), self.dataset_path),
            "tokenizer_source": (config.data.tokenizer_source, self.tokenizer_source),
            "backbone_checkpoint": (config.backbone_checkpoint, self.backbone_checkpoint),
        }
        for name, (observed, expected) in checks.items():
            if expected is not None and observed != expected:
                mismatches.append(f"{name}: observed {observed!r}, expected {expected!r}")
        dataset = Path(config.data.dataset or "").expanduser()
        if not dataset.is_file() or _sha256_file(dataset) != self.dataset_sha256:
            mismatches.append("dataset_sha256 does not match the campaign contract")
        if mismatches:
            raise ValueError("campaign contract mismatch: " + "; ".join(mismatches))


def validate_campaign(config: TrainConfig) -> CampaignContract | None:
    if config.data.campaign_manifest is None:
        return None
    contract = CampaignContract.load(config.data.campaign_manifest)
    if contract.schema != "gcw.math_campaign.v1":
        raise ValueError(f"unsupported campaign schema: {contract.schema}")
    contract.validate(config)
    return contract


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
