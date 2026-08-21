from __future__ import annotations

import pytest
import torch

from gcw.config import DataConfig, ModelConfig, TrainConfig
from gcw.performance.config import TritonCacheMovementConfig
from gcw.performance.kernel_manifest import kernel_manifest_sha256
from gcw.performance.triton.execution import (
    triton_cache_movement_metadata,
    validate_triton_cache_movement_request,
)


def _train_config(**overrides) -> TrainConfig:
    values = {
        "model": ModelConfig(variant="bgt", family="qwen3"),
        "data": DataConfig(kind="random"),
        "trainable": "cache_processor",
        "batch_size": 2,
        "bf16": True,
        "performance": {"triton_cache_movement": {"enabled": True}},
    }
    values.update(overrides)
    return TrainConfig(**values)


def test_triton_cache_policy_is_typed_default_off_and_fingerprinted() -> None:
    assert TritonCacheMovementConfig().enabled is False
    config = _train_config()
    assert config.performance.triton_cache_movement.enabled is True
    metadata = triton_cache_movement_metadata(
        config.performance.triton_cache_movement
    )
    assert metadata["status"] == "benchmark_only_not_campaign_admitted"
    assert metadata["kernel_manifest_sha256"] == kernel_manifest_sha256()
    assert metadata["runtime_autotuning"] is False
    assert len(str(metadata["fingerprint_sha256"])) == 64


def test_unqualified_triton_request_fails_before_runtime(monkeypatch) -> None:
    monkeypatch.delenv("GCW_ALLOW_UNQUALIFIED_TRITON_CACHE", raising=False)
    with pytest.raises(RuntimeError, match="benchmark-only"):
        validate_triton_cache_movement_request(
            TritonCacheMovementConfig(enabled=True),
            torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"batch_size": 4}, "fixed batch two"),
        ({"distributed": "ddp"}, "distributed='none'"),
        ({"trainable": "all"}, "cache-processor-only"),
        ({"bf16": False}, "BF16"),
    ),
)
def test_triton_cache_policy_rejects_unsupported_training_composition(
    overrides, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _train_config(**overrides)
