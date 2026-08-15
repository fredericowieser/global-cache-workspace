from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gcw.architecture import resolve_experiment, resolve_model
from gcw.config import CacheProcessorConfig, DataConfig, ModelConfig, TrainConfig, to_dict
from gcw.data.training import RandomCursor
from gcw.models import build_model
from gcw.performance.config import LongshotFullHistoryConfig, PerformanceConfig
from gcw.performance.longshot_cache import longshot_full_history_execution_metadata
from gcw.provenance import canonical_json_sha256
from gcw.training.checkpoint import (
    CheckpointCompatibilityError,
    ResumeState,
    TrainingCheckpoint,
    save_training_checkpoint,
)
from gcw.training.core import _training_fingerprint


def _train_config(
    *,
    history: LongshotFullHistoryConfig = LongshotFullHistoryConfig(),
    processor: CacheProcessorConfig | None = None,
    distributed: str = "none",
    trainable: str = "cache_processor",
    family: str = "smollm",
) -> TrainConfig:
    processor = processor or CacheProcessorConfig(
        process_every=2,
        topk=2,
        topk_heads=2,
        score="kv_norm",
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        gradient_checkpointing=False,
    )
    return TrainConfig(
        model=ModelConfig(
            variant="longshot",
            family=family,  # type: ignore[arg-type]
            sequence_len=128,
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            cache_processor=processor,
        ),
        data=DataConfig(kind="random"),
        performance=PerformanceConfig(longshot_full_history=history),
        distributed=distributed,  # type: ignore[arg-type]
        trainable=trainable,  # type: ignore[arg-type]
        sequence_len=128,
        batch_size=2,
        total_batch_size=256,
        max_steps=1,
    )


def _fingerprint(config: TrainConfig) -> dict[str, object]:
    return _training_fingerprint(config, 1, 256, 256, 1)


def test_disabled_policy_preserves_normalized_payload_and_same_environment_identities() -> None:
    config = _train_config()
    explicit_default = replace(config, performance=PerformanceConfig())

    performance_payload = to_dict(config.performance)
    assert "longshot_full_history" not in performance_payload
    assert "longshot_full_history_execution" not in _fingerprint(config)
    assert to_dict(config) == to_dict(explicit_default)
    assert _fingerprint(config) == _fingerprint(explicit_default)

    model_receipt = resolve_model(config.model).require()[1]
    explicit_model_receipt = resolve_model(explicit_default.model).require()[1]
    experiment_receipt = resolve_experiment(config, model_receipt).require()[1]
    explicit_experiment_receipt = resolve_experiment(
        explicit_default, explicit_model_receipt
    ).require()[1]
    assert model_receipt == explicit_model_receipt
    assert experiment_receipt == explicit_experiment_receipt


def test_enabled_policy_is_authenticated_in_metadata_and_resume_fingerprint() -> None:
    first = LongshotFullHistoryConfig(enabled=True, max_temp_bytes=4096)
    second = replace(first, max_temp_bytes=8192)
    first_config = _train_config(history=first)
    second_config = _train_config(history=second)

    first_metadata = longshot_full_history_execution_metadata(first)
    assert first_metadata["requested_via"] == "TrainConfig.performance.longshot_full_history"
    assert (
        first_metadata["fingerprint_sha256"] != longshot_full_history_execution_metadata(second)["fingerprint_sha256"]
    )
    assert _fingerprint(first_config)["longshot_full_history_execution"] == first_metadata
    assert canonical_json_sha256(_fingerprint(first_config)) != canonical_json_sha256(_fingerprint(second_config))


def test_production_checkpoint_rejects_full_history_policy_mismatch(tmp_path) -> None:
    first = _train_config(
        history=LongshotFullHistoryConfig(enabled=True, max_temp_bytes=4096)
    )
    changed = replace(
        first,
        performance=PerformanceConfig(
            longshot_full_history=LongshotFullHistoryConfig(
                enabled=True, max_temp_bytes=8192
            )
        ),
    )
    model = build_model(first.model)
    model.init_weights()
    optimizer = torch.optim.AdamW(model.cache_processor.parameters(), lr=1e-3)
    resume = ResumeState(
        step=1,
        smooth_loss=0.5,
        loader_cursor=RandomCursor(rng_state=(1, 2, 3)),
        saved_world_size=1,
        exact_topology=True,
    )
    checkpoint_path = tmp_path / "checkpoint"
    save_training_checkpoint(
        checkpoint_path, model, first.model, optimizer, resume, _fingerprint(first)
    )

    for expected in (changed, _train_config()):
        with pytest.raises(
            CheckpointCompatibilityError, match="longshot_full_history_execution"
        ):
            TrainingCheckpoint.open(
                checkpoint_path,
                expected_fingerprint=_fingerprint(expected),
                topology="same",
                current_world_size=1,
            )


def test_verified_checkpoint_recomputation_and_external_recurrence_are_admitted() -> None:
    processor = replace(
        _train_config().model.cache_processor,
        gradient_checkpointing=True,
        iterations=2,
        iteration_mode="external",
    )
    config = _train_config(
        history=LongshotFullHistoryConfig(enabled=True, max_temp_bytes=4096),
        processor=processor,
        family="qwen3",
    )

    segmented = replace(config, activation_checkpoint_windows=2)

    assert segmented.activation_checkpoint_windows == 2
    with pytest.raises(ValueError, match="one FP32 K/V row"):
        _train_config(history=LongshotFullHistoryConfig(enabled=True, max_temp_bytes=1))


def test_enabled_policy_rejects_compiled_backbone_execution() -> None:
    processor = replace(_train_config().model.cache_processor, process_every=64)
    eager = _train_config(
        history=LongshotFullHistoryConfig(enabled=True, max_temp_bytes=4096),
        processor=processor,
        family="qwen3",
    )
    with pytest.raises(ValueError, match="requires backbone_chunk_backend='eager'"):
        replace(eager, backbone_chunk_backend="compile")


@pytest.mark.parametrize(
    ("processor", "distributed", "trainable", "message"),
    [
        (replace(_train_config().model.cache_processor, score="attn_l2"), "none", "cache_processor", "score='kv_norm'"),
        (
            replace(_train_config().model.cache_processor, attention_dropout=0.1),
            "none",
            "cache_processor",
            "attention_dropout=0",
        ),
        (
            replace(
                _train_config().model.cache_processor,
                iterations=2,
                iteration_mode="internal",
                internal_iteration_score="attn_l2",
            ),
            "none",
            "cache_processor",
            "internal recurrence",
        ),
        (
            replace(_train_config().model.cache_processor, unrope_keys=True),
            "none",
            "cache_processor",
            "unrope_keys",
        ),
        (_train_config().model.cache_processor, "ddp", "cache_processor", "single-process"),
        (_train_config().model.cache_processor, "none", "all", "trainable='cache_processor'"),
    ],
)
def test_enabled_policy_fails_closed_for_unverified_execution(
    processor: CacheProcessorConfig,
    distributed: str,
    trainable: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _train_config(
            history=LongshotFullHistoryConfig(enabled=True),
            processor=processor,
            distributed=distributed,
            trainable=trainable,
        )
