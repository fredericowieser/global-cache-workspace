from __future__ import annotations

import copy

import pytest
import torch

from gcw.config import (
    CacheProcessorConfig,
    ModelConfig,
    TrainConfig,
    to_dict,
)
from gcw.models import build_model
from gcw.models.cache_processors.windowed import WindowedCache


def _model_config(variant: str) -> ModelConfig:
    return ModelConfig(
        variant=variant,  # type: ignore[arg-type]
        family="smollm",
        sequence_len=32,
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=2,
            topk_heads=2,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
        ),
    )


@pytest.mark.parametrize("variant", ["bt", "bgt", "icarus", "longshot"])
def test_truncated_bptt_preserves_forward_loss_and_is_opt_in(
    variant: str,
) -> None:
    torch.manual_seed(41)
    full = build_model(_model_config(variant))
    full.init_weights()
    truncated = copy.deepcopy(full)
    input_ids = torch.randint(0, 128, (2, 12))
    targets = input_ids.roll(-1, dims=1)

    full_loss = full(input_ids, targets)
    truncated_loss = truncated(
        input_ids,
        targets,
        truncated_bptt_windows=1,
    )

    assert torch.equal(truncated_loss, full_loss)


@pytest.mark.parametrize("horizon, expected_detaches", [(1, 2), (2, 1), (3, 0)])
def test_truncated_bptt_detaches_only_at_configured_window_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    horizon: int,
    expected_detaches: int,
) -> None:
    model = build_model(_model_config("bgt"))
    model.init_weights()
    input_ids = torch.randint(0, 128, (1, 12))
    calls = 0
    original_detach = WindowedCache.detach

    def track_detach(cache: WindowedCache) -> WindowedCache:
        nonlocal calls
        calls += 1
        return original_detach(cache)

    monkeypatch.setattr(WindowedCache, "detach", track_detach)
    model(
        input_ids,
        input_ids,
        truncated_bptt_windows=horizon,
    ).backward()

    assert calls == expected_detaches


def test_truncated_bptt_config_is_serializable_and_rejects_processor_free_models() -> (
    None
):
    config = TrainConfig(
        model=_model_config("bgt"),
        truncated_bptt_windows=2,
    )
    assert config.truncated_bptt_windows == 2
    assert to_dict(config)["truncated_bptt_windows"] == 2
    assert "truncated_bptt_windows" not in to_dict(TrainConfig())

    with pytest.raises(
        ValueError,
        match="truncated_bptt_windows requires a cache-processor variant",
    ):
        TrainConfig(
            model=ModelConfig(
                variant="tf",
                sequence_len=32,
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                head_dim=8,
            ),
            truncated_bptt_windows=1,
        )
