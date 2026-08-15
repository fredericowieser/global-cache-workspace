from __future__ import annotations

import sys
from pathlib import Path

import torch

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models import build_model
from gcw.performance.config import KVScoreSidecarConfig
from gcw.performance.kv_score_sidecar import (
    configure_kv_score_sidecar_execution,
    kv_score_sidecar_execution_metadata,
)
from gcw.training.checkpoint import TrainingCheckpoint


def model_config() -> ModelConfig:
    return ModelConfig(
        variant="longshot",
        family="qwen3",
        sequence_len=16,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=2,
            topk_heads=1,
            score="kv_norm",
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
            attention_dropout=0.0,
            update_stream="kv",
            encoder_method="query_attention",
            encoder_query_source="final_token",
        ),
    )


def fingerprint() -> dict[str, object]:
    policy = KVScoreSidecarConfig(enabled=True)
    return {
        "distributed": "none",
        "kv_score_sidecar_execution": kv_score_sidecar_execution_metadata(policy, unrope_keys=False),
    }


def main() -> None:
    checkpoint_path, batch_path, output_path = map(Path, sys.argv[1:4])
    config = model_config()
    model = build_model(config).train()
    checkpoint = TrainingCheckpoint.open(
        checkpoint_path,
        expected_fingerprint=fingerprint(),
        topology="same",
        current_world_size=1,
    )
    checkpoint.restore_model(model)
    configure_kv_score_sidecar_execution(model, KVScoreSidecarConfig(enabled=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    checkpoint.restore_optimizer(optimizer, device="cpu")
    input_ids, targets = torch.load(batch_path, weights_only=True)
    optimizer.zero_grad(set_to_none=True)
    model(input_ids, targets).backward()
    optimizer.step()
    torch.save(
        {
            "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
        },
        output_path,
    )


if __name__ == "__main__":
    main()
