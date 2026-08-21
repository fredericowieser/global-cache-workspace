from __future__ import annotations

import torch

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models.cache_processors.longshot import GlobalCacheProcessor
from gcw.models.cache_processors.windowed import WindowedCache


def _processor() -> GlobalCacheProcessor:
    config = ModelConfig(
        variant="longshot",
        family="smollm",
        sequence_len=16,
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=0,
            topk_heads=4,
            score="kv_norm",
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            encoder_method="query_attention",
            encoder_query_source="final_token",
            update_stream="v",
            gate_init=0.0,
        ),
    )
    return GlobalCacheProcessor(config)


def _cache() -> WindowedCache:
    torch.manual_seed(7)
    return WindowedCache(
        keys=torch.randn(2, 2, 2, 4, 8),
        values=torch.randn(2, 2, 2, 4, 8),
        scores=torch.randn(2, 4),
    )


def test_workspace_cycle_preserves_source_and_writes_differentiable_values() -> (
    None
):
    processor = _processor()
    cache = _cache()
    original_keys = cache.keys.clone()
    original_values = cache.values.clone()

    source = processor.encode_source(cache, select_all=True)
    workspace_0 = processor.initialise_workspace(source)
    workspace_1 = processor.deliberate(workspace_0, source)
    branch = processor.write_for_readout(cache, source, workspace_1)

    assert workspace_0.shape == (2, 4, 8)
    assert workspace_1.shape == workspace_0.shape
    assert torch.equal(cache.keys, original_keys)
    assert torch.equal(cache.values, original_values)
    assert torch.equal(branch.keys, original_keys)
    assert not torch.equal(branch.values, original_values)

    branch.values.sum().backward()
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in processor.parameters()
    )
