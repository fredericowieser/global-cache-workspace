from dataclasses import dataclass

import pytest
import torch

from gcw.models.cache_processors.base import CacheProcessorBase
from gcw.models.cache_processors.state import (
    processor_call_count,
    select_l2_indices,
)
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import WindowedCache
from gcw.performance.kv_score_sidecar import (
    KVScoreSidecarSpec,
    KVScoreSidecarState,
)


@dataclass
class ExtendedCacheState(CacheState):
    provenance: str = "preserved"


@dataclass
class ExtendedWindowedCache(WindowedCache):
    provenance: str = "preserved"


@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("optional_state", [False, True])
def test_cache_detach_preserves_metadata_and_cuts_all_tensor_history(
    windowed: bool, optional_state: bool
) -> None:
    source = torch.randn(1, 2, 3, 4, requires_grad=True)
    keys = [source * 2, source * 3]
    values = [source * 4, source * 5]
    scores = source.sum(dim=(1, 3))
    cache: ExtendedCacheState | ExtendedWindowedCache
    if windowed:
        cache = ExtendedWindowedCache(
            keys=torch.stack(keys), values=torch.stack(values), scores=scores
        )
    else:
        cache = ExtendedCacheState(keys=keys, values=values, scores=scores)
    cache.provenance = "original-window"
    if optional_state:
        cache.linear_states = {
            0: (source * 6, source.sum(dim=1)),
            1: (source * 7, None),
        }
        cache.kv_score_sidecar = KVScoreSidecarState(
            spec=KVScoreSidecarSpec(
                schema="gcw.kv_score_sidecar_state.v1",
                num_layers=2,
                num_kv_heads=2,
                head_dim=4,
                unrope_keys=False,
                rope_theta=10000.0,
                rotary_style="llama",
                rotary_dim=4,
            ),
            scores=scores[:, None, :].expand(-1, 4, -1),
        )

    detached = cache.detach()
    assert detached is not cache and type(detached) is type(cache)
    assert isinstance(detached, (ExtendedCacheState, ExtendedWindowedCache))
    assert detached.provenance == cache.provenance
    originals = [cache.scores]
    boundaries = [detached.scores]
    for layer in range(cache.num_layers):
        originals.extend(cache.layer(layer))
        boundaries.extend(detached.layer(layer))
    if optional_state:
        assert cache.linear_states is not None
        assert detached.linear_states is not None
        assert detached.linear_states.keys() == cache.linear_states.keys()
        for layer, original in cache.linear_states.items():
            boundary = detached.linear_states[layer]
            for before, after in zip(original, boundary, strict=True):
                if before is None:
                    assert after is None
                else:
                    assert after is not None
                    originals.append(before)
                    boundaries.append(after)
        assert cache.kv_score_sidecar is not None
        assert detached.kv_score_sidecar is not None
        assert detached.kv_score_sidecar.spec is cache.kv_score_sidecar.spec
        originals.append(cache.kv_score_sidecar.scores)
        boundaries.append(detached.kv_score_sidecar.scores)
    else:
        assert detached.linear_states is None
        assert detached.kv_score_sidecar is None
    for before, after in zip(originals, boundaries, strict=True):
        assert before.requires_grad and before.grad_fn is not None
        assert not after.requires_grad and after.grad_fn is None
        assert torch.equal(before, after)
        assert before.data_ptr() == after.data_ptr()
    sum(tensor.sum() for tensor in originals).backward()
    assert source.grad is not None and torch.count_nonzero(source.grad) > 0


def test_processor_call_counter_is_consumable():
    class Processor(CacheProcessorBase):
        def forward(self, cache, *, select_all=False):
            return cache

    processor = Processor()
    marker = object()
    assert processor(marker) is marker
    assert processor.pop_call_count() == 1
    assert processor.pop_call_count() == 0


def test_processor_call_count_matches_window_rule():
    assert processor_call_count(1, 64) == 0
    assert processor_call_count(64, 64) == 0
    assert processor_call_count(65, 64) == 1
    assert processor_call_count(128, 64) == 1
    assert processor_call_count(129, 64) == 2


def test_l2_selector_keeps_recent_and_top_history_sorted():
    scores = torch.tensor([[0.0, 9.0, 1.0, 7.0, 2.0, 3.0]])
    idx = select_l2_indices(scores, process_every=2, topk=2)
    assert idx.tolist() == [[1, 3, 4, 5]]


def test_cache_storage_selects_all_heads():
    keys = [torch.zeros(1, 3, 4, 2) for _ in range(2)]
    values = [torch.zeros(1, 3, 4, 2) for _ in range(2)]
    scores = torch.zeros(1, 4)
    cache = CacheState(keys, values, scores)
    static = StaticCacheState(
        [k.clone() for k in keys],
        [v.clone() for v in values],
        scores.clone(),
        4,
    )
    windowed = WindowedCache.from_cache_state(cache)

    assert cache.all_heads().tolist() == [[0, 1, 2, 3, 4, 5]]
    assert static.all_heads().tolist() == [[0, 1, 2, 3, 4, 5]]
    assert windowed.all_heads().tolist() == [[0, 1, 2, 3, 4, 5]]
