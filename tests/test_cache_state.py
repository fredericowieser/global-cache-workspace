import torch

from gcw.models.cache_processors.state import processor_call_count, select_l2_indices
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.cache_processors.windowed import WindowedCache


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
    static = StaticCacheState([k.clone() for k in keys], [v.clone() for v in values], scores.clone(), 4)
    windowed = WindowedCache.from_cache_state(cache)

    assert cache.all_heads().tolist() == [[0, 1, 2, 3, 4, 5]]
    assert static.all_heads().tolist() == [[0, 1, 2, 3, 4, 5]]
    assert windowed.all_heads().tolist() == [[0, 1, 2, 3, 4, 5]]
