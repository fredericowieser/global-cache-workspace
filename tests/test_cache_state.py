import torch

from gcw.models.cache_state import CacheState, processor_call_count, select_l2_indices
from gcw.config import CacheProcessorConfig
from gcw.models.icarus import ProcessorAttention, _gather_cache_heads, _select_heads_from_merged


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


def test_cache_state_selects_all_heads():
    keys = [torch.zeros(1, 3, 4, 2) for _ in range(2)]
    values = [torch.zeros(1, 3, 4, 2) for _ in range(2)]
    scores = torch.zeros(1, 4)
    cache = CacheState(keys, values, scores)

    assert cache.all_heads().tolist() == [[0, 1, 2, 3, 4, 5]]


def test_icarus_selects_top_heads_from_gathered_kv_norms():
    parts = [torch.zeros(1, 3, 2, 4) for _ in range(2)]
    parts[0][0, 2] = 10.0
    parts[1][0, 1] = 9.0
    merged = torch.cat(parts, dim=1)

    assert _select_heads_from_merged(merged, topk=2).tolist() == [[2, 4]]


def test_icarus_gathers_cache_heads_per_batch():
    x = torch.arange(2 * 4 * 3 * 2 * 2).view(2, 4, 3, 2, 2)
    selected = torch.tensor([[1, 3], [0, 2]])

    gathered = _gather_cache_heads(x, selected)

    assert torch.equal(gathered[0], x[0, [1, 3]])
    assert torch.equal(gathered[1], x[1, [0, 2]])


def test_icarus_attention_preserves_cache_layout():
    attn = ProcessorAttention(CacheProcessorConfig(hidden_size=8, num_heads=4, num_kv_heads=2, attention_dropout=0.0))
    x = torch.randn(2, 5, 3, 8)
    positions = torch.tensor([[0, 1, 2], [0, 2, 4]])
    selected = torch.tensor([[1, 3], [0, 4]])

    y = attn(x, positions, theta=10000.0, selected_heads=selected)

    assert y.shape == x.shape
