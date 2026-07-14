import torch

from gcw.config import CacheProcessorConfig
from gcw.models.longshot import ProcessorAttention


def test_processor_attention_self_head_kv_matches_dense_gqa_reference():
    torch.manual_seed(21)
    attn = ProcessorAttention(
        CacheProcessorConfig(
            hidden_size=12,
            intermediate_size=24,
            num_heads=4,
            num_kv_heads=2,
            attention_dropout=0.0,
        )
    ).eval()
    x = torch.randn(2, 4, 12)
    selected_heads = torch.tensor([[0, 2], [1, 3]])

    actual = attn.forward_with_self_head_kv(x, selected_heads)
    expected = _dense_self_head_kv_reference(attn, x, selected_heads)

    assert actual.shape == (2, 4, 12)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_processor_attention_self_head_kv_masks_appended_duplicate_self():
    torch.manual_seed(22)
    attn = ProcessorAttention(
        CacheProcessorConfig(
            hidden_size=8,
            intermediate_size=16,
            num_heads=2,
            num_kv_heads=1,
            attention_dropout=0.0,
        )
    ).eval()
    x = torch.randn(1, 3, 8)

    without_duplicate = attn.forward_with_self_head_kv(x, torch.tensor([[0, 2]]))
    with_duplicate = attn.forward_with_self_head_kv(x, torch.tensor([[0, 2, 1]]))
    dense_with_duplicate = _dense_self_head_kv_reference(attn, x, torch.tensor([[0, 2, 1]]))

    assert torch.allclose(with_duplicate, dense_with_duplicate, atol=1e-6, rtol=1e-6)
    assert torch.allclose(without_duplicate[:, 1], with_duplicate[:, 1], atol=1e-6, rtol=1e-6)


def test_processor_attention_self_head_kv_empty_selection_is_self_only():
    torch.manual_seed(23)
    attn = ProcessorAttention(
        CacheProcessorConfig(
            hidden_size=12,
            intermediate_size=24,
            num_heads=3,
            num_kv_heads=3,
            attention_dropout=0.0,
        )
    ).eval()
    x = torch.randn(2, 5, 12)
    selected_heads = torch.empty(2, 0, dtype=torch.long)

    actual = attn.forward_with_self_head_kv(x, selected_heads)
    expected = _dense_self_head_kv_reference(attn, x, selected_heads)
    self_only = _self_only_reference(attn, x)

    assert actual.shape == (2, 5, 12)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual, self_only, atol=1e-6, rtol=1e-6)


def _dense_self_head_kv_reference(
    attn: ProcessorAttention,
    x: torch.Tensor,
    selected_heads: torch.Tensor,
) -> torch.Tensor:
    batch, cache_heads, _ = x.shape
    groups = attn.num_heads // attn.num_kv_heads
    q = attn.q_proj(x).view(batch, cache_heads, attn.num_kv_heads, groups, attn.head_dim)
    k = attn.k_proj(x).view(batch, cache_heads, attn.num_kv_heads, attn.head_dim)
    v = attn.v_proj(x).view(batch, cache_heads, attn.num_kv_heads, attn.head_dim)
    q, k = q.to(v.dtype), k.to(v.dtype)

    scores = torch.einsum("bckgd,bskd->bckgs", q, k) * (attn.head_dim**-0.5)
    query_heads = torch.arange(cache_heads, device=x.device)
    selected = (selected_heads[:, None, :] == query_heads[None, :, None]).any(dim=-1)
    self_mask = torch.eye(cache_heads, dtype=torch.bool, device=x.device)[None, :, :]
    mask = selected[:, None, :] | self_mask
    scores = scores.masked_fill(~mask[:, :, None, None, :], float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    y = torch.einsum("bckgs,bskd->bckgd", probs, v)
    y = y.contiguous().view(batch, cache_heads, attn.num_heads * attn.head_dim)
    return attn.o_proj(y)


def _self_only_reference(attn: ProcessorAttention, x: torch.Tensor) -> torch.Tensor:
    batch, cache_heads, _ = x.shape
    groups = attn.num_heads // attn.num_kv_heads
    v = attn.v_proj(x).view(batch, cache_heads, attn.num_kv_heads, attn.head_dim)
    y = v[:, :, :, None, :].expand(-1, -1, -1, groups, -1)
    y = y.contiguous().view(batch, cache_heads, attn.num_heads * attn.head_dim)
    return attn.o_proj(y)
