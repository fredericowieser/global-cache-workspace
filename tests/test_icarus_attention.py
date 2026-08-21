import torch

from gcw.config import CacheProcessorConfig
from gcw.models.cache_processors.ops import rotary_for_indices
from gcw.models.cache_processors.icarus import ProcessorAttention
from gcw.models.backbones.transformer import apply_rotary


def test_processor_attention_self_head_kv_matches_dense_gqa_reference_with_positions():
    torch.manual_seed(31)
    attn = ProcessorAttention(
        CacheProcessorConfig(
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            num_kv_heads=2,
            attention_dropout=0.0,
        )
    ).eval()
    x = torch.randn(2, 4, 3, 16)
    positions = torch.tensor(
        [
            [[0, 2, 5], [1, 3, 6], [2, 4, 7], [3, 5, 8]],
            [[4, 6, 9], [5, 7, 10], [6, 8, 11], [7, 9, 12]],
        ]
    )
    selected_heads = torch.tensor([[0, 2], [1, 3]])

    actual = attn.forward_with_self_head_kv(
        x, positions, 10000.0, selected_heads
    )
    expected = _dense_self_head_kv_reference(
        attn, x, positions, 10000.0, selected_heads
    )

    assert actual.shape == (2, 4, 3, 16)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_processor_attention_self_head_kv_masks_duplicate_self_head_sequence():
    torch.manual_seed(32)
    attn = ProcessorAttention(
        CacheProcessorConfig(
            hidden_size=8,
            intermediate_size=16,
            num_heads=2,
            num_kv_heads=1,
            attention_dropout=0.0,
        )
    ).eval()
    x = torch.randn(1, 3, 4, 8)
    positions = torch.tensor([[[0, 1, 4, 7], [2, 3, 6, 9], [5, 8, 10, 12]]])

    without_duplicate = attn.forward_with_self_head_kv(
        x, positions, 10000.0, torch.tensor([[0, 2]])
    )
    with_duplicate = attn.forward_with_self_head_kv(
        x, positions, 10000.0, torch.tensor([[0, 2, 1]])
    )
    dense_with_duplicate = _dense_self_head_kv_reference(
        attn, x, positions, 10000.0, torch.tensor([[0, 2, 1]])
    )

    assert torch.allclose(
        with_duplicate, dense_with_duplicate, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        without_duplicate[:, 1], with_duplicate[:, 1], atol=1e-6, rtol=1e-6
    )


def _dense_self_head_kv_reference(
    attn: ProcessorAttention,
    x: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
    selected_heads: torch.Tensor,
) -> torch.Tensor:
    batch, cache_heads, seq, _ = x.shape
    groups = attn.num_heads // attn.num_kv_heads
    q = attn.q_proj(x).view(
        batch, cache_heads, seq, attn.num_kv_heads, groups, attn.head_dim
    )
    k = attn.k_proj(x).view(
        batch, cache_heads, seq, attn.num_kv_heads, attn.head_dim
    )
    v = attn.v_proj(x).view(
        batch, cache_heads, seq, attn.num_kv_heads, attn.head_dim
    )
    q_cos, q_sin = rotary_for_indices(positions, attn.head_dim, theta, q.dtype)
    k_cos, k_sin = rotary_for_indices(positions, attn.head_dim, theta, k.dtype)
    q = apply_rotary(q, q_cos.unsqueeze(-2), q_sin.unsqueeze(-2))
    k = apply_rotary(k, k_cos, k_sin)
    q, k = q.to(v.dtype), k.to(v.dtype)

    scores = torch.einsum("bctkgd,bsmkd->bctkgsm", q, k) * (attn.head_dim**-0.5)
    query_heads = torch.arange(cache_heads, device=x.device)
    selected = (selected_heads[:, None, :] == query_heads[None, :, None]).any(
        dim=-1
    )
    self_mask = torch.eye(cache_heads, dtype=torch.bool, device=x.device)[
        None, :, :
    ]
    mask = selected[:, None, :] | self_mask
    scores = scores.masked_fill(
        ~mask[:, :, None, None, None, :, None], float("-inf")
    )

    probs = torch.softmax(scores.flatten(start_dim=-2), dim=-1).view_as(scores)
    y = torch.einsum("bctkgsm,bsmkd->bctkgd", probs, v)
    y = y.contiguous().view(
        batch, cache_heads, seq, attn.num_heads * attn.head_dim
    )
    return attn.o_proj(y)
