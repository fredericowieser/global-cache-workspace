import pytest
import torch

from gcw.config import CacheProcessorConfig
from gcw.models.cache_processors.ops import rotary_for_indices
from gcw.models.cache_processors.icarus import ProcessorAttention
from gcw.models.backbones.transformer import apply_rotary


@pytest.mark.parametrize("self_kv", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_processor_attention_matches_dense_gqa_reference_with_positions(
    self_kv, kv_heads
):
    torch.manual_seed(31)
    attn = ProcessorAttention(
        CacheProcessorConfig(
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            num_kv_heads=kv_heads,
            attention_dropout=0.0,
            include_self_head_kv=self_kv,
        )
    ).eval()
    x = torch.randn(2, 4, 3, 16, requires_grad=True)
    positions = torch.tensor(
        [
            [[0, 2, 5], [1, 3, 6], [2, 4, 7], [3, 5, 8]],
            [[4, 6, 9], [5, 7, 10], [6, 8, 11], [7, 9, 12]],
        ]
    )
    selected_heads = torch.tensor([[0, 2], [1, 3]])

    actual = _workspace_attention(attn, x, positions, 10000.0, selected_heads)
    expected = _dense_self_head_kv_reference(
        attn, x, positions, 10000.0, selected_heads
    )

    assert actual.shape == (2, 4, 3, 16)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    weights = torch.randn_like(actual)
    inputs = (x, *attn.parameters())
    actual_grad = torch.autograd.grad(
        (actual * weights).sum(), inputs, retain_graph=True
    )
    expected_grad = torch.autograd.grad((expected * weights).sum(), inputs)
    for actual_value, expected_value in zip(
        actual_grad, expected_grad, strict=True
    ):
        torch.testing.assert_close(
            actual_value, expected_value, atol=2e-6, rtol=2e-5
        )


def test_processor_attention_self_head_kv_masks_duplicate_self_head_sequence():
    torch.manual_seed(32)
    attn = ProcessorAttention(
        CacheProcessorConfig(
            hidden_size=8,
            intermediate_size=16,
            num_heads=2,
            num_kv_heads=1,
            attention_dropout=0.0,
            include_self_head_kv=True,
        )
    ).eval()
    x = torch.randn(1, 3, 4, 8)
    positions = torch.tensor([[[0, 1, 4, 7], [2, 3, 6, 9], [5, 8, 10, 12]]])

    without_duplicate = _workspace_attention(
        attn, x, positions, 10000.0, torch.tensor([[0, 2]])
    )
    with_duplicate = _workspace_attention(
        attn, x, positions, 10000.0, torch.tensor([[0, 2, 1]])
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


def _workspace_attention(attn, x, positions, theta, selected_heads):
    _, heads, length, _ = x.shape
    tokens = (
        selected_heads[..., None] * length + torch.arange(length)
    ).flatten(1)
    return attn(
        x.flatten(1, 2), positions.flatten(1), theta, tokens, ((heads, length),)
    ).reshape_as(x)


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
    mask = selected[:, None, :]
    if attn.include_self_head_kv:
        mask = mask | self_mask
    scores = scores.masked_fill(
        ~mask[:, :, None, None, None, :, None], float("-inf")
    )

    probs = torch.softmax(scores.flatten(start_dim=-2), dim=-1).view_as(scores)
    y = torch.einsum("bctkgsm,bsmkd->bctkgd", probs, v)
    y = y.contiguous().view(
        batch, cache_heads, seq, attn.num_heads * attn.head_dim
    )
    return attn.o_proj(y)


@pytest.mark.parametrize("linear_heads", [0, 3])
def test_self_head_attention_work_scales_with_selected_and_own_tokens(
    monkeypatch, linear_heads
):
    attn = ProcessorAttention(
        CacheProcessorConfig(
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            num_kv_heads=2,
            include_self_head_kv=True,
        )
    ).eval()
    original = torch.einsum
    logits = []

    def observe(equation, *args, **kwargs):
        result = original(equation, *args, **kwargs)
        if equation.endswith("->bhtkgs"):
            logits.append(result.numel())
        return result

    monkeypatch.setattr(torch, "einsum", observe)
    length, selected_count = 5, 10
    for kv_heads in (8, 16):
        logits.clear()
        tokens = kv_heads * length + linear_heads
        groups = ((kv_heads, length),) + (
            ((linear_heads, 1),) if linear_heads else ()
        )
        x = torch.randn(2, tokens, 16)
        positions = torch.arange(tokens).expand(2, -1)
        selected = torch.arange(selected_count).expand(2, -1)
        with torch.no_grad():
            result = attn(x, positions, 10000.0, selected, groups)
        expected_selected = 2 * 4 * tokens * selected_count
        expected_self = 2 * 4 * (kv_heads * length**2 + linear_heads)
        assert sum(logits) == expected_selected + expected_self
        assert max(logits) < 2 * 4 * tokens**2
        assert result.shape == x.shape
