import warnings

import pytest
import torch
import torch.nn as nn

from gcw.config import CacheProcessorConfig
from gcw.models.cache_processors.longshot_layers import (
    PerCacheHeadLinear,
    ProcessorAttention,
    _fit_time_length,
    _zero_projection,
)
from gcw.models.cache_processors.longshot_selection import (
    _select_heads_from_head_scores,
)


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

    without_duplicate = attn.forward_with_self_head_kv(
        x, torch.tensor([[0, 2]])
    )
    with_duplicate = attn.forward_with_self_head_kv(
        x, torch.tensor([[0, 2, 1]])
    )
    dense_with_duplicate = _dense_self_head_kv_reference(
        attn, x, torch.tensor([[0, 2, 1]])
    )

    assert torch.allclose(
        with_duplicate, dense_with_duplicate, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        without_duplicate[:, 1], with_duplicate[:, 1], atol=1e-6, rtol=1e-6
    )


def test_processor_attention_can_return_attention_output_scores():
    torch.manual_seed(24)
    x = torch.randn(2, 4, 8)
    selected_heads = torch.tensor([[0, 2], [1, 3]])

    for include_self_head_kv in (False, True):
        attn = ProcessorAttention(
            CacheProcessorConfig(
                hidden_size=8,
                intermediate_size=16,
                num_heads=2,
                num_kv_heads=1,
                attention_dropout=0.0,
                include_self_head_kv=include_self_head_kv,
            )
        ).eval()

        output = attn(x, selected_heads)
        scored_output, scores = attn(x, selected_heads, return_attn_scores=True)

        assert torch.allclose(scored_output, output, atol=1e-6, rtol=1e-6)
        assert scores.shape == (2, 4)
        assert torch.allclose(
            scores, output.float().pow(2).sum(dim=-1), atol=1e-6, rtol=1e-6
        )


def test_processor_attention_cache_head_attention_scores_match_existing_outputs():
    torch.manual_seed(25)
    attn = ProcessorAttention(
        CacheProcessorConfig(
            hidden_size=12,
            intermediate_size=24,
            num_heads=4,
            num_kv_heads=2,
            attention_dropout=0.0,
        )
    ).eval()
    x = torch.randn(2, 5, 12)
    selected_heads = torch.tensor([[0, 3], [1, 4]])

    no_self_output, no_self_scores = (
        attn.forward_with_cache_head_attention_scores(
            x,
            selected_heads,
            include_self_head_kv=False,
        )
    )
    self_output, self_scores = attn.forward_with_cache_head_attention_scores(
        x,
        selected_heads,
        include_self_head_kv=True,
    )

    assert torch.allclose(
        no_self_output,
        attn.forward_selected_head_kv(x, selected_heads),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        self_output,
        attn.forward_with_self_head_kv(x, selected_heads),
        atol=1e-6,
        rtol=1e-6,
    )
    assert no_self_scores.shape == (2, 5)
    assert torch.allclose(no_self_scores, self_scores, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        no_self_scores.sum(dim=-1),
        torch.full(
            (2,),
            5 * attn.num_heads,
            device=x.device,
            dtype=no_self_scores.dtype,
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_processor_attention_kv_norm_scores_match_existing_outputs():
    torch.manual_seed(27)
    x = torch.randn(2, 5, 12)
    selected_heads = torch.tensor([[0, 3], [1, 4]])

    for include_self_head_kv in (False, True):
        attn = ProcessorAttention(
            CacheProcessorConfig(
                hidden_size=12,
                intermediate_size=24,
                num_heads=4,
                num_kv_heads=2,
                attention_dropout=0.0,
                include_self_head_kv=include_self_head_kv,
                internal_iteration_score="kv_norm",
            )
        ).eval()

        output = attn(x, selected_heads)
        scored_output, scores = attn.forward_with_iteration_scores(
            x, selected_heads
        )
        k_all = attn.k_proj(x).view(2, 5, attn.num_kv_heads, attn.head_dim)
        v_all = attn.v_proj(x).view(2, 5, attn.num_kv_heads, attn.head_dim)
        expected_scores = k_all.float().pow(2).sum(
            dim=(2, 3)
        ) + v_all.float().pow(2).sum(dim=(2, 3))

        assert torch.allclose(scored_output, output, atol=1e-6, rtol=1e-6)
        assert scores.shape == (2, 5)
        assert torch.allclose(scores, expected_scores, atol=1e-6, rtol=1e-6)


def _dense_self_head_kv_reference(
    attn: ProcessorAttention,
    x: torch.Tensor,
    selected_heads: torch.Tensor,
) -> torch.Tensor:
    batch, cache_heads, _ = x.shape
    groups = attn.num_heads // attn.num_kv_heads
    q = attn.q_proj(x).view(
        batch, cache_heads, attn.num_kv_heads, groups, attn.head_dim
    )
    k = attn.k_proj(x).view(
        batch, cache_heads, attn.num_kv_heads, attn.head_dim
    )
    v = attn.v_proj(x).view(
        batch, cache_heads, attn.num_kv_heads, attn.head_dim
    )
    q, k = q.to(v.dtype), k.to(v.dtype)

    scores = torch.einsum("bckgd,bskd->bckgs", q, k) * (attn.head_dim**-0.5)
    query_heads = torch.arange(cache_heads, device=x.device)
    selected = (selected_heads[:, None, :] == query_heads[None, :, None]).any(
        dim=-1
    )
    self_mask = torch.eye(cache_heads, dtype=torch.bool, device=x.device)[
        None, :, :
    ]
    mask = selected[:, None, :] | self_mask
    scores = scores.masked_fill(~mask[:, :, None, None, :], float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    y = torch.einsum("bckgs,bskd->bckgd", probs, v)
    y = y.contiguous().view(batch, cache_heads, attn.num_heads * attn.head_dim)
    return attn.o_proj(y)


def test_select_heads_rejects_a_non_positive_head_budget():
    scores = torch.randn(2, 8)

    with pytest.raises(ValueError, match="at least one cache head"):
        _select_heads_from_head_scores(scores, 0)


def test_select_heads_returns_every_head_when_the_budget_covers_them():
    scores = torch.randn(3, 5)

    for topk in (5, 9):
        selected = _select_heads_from_head_scores(scores, topk)
        assert torch.equal(selected, torch.arange(5).expand(3, -1))


def test_zero_projection_rejects_a_projection_it_cannot_zero():
    class Opaque(nn.Module):
        pass

    with pytest.raises(TypeError, match="unsupported projection type"):
        _zero_projection(Opaque())


def test_zero_projection_zeroes_both_supported_projections():
    linear = nn.Linear(4, 3)
    per_head = PerCacheHeadLinear(2, 4, 3)

    for projection in (linear, per_head):
        nn.init.normal_(projection.weight)
        _zero_projection(projection)
        assert torch.count_nonzero(projection.weight) == 0
        assert torch.count_nonzero(projection.bias) == 0


def test_fit_time_length_warns_when_it_changes_the_history_length():
    x = torch.randn(1, 2, 4, 3)

    with pytest.warns(
        UserWarning, match="expected 6 selected positions but got 4"
    ):
        padded = _fit_time_length(x, 6)
    assert torch.equal(padded[:, :, 2:], x)
    assert torch.count_nonzero(padded[:, :, :2]) == 0

    with pytest.warns(
        UserWarning, match="expected 3 selected positions but got 4"
    ):
        trimmed = _fit_time_length(x, 3)
    assert torch.equal(trimmed, x[:, :, -3:])


def test_fit_time_length_is_silent_when_the_history_already_fits():
    x = torch.randn(1, 2, 5, 3)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _fit_time_length(x, 5) is x
