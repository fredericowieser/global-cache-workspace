from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from gcw.config import ModelConfig
from gcw.models.backbones.transformer import TF


def _make_model(*, tied: bool, dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> TF:
    config = ModelConfig(
        variant="tf",
        vocab_size=37,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=128,
        tie_embeddings=tied,
    )
    torch.manual_seed(412)
    model = TF(config, pad_vocab_size_to=1)
    model.init_weights()
    return model.to(device=device or torch.device("cpu"), dtype=dtype)


def _head_weight(model: TF) -> torch.Tensor:
    if model.config.tie_embeddings:
        return model.embed_tokens.weight
    assert model.lm_head is not None
    return model.lm_head.weight


def _targets(*, masked: bool, device: torch.device) -> torch.Tensor:
    targets = torch.arange(2 * 70, device=device).reshape(2, 70) % 37
    if masked:
        targets[:, :64] = -1  # Exercise a whole chunk with no targets.
        targets[0, 65::2] = -1
        targets[1, 66::2] = -1
    return targets


def _backward(loss: torch.Tensor) -> None:
    if loss.ndim == 0:
        loss.backward()
        return
    upstream = torch.linspace(-0.5, 0.5, loss.numel(), device=loss.device).reshape_as(loss)
    (loss * upstream).sum().backward()


def _assert_loss_and_gradient_parity(
    model: TF,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    reduction: str,
    *,
    atol: float,
    rtol: float,
) -> None:
    optimized_hidden = hidden.detach().clone().requires_grad_(True)
    optimized = model.loss_from_hidden(optimized_hidden, targets, reduction)
    _backward(optimized)
    assert optimized_hidden.grad is not None
    optimized_hidden_grad = optimized_hidden.grad.detach().clone()
    optimized_head_grad = _head_weight(model).grad
    assert optimized_head_grad is not None
    optimized_head_grad = optimized_head_grad.detach().clone()

    model.zero_grad(set_to_none=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference = model.loss_from_hidden_reference(reference_hidden, targets, reduction)
    _backward(reference)
    assert reference_hidden.grad is not None
    reference_head_grad = _head_weight(model).grad
    assert reference_head_grad is not None

    torch.testing.assert_close(optimized, reference, atol=atol, rtol=rtol)
    torch.testing.assert_close(optimized_hidden_grad, reference_hidden.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(optimized_head_grad, reference_head_grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none", "row_sum"])
@pytest.mark.parametrize("masked", [False, True], ids=["unmasked", "masked"])
@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_answer_only_loss_forward_backward_matches_eager_reference(reduction: str, masked: bool, tied: bool):
    model = _make_model(tied=tied)
    hidden = torch.randn(2, 70, model.config.hidden_size)
    targets = _targets(masked=masked, device=hidden.device)

    _assert_loss_and_gradient_parity(model, hidden, targets, reduction, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none", "row_sum"])
@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_answer_only_loss_all_masked_is_differentiable_and_matches_reference(reduction: str, tied: bool):
    model = _make_model(tied=tied)
    hidden = torch.randn(2, 70, model.config.hidden_size)
    targets = torch.full((2, 70), -1, dtype=torch.long)

    _assert_loss_and_gradient_parity(model, hidden, targets, reduction, atol=0.0, rtol=0.0)


def test_answer_only_loss_projects_only_valid_positions(monkeypatch: pytest.MonkeyPatch):
    model = _make_model(tied=True)
    hidden = torch.randn(2, 70, model.config.hidden_size)
    targets = torch.full((2, 70), -1, dtype=torch.long)
    targets[0, 64] = 3
    targets[1, 65:67] = torch.tensor([4, 5])
    projected_shapes: list[torch.Size] = []
    eager_logits: Callable[[torch.Tensor], torch.Tensor] = model.logits_from_hidden

    def record_logits(selected: torch.Tensor) -> torch.Tensor:
        projected_shapes.append(selected.shape)
        return eager_logits(selected)

    monkeypatch.setattr(model, "logits_from_hidden", record_logits)
    model.loss_from_hidden(hidden, targets, "mean")
    assert projected_shapes == [torch.Size((3, model.config.hidden_size))]

    projected_shapes.clear()
    targets.fill_(-1)
    loss = model.loss_from_hidden(hidden.requires_grad_(True), targets, "none")
    assert projected_shapes == []
    assert loss.shape == targets.shape
    assert torch.count_nonzero(loss) == 0
    loss.sum().backward()


@pytest.mark.parametrize("reduction", ["mean", "sum", "none", "row_sum"])
@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_answer_only_loss_bfloat16_matches_eager_reference(reduction: str, tied: bool):
    device = torch.device("cuda") if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.device("cpu")
    try:
        model = _make_model(tied=tied, dtype=torch.bfloat16, device=device)
        hidden = torch.randn(2, 70, model.config.hidden_size, device=device, dtype=torch.bfloat16)
        targets = _targets(masked=True, device=device)
        _assert_loss_and_gradient_parity(model, hidden, targets, reduction, atol=3e-2, rtol=3e-2)
    except RuntimeError as error:
        if device.type == "cpu" and "bfloat16" in str(error).lower():
            pytest.skip(f"CPU BF16 kernels unavailable: {error}")
        raise
