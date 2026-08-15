from __future__ import annotations

import os
from importlib import metadata
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from gcw.architecture import _config_payload
from gcw.config import DataConfig, ModelConfig, TrainConfig
from gcw.models.backbones import transformer as transformer_module
from gcw.models.backbones.transformer import TF
from gcw.models.backbones.gemma4 import Gemma4TF
from gcw.performance import PerformanceConfig, ReferenceLossConfig
from gcw.performance import losses as losses_module
from gcw.performance.losses import (
    LIGER_KERNEL_VERSION,
    bounded_reference_loss,
    checkpoint_loss_backend,
    configure_reference_loss_execution,
    loss_backend_metadata,
    reference_loss_execution_metadata,
    require_loss_backend,
)
from gcw.training import core as training_module
from gcw.training.core import _checkpoint_metadata, _training_fingerprint


def _model(*, tied: bool, dtype: torch.dtype = torch.float32) -> TF:
    config = ModelConfig(
        vocab_size=11,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        sequence_len=16,
        tie_embeddings=tied,
    )
    torch.manual_seed(7)
    model = TF(config, pad_vocab_size_to=8)
    model.init_weights()
    return model.to(dtype=dtype)


def _head_weight(model: TF) -> torch.Tensor:
    if model.config.tie_embeddings:
        return model.embed_tokens.weight
    assert model.lm_head is not None
    return model.lm_head.weight


def _install_fake_liger(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(transformer_module, "require_loss_backend", lambda backend, device: None)
    monkeypatch.setattr(losses_module, "require_loss_backend", lambda backend, device: None)
    monkeypatch.setattr(losses_module.metadata, "version", lambda package: LIGER_KERNEL_VERSION)

    class FakeLiger:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(
            self,
            weight: torch.Tensor,
            hidden: torch.Tensor,
            targets: torch.Tensor,
            bias: torch.Tensor | None,
        ) -> torch.Tensor:
            calls.append(
                {
                    "kwargs": self.kwargs,
                    "weight_shape": tuple(weight.shape),
                    "hidden_shape": tuple(hidden.shape),
                    "targets": targets.detach().clone(),
                    "bias": bias,
                }
            )
            logits = F.linear(hidden, weight, bias).float()
            softcap = self.kwargs["softcap"]
            if softcap is not None:
                logits = torch.tanh(logits / softcap) * softcap
            return F.cross_entropy(
                logits,
                targets,
                ignore_index=self.kwargs["ignore_index"],
                reduction=self.kwargs["reduction"],
            )

    monkeypatch.setattr(losses_module, "_load_liger_factory", lambda: FakeLiger)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_fake_liger_matches_reference_with_vocab_truncation_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
    tied: bool,
    reduction: str,
    dtype: torch.dtype,
) -> None:
    calls: list[dict[str, Any]] = []
    _install_fake_liger(monkeypatch, calls)
    model = _model(tied=tied, dtype=dtype)
    hidden = torch.randn(2, 5, model.config.hidden_size, dtype=dtype)
    targets = torch.tensor([[0, 1, -1, 10, 2], [3, -100, 4, 5, 6]])

    if tied:
        head = model.embed_tokens.weight
    else:
        assert model.lm_head is not None
        head = model.lm_head.weight
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference = model.loss_from_hidden(reference_hidden, targets, reduction, "reference")
    reference.backward()
    assert reference_hidden.grad is not None
    assert head.grad is not None
    reference_hidden_grad = reference_hidden.grad.detach().clone()
    reference_head_grad = head.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    liger_hidden = hidden.detach().clone().requires_grad_(True)
    liger = model.loss_from_hidden(liger_hidden, targets, reduction, "liger")
    liger.backward()

    torch.testing.assert_close(liger, reference)
    torch.testing.assert_close(liger_hidden.grad, reference_hidden_grad)
    torch.testing.assert_close(head.grad, reference_head_grad)
    assert calls[0]["weight_shape"] == (model.config.vocab_size, model.config.hidden_size)
    assert calls[0]["kwargs"] == {
        "ignore_index": -100,
        "reduction": reduction,
        "accum_dtype": torch.float32,
        "softcap": None,
    }
    assert head.shape[0] == 16
    assert head.grad is not None
    assert torch.count_nonzero(head.grad[model.config.vocab_size :]) == 0


def test_fake_liger_receives_gemma_logit_softcap(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    _install_fake_liger(monkeypatch, calls)
    config = ModelConfig(
        family="gemma4",
        vocab_size=11,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        sequence_len=8,
        rotary_style="llama",
        layer_types=["full_attention"],
        global_head_dim=4,
        final_logit_softcapping=2.0,
        hidden_act="gelu_pytorch_tanh",
    )
    model = Gemma4TF(config, pad_vocab_size_to=8)
    model.init_weights()
    hidden = torch.randn(2, 3, config.hidden_size)
    targets = torch.tensor([[1, 2, -100], [3, 4, 5]])

    expected = model.loss_from_hidden(hidden, targets, "mean", "reference")
    actual = model.loss_from_hidden(hidden, targets, "mean", "liger")

    torch.testing.assert_close(actual, expected)
    assert calls[0]["kwargs"]["softcap"] == 2.0


def test_liger_none_uses_explicit_target_only_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    _install_fake_liger(monkeypatch, calls)
    model = _model(tied=True)
    hidden = torch.randn(2, 3, model.config.hidden_size, requires_grad=True)
    targets = torch.tensor([[1, -1, 2], [3, 4, -100]])

    actual = model.loss_from_hidden(hidden, targets, "none", "liger")
    expected = model.loss_from_hidden(hidden, targets, "none", "target_only")

    torch.testing.assert_close(actual, expected)
    assert calls == []
    provenance = loss_backend_metadata("liger")
    assert provenance["reduction_none"]["effective_backend"] == "target_only"
    assert "no per-token reduction" in provenance["reduction_none"]["reason"]


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_fake_liger_all_ignored_is_zero_and_differentiable(
    monkeypatch: pytest.MonkeyPatch,
    reduction: str,
) -> None:
    calls: list[dict[str, Any]] = []
    _install_fake_liger(monkeypatch, calls)
    model = _model(tied=False)
    hidden = torch.randn(2, 4, model.config.hidden_size, requires_grad=True)
    targets = torch.full((2, 4), -100, dtype=torch.long)

    loss = model.loss_from_hidden(hidden, targets, reduction, "liger")
    loss.sum().backward()

    assert torch.count_nonzero(loss) == 0
    assert hidden.grad is not None and torch.count_nonzero(hidden.grad) == 0
    assert model.lm_head is not None
    assert model.lm_head.weight.grad is not None
    assert torch.count_nonzero(model.lm_head.weight.grad) == 0
    assert calls == []


def test_liger_fails_closed_for_cpu_missing_and_wrong_version(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="CUDA device"):
        require_loss_backend("liger", torch.device("cpu"))

    def missing(_: str) -> str:
        raise metadata.PackageNotFoundError("liger-kernel")

    monkeypatch.setattr(losses_module.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="liger-kernel==0.8.1"):
        require_loss_backend("liger", torch.device("cuda"))

    monkeypatch.setattr(losses_module.metadata, "version", lambda _: "0.8.0")
    with pytest.raises(RuntimeError, match="found 0.8.0"):
        require_loss_backend("liger", torch.device("cuda"))


def test_backend_is_state_dict_neutral_and_in_runtime_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    model = _model(tied=True)
    keys = tuple(model.state_dict())
    config = TrainConfig(
        model=model.config,
        data=DataConfig(kind="random"),
        output_dir=str(tmp_path),
        loss_backend="target_only",
    )

    fingerprint = _training_fingerprint(config, 1, 16, 16, 1)
    checkpoint = _checkpoint_metadata(config, {"step": 0})

    assert tuple(model.state_dict()) == keys
    assert fingerprint["loss_backend"] == "target_only"
    assert checkpoint["loss_backend"]["name"] == "target_only"
    assert TrainConfig(model=model.config, data=DataConfig(kind="random")).loss_backend == "reference"
    assert checkpoint_loss_backend(checkpoint, torch.device("cpu")) == "target_only"


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none", "row_sum"])
def test_bounded_reference_loss_matches_legacy_values_and_gradients(tied: bool, reduction: str) -> None:
    model = _model(tied=tied)
    hidden = torch.randn(2, 9, model.config.hidden_size)
    targets = torch.tensor([[0, 1, -100, 10, 2, 3, -1, 4, 5], [3, -100, 4, 5, 6, 7, 8, 9, 10]])

    legacy_hidden = hidden.detach().clone().requires_grad_(True)
    legacy = model.loss_from_hidden(legacy_hidden, targets, reduction, "reference")
    legacy.sum().backward()
    legacy_hidden_grad = legacy_hidden.grad.detach().clone()
    legacy_head_grad = _head_weight(model).grad.detach().clone()

    model.zero_grad(set_to_none=True)
    configure_reference_loss_execution(
        model,
        ReferenceLossConfig(mode="bounded", max_projected_positions=3),
    )
    bounded_hidden = hidden.detach().clone().requires_grad_(True)
    bounded = model.loss_from_hidden(bounded_hidden, targets, reduction, "reference")
    bounded.sum().backward()

    torch.testing.assert_close(bounded, legacy, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(bounded_hidden.grad, legacy_hidden_grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(_head_weight(model).grad, legacy_head_grad, rtol=1e-5, atol=1e-6)


def test_bounded_reference_loss_caps_projected_positions_and_preserves_masked_shape() -> None:
    hidden = torch.randn(4, 11, 8, requires_grad=True)
    targets = torch.arange(44).reshape(4, 11) % 13
    targets[:, 3] = -100
    projected_shapes: list[tuple[int, ...]] = []

    def logits(values: torch.Tensor) -> torch.Tensor:
        projected_shapes.append(tuple(values.shape))
        return F.linear(values, torch.randn(13, 8))

    losses = bounded_reference_loss(
        logits,
        hidden,
        targets,
        "none",
        ReferenceLossConfig(mode="bounded", max_projected_positions=9),
    )

    assert losses.shape == targets.shape
    assert torch.count_nonzero(losses[:, 3]) == 0
    assert sum(shape[0] * shape[1] for shape in projected_shapes) == targets.numel()
    assert max(shape[0] * shape[1] for shape in projected_shapes) <= 9


def test_checkpointed_bounded_reference_loss_matches_uncheckpointed_gradients() -> None:
    model = _model(tied=True)
    hidden = torch.randn(2, 9, model.config.hidden_size)
    targets = torch.tensor([[0, 1, -100, 10, 2, 3, -1, 4, 5], [3, -100, 4, 5, 6, 7, 8, 9, 10]])

    configure_reference_loss_execution(
        model,
        ReferenceLossConfig(mode="bounded", max_projected_positions=4),
    )
    eager_hidden = hidden.detach().clone().requires_grad_(True)
    eager = model.loss_from_hidden(eager_hidden, targets, "row_sum", "reference")
    eager.sum().backward()
    eager_hidden_grad = eager_hidden.grad.detach().clone()
    eager_head_grad = _head_weight(model).grad.detach().clone()

    model.zero_grad(set_to_none=True)
    configure_reference_loss_execution(
        model,
        ReferenceLossConfig(mode="bounded", max_projected_positions=4, checkpoint_chunks=True),
    )
    checkpointed_hidden = hidden.detach().clone().requires_grad_(True)
    checkpointed = model.loss_from_hidden(checkpointed_hidden, targets, "row_sum", "reference")
    checkpointed.sum().backward()

    torch.testing.assert_close(checkpointed, eager)
    torch.testing.assert_close(checkpointed_hidden.grad, eager_hidden_grad)
    torch.testing.assert_close(_head_weight(model).grad, eager_head_grad)


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
@pytest.mark.parametrize("checkpoint_chunks", [False, True], ids=["eager", "checkpointed"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none", "row_sum"])
def test_bounded_reference_all_masked_is_zero_and_differentiable(
    tied: bool,
    checkpoint_chunks: bool,
    reduction: str,
) -> None:
    model = _model(tied=tied)
    configure_reference_loss_execution(
        model,
        ReferenceLossConfig(
            mode="bounded",
            max_projected_positions=4,
            checkpoint_chunks=checkpoint_chunks,
        ),
    )
    hidden = torch.randn(2, 9, model.config.hidden_size, requires_grad=True)
    targets = torch.full((2, 9), -100, dtype=torch.long)

    loss = model.loss_from_hidden(hidden, targets, reduction, "reference")
    loss.sum().backward()

    assert torch.count_nonzero(loss) == 0
    assert hidden.grad is not None and torch.count_nonzero(hidden.grad) == 0
    assert _head_weight(model).grad is not None
    assert torch.count_nonzero(_head_weight(model).grad) == 0


def test_reference_loss_execution_is_state_neutral_and_conditionally_fingerprinted(tmp_path) -> None:
    model = _model(tied=True)
    state_keys = tuple(model.state_dict())
    bounded = ReferenceLossConfig(mode="bounded", max_projected_positions=512, checkpoint_chunks=True)
    configure_reference_loss_execution(model, bounded)

    default_config = TrainConfig(model=model.config, data=DataConfig(kind="random"), output_dir=str(tmp_path))
    bounded_config = TrainConfig(
        model=model.config,
        data=DataConfig(kind="random"),
        output_dir=str(tmp_path),
        performance=PerformanceConfig(reference_loss=bounded),
    )
    default_fingerprint = _training_fingerprint(default_config, 1, 16, 16, 1)
    bounded_fingerprint = _training_fingerprint(bounded_config, 1, 16, 16, 1)
    default_payload = _config_payload(default_config)
    bounded_payload = _config_payload(bounded_config)
    metadata = reference_loss_execution_metadata(bounded)

    assert tuple(model.state_dict()) == state_keys
    assert isinstance(default_payload, dict) and isinstance(bounded_payload, dict)
    assert "reference_loss" not in default_payload["performance"]
    assert bounded_payload["performance"]["reference_loss"]["mode"] == "bounded"
    assert "reference_loss_execution" not in default_fingerprint
    assert bounded_fingerprint["reference_loss_execution"]["schema"] == "gcw.reference_loss_execution.v1"
    assert metadata["fingerprint_sha256"] == bounded_fingerprint["reference_loss_execution"]["fingerprint_sha256"]


def test_reference_loss_configuration_fails_closed(tmp_path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ReferenceLossConfig(mode="bounded", max_projected_positions=0)
    with pytest.raises(ValueError, match="requires mode='bounded'"):
        ReferenceLossConfig(checkpoint_chunks=True)
    with pytest.raises(ValueError, match="requires loss_backend='reference'"):
        TrainConfig(
            model=_model(tied=True).config,
            data=DataConfig(kind="random"),
            output_dir=str(tmp_path),
            loss_backend="target_only",
            performance=PerformanceConfig(reference_loss=ReferenceLossConfig(mode="bounded")),
        )
    with pytest.raises(ValueError, match="must cover the maximum physical rows"):
        TrainConfig(
            model=_model(tied=True).config,
            data=DataConfig(kind="random"),
            output_dir=str(tmp_path),
            batch_size=2,
            performance=PerformanceConfig(
                reference_loss=ReferenceLossConfig(mode="bounded", max_projected_positions=1)
            ),
        )
    with pytest.raises(ValueError, match="supports only distributed='none'"):
        TrainConfig(
            model=_model(tied=True).config,
            data=DataConfig(kind="random"),
            output_dir=str(tmp_path),
            distributed="ddp",
            performance=PerformanceConfig(reference_loss=ReferenceLossConfig(mode="bounded")),
        )


def test_training_configures_bounded_reference_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: list[ReferenceLossConfig] = []
    original = transformer_module.bounded_reference_loss

    def record(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(args[-1])
        return original(*args, **kwargs)

    monkeypatch.setattr(transformer_module, "bounded_reference_loss", record)
    monkeypatch.setattr(training_module, "init_distributed", lambda _kind: torch.device("cpu"))
    config = TrainConfig(
        model=_model(tied=True).config,
        data=DataConfig(kind="random"),
        output_dir=str(tmp_path / "bounded-train"),
        batch_size=2,
        sequence_len=16,
        max_steps=1,
        save_final=False,
        eval_every=0,
        eval_on_start=False,
        performance=PerformanceConfig(reference_loss=ReferenceLossConfig(mode="bounded", max_projected_positions=4)),
    )

    training_module.train(config)

    assert calls
    assert all(config.mode == "bounded" for config in calls)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("GCW_RUN_LIGER_CUDA_SMOKE") != "1"
    or not torch.cuda.is_available()
    or not torch.cuda.is_bf16_supported(),
    reason="set GCW_RUN_LIGER_CUDA_SMOKE=1 on a BF16 CUDA host with the GPU extra",
)
def test_liger_cuda_bfloat16_forward_backward_parity_smoke() -> None:
    try:
        require_loss_backend("liger", torch.device("cuda"))
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert metadata.version("liger-kernel") == LIGER_KERNEL_VERSION

    model = _model(tied=False, dtype=torch.bfloat16).cuda()
    hidden = torch.randn(2, 5, model.config.hidden_size, device="cuda", dtype=torch.bfloat16)
    targets = torch.tensor([[0, 1, -100, 10, 2], [3, -1, 4, 5, 6]], device="cuda")

    assert model.lm_head is not None
    head = model.lm_head.weight
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference = model.loss_from_hidden(reference_hidden, targets, "mean", "reference")
    reference.backward()
    assert reference_hidden.grad is not None
    assert head.grad is not None
    reference_hidden_grad = reference_hidden.grad.detach().clone()
    reference_head_grad = head.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    liger_hidden = hidden.detach().clone().requires_grad_(True)
    liger = model.loss_from_hidden(liger_hidden, targets, "mean", "liger")
    liger.backward()

    torch.testing.assert_close(liger, reference, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(liger_hidden.grad, reference_hidden_grad, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(head.grad, reference_head_grad, atol=3e-2, rtol=3e-2)
