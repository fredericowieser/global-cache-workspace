from __future__ import annotations

import copy
import os

import pytest
import torch

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models import build_model
from gcw.models.model import LanguageModel
from gcw.performance.config import TritonCacheMovementConfig
from gcw.performance.triton.execution import (
    TritonCacheExecutionContext,
    configure_triton_cache_movement,
)
from gcw.training.core import _apply_trainable_policy

pytestmark = [pytest.mark.gpu, pytest.mark.performance]


def _require_cuda_gate() -> None:
    if os.environ.get("GCW_RUN_TRITON_CACHE_TRAINING_TESTS") != "1":
        pytest.skip("set GCW_RUN_TRITON_CACHE_TRAINING_TESTS=1 for full-model Triton tests")
    if os.environ.get("GCW_ALLOW_UNQUALIFIED_TRITON_CACHE") != "1":
        pytest.skip("explicit unqualified Triton qualification override is required")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("full-model Triton tests require H100/H200")


def _config(variant: str) -> ModelConfig:
    return ModelConfig(
        variant=variant,  # type: ignore[arg-type]
        family="qwen3",
        qk_norm=True,
        rotary_style="llama",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        sequence_len=128,
        cache_processor=CacheProcessorConfig(
            process_every=64,
            topk=8,
            topk_heads=2,
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            gate_mode="direct",
            gate_init=0.5,
            attention_dropout=0.0,
            update_stream="kv",
            encoder_seq_len=64,
        ),
    )


def _configure(model: LanguageModel) -> None:
    configure_triton_cache_movement(
        model,
        TritonCacheMovementConfig(enabled=True),
        context=TritonCacheExecutionContext(
            model_variant=model.config.variant,
            model_family=model.config.family,
            physical_batch_size=2,
            distributed="none",
            trainable="cache_processor",
            bf16=True,
            backbone_chunk_backend="eager",
            adaptive_batching_enabled=False,
            whole_model_compile_requested=False,
        ),
    )


@pytest.mark.parametrize("variant", ("bt", "bgt", "longshot", "icarus"))
def test_triton_cache_training_forward_gradient_and_adam_match_eager(variant: str) -> None:
    _require_cuda_gate()
    torch.manual_seed(123)
    eager = build_model(_config(variant)).cuda()
    assert isinstance(eager, LanguageModel)
    eager.init_weights()
    candidate = copy.deepcopy(eager)
    _apply_trainable_policy(eager, "cache_processor")
    _apply_trainable_policy(candidate, "cache_processor")
    _configure(candidate)
    eager.train()
    candidate.train()

    generator = torch.Generator(device="cuda").manual_seed(456)
    inputs = torch.randint(0, 64, (2, 128), device="cuda", generator=generator)
    targets = torch.randint(0, 64, (2, 128), device="cuda", generator=generator)
    eager_parameters = tuple(parameter for parameter in eager.parameters() if parameter.requires_grad)
    candidate_parameters = tuple(parameter for parameter in candidate.parameters() if parameter.requires_grad)
    eager_optimizer = torch.optim.AdamW(eager_parameters, lr=1e-3)
    candidate_optimizer = torch.optim.AdamW(candidate_parameters, lr=1e-3)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager_loss = eager(inputs, targets, loss_reduction="row_sum").sum()
        candidate_loss = candidate(inputs, targets, loss_reduction="row_sum").sum()
    torch.testing.assert_close(candidate_loss, eager_loss, atol=0.0, rtol=0.0)
    eager_loss.backward()
    candidate_loss.backward()
    for eager_parameter, candidate_parameter in zip(
        eager_parameters, candidate_parameters, strict=True
    ):
        if eager_parameter.grad is None or candidate_parameter.grad is None:
            assert eager_parameter.grad is candidate_parameter.grad is None
        else:
            torch.testing.assert_close(
                candidate_parameter.grad,
                eager_parameter.grad,
                atol=0.0,
                rtol=0.0,
            )
    eager_optimizer.step()
    candidate_optimizer.step()
    for eager_parameter, candidate_parameter in zip(
        eager_parameters, candidate_parameters, strict=True
    ):
        torch.testing.assert_close(candidate_parameter, eager_parameter, atol=0.0, rtol=0.0)
