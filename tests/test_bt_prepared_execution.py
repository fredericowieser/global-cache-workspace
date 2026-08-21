from __future__ import annotations

import copy

import pytest
import torch

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models import build_model
from gcw.models.cache_processors import standard as standard_module
from gcw.models.cache_processors.state import CacheState
from gcw.models.model import LanguageModel


def _config(*, update_stream: str = "kv") -> ModelConfig:
    return ModelConfig(
        variant="bt",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=32,
        tie_embeddings=True,
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=2,
            hidden_size=8,
            intermediate_size=16,
            num_layers=2,
            num_heads=1,
            num_kv_heads=1,
            attention_dropout=0.0,
            update_stream=update_stream,
            gate_mode="direct",
            gate_init=0.5,
        ),
    )


def _model(*, update_stream: str = "kv") -> LanguageModel:
    model = build_model(_config(update_stream=update_stream))
    assert isinstance(model, LanguageModel)
    model.init_weights()
    return model


def _cache(config: ModelConfig, *, seq_len: int = 9) -> CacheState:
    shape = (2, config.num_kv_heads, seq_len, int(config.head_dim))
    return CacheState(
        [torch.randn(shape) for _ in range(config.cache_num_layers)],
        [torch.randn(shape) for _ in range(config.cache_num_layers)],
        torch.randn(shape[0], seq_len),
    )


def _clone_cache(cache: CacheState) -> CacheState:
    return CacheState(
        [key.clone() for key in cache.keys],
        [value.clone() for value in cache.values],
        cache.scores.clone(),
    )


def _assert_cache_close(actual: CacheState, expected: CacheState) -> None:
    for actual_key, expected_key in zip(
        actual.keys, expected.keys, strict=True
    ):
        torch.testing.assert_close(actual_key, expected_key)
    for actual_value, expected_value in zip(
        actual.values, expected.values, strict=True
    ):
        torch.testing.assert_close(actual_value, expected_value)
    torch.testing.assert_close(actual.scores, expected.scores)


def _eager_forward(
    model: LanguageModel,
    input_ids: torch.Tensor,
    targets: torch.Tensor | None = None,
) -> torch.Tensor:
    processor = model.config.cache_processor
    assert processor is not None
    return model.backbone(
        input_ids,
        targets,
        processor=model.cache_processor,
        process_every=processor.process_every,
    )


def test_bt_prepared_forward_matches_eager_forward_backward_and_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(11)
    prepared_model = _model()
    eager_model = copy.deepcopy(prepared_model)
    input_ids = torch.randint(0, prepared_model.config.vocab_size, (2, 13))
    targets = torch.randint(0, prepared_model.config.vocab_size, (2, 13))

    prepare_calls = 0
    original_prepare = (
        standard_module.prepare_batched_cache_processor_parameters
    )

    def counted_prepare(blocks):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(blocks)

    monkeypatch.setattr(
        standard_module,
        "prepare_batched_cache_processor_parameters",
        counted_prepare,
    )
    prepared_loss = prepared_model(input_ids, targets)
    eager_loss = _eager_forward(eager_model, input_ids, targets)
    prepared_loss.backward()
    eager_loss.backward()

    torch.testing.assert_close(prepared_loss, eager_loss)
    assert prepare_calls == 1
    assert prepared_model.cache_processor.pop_call_count() == 3
    assert eager_model.cache_processor.pop_call_count() == 3
    for (prepared_name, prepared_parameter), (
        eager_name,
        eager_parameter,
    ) in zip(
        prepared_model.named_parameters(),
        eager_model.named_parameters(),
        strict=True,
    ):
        assert prepared_name == eager_name
        assert prepared_parameter.grad is not None, prepared_name
        assert eager_parameter.grad is not None, eager_name
        torch.testing.assert_close(
            prepared_parameter.grad, eager_parameter.grad, atol=2e-5, rtol=2e-4
        )


@pytest.mark.parametrize("update_stream", ["kv", "v"])
def test_bt_prepared_cache_matches_eager_and_reuses_one_preparation(
    update_stream: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(17)
    prepared_processor = _model(update_stream=update_stream).cache_processor
    eager_processor = copy.deepcopy(prepared_processor)
    source_caches = [
        _cache(_config(update_stream=update_stream)) for _ in range(2)
    ]
    actual_caches = [_clone_cache(cache) for cache in source_caches]
    expected_caches = [_clone_cache(cache) for cache in source_caches]

    prepare_calls = 0
    original_prepare = (
        standard_module.prepare_batched_cache_processor_parameters
    )

    def counted_prepare(blocks):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(blocks)

    monkeypatch.setattr(
        standard_module,
        "prepare_batched_cache_processor_parameters",
        counted_prepare,
    )
    with prepared_processor.prepare_forward_execution() as execution:
        for cache in actual_caches:
            execution(cache)
    for cache in expected_caches:
        eager_processor(cache)

    assert prepare_calls == 1
    for actual, expected in zip(actual_caches, expected_caches, strict=True):
        _assert_cache_close(actual, expected)


def test_bt_repeated_training_forwards_do_not_reuse_a_stale_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(23)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 9))
    targets = torch.randint(0, model.config.vocab_size, (2, 9))
    executions = []
    original_factory = type(model.cache_processor).prepare_forward_execution

    def capture_execution(processor):
        execution = original_factory(processor)
        executions.append(execution)
        return execution

    monkeypatch.setattr(
        type(model.cache_processor),
        "prepare_forward_execution",
        capture_execution,
    )
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())

    assert len(executions) == 2
    assert executions[0] is not executions[1]
    assert all(
        execution._closed and execution._prepared is None
        for execution in executions
    )
    assert all(torch.isfinite(loss) for loss in losses)
    with pytest.raises(RuntimeError, match="cannot be reused across forwards"):
        executions[0](_cache(model.config))


def test_bt_preparation_preserves_parameter_and_state_dict_identity() -> None:
    torch.manual_seed(29)
    model = _model()
    parameter_ids = {
        name: id(parameter) for name, parameter in model.named_parameters()
    }
    state_before = {
        name: tensor.clone() for name, tensor in model.state_dict().items()
    }
    state_keys = tuple(state_before)

    model(torch.randint(0, model.config.vocab_size, (1, 9)))

    assert {
        name: id(parameter) for name, parameter in model.named_parameters()
    } == parameter_ids
    assert tuple(model.state_dict()) == state_keys
    assert all("prepared" not in name for name in state_keys)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, state_before[name]), name
    restored = _model()
    incompatible = restored.load_state_dict(state_before, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_bt_eval_forward_matches_eager_and_generation_keeps_eager_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(31)
    model = _model().eval()
    eager_model = copy.deepcopy(model)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 9))

    with torch.no_grad():
        prepared_logits = model(input_ids)
        eager_logits = _eager_forward(eager_model, input_ids)
    torch.testing.assert_close(prepared_logits, eager_logits)

    def fail_prepare():
        pytest.fail("generation must use the eager processor fallback")

    monkeypatch.setattr(
        model.cache_processor, "prepare_forward_execution", fail_prepare
    )
    eager_calls = 0
    original_eager = standard_module.run_batched_cache_processors

    def counted_eager(blocks, x, positions):
        nonlocal eager_calls
        eager_calls += 1
        return original_eager(blocks, x, positions)

    monkeypatch.setattr(
        standard_module, "run_batched_cache_processors", counted_eager
    )
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=1)
        generated_fast = model.generate_fast(input_ids, max_new_tokens=1)

    assert generated.shape == generated_fast.shape == (1, 10)
    assert torch.equal(generated, generated_fast)
    assert eager_calls > 0
