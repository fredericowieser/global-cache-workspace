from __future__ import annotations

import pytest
import torch

from benchmarks.parity import (
    assert_cache_close,
    assert_parameter_gradients_close,
    assert_tensor_gradients_close,
    cache_objective,
    deterministic_cache,
    patch_eager_reference,
    run_processor_parity_path,
    small_cpu_config,
)
from gcw.models import build_model


@pytest.mark.parametrize("variant", ("bt", "bgt"))
def test_processor_eager_parity_covers_state_indices_inputs_and_parameter_gradients(variant: str):
    torch.manual_seed(2025)
    config = small_cpu_config(variant)
    candidate = build_model(config)
    candidate.init_weights()
    reference = build_model(config)
    reference.load_state_dict(candidate.state_dict())
    source = deterministic_cache(config, requires_grad=False)

    actual = run_processor_parity_path(candidate.cache_processor, source, eager=False)
    expected = run_processor_parity_path(reference.cache_processor, source, eager=True)
    actual_loss = cache_objective(actual.cache)
    expected_loss = cache_objective(expected.cache)
    actual_loss.backward()
    expected_loss.backward()

    assert torch.equal(actual.selected_indices, expected.selected_indices)
    assert actual.selected_indices.tolist() == [[1, 3, 4, 5, 6], [0, 2, 4, 5, 6]]
    assert_cache_close(actual.cache, expected.cache)
    assert_tensor_gradients_close(actual.input_keys, expected.input_keys)
    assert_tensor_gradients_close(actual.input_values, expected.input_values)
    assert_parameter_gradients_close(candidate.cache_processor, reference.cache_processor)


@pytest.mark.parametrize("variant", ("bt", "bgt"))
def test_complete_model_loss_and_processor_gradients_match_eager_reference(variant: str):
    torch.manual_seed(17)
    config = small_cpu_config(variant)
    candidate = build_model(config)
    candidate.init_weights()
    reference = build_model(config)
    reference.load_state_dict(candidate.state_dict())
    patch_eager_reference(reference)
    generator = torch.Generator().manual_seed(23)
    input_ids = torch.randint(0, config.vocab_size, (2, 10), generator=generator)
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    targets[:, -1] = -100

    actual_loss = candidate(input_ids, targets)
    expected_loss = reference(input_ids, targets)
    actual_loss.backward()
    expected_loss.backward()

    torch.testing.assert_close(actual_loss, expected_loss, atol=1e-5, rtol=1e-5)
    assert_parameter_gradients_close(candidate.cache_processor, reference.cache_processor)
