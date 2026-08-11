from __future__ import annotations

import sys
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
if str(SPIKE_DIR) not in sys.path:
    sys.path.insert(0, str(SPIKE_DIR))

from continuous_thought_positions import run_checks  # noqa: E402


def test_continuous_thought_positions_feasibility() -> None:
    result = run_checks()

    assert result.backbone_parameters_frozen
    assert result.no_op_cache_identity
    assert result.no_op_logit_max_abs_delta == 0.0
    assert result.old_cache_prefix_unchanged_after_append
    assert result.one_step_cache_length == result.prompt_length + 1
    assert result.one_step_start_positions == (result.prompt_length,)
    assert result.one_step_vs_no_thought_logit_max_abs_delta > 0.0
    assert result.one_step_vs_zero_latent_logit_max_abs_delta > 0.0
    assert result.repeated_start_positions == tuple(
        range(result.prompt_length, result.prompt_length + result.repeated_k)
    )
    assert result.repeated_final_cache_length == result.prompt_length + result.repeated_k + 1
    assert result.repeated_vs_one_step_logit_max_abs_delta > 0.0
    assert result.emitter_grad_l1 > 0.0
    assert result.emitter_parameters_with_nonzero_grad > 0
    assert result.backbone_parameters_with_grad == 0
