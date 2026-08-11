from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from gcw.config import CacheProcessorConfig, ModelConfig, WorkspaceFlowRefinementConfig
from gcw.continuous_thought import frozen_qwen_backbone, generate_continuous_thought_answer
from gcw.models import build_model
from gcw.models.windowed_cache import WindowedCache
from gcw.process_branch import branch_next_token_loss
from gcw.unified_flow_position import (
    UnifiedFlowPositionConfig,
    build_unified_flow_position_processor,
    generate_unified_answer,
    load_unified_weights,
    save_unified_checkpoint,
    unified_answer_loss,
)


def _model_config() -> ModelConfig:
    return ModelConfig(
        variant="tf",
        family="qwen3",
        sequence_len=64,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        rope_theta=10_000.0,
        norm_eps=1e-6,
        tie_embeddings=False,
        qk_norm=True,
        rotary_style="llama",
    )


def _cache_config() -> CacheProcessorConfig:
    return CacheProcessorConfig(
        process_every=4,
        topk=0,
        topk_heads=4,
        score="kv_norm",
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        encoder_method="query_attention",
        encoder_query_source="final_token",
        update_stream="v",
        out_proj_init="normal",
        gate_init=0.0,
        unrope_keys=True,
    )


def _setup():
    torch.manual_seed(108)
    model = build_model(_model_config())
    model.init_weights()
    backbone = frozen_qwen_backbone(model)
    hybrid = UnifiedFlowPositionConfig(
        model_width=32,
        workspace_cycles=2,
        position_cycles=2,
        latent_tokens_per_cycle=2,
        emitter_width=24,
        source_delta_bound=0.05,
    )
    processor = build_unified_flow_position_processor(
        backbone.config,
        cache_processor=_cache_config(),
        hybrid=hybrid,
        flow_refinement=WorkspaceFlowRefinementConfig(
            solver="midpoint",
            solver_steps=2,
            max_velocity=0.1,
        ),
    )
    return model, backbone, processor


def _example():
    return (
        torch.tensor([[3, 5, 8, 13]]),
        torch.tensor([[21, 22]]),
        torch.tensor([[34, 35, 36]]),
    )


def test_off_is_exact_object_identity_skip_and_ordinary_ce() -> None:
    _, backbone, processor = _setup()
    prompt, prefix, targets = _example()
    with torch.no_grad():
        hidden, cache = backbone.forward_chunk(prompt, 0, None, logits=False, cache_format="windowed")
    assert isinstance(cache, WindowedCache)
    calls = 0

    def count_call(_module, _inputs, _output) -> None:
        nonlocal calls
        calls += 1

    handles = [
        processor.workspace_processor.register_forward_hook(count_call),
        processor.emitter.register_forward_hook(count_call),
    ]
    final_hidden = hidden[:, -1]
    readout, skipped, trace = processor.process_cache(
        backbone, final_hidden, cache, active=False
    )
    result = unified_answer_loss(
        backbone,
        processor,
        prompt_ids=prompt,
        response_prefix_ids=prefix,
        target_ids=targets,
        active=False,
    )
    for handle in handles:
        handle.remove()
    ordinary = branch_next_token_loss(
        backbone,
        cache,
        start_pos=prompt.shape[1],
        response_prefix_ids=prefix,
        target_ids=targets,
    )

    assert skipped is cache
    assert readout is final_hidden
    assert calls == 0
    assert not trace.active and trace.added_positions == 0
    assert result.answer_start_position == prompt.shape[1]
    assert torch.equal(result.loss, ordinary)


def test_active_path_jointly_edits_values_and_adds_bounded_causal_positions() -> None:
    _, backbone, processor = _setup()
    prompt, _, _ = _example()
    with torch.no_grad():
        hidden, cache = backbone.forward_chunk(prompt, 0, None, logits=False, cache_format="windowed")
    assert isinstance(cache, WindowedCache)
    old_keys = cache.keys.clone()
    old_values = cache.values.clone()

    _, processed, trace = processor.process_cache(
        backbone, hidden[:, -1], cache, active=True
    )

    assert trace.active and trace.source_mode == "matched"
    assert trace.continuous.start_positions == (4, 6)
    assert trace.continuous.cache_lengths == (6, 8)
    assert trace.added_positions == 4 and processed.seq_len == 8
    assert trace.workspace_flow_max_displacement <= 0.1 + 1e-6
    assert trace.source_delta_max_abs <= 0.05 + 1e-6
    assert torch.equal(processed.keys[:, :, :, :4], old_keys)
    assert not torch.equal(processed.values[:, :, :, :4], old_values)


def test_generation_off_has_ordinary_parity_and_active_accounts_for_positions() -> None:
    _, backbone, processor = _setup()
    prompt, prefix, _ = _example()
    ordinary = generate_continuous_thought_answer(
        backbone,
        processor.emitter,
        prompt_ids=prompt[0],
        response_prefix_ids=prefix[0],
        cycles=0,
        max_new_tokens=3,
        eos_token_id=None,
        valid_vocab_size=64,
    )
    off = generate_unified_answer(
        backbone,
        processor,
        prompt_ids=prompt[0],
        response_prefix_ids=prefix[0],
        active=False,
        max_new_tokens=3,
        eos_token_id=None,
        valid_vocab_size=64,
    )
    active = generate_unified_answer(
        backbone,
        processor,
        prompt_ids=prompt[0],
        response_prefix_ids=prefix[0],
        active=True,
        max_new_tokens=3,
        eos_token_id=None,
        valid_vocab_size=64,
    )
    with torch.no_grad():
        full_hidden, _ = backbone.forward_chunk(
            prompt,
            0,
            None,
            logits=False,
            cache_format="windowed",
        )
        expected_first = torch.argmax(backbone.logits_from_hidden(full_hidden[:, -1:])[:, -1], dim=-1)
    template_off = generate_unified_answer(
        backbone,
        processor,
        prompt_ids=prompt[0],
        response_prefix_ids=torch.empty(0, dtype=torch.long),
        active=False,
        max_new_tokens=1,
        eos_token_id=None,
        valid_vocab_size=64,
    )

    assert torch.equal(off.token_ids, ordinary.token_ids)
    assert torch.equal(template_off.token_ids, expected_first)
    assert off.answer_start_position == ordinary.answer_start_position == prompt.shape[1]
    assert active.answer_start_position == prompt.shape[1] + 4
    assert active.trace.added_positions == 4


def test_shuffled_source_uses_donor_content_but_recipient_write_positions() -> None:
    _, backbone, processor = _setup()
    prompt, prefix, targets = _example()
    donor_prompt = torch.tensor([[55, 3, 2, 1, 7, 9]])
    donor = processor.prepare_source(backbone, donor_prompt)
    with torch.no_grad():
        _, recipient_cache = backbone.forward_chunk(
            prompt, 0, None, logits=False, cache_format="windowed"
        )
    assert isinstance(recipient_cache, WindowedCache)
    recipient_source = processor.workspace_processor.encode_source(recipient_cache, select_all=True)

    matched = unified_answer_loss(
        backbone,
        processor,
        prompt_ids=prompt,
        response_prefix_ids=prefix,
        target_ids=targets,
        active=True,
    )
    shuffled = unified_answer_loss(
        backbone,
        processor,
        prompt_ids=prompt,
        response_prefix_ids=prefix,
        target_ids=targets,
        active=True,
        source_override=donor,
    )

    assert shuffled.trace.source_mode == "shuffled"
    assert torch.equal(shuffled.trace.edited_positions, recipient_source.latest_indices)
    assert shuffled.answer_start_position == matched.answer_start_position
    assert not torch.equal(shuffled.loss, matched.loss)


def test_only_hybrid_trains_and_processor_only_checkpoint_round_trips(tmp_path: Path) -> None:
    model, backbone, processor = _setup()
    prompt, prefix, targets = _example()
    backbone_before = {name: value.clone() for name, value in backbone.state_dict().items()}
    processor_before = {name: value.clone() for name, value in processor.state_dict().items()}
    optimizer = torch.optim.AdamW(processor.parameters(), lr=1e-2)

    result = unified_answer_loss(
        backbone,
        processor,
        prompt_ids=prompt,
        response_prefix_ids=prefix,
        target_ids=targets,
        active=True,
    )
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in processor.parameters()
    )
    optimizer.step()
    assert all(torch.equal(backbone_before[name], value) for name, value in backbone.state_dict().items())
    assert any(not torch.equal(processor_before[name], value) for name, value in processor.state_dict().items())

    checkpoint = tmp_path / "hybrid"
    construction = {
        "cache_processor": asdict(_cache_config()),
        "flow_refinement": {"solver": "midpoint", "solver_steps": 2, "max_velocity": 0.1},
    }
    save_unified_checkpoint(
        checkpoint,
        processor,
        construction=construction,
        train_state={"step": 1},
    )
    _, _, restored = _setup()
    state = load_unified_weights(checkpoint, restored)
    assert state == {"step": 1}
    assert restored.trainable_parameter_count == processor.trainable_parameter_count
    assert all(torch.equal(restored.state_dict()[name], value) for name, value in processor.state_dict().items())
    assert {path.name for path in checkpoint.iterdir()} == {
        "config.json",
        "processor.safetensors",
        "train_state.json",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"workspace_cycles": -1},
        {"position_cycles": -1},
        {"latent_tokens_per_cycle": 0},
        {"source_delta_bound": 0.0},
        {"source_delta_bound": float("inf")},
    ],
)
def test_config_rejects_unstable_or_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        UnifiedFlowPositionConfig(model_width=32, **kwargs)  # type: ignore[arg-type]
