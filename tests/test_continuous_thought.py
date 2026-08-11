from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.continuous_thought import (
    ContinuousThoughtEmitter,
    ContinuousThoughtEmitterConfig,
    append_continuous_thoughts,
    continuous_thought_answer_loss,
    frozen_qwen_backbone,
    load_emitter_checkpoint,
    save_emitter_checkpoint,
)
from gcw.models import build_model
from gcw.models.transformer import TF
from gcw.models.windowed_cache import WindowedCache
from gcw.process_branch import branch_next_token_loss


def _config(variant: str = "tf") -> ModelConfig:
    return ModelConfig(
        variant=variant,
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
        cache_processor=(
            CacheProcessorConfig(
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
            )
            if variant == "longshot"
            else None
        ),
    )


def _frozen_backbone(variant: str = "tf") -> tuple[torch.nn.Module, TF]:
    torch.manual_seed(101)
    model = build_model(_config(variant))
    model.init_weights()
    return model, frozen_qwen_backbone(model)


def _emitter(latent_tokens: int = 2) -> ContinuousThoughtEmitter:
    torch.manual_seed(202)
    return ContinuousThoughtEmitter(
        ContinuousThoughtEmitterConfig(
            model_width=32,
            processor_width=24,
            latent_tokens_per_cycle=latent_tokens,
        )
    )


def _example() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[3, 5, 8, 13]]),
        torch.tensor([[21, 22]]),
        torch.tensor([[34, 35, 36]]),
    )


def test_forward_embedding_chunk_is_exact_forward_chunk_inputs_embeds_path() -> None:
    _, backbone = _frozen_backbone()
    ids = torch.tensor([[2, 4, 6, 8]])

    token_hidden, token_cache = backbone.forward_chunk(ids, 0, None, logits=False, cache_format="windowed")
    embed_hidden, embed_cache = backbone.forward_embedding_chunk(
        backbone.embed_tokens(ids), 0, None, logits=False, cache_format="windowed"
    )

    assert isinstance(token_cache, WindowedCache)
    assert isinstance(embed_cache, WindowedCache)
    assert torch.equal(token_hidden, embed_hidden)
    assert torch.equal(token_cache.keys, embed_cache.keys)
    assert torch.equal(token_cache.values, embed_cache.values)
    assert torch.equal(token_cache.scores, embed_cache.scores)
    with pytest.raises(ValueError, match="width"):
        backbone.forward_embedding_chunk(torch.zeros(1, 1, 31), 0)


def test_k_zero_is_exact_skip_and_exact_ordinary_answer_ce() -> None:
    _, backbone = _frozen_backbone()
    emitter = _emitter()
    prompt, prefix, targets = _example()
    calls = 0

    def count_call(_module, _inputs, _output) -> None:
        nonlocal calls
        calls += 1

    handle = emitter.register_forward_hook(count_call)
    result = continuous_thought_answer_loss(
        backbone,
        emitter,
        prompt_ids=prompt,
        response_prefix_ids=prefix,
        target_ids=targets,
        cycles=0,
    )
    handle.remove()

    with torch.no_grad():
        _, cache = backbone.forward_chunk(prompt, 0, None, logits=False, cache_format="windowed")
    assert isinstance(cache, WindowedCache)
    ordinary = branch_next_token_loss(
        backbone,
        cache,
        start_pos=prompt.shape[1],
        response_prefix_ids=prefix,
        target_ids=targets,
    )

    assert calls == 0
    assert result.trace.start_positions == ()
    assert result.trace.cache_lengths == ()
    assert result.answer_start_position == prompt.shape[1]
    assert torch.equal(result.loss, ordinary)


def test_repeated_cycles_append_genuine_positions_without_rewriting_prefix() -> None:
    _, backbone = _frozen_backbone()
    emitter = _emitter(latent_tokens=2)
    prompt, _, _ = _example()
    with torch.no_grad():
        hidden, cache = backbone.forward_chunk(prompt, 0, None, logits=False, cache_format="windowed")
    assert isinstance(cache, WindowedCache)
    old_keys = cache.keys.clone()
    old_values = cache.values.clone()
    old_scores = cache.scores.clone()

    _, thought_cache, trace = append_continuous_thoughts(
        backbone,
        emitter,
        hidden[:, -1],
        cache,
        cycles=3,
    )

    assert trace.start_positions == (4, 6, 8)
    assert trace.cache_lengths == (6, 8, 10)
    assert trace.added_positions == 6
    assert thought_cache.seq_len == 10
    assert torch.equal(thought_cache.keys[:, :, :, :4], old_keys)
    assert torch.equal(thought_cache.values[:, :, :, :4], old_values)
    assert torch.equal(thought_cache.scores[:, :4], old_scores)


def test_answer_ce_trains_only_emitter_and_checkpoint_round_trips(tmp_path: Path) -> None:
    model, backbone = _frozen_backbone(variant="longshot")
    emitter = _emitter(latent_tokens=2)
    prompt, prefix, targets = _example()
    backbone_before = {name: value.clone() for name, value in backbone.state_dict().items()}
    processor_before = {name: value.clone() for name, value in model.cache_processor.state_dict().items()}
    emitter_before = {name: value.clone() for name, value in emitter.state_dict().items()}
    optimizer = torch.optim.AdamW(emitter.parameters(), lr=1e-2)

    result = continuous_thought_answer_loss(
        backbone,
        emitter,
        prompt_ids=prompt,
        response_prefix_ids=prefix,
        target_ids=targets,
        cycles=2,
    )
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)) for parameter in emitter.parameters()
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    optimizer.step()

    assert any(not torch.equal(emitter_before[name], value) for name, value in emitter.state_dict().items())
    assert all(torch.equal(backbone_before[name], value) for name, value in backbone.state_dict().items())
    assert all(torch.equal(processor_before[name], value) for name, value in model.cache_processor.state_dict().items())

    checkpoint = tmp_path / "emitter-checkpoint"
    save_emitter_checkpoint(checkpoint, emitter, train_state={"step": 1, "k": 2})
    restored, train_state = load_emitter_checkpoint(checkpoint)
    assert restored.config == emitter.config
    assert train_state == {"step": 1, "k": 2}
    assert set(restored.state_dict()) == set(emitter.state_dict())
    assert all(torch.equal(restored.state_dict()[name], value) for name, value in emitter.state_dict().items())
    assert set(path.name for path in checkpoint.iterdir()) == {
        "config.json",
        "emitter.safetensors",
        "train_state.json",
    }


def test_source_transplant_changes_thoughts_but_preserves_position_contract() -> None:
    _, backbone = _frozen_backbone()
    emitter = _emitter(latent_tokens=2)
    prompt, prefix, targets = _example()
    donor_prompt = torch.tensor([[13, 8, 5, 3]])
    with torch.no_grad():
        donor_hidden, _ = backbone.forward_chunk(
            donor_prompt,
            0,
            None,
            logits=False,
            cache_format="windowed",
        )
    matched = continuous_thought_answer_loss(
        backbone,
        emitter,
        prompt_ids=prompt,
        response_prefix_ids=prefix,
        target_ids=targets,
        cycles=2,
    )
    transplanted = continuous_thought_answer_loss(
        backbone,
        emitter,
        prompt_ids=prompt,
        response_prefix_ids=prefix,
        target_ids=targets,
        cycles=2,
        source_override=donor_hidden[:, -1],
    )
    assert matched.trace == transplanted.trace
    assert matched.answer_start_position == transplanted.answer_start_position
    assert not torch.equal(matched.loss, transplanted.loss)
    transplanted.loss.backward()
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)) for parameter in emitter.parameters()
    )


def test_cli_exposes_required_experiment_controls() -> None:
    import importlib.util

    script = Path(__file__).parents[1] / "scripts" / "explore_continuous_thought.py"
    spec = importlib.util.spec_from_file_location("explore_continuous_thought", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(
        [
            "--output",
            "run",
            "--k",
            "3",
            "--latent-tokens-per-cycle",
            "2",
            "--steps",
            "7",
            "--lr",
            "0.001",
            "--train-rows",
            "11",
            "--heldout-rows",
            "5",
        ]
    )
    assert (args.k, args.latent_tokens_per_cycle, args.steps) == (3, 2, 7)
    assert args.lr == 0.001
    assert (args.train_rows, args.heldout_rows) == (11, 5)
